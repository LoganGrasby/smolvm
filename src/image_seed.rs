//! Shared storage seeds for registry images.
//!
//! A machine created with `--image` pulls that image inside the guest on its
//! first start: the manifest, config and every layer come over the network and
//! are extracted onto the machine's own storage disk. Nothing is shared, so every
//! new machine of the same image repeats the whole pull (seconds, and two
//! rate-limited manifest GETs against Docker Hub).
//!
//! A seed is that pulled state, captured once per image: the storage disk of a
//! throwaway machine that pulled the image and never ran a workload. It is a
//! read-only qcow2 over the storage template. A new machine's `storage.qcow2` is
//! created as a copy-on-write overlay on the seed instead of on the bare template,
//! so its guest finds the image already present and its first start skips the
//! pull, exactly as a restart does.
//!
//! The key includes the digest the image reference points to, resolved on the
//! host with the caller's credentials on every first start (one manifest HEAD,
//! the same registry authorization a pull gets). A moved tag gets a new seed, and
//! a caller the registry refuses never reaches a cached one. Portable checkpoints
//! copy the whole disk chain, so they do not depend on a seed staying on disk.
//! Seeding is best-effort: any failure falls back to the in-guest pull.
//! `SMOLVM_IMAGE_SEEDS=0` turns it off.

#[cfg(target_os = "linux")]
pub use linux::{
    revalidate_seed, seed_root, seed_storage, seedable_image, wants_seed, SEED_MACHINE_PREFIX,
};

/// Seeds need the Linux storage-template overlay; elsewhere nothing seeds.
#[cfg(not(target_os = "linux"))]
pub fn wants_seed(_: &str, _: &crate::config::VmRecord, _: bool) -> Option<String> {
    None
}

/// Seeds need the Linux storage-template overlay; elsewhere nothing seeds.
#[cfg(not(target_os = "linux"))]
pub fn seedable_image(_: &str, _: Option<&str>, _: Option<u64>) -> Option<String> {
    None
}

/// Seeds need the Linux storage-template overlay; elsewhere nothing seeds.
#[cfg(not(target_os = "linux"))]
pub fn seed_storage(
    _: &std::path::Path,
    _: &str,
    _: &str,
    _: &crate::registry::PullAuth,
    _: Option<&str>,
    _: Option<&str>,
) -> crate::Result<bool> {
    Ok(false)
}

#[cfg(not(target_os = "linux"))]
/// Seeds only exist on Linux, so other platforms have nothing to revalidate.
pub fn revalidate_seed(_: &str, _: &str, _: &crate::registry::PullAuth) -> crate::Result<bool> {
    Ok(false)
}

#[cfg(target_os = "linux")]
mod linux {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    use std::path::{Path, PathBuf};

    use sha2::{Digest, Sha256};

    use crate::config::VmRecord;
    use crate::registry::PullAuth;
    use crate::storage::DiskFormat;
    use crate::{Error, Result};

    /// Bumped when the guest's storage layout changes in a way an old seed would not
    /// satisfy.
    const SEED_FORMAT: &str = "image-seed-v2";

    /// Name prefix of the throwaway machines that build seeds.
    pub const SEED_MACHINE_PREFIX: &str = "image-seed-";

    /// A builder machine older than this is left over from a crashed build.
    const STALE_BUILDER_SECS: u64 = 30 * 60;

    /// Default cap on the seed cache before unreferenced seeds are evicted;
    /// `SMOLVM_IMAGE_SEED_MAX_BYTES` overrides it.
    const DEFAULT_MAX_BYTES: u64 = 20 * 1024 * 1024 * 1024;

    /// The image a machine's first start can seed from, or `None` when it cannot
    /// (not a fresh registry-image machine, a clone, a checkpoint restore) or
    /// seeding is off.
    pub fn wants_seed(name: &str, record: &VmRecord, from_snapshot: bool) -> Option<String> {
        if record.init_completed
            || from_snapshot
            || record.source_smolmachine.is_some()
            || record.vm_uid_owner().is_some()
            || record.golden.is_some()
        {
            return None;
        }
        seedable_image(name, record.image.as_deref(), record.storage_gb)
    }

    /// `image` when a machine called `name` with that image and storage size can
    /// start on a seed: a registry image, the default storage size, no storage
    /// disk yet, and seeding on.
    pub fn seedable_image(
        name: &str,
        image: Option<&str>,
        storage_gb: Option<u64>,
    ) -> Option<String> {
        // The builder's own machine pulls the normal way.
        if name.starts_with(SEED_MACHINE_PREFIX)
            || std::env::var("SMOLVM_IMAGE_SEEDS").is_ok_and(|v| v.trim() == "0")
            || storage_gb.is_some_and(|gb| gb != crate::storage::DEFAULT_STORAGE_SIZE_GIB)
        {
            return None;
        }
        let image = image?;
        if crate::data::image_source::is_local_ref(image)
            || crate::data::image_source::packed_layers_dir_for_ref(image).is_some()
        {
            return None;
        }
        // An existing disk already holds whatever it pulled.
        let storage = crate::agent::vm_data_dir(name).join(crate::storage::STORAGE_DISK_FILENAME);
        if storage.exists()
            || storage
                .with_extension(DiskFormat::Qcow2.extension())
                .exists()
        {
            return None;
        }
        Some(image.to_string())
    }

    /// Give machine `name` a storage disk over the seed for `image`, building the
    /// seed first (with `exe`, this smolvm binary) if its digest has none yet.
    /// Returns `Ok(false)` when the storage template cannot back a seed.
    pub fn seed_storage(
        exe: &Path,
        name: &str,
        image: &str,
        auth: &PullAuth,
        proxy: Option<&str>,
        no_proxy: Option<&str>,
    ) -> Result<bool> {
        let Some(template) = storage_template() else {
            return Ok(false);
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::config("image seed", e.to_string()))?;
        let resolve = || rt.block_on(crate::image_store::authorized_reference_digest(image, auth));
        let digest = resolve()?;
        let key = seed_key(image, &digest, &template)?;
        let root = seed_root();
        std::fs::create_dir_all(&root).map_err(|e| Error::config("image seed", e.to_string()))?;
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o711))
            .map_err(|e| Error::config("image seed", e.to_string()))?;
        let seed = root.join(&key).join("storage.qcow2");
        let mut built = false;
        // A cache hit holds the shared lock through overlay publication. Prune
        // takes the exclusive lock before scanning backing references, so it
        // cannot miss an overlay being created and then delete its base.
        let mut cache = CacheLock::shared(&root)?;
        if !seed.exists() {
            drop(cache);
            // One builder per image; concurrent first starts wait for it and
            // then share it. This lock file is permanent: unlinking a locked
            // file would let new callers lock a different inode for this key.
            let _build = Lock::exclusive(&root.join(format!("{key}.lock")))?;
            if !seed.exists() {
                build_seed(exe, image, auth, &key, &seed, proxy, no_proxy)?;
                // The builder pulled the tag, not the digest. If the tag moved
                // in the meantime, discard the seed rather than miskey it.
                if resolve()? != digest {
                    let _ = std::fs::remove_dir_all(seed.parent().expect("seed path has a parent"));
                    return Err(Error::config(
                        "image seed",
                        format!("{image} moved during the seed build"),
                    ));
                }
                built = true;
            }
            cache = CacheLock::shared(&root)?;
        }
        if !seed.exists() {
            return Err(Error::config(
                "image seed",
                "seed disappeared before overlay creation",
            ));
        }
        let dir = crate::agent::ensure_vm_dir(name)
            .map_err(|e| Error::config("image seed", e.to_string()))?;
        let storage = dir
            .join(crate::storage::STORAGE_DISK_FILENAME)
            .with_extension(DiskFormat::Qcow2.extension());
        // Build under a unique name and publish with a no-replace hard link.
        // A failed libkrun call cannot leave a partial final disk, and a
        // concurrent start cannot have its finished disk removed or replaced.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let staging = dir.join(format!(
            ".seed-overlay-{}-{nonce}.qcow2",
            std::process::id()
        ));
        let created = crate::agent::create_disk_overlays(&[(
            staging.clone(),
            seed.clone(),
            DiskFormat::Qcow2,
        )]);
        if let Err(error) = created {
            let _ = std::fs::remove_file(staging);
            return Err(error);
        }
        let published = std::fs::hard_link(&staging, &storage)
            .map_err(|e| Error::config("image seed", e.to_string()));
        let _ = std::fs::remove_file(staging);
        published?;
        // Recently used seeds are the last to be evicted.
        let _ =
            std::fs::File::open(&seed).and_then(|f| f.set_modified(std::time::SystemTime::now()));
        drop(cache);
        if built {
            prune(&root, max_bytes(), &seed);
        }
        Ok(true)
    }

    /// Reauthorize a seed attached at API create with the credentials supplied
    /// at start. A moved tag or denied request discards the untouched overlay;
    /// the start path can then seed again or let the guest pull normally.
    pub fn revalidate_seed(name: &str, image: &str, auth: &PullAuth) -> Result<bool> {
        let storage = crate::agent::vm_data_dir(name)
            .join(crate::storage::STORAGE_DISK_FILENAME)
            .with_extension(DiskFormat::Qcow2.extension());
        let Some(backing) = qcow2_backing(&storage) else {
            return Ok(false);
        };
        let root = seed_root();
        let Some(key_dir) = backing.parent() else {
            return Ok(false);
        };
        if key_dir.parent() != Some(root.as_path())
            || backing.file_name().is_none_or(|n| n != "storage.qcow2")
        {
            return Ok(false);
        }
        let expected = (|| -> Result<PathBuf> {
            let template = storage_template()
                .ok_or_else(|| Error::config("image seed", "storage template unavailable"))?;
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| Error::config("image seed", e.to_string()))?;
            let digest =
                rt.block_on(crate::image_store::authorized_reference_digest(image, auth))?;
            Ok(root
                .join(seed_key(image, &digest, &template)?)
                .join("storage.qcow2"))
        })();
        match expected {
            Ok(expected)
                if expected.canonicalize().ok() == backing.canonicalize().ok()
                    && backing.is_file() =>
            {
                Ok(true)
            }
            result => {
                // Only a fresh machine's storage overlay is passed here. Do not
                // leave unauthorized or stale image contents for the guest.
                std::fs::remove_file(&storage)
                    .map_err(|e| Error::config("image seed", e.to_string()))?;
                result?;
                Ok(false)
            }
        }
    }

    /// Pull `image` once in a throwaway machine that never runs a workload, and
    /// publish its storage disk as the seed.
    fn build_seed(
        exe: &Path,
        image: &str,
        auth: &PullAuth,
        key: &str,
        seed: &Path,
        proxy: Option<&str>,
        no_proxy: Option<&str>,
    ) -> Result<()> {
        reap_stale_builders(exe);
        let tmp = format!("{SEED_MACHINE_PREFIX}{}-{}", &key[..16], std::process::id());
        let _ = run(exe, &["machine", "delete", "--name", &tmp, "-f"]);
        let staging = seed.with_extension(format!("qcow2.{}", std::process::id()));
        let started = std::time::Instant::now();
        let built = (|| -> Result<()> {
            run(
                exe,
                &[
                    "machine", "create", "--name", &tmp, "--image", image, "--net",
                ],
            )?;
            let mut start = vec!["machine", "start", "--name", &tmp, "--no-workload"];
            if let Some(proxy) = proxy {
                start.extend(["--proxy", proxy]);
            }
            if let Some(no_proxy) = no_proxy {
                start.extend(["--no-proxy", no_proxy]);
            }
            run(exe, &start)?;
            // Compare the guest's stored config and layer digests to the
            // caller-authorized platform manifest. A second HEAD alone misses
            // tag ABA and a registry mirror serving different image bytes.
            let manager = crate::agent::AgentManager::for_vm(&tmp)?;
            let mut client = manager.connect()?;
            let pulled = client
                .query(image)?
                .ok_or_else(|| Error::config("image seed", "builder did not cache its image"))?;
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| Error::config("image seed", e.to_string()))?;
            let (config, layers) =
                rt.block_on(crate::image_store::authorized_image_content(image, auth))?;
            if pulled.digest != config || pulled.layers != layers {
                return Err(Error::config(
                    "image seed",
                    "builder image differs from authorized registry image",
                ));
            }
            run(exe, &["machine", "stop", "--name", &tmp])?;
            let disk = crate::agent::vm_data_dir(&tmp)
                .join(crate::storage::STORAGE_DISK_FILENAME)
                .with_extension(DiskFormat::Qcow2.extension());
            if !disk.exists() {
                return Err(Error::config(
                    "image seed",
                    "the builder's storage is not a template overlay",
                ));
            }
            let dir = seed.parent().expect("seed path has a parent");
            std::fs::create_dir_all(dir).map_err(|e| Error::config("image seed", e.to_string()))?;
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
                .map_err(|e| Error::config("image seed", e.to_string()))?;
            std::fs::rename(&disk, &staging).map_err(|e| Error::config("image seed", e.to_string()))
        })();
        let _ = run(exe, &["machine", "delete", "--name", &tmp, "-f"]);
        let published = built.and_then(|()| publish(&staging, seed));
        if published.is_err() {
            let _ = std::fs::remove_file(&staging);
        }
        published?;
        tracing::info!(
            image,
            key,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "built image seed"
        );
        Ok(())
    }

    /// Publish the seed privately. A dropped VMM uid sees its selected seed
    /// through a read-only idmapped mount in its private mount namespace.
    fn publish(staging: &Path, seed: &Path) -> Result<()> {
        let seed_error = |e: std::io::Error| Error::config("image seed", e.to_string());
        let file = std::fs::File::open(staging).map_err(seed_error)?;
        if unsafe { libc::fchown(file.as_raw_fd(), libc::geteuid(), libc::getegid()) } != 0 {
            return Err(seed_error(std::io::Error::last_os_error()));
        }
        file.sync_all().map_err(seed_error)?;
        std::fs::set_permissions(staging, std::fs::Permissions::from_mode(0o400))
            .map_err(seed_error)?;
        std::fs::set_permissions(
            seed.parent().expect("seed path has a parent"),
            std::fs::Permissions::from_mode(0o700),
        )
        .map_err(seed_error)?;
        std::fs::rename(staging, seed).map_err(seed_error)
    }

    /// A seed depends on the exact template bytes under it, the image content, the
    /// guest architecture and the guest's storage layout.
    pub(super) fn seed_key(image: &str, digest: &str, template: &Path) -> Result<String> {
        let meta =
            std::fs::metadata(template).map_err(|e| Error::config("image seed", e.to_string()))?;
        let mut hash = Sha256::new();
        for part in [
            SEED_FORMAT,
            env!("CARGO_PKG_VERSION"),
            image,
            digest,
            std::env::consts::ARCH,
            &template.display().to_string(),
            &format!(
                "{}:{}:{}:{}",
                meta.dev(),
                meta.ino(),
                meta.len(),
                meta.mtime()
            ),
        ] {
            hash.update(part.as_bytes());
            hash.update([0]);
        }
        Ok(hex::encode(hash.finalize()))
    }

    /// The storage template, when it can back a copy-on-write disk (sized at
    /// install time, the same condition as a machine's own template overlay).
    fn storage_template() -> Option<PathBuf> {
        let template = smolvm_pack::assets::find_existing_template("storage-template.ext4")?;
        let size = crate::storage::DEFAULT_STORAGE_SIZE_GIB * 1024 * 1024 * 1024;
        if std::fs::metadata(&template).ok()?.len() < size {
            return None;
        }
        template.canonicalize().ok()
    }

    /// `~/.cache/smolvm/image-seeds`.
    pub fn seed_root() -> PathBuf {
        crate::agent::vm_cache_root()
            .parent()
            .map(|root| root.join("image-seeds"))
            .unwrap_or_else(|| PathBuf::from("/tmp/smolvm-image-seeds"))
    }

    fn max_bytes() -> u64 {
        std::env::var("SMOLVM_IMAGE_SEED_MAX_BYTES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DEFAULT_MAX_BYTES)
    }

    /// Evict least recently used seeds while the cache is over `max_bytes`. A seed
    /// is only evicted when no disk image under smolvm's cache backs onto it: every
    /// machine, fork generation and paused disk that uses one keeps it.
    fn prune(root: &Path, max_bytes: u64, keep: &Path) {
        let Ok(_cache) = CacheLock::exclusive(root) else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };
        let mut seeds: Vec<(PathBuf, u64, std::time::SystemTime)> = entries
            .flatten()
            .map(|entry| entry.path().join("storage.qcow2"))
            .filter_map(|seed| {
                let meta = std::fs::metadata(&seed).ok()?;
                Some((seed, meta.blocks() * 512, meta.modified().ok()?))
            })
            .collect();
        let mut total: u64 = seeds.iter().map(|(_, bytes, _)| bytes).sum();
        if total <= max_bytes {
            return;
        }
        let Ok(referenced) = backing_references(
            &crate::agent::vm_cache_root()
                .parent()
                .map_or_else(crate::agent::vm_cache_root, Path::to_path_buf),
        ) else {
            // An incomplete scan cannot prove an old seed is unreferenced.
            return;
        };
        seeds.sort_by_key(|(_, _, used)| *used);
        for (seed, bytes, _) in seeds {
            if total <= max_bytes {
                break;
            }
            let canonical = seed.canonicalize().unwrap_or_else(|_| seed.clone());
            if seed == keep || referenced.contains(&canonical) {
                continue;
            }
            let dir = seed.parent().expect("seed path has a parent");
            let lock = root.join(format!(
                "{}.lock",
                dir.file_name().unwrap_or_default().to_string_lossy()
            ));
            // Skip a seed another process is building or about to use.
            let Some(_lock) = Lock::try_exclusive(&lock) else {
                continue;
            };
            if std::fs::remove_dir_all(dir).is_ok() {
                total = total.saturating_sub(bytes);
                tracing::info!(seed = %seed.display(), "evicted image seed");
            }
        }
    }

    /// Every backing file named by a qcow2 image under `dir`, recursively (seeds
    /// themselves excluded).
    fn backing_references(dir: &Path) -> std::io::Result<std::collections::HashSet<PathBuf>> {
        let mut found = std::collections::HashSet::new();
        let mut stack = vec![dir.to_path_buf()];
        let seeds = seed_root();
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir)?;
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                let kind = entry.file_type()?;
                if kind.is_dir() {
                    if path != seeds {
                        stack.push(path);
                    }
                } else if kind.is_file() {
                    if path.extension().is_some_and(|ext| ext == "qcow2") {
                        // Treat unreadable disk headers as an incomplete scan.
                        let _ = std::fs::File::open(&path)?;
                    }
                    if let Some(backing) = qcow2_backing(&path) {
                        let backing = if backing.is_absolute() {
                            backing
                        } else {
                            dir.join(backing)
                        };
                        found.insert(backing.canonicalize().unwrap_or(backing));
                    }
                }
            }
        }
        Ok(found)
    }

    /// The backing file a qcow2 image names in its header, if it is one.
    pub(super) fn qcow2_backing(path: &Path) -> Option<PathBuf> {
        use std::io::{Read, Seek, SeekFrom};
        use std::os::unix::ffi::OsStrExt;
        let mut file = std::fs::File::open(path).ok()?;
        let mut header = [0u8; 20];
        file.read_exact(&mut header).ok()?;
        if header[..4] != *b"QFI\xfb" {
            return None;
        }
        let offset = u64::from_be_bytes(header[8..16].try_into().ok()?);
        let len = u32::from_be_bytes(header[16..20].try_into().ok()?) as usize;
        if offset == 0 || len == 0 || len > 4096 {
            return None;
        }
        let mut name = vec![0u8; len];
        file.seek(SeekFrom::Start(offset)).ok()?;
        file.read_exact(&mut name).ok()?;
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(&name)))
    }

    /// Delete builder machines left behind by a build that crashed.
    fn reap_stale_builders(exe: &Path) {
        let Ok(config) = crate::config::SmolvmConfig::load() else {
            return;
        };
        let now = crate::util::current_timestamp();
        let stale: Vec<String> = config
            .list_vms()
            .filter(|(name, _)| name.starts_with(SEED_MACHINE_PREFIX))
            .filter(|(_, record)| now.saturating_sub(record.created_at) >= STALE_BUILDER_SECS)
            .map(|(name, _)| name.clone())
            .collect();
        for name in stale {
            let _ = run(exe, &["machine", "delete", "--name", &name, "-f"]);
        }
    }

    /// Run one step of a seed build with this smolvm binary, capturing its output so
    /// the builder's chatter stays off the caller's terminal.
    fn run(exe: &Path, args: &[&str]) -> Result<()> {
        let out = std::process::Command::new(exe)
            .args(args)
            .env("SMOLVM_IMAGE_SEEDS", "0")
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| Error::config("image seed", e.to_string()))?;
        if out.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
        Err(Error::config(
            "image seed",
            format!(
                "`smolvm {}` failed ({}): {}",
                args.join(" "),
                out.status,
                tail[tail.len().saturating_sub(6)..].join("\n")
            ),
        ))
    }

    struct Lock(std::fs::File);

    struct CacheLock {
        _lock: Lock,
    }

    impl CacheLock {
        fn shared(root: &Path) -> Result<Self> {
            Lock::with_mode(&root.join(".cache.lock"), libc::LOCK_SH)
                .map(|lock| Self { _lock: lock })
        }

        fn exclusive(root: &Path) -> Result<Self> {
            Lock::with_mode(&root.join(".cache.lock"), libc::LOCK_EX)
                .map(|lock| Self { _lock: lock })
        }
    }

    impl Lock {
        fn open(path: &Path) -> Result<std::fs::File> {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .write(true)
                .open(path)
                .map_err(|e| Error::config("image seed lock", e.to_string()))
        }

        fn with_mode(path: &Path, mode: libc::c_int) -> Result<Self> {
            let file = Self::open(path)?;
            file.set_permissions(std::fs::Permissions::from_mode(0o600))
                .map_err(|e| Error::config("image seed lock", e.to_string()))?;
            if unsafe { libc::flock(file.as_raw_fd(), mode) } != 0 {
                return Err(Error::config(
                    "image seed lock",
                    std::io::Error::last_os_error().to_string(),
                ));
            }
            Ok(Self(file))
        }

        fn exclusive(path: &Path) -> Result<Self> {
            Self::with_mode(path, libc::LOCK_EX)
        }

        fn try_exclusive(path: &Path) -> Option<Self> {
            let file = Self::open(path).ok()?;
            (unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0)
                .then_some(Self(file))
        }
    }

    impl Drop for Lock {
        fn drop(&mut self) {
            let _ = unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::linux::*;
    use std::path::PathBuf;

    #[test]
    fn key_changes_with_digest_and_image() {
        let template = std::env::temp_dir().join("seed-key-template");
        std::fs::write(&template, b"t").unwrap();
        let a = seed_key("alpine", "sha256:aa", &template).unwrap();
        assert_eq!(a, seed_key("alpine", "sha256:aa", &template).unwrap());
        assert_ne!(a, seed_key("alpine", "sha256:bb", &template).unwrap());
        assert_ne!(a, seed_key("busybox", "sha256:aa", &template).unwrap());
    }

    #[test]
    fn reads_qcow2_backing_and_ignores_other_files() {
        let dir = tempfile::tempdir().unwrap();
        let backing = b"/seeds/k/storage.qcow2";
        let mut image = vec![0u8; 512];
        image[..4].copy_from_slice(b"QFI\xfb");
        image[8..16].copy_from_slice(&256u64.to_be_bytes());
        image[16..20].copy_from_slice(&(backing.len() as u32).to_be_bytes());
        image[256..256 + backing.len()].copy_from_slice(backing);
        std::fs::write(dir.path().join("a.qcow2"), &image).unwrap();
        std::fs::write(dir.path().join("b.raw"), b"not an image").unwrap();
        assert_eq!(
            qcow2_backing(&dir.path().join("a.qcow2")),
            Some(PathBuf::from("/seeds/k/storage.qcow2"))
        );
        assert_eq!(qcow2_backing(&dir.path().join("b.raw")), None);
    }
}

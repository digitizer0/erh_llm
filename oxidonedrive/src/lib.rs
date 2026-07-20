#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncTarget {
    pub remote_path: String,
}

impl SyncTarget {
    pub fn new(remote_path: impl Into<String>) -> Self {
        Self {
            remote_path: remote_path.into(),
        }
    }
}

pub fn sync_preview(target: &SyncTarget) -> String {
    format!("Planned sync target: {}", target.remote_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_sync_target() {
        let target = SyncTarget::new("/Documents");
        assert_eq!(target.remote_path, "/Documents");
    }

    #[test]
    fn generates_sync_preview() {
        let target = SyncTarget::new("/Photos");
        assert_eq!(sync_preview(&target), "Planned sync target: /Photos");
    }
}

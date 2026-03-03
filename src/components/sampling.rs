/// Module for defining sampling strategies used in the system.
/// Each sampling strategy is encapsulated in a `Sampling` struct.
use std::sync::Arc;

/// Represents a sampling strategy with a name, description, and implementation.
///
/// This struct is used to encapsulate different sampling methods where:
/// 1. `name` identifies the sampling type
/// 2. `description` provides human-readable documentation
/// 3. `func` contains the actual implementation logic
#[derive(Clone)]
pub struct Sampling {
    /// Unique identifier for this sampling method (e.g., "random", "stratified")
    pub name: String,

    /// Human-readable description explaining the sampling behavior
    /// This should clarify what the sampling method does and any parameters it uses
    pub description: String,

    /// Sampling implementation function
    ///
    /// Takes input data as a string and returns sampled output as a string.
    /// The function is wrapped in an `Arc` to enable:
    /// - Thread-safe shared ownership (Send + Sync)
    /// - Efficient cloning without data duplication
    ///
    /// The signature Fn(&str) -> String means:
    /// - Accepts input as a string slice
    /// - Returns a new owned string as output
    pub func: Arc<dyn Fn(&str) -> String + Send + Sync>,
}

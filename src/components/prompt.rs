use std::sync::Arc;

/// A configurable prompt template with associated metadata and processing function.
///
/// The Prompt struct encapsulates a named prompt template along with a description and a function
/// that processes input text through the prompt. The function is wrapped in an Arc to enable
/// thread-safe sharing and cloning.
#[derive(Clone)]
pub struct Prompt {
    /// Unique identifier for this prompt template
    pub name: String,

    /// Human-readable description explaining this prompt's purpose and functionality
    pub description: String,

    /// Function that applies this prompt template to input text.
    ///
    /// This is a thread-safe, cloneable function that takes a &str input and returns a processed
    /// String. The function is wrapped in an Arc to enable safe sharing across threads.
    ///
    /// The function signature follows the pattern:
    /// |input: &str| -> String { ... }
    pub func: Arc<dyn Fn(&str) -> String + Send + Sync>,
}

/// Module for defining and managing async tool components in the system.
use std::{pin::Pin, sync::Arc};
use futures::Future;

/// Represents an async tool with a name, description, and implementation.
///
/// This struct encapsulates a tool that can be invoked by the system.
/// The function field is an asynchronous operation that returns a String result.
#[derive(Clone)]
pub struct Tool  {
    /// Unique identifier for the tool (e.g., "search", "calculate")
    pub name: String,

    /// Human-readable description of the tool's purpose and behavior
    pub description: String,

    /// Asynchronous implementation of the tool
    ///
    /// Takes a String parameter and returns a boxed future that resolves to a String result.
    /// The function is wrapped in an Arc for shared ownership and thread safety.
    #[allow(clippy::type_complexity)]
    pub func: Arc<dyn for<'a> Fn(&'a String) -> Pin<Box<dyn Future<Output = String> + Send + Sync + 'a>> + Send + Sync>
}

impl Tool {
    /// Creates a new Tool instance
    ///
    /// # Parameters
    /// - `name`: The unique identifier for this tool
    /// - `description`: Documentation describing the tool's purpose
    /// - `func`: The implementation function that takes a String parameter
    ///   and returns a Future<String>
    ///
    /// # Generics
    /// - `F`: A function type that matches the required signature
    /// - `Fut`: The future type returned by the function
    ///
    /// # Returns
    /// A new Tool instance with the provided configuration
    pub fn new<F, Fut>(name: &str, description: &str, func: F) -> Self
    where
        F: for<'a> Fn(&'a String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + Sync + 'static,
    {
        Tool {
            name: name.to_string(),
            description: description.to_string(),
            func: Arc::new(move |param: &String| {
                Box::pin(func(param))
            }),
        }
    }

    /// Executes the tool with the provided parameter
    ///
    /// # Parameters
    /// - `param`: The input parameter to pass to the tool's function
    ///
    /// # Returns
    /// Some(String) if execution succeeds, None if it fails
    pub async fn execute(&self, param: &String) -> Option<String> {
        let fut = (self.func)(param).await;
        Some(fut)
    }
}



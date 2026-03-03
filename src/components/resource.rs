use std::{pin::Pin, sync::Arc};
use futures::Future;
use ollama_rs::generation::tools::ToolHolder;

/// A reusable resource component with async execution capabilities.
///
/// Resources encapsulate named functions that process string parameters asynchronously.
/// They can be used as tools in the LLM coordination system by implementing ToolHolder.
#[derive(Clone)]
pub struct Resource {
    /// Unique identifier for this resource
    pub name: String,

    /// Human-readable description explaining this resource's purpose and functionality
    pub description: String,

    /// Asynchronous function that processes string parameters
    ///
    /// This is a thread-safe, cloneable function that takes a string reference and returns
    /// a future that resolves to a String. The function is wrapped in an Arc for safe sharing.
    pub func: Arc<dyn for<'a> Fn(&'a String) -> Pin<Box<dyn Future<Output = String> + Send + Sync + 'a>> + Send + Sync>,
}

impl Resource {
    /// Creates a new Resource instance with the specified function.
    ///
    /// Parameters:
    ///     name: Identifier for this resource
    ///     description: Documentation string explaining the resource's behavior
    ///     func: Asynchronous function that takes a String parameter and returns a String
    ///
    /// Returns:
    ///     Resource: A new Resource instance with the provided configuration
    ///
    /// The function parameter must:
    /// 1. Take a &String as input
    /// 2. Return a future that resolves to a String
    /// 3. Be Send + Sync + 'static
    pub fn new<F, Fut>(name: &str, description: &str, func: F) -> Self
    where
        F: for<'a> Fn(&'a String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + Sync + 'static,
    {
        Resource {
            name: name.to_string(),
            description: description.to_string(),
            func: Arc::new(move |param: &String| {
                Box::pin(func(param))
            }),
        }
    }

    /// Executes the resource's async function with the given parameter.
    ///
    /// Parameters:
    ///     param: String parameter to pass to the resource function
    ///
    /// Returns:
    ///     Option<String>: The result of the function execution, or None if it failed
    ///
    /// This method will:
    /// 1. Call the stored async function with the provided parameter
    /// 2. Await the result of the returned future
    /// 3. Return the resulting String wrapped in Some()
    ///
    /// Note: This function must be called with `.await` to execute the asynchronous operation
    pub async fn execute(&self, param: &String) -> Option<String> {
        let fut = (self.func)(param).await;
        Some(fut)
    }
}

impl ToolHolder for Resource {
    /// Implements the ToolHolder trait for Resource execution.
    ///
    /// Parameters:
    ///     parameters: JSON-encoded input parameters
    ///
    /// Returns:
    ///     Future<Output = Result<String, Error>>: Asynchronous operation that:
    ///     - Serializes parameters to JSON string
    ///     - Calls the execute() method with the serialized parameters
    ///     - Returns the result or an error if serialization failed
    ///
    /// This implementation:
    /// 1. Converts input parameters to JSON string
    /// 2. Executes the resource with the serialized parameters
    /// 3. Returns successful result or error message
    ///
    /// Note: This must be awaited to get the final result
    fn call(
        &mut self,
        parameters: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>> + '_ + Send + Sync>> {
        Box::pin(async move {
            let param_str = serde_json::to_string(&parameters)?;
            let result = self.execute(&param_str).await;
            match result {
                Some(res) => Ok(res),
                None => Err("Tool execution failed".into()),
            }
        })
    }
}

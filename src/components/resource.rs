use std::{pin::Pin, sync::Arc};
use futures::Future;

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
    #[allow(clippy::type_complexity)]
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



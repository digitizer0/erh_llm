/// Module for defining and managing async tool components in the system.
/// Contains the Tool struct implementation and integration with ollama_rs's ToolHolder trait.
use std::{pin::Pin, sync::Arc};
use futures::Future;
use ollama_rs::generation::tools::ToolHolder;

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

impl ToolHolder for Tool {
    /// Invokes the tool with provided JSON parameters
    ///
    /// # Parameters
    /// - `parameters`: A JSON value containing the tool's input parameters
    ///
    /// # Returns
    /// A future that resolves to either:
    /// - Ok(String): The tool's successful result
    /// - Err(Box<dyn Error>): If serialization or execution fails
    fn call(
        &mut self,
        parameters: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>> + '_ + Send + Sync>> {
        Box::pin(async move {
            // Extract the first string value from the JSON object, or fall back to
            // serialising the whole value. This handles models that wrap the single
            // string parameter in an object with an arbitrary key name, e.g.
            // {"query": "…"}, {"whereclause": "…"}, {"param": "…"}, etc.
            log::debug!("Tool '{}' called with parameters: {}", self.name, parameters);
            let param_str = match &parameters {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(map) => {
                    // Pick the first string value; fall back to serialising the object.
                    map.values()
                        .find_map(|v| v.as_str().map(str::to_string))
                        .unwrap_or_else(|| serde_json::to_string(&parameters).unwrap_or_default())
                }
                _ => serde_json::to_string(&parameters).unwrap_or_default(),
            };

            // Execute the tool with the extracted parameter
            let result = self.execute(&param_str).await;

            // Return the result or a failure error
            match result {
                Some(res) => Ok(res),
                None => Err("Tool execution failed".into()),
            }
        })
    }
}

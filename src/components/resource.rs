use std::{pin::Pin, sync::Arc};
use futures::Future;
use ollama_rs::generation::tools::ToolHolder;

#[derive(Clone)]
pub struct Resource  {
    pub name: String,
    pub description: String,
    pub func: Arc<dyn for<'a> Fn(&'a String) -> Pin<Box<dyn Future<Output = String> + Send + Sync + 'a>> + Send + Sync>
}



impl Resource {
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


    pub async fn execute(&self, param: &String) -> Option<String> {
        let fut = (self.func)(param).await;
        Some(fut)
    }
}   

impl ToolHolder for Resource {
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

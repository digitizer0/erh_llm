use std::{pin::Pin, sync::Arc};
use futures::Future;

#[derive(Clone)]
pub struct Resource {
    pub name: String,
    pub description: String,
    pub func: Arc<dyn for<'a> Fn(&'a String) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> + Send + Sync>
}

impl Resource {
    pub fn new<F, Fut>(name: &str, description: &str, func: F) -> Self
    where
        F: for<'a> Fn(&'a String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        Resource {
            name: name.to_string(),
            description: description.to_string(),
            func: Arc::new(move |param: &String| {
                Box::pin(func(param))
            }),
        }
    }

    pub async fn execute(&self, input: &String) -> Option<String> {
        let fut = (self.func)(input).await;
        Some(fut)
    }
}
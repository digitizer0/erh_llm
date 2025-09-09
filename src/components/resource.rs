use std::sync::Arc;

#[derive(Clone)]
pub struct Resource {
    pub name: String,
    pub description: String,
    pub func: Arc<dyn Fn(&str) -> String + Send + Sync>,
}
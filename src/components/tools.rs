use std::sync::Arc;

#[derive(Clone)]
pub struct Tool  {
    pub name: String,
    pub description: String,
    pub func: Arc<dyn Fn(&str) -> String + Send + Sync>,
}




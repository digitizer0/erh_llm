use std::sync::Arc;

#[derive(Clone)]
pub struct ComponentRegistry {
    pub components: Vec<Component>,
}

#[derive(Clone, Debug, PartialEq,Default)]
pub enum ComponentSource {
    MCP,
    Internal,
    #[default]
    Unknown
}

#[derive(Clone, Debug, PartialEq, Default)]
pub enum ComponentType {
    Tool,
    Resource,
    Prompt,
    Sampling,
    #[default]
    Unknown
}

#[derive(Clone)]
pub struct Component {
    pub source: ComponentSource,
    pub component_type: ComponentType,
    pub name: String,
    pub description: String,
    pub access: String,
    pub run: Arc<dyn Fn(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, Box<dyn std::error::Error>>> + Send>> + Send + Sync>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        ComponentRegistry {
            components: Vec::new(),
        }
    }

    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }

    pub fn get_components(&self) -> Vec<Component> {
        self.components.clone()
    }

    pub fn list(&self) -> Vec<String> {
        self.components.iter().map(|t| t.name.clone()).collect()
    }

    pub fn get(&self, name: &str) -> Option<Component> {
        self.components.iter().find(|t| t.name == name).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    
    
}


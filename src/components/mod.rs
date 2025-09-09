pub (crate) mod prompt;
pub (crate) mod resource;
pub (crate) mod sampling;
pub (crate) mod tools;

use crate::components::prompt::Prompt;
use crate::components::resource::Resource;
use crate::components::sampling::Sampling;
use crate::components::tools::Tool;

#[derive(Clone, Debug, PartialEq,Default)]
pub enum ComponentSource {
    McpStdio(String),
    McpSse(String),
    Internal(String),
    #[default]
    Unknown
}


#[derive(Clone)]
pub struct ComponentRegistry {
    pub components: Vec<Component>,
}

#[derive(Clone)]
pub struct Component {
    pub source: ComponentSource,
    pub tools: Vec<Tool>,
    pub resources: Vec<Resource>,
    pub prompts: Vec<Prompt>,
    pub samplings: Vec<Sampling>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    pub fn register(&mut self, component: Component) {
        self.components.push(component);
    }

    pub fn list(&self) -> Vec<String> {
        self.components.iter().map(|c| format!("{:?}", c.source)).collect()
    }
    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }
    
}
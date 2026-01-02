pub (crate) mod prompt;
pub (crate) mod resource;
pub (crate) mod sampling;
pub (crate) mod tools;

use crate::components::prompt::Prompt;
use crate::components::resource::Resource;
use crate::components::sampling::Sampling;
use crate::components::tools::Tool;

#[derive(Clone,Default)]
pub struct ComponentRegistry {
    pub components: Vec<Component>,
}

#[derive(Clone)]
pub struct Component {
    //pub source: ComponentSource,
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

    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }
    
}


/// Get the CPU temperature in Celsius.
#[ollama_rs::function]
pub (crate) async fn get_cpu_temperature() -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    Ok("42.7".to_string())
}
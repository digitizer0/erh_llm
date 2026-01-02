pub (crate) mod prompt;
pub (crate) mod resource;
pub (crate) mod sampling;
pub (crate) mod tools;

use ollama_rs::coordinator::Coordinator;
use ollama_rs::history::ChatHistory;

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
        log::debug!("Registering component with {} tools", component.tools.len());
        self.components.push(component);
    }

    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }
    
    pub fn add_tools<T: ChatHistory>(&mut self, coordinator : Coordinator<T>) -> Coordinator<T> {
        let mut cd = coordinator;
        log::debug!("Adding tools from ComponentRegistry with {} components", self.components.len());
        for component in &self.components {   
            for tool in &component.tools {
                log::debug!("Adding tool: {}", tool.name);
                cd = cd.add_tool_custom(tool.name.as_str(), tool.description.as_str(), Box::new(tool.clone()));
                
            }
            for resource in &component.resources {
                log::debug!("Adding resource: {}", resource.name);
                cd = cd.add_tool_custom(resource.name.as_str(), resource.description.as_str(), Box::new(resource.clone()));
                
            }
        };
        cd
    }
}


/// Get the CPU temperature in Celsius.
#[ollama_rs::function]
pub (crate) async fn get_cpu_temperature() -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    Ok("42.7".to_string())
}
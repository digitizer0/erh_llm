pub (crate) mod prompt;
pub (crate) mod resource;
pub (crate) mod sampling;
pub (crate) mod tools;

use log::debug;

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


#[derive(Clone,Default)]
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

    pub(crate) async fn get_tools(&self) -> String {
        let mut selected_tools = Vec::new();
        for component in &self.components {
            for tool in &component.tools {
                selected_tools.push(format!("Tool Name: {} Description: {}", tool.name, tool.description));
            }
        }
        for component in &self.components {
            for resource in &component.resources {
                selected_tools.push(format!("Resource Name: {} Description: {}", resource.name, resource.description));
            }
        }
        selected_tools.join(", ")
    }

    pub(crate) async fn execute_tools(&self, tool_list: &std::collections::HashMap<String, String>) -> Option<String> {
        let mut results = Vec::new();
        for (tool_name, param) in tool_list {
            for component in &self.components {
                for tool in &component.tools {
                    if &tool.name == tool_name {
                        debug!("Executing tool: {} with param: {}", tool_name, param);
                        if let Some(result) = tool.execute(param).await {
                            results.push(format!("Result from {}: {}", tool_name, result));
                        }
                    }
                }
                for resource in &component.resources {
                    if &resource.name == tool_name {
                        debug!("Executing resource: {} with param: {}", tool_name, param);
                        if let Some(result) = resource.execute(param).await {
                            results.push(format!("Result from {}: {}", tool_name, result));
                        }
                    }
                }
            }
        }
        if results.is_empty() {
            None
        } else {
            Some(results.join("\n"))
        }
    }
    pub fn len(&self) -> usize {
        self.components.len()
    }
    
}
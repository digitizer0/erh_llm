use std::sync::Arc;

#[derive(Clone)]
pub struct ToolRegistry {
    pub tools: Vec<Tool>,
}

#[derive(Clone, Debug, PartialEq,Default)]
pub enum ToolSource {
    MCP,
    Internal,
    #[default]
    Unknown
}

#[derive(Clone)]
pub struct Tool {
    pub source: ToolSource,
    pub name: String,
    pub description: String,
    pub access: String,
    pub run: Arc<dyn Fn(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, Box<dyn std::error::Error>>> + Send>> + Send + Sync>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry {
            tools: Vec::new(),
        }
    }

    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.push(tool);
    }

    pub fn get_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    pub fn list(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    pub fn get(&self, name: &str) -> Option<Tool> {
        self.tools.iter().find(|t| t.name == name).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    
    
}


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
    /// Creates a new empty ComponentRegistry.
    ///
    /// Initializes with an empty components vector.
    ///
    /// Returns:
    ///     Self: A new ComponentRegistry instance with empty components.
    ///
    /// Example:
    ///     let registry = ComponentRegistry::new();
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Registers a component with the registry.
    ///
    /// Adds the given component to the registry's components vector and logs the action.
    ///
    /// Parameters:
    ///     component: Component to be registered
    ///
    /// Side Effects:
    ///     - Logs debug message with component tool count
    ///     - Modifies internal components vector
    ///
    /// Example:
    ///     registry.register(Component { ... });
    pub fn register(&mut self, component: Component) {
        log::debug!("Registering component with {} tools", component.tools.len());
        self.components.push(component);
    }

    /// Adds a component to the registry (deprecated).
    ///
    /// This is a redundant alternative to register(). Use register() for better clarity.
    ///
    /// Parameters:
    ///     component: Component to be added
    ///
    /// Side Effects:
    ///     - Modifies internal components vector
    ///
    /// Note: register() is preferred for component registration
    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }

    /// Returns tool definitions as JSON for the Ollama API `tools` field.
    ///
    /// Each definition has `name`, `description`, and `parameters` keys.
    pub fn get_tool_definitions(&self) -> Vec<serde_json::Value> {
        let mut defs = Vec::new();
        let default_params = serde_json::json!({
            "type": "object",
            "properties": {
                "param": { "type": "string", "description": "The input parameter for this tool." }
            },
            "required": ["param"]
        });
        for component in &self.components {
            for tool in &component.tools {
                defs.push(serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": default_params.clone()
                }));
            }
            for resource in &component.resources {
                defs.push(serde_json::json!({
                    "name": resource.name,
                    "description": resource.description,
                    "parameters": default_params.clone()
                }));
            }
        }
        defs
    }

    /// Executes a tool or resource by name with the given JSON arguments.
    ///
    /// Extracts the first string value from the arguments object (or serialises the whole
    /// value) and passes it to the matching tool/resource.
    pub async fn execute_tool(&self, name: &str, arguments: serde_json::Value) -> Option<String> {
        let param = match &arguments {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(map) => map
                .values()
                .find_map(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| serde_json::to_string(&arguments).unwrap_or_default()),
            _ => serde_json::to_string(&arguments).unwrap_or_default(),
        };

        for component in &self.components {
            for tool in &component.tools {
                if tool.name == name {
                    log::debug!("Executing tool '{}' with param: {}", name, param);
                    return tool.execute(&param).await;
                }
            }
            for resource in &component.resources {
                if resource.name == name {
                    log::debug!("Executing resource '{}' with param: {}", name, param);
                    return resource.execute(&param).await;
                }
            }
        }
        log::debug!("Tool '{}' not found in registry", name);
        None
    }
}

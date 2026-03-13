pub (crate) mod prompt;
pub (crate) mod resource;
pub (crate) mod sampling;
pub (crate) mod tools;

use ollama_rs::coordinator::Coordinator;
use ollama_rs::history::ChatHistory;
use schemars::Schema;

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

    /// Adds all tools and resources from registry components to a coordinator.
    ///
    /// Parameters:
    ///     coordinator: Coordinator<T> to which tools will be added
    ///
    /// Returns:
    ///     Coordinator<T>: Updated coordinator with added tools/resources
    ///
    /// Side Effects:
    ///     - Logs debug messages for each added tool/resource
    ///     - Modifies the provided coordinator by adding tools
    ///
    /// Process:
    ///     1. Iterates through each component in the registry
    ///     2. Adds all component tools to the coordinator
    ///     3. Adds all component resources to the coordinator
    ///     4. Returns the modified coordinator
    pub fn add_tools<T: ChatHistory>(&mut self, coordinator : Coordinator<T>) -> Coordinator<T> {
        let mut cd = coordinator;
        log::debug!("Adding tools from ComponentRegistry with {} components", self.components.len());
        // Build a reusable schema for tools that accept a single string parameter.
        // Schema::default() serialises as `{}` which Ollama cannot parse; we need
        // a proper JSON-Schema object with type+properties so Ollama accepts the
        // tool definition and knows how to call it.
        let single_string_schema = Schema::from(
            serde_json::json!({
                "type": "object",
                "properties": {
                    "param": {
                        "type": "string",
                        "description": "The input parameter for this tool."
                    }
                },
                "required": ["param"]
            })
            .as_object()
            .cloned()
            .unwrap_or_default()
        );

        for component in &self.components {
            for tool in &component.tools {
                log::debug!("Adding tool: {}", tool.name);
                cd = cd.add_tool_custom_schema(tool.name.as_str(), tool.description.as_str(), single_string_schema.clone(), Box::new(tool.clone()));

            }
            for resource in &component.resources {
                log::debug!("Adding resource: {}", resource.name);
                cd = cd.add_tool_custom_schema(resource.name.as_str(), resource.description.as_str(), single_string_schema.clone(), Box::new(resource.clone()));

            }
        };
        cd
    }
}

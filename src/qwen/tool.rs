use serde::{Deserialize, Serialize};

/// A callable tool that the model can invoke during generation.
///
/// Tools are described using a JSON Schema for their `parameters`, matching the
/// OpenAI / Qwen tool-calling format understood by llama.cpp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Unique name used to identify the tool in a `ToolCall`.
    pub name: String,

    /// Human-readable description shown to the model so it knows when to call this tool.
    pub description: String,

    /// JSON Schema describing the tool's input parameters.
    ///
    /// Example:
    /// ```json
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "location": { "type": "string", "description": "City name" }
    ///   },
    ///   "required": ["location"]
    /// }
    /// ```
    pub parameters: serde_json::Value,
}

impl Tool {
    /// Create a new tool with the given name, description, and parameter schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Tool {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Serialize the tool into the JSON format expected by the Qwen3 chat template.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// A single tool invocation requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this call (provided by the model).
    pub id: String,

    /// The name of the tool to call.
    pub name: String,

    /// Arguments passed to the tool, as a JSON object.
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Try to parse a `ToolCall` from the raw JSON string emitted by the model.
    pub fn from_json_str(s: &str) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        let name = v["name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'name' field in tool call"))?
            .to_string();
        let arguments = v["arguments"].clone();
        let id = v["id"]
            .as_str()
            .unwrap_or("call_0")
            .to_string();
        Ok(ToolCall { id, name, arguments })
    }
}

/// The result returned after executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches the `id` field of the originating [`ToolCall`].
    pub call_id: String,

    /// Plain-text (or JSON-encoded) output produced by the tool.
    pub content: String,
}

impl ToolResult {
    pub fn new(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        ToolResult {
            call_id: call_id.into(),
            content: content.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_to_json() {
        let tool = Tool::new(
            "get_weather",
            "Get current weather for a city",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
        );
        let json = tool.to_json();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "get_weather");
    }

    #[test]
    fn test_tool_call_parse() {
        let raw = r#"{"id":"call_1","name":"get_weather","arguments":{"location":"London"}}"#;
        let call = ToolCall::from_json_str(raw).unwrap();
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments["location"], "London");
    }
}

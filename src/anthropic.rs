//! Anthropic backend implementation for erh_llm.
//!
//! Provides [`anthropic_chat`] and [`anthropic_chat_with_system`] as thin
//! wrappers around `anthropic_rust` so that [`crate::Query`] does not need to
//! import Anthropic types directly.
//!
//! Also provides [`AnthropicProvider`] which implements the [`crate::provider::LlmProvider`] trait.

use anthropic_rust::{
    ClientBuilder,
    types::{ChatRequestBuilder, ContentBlock, Model, Role},
    Tool, ToolBuilder,
};
use log::debug;

use crate::errors::{ErhLlmError, Result};
use crate::provider::{LlmProvider, AnthropicConfig};
use crate::{ChatMessage as ErhChatMessage, ModelConfig};
#[cfg(feature = "tools")]
use crate::ComponentRegistry;

/// Maps a model name string to the corresponding [`Model`] enum variant.
/// Unknown names are passed through as-is via [`Model::Other`].
fn model_from_str(name: &str) -> Model {
    match name {
        "claude-3-haiku-20240307"    => Model::Claude3Haiku20240307,
        "claude-3-5-haiku-20241022" => Model::Claude35Haiku20241022,
        "claude-3-sonnet-20240229"  => Model::Claude3Sonnet20240229,
        "claude-3-opus-20240229"    => Model::Claude3Opus20240229,
        "claude-3-5-sonnet-20241022" => Model::Claude35Sonnet20241022,
        "claude-3-5-sonnet-20250114" => Model::Claude35Sonnet20250114,
        "claude-4-sonnet-20250514"  => Model::Claude4Sonnet20250514,
        "claude-sonnet-4-5"         => Model::ClaudeSonnet45,
        "claude-sonnet-4-6"         => Model::ClaudeSonnet46,
        "claude-opus-4-5"           => Model::ClaudeOpus45,
        other => {
            debug!("Passing model name '{}' as-is to Anthropic API", other);
            Model::Other(other.to_string())
        }
    }
}

// ── Tool conversion helpers ─────────────────────────────────────────────────

#[cfg(feature = "tools")]
fn convert_tools_to_anthropic(components: &ComponentRegistry) -> Vec<Tool> {
    let mut tools = Vec::new();
    
    for component in &components.components {
        for tool in &component.tools {
            let anthropic_tool = ToolBuilder::new(&tool.name)
                .description(&tool.description)
                .property("param", "string", Some("The input parameter for this tool."), true)
                .build();
            tools.push(anthropic_tool);
        }
        
        for resource in &component.resources {
            let anthropic_tool = ToolBuilder::new(&resource.name)
                .description(&resource.description)
                .property("param", "string", Some("The input parameter for this resource."), true)
                .build();
            tools.push(anthropic_tool);
        }
    }
    
    tools
}

#[cfg(feature = "tools")]
async fn execute_tool(
    components: &ComponentRegistry,
    tool_name: &str,
    input: &serde_json::Value,
) -> Result<String> {
    // Extract the parameter from the input JSON
    let param_str = match input {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => {
            // Try to find "param" key or use the first string value
            map.get("param")
                .and_then(|v| v.as_str())
                .or_else(|| map.values().find_map(|v| v.as_str()))
                .unwrap_or("")
                .to_string()
        }
        _ => serde_json::to_string(input).unwrap_or_default(),
    };

    // Search for the tool in the component registry
    for component in &components.components {
        // Check tools
        for tool in &component.tools {
            if tool.name == tool_name {
                debug!("Executing tool '{}' with param: {}", tool_name, param_str);
                return tool.execute(&param_str)
                    .await
                    .ok_or_else(|| ErhLlmError::AnthropicError(format!("Tool '{}' returned None", tool_name)));
            }
        }
        
        // Check resources
        for resource in &component.resources {
            if resource.name == tool_name {
                debug!("Executing resource '{}' with param: {}", tool_name, param_str);
                return resource.execute(&param_str)
                    .await
                    .ok_or_else(|| ErhLlmError::AnthropicError(format!("Resource '{}' returned None", tool_name)));
            }
        }
    }

    Err(ErhLlmError::AnthropicError(format!("Tool '{}' not found", tool_name)))
}

/// Sends a single user message to the Anthropic API, optionally injecting
/// chat history as alternating user/assistant turns and tools.
///
/// Returns the raw text response. History is **not** persisted by this function.
pub async fn anthropic_chat(
    api_key: &str,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    user_text: String,
    #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
) -> Result<String> {
    let client = ClientBuilder::new()
        .api_key(api_key)
        .build()
        .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

    let mut builder = ChatRequestBuilder::new();

    // Inject prior turns as alternating user/assistant messages.
    // Skip turns with no plain-text response — these occurred when the model
    // used tools and the tool result (not the final text) was what got stored,
    // or the turn was interrupted. Replaying such turns would produce a
    // `tool_use` without a following `tool_result`, which Anthropic rejects.
    for msg in history {
        if msg.bot_response.trim().is_empty() {
            debug!("Skipping history turn with empty bot_response for user: {}", msg.user_message);
            continue;
        }
        builder = builder
            .message(Role::User, ContentBlock::text(msg.user_message))
            .message(Role::Assistant, ContentBlock::text(msg.bot_response));
    }

    builder = builder.user_message(ContentBlock::text(user_text.clone()));

    if let Some(temp) = model.temperature {
        builder = builder.temperature(temp);
    }

    // Add tools if available and enabled
    #[cfg(feature = "tools")]
    if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            let tools = convert_tools_to_anthropic(comp);
            if !tools.is_empty() {
                debug!("Adding {} tools to Anthropic request", tools.len());
                builder = builder.tools(tools);
            }
        }
    }

    let request = builder.build();
    // Keep the initial messages so the tool loop can reconstruct the full
    // conversation (history + user turn + assistant tool_use + tool_result...).
    let initial_messages = request.messages.clone();
    debug!("Sending prompt to Anthropic: {user_text}");

    let anthropic_model = model_from_str(&model.model);
    let mut response: anthropic_rust::types::Message = client
        .execute_chat_with_model(anthropic_model.clone(), request)
        .await
        .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

    // Handle tool use loop
    #[cfg(feature = "tools")]
    if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            // Accumulates (assistant tool_use, user tool_result) pairs added
            // after the initial messages during this request's tool loop.
            let mut extra_messages: Vec<(Role, Vec<ContentBlock>)> = vec![];
            let max_iterations = 10;
            let mut iteration = 0;

            while iteration < max_iterations {
                let tool_uses: Vec<_> = response
                    .content
                    .iter()
                    .filter_map(|block| {
                        if let ContentBlock::ToolUse { id, name, input } = block {
                            Some((id.clone(), name.clone(), input.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();

                if tool_uses.is_empty() {
                    break;
                }

                debug!("Anthropic requested {} tool calls", tool_uses.len());

                // Append the assistant turn containing the tool_use blocks.
                extra_messages.push((Role::Assistant, response.content.clone()));

                // Execute tools and collect results.
                let mut tool_results = vec![];
                for (tool_id, tool_name, input) in tool_uses {
                    debug!("Executing tool: {} with id: {}", tool_name, tool_id);
                    match execute_tool(comp, &tool_name, &input).await {
                        Ok(result) => {
                            tool_results.push(ContentBlock::tool_result(tool_id, result));
                        }
                        Err(e) => {
                            let error_msg = format!("Error executing tool '{}': {}", tool_name, e);
                            debug!("{}", error_msg);
                            tool_results.push(ContentBlock::tool_result(tool_id, error_msg));
                        }
                    }
                }
                extra_messages.push((Role::User, tool_results));

                // Rebuild the full conversation: initial messages + all
                // tool_use/tool_result pairs accumulated so far.
                let mut new_builder = ChatRequestBuilder::new();
                new_builder = new_builder.messages(initial_messages.clone());
                for (role, content) in &extra_messages {
                    new_builder = new_builder.message_with_content(role.clone(), content.clone());
                }

                if let Some(temp) = model.temperature {
                    new_builder = new_builder.temperature(temp);
                }
                let tools = convert_tools_to_anthropic(comp);
                if !tools.is_empty() {
                    new_builder = new_builder.tools(tools);
                }

                let new_request = new_builder.build();
                response = client
                    .execute_chat_with_model(anthropic_model.clone(), new_request)
                    .await
                    .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

                iteration += 1;
            }

            if iteration >= max_iterations {
                debug!("Warning: Tool use loop reached max iterations");
            }
        }
    }

    // Extract final text response
    let text = response
        .content
        .into_iter()
        .filter_map(|block| {
            if let ContentBlock::Text { text, .. } = block {
                Some(text)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("");

    debug!("Received response from Anthropic: {text}");
    Ok(text)
}

/// Like [`anthropic_chat`] but prepends a system prompt before the
/// conversational turns.
///
/// Returns the raw text response. History is **not** persisted by this function.
pub async fn anthropic_chat_with_system(
    api_key: &str,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    system: String,
    user_query: String,
    #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
) -> Result<String> {
    let client = ClientBuilder::new()
        .api_key(api_key)
        .build()
        .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

    let mut builder = ChatRequestBuilder::new().system(system.clone());

    // Skip turns with no plain-text response — these occurred when the model
    // used tools and the tool result (not the final text) was what got stored.
    // Replaying such turns would produce a `tool_use` without a following
    // `tool_result`, which Anthropic rejects with a 400 error.
    for msg in history {
        if msg.bot_response.trim().is_empty() {
            debug!("Skipping history turn with empty bot_response for user: {}", msg.user_message);
            continue;
        }
        builder = builder
            .message(Role::User, ContentBlock::text(msg.user_message))
            .message(Role::Assistant, ContentBlock::text(msg.bot_response));
    }

    builder = builder.user_message(ContentBlock::text(user_query.clone()));

    if let Some(temp) = model.temperature {
        builder = builder.temperature(temp);
    }

    // Add tools if available and enabled
    #[cfg(feature = "tools")]
    if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            let tools = convert_tools_to_anthropic(comp);
            if !tools.is_empty() {
                debug!("Adding {} tools to Anthropic request (with system)", tools.len());
                builder = builder.tools(tools);
            }
        }
    }

    let request = builder.build();
    let initial_messages = request.messages.clone();
    debug!("Sending prompt to Anthropic (with system): {user_query}");

    let anthropic_model = model_from_str(&model.model);
    let mut response = client
        .execute_chat_with_model(anthropic_model.clone(), request)
        .await
        .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

    // Handle tool use loop
    #[cfg(feature = "tools")]
    if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            let mut extra_messages: Vec<(Role, Vec<ContentBlock>)> = vec![];
            let max_iterations = 10;
            let mut iteration = 0;

            while iteration < max_iterations {
                let tool_uses: Vec<_> = response
                    .content
                    .iter()
                    .filter_map(|block| {
                        if let ContentBlock::ToolUse { id, name, input } = block {
                            Some((id.clone(), name.clone(), input.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();

                if tool_uses.is_empty() {
                    break;
                }

                debug!("Anthropic requested {} tool calls", tool_uses.len());

                extra_messages.push((Role::Assistant, response.content.clone()));

                let mut tool_results = vec![];
                for (tool_id, tool_name, input) in tool_uses {
                    debug!("Executing tool: {} with id: {}", tool_name, tool_id);
                    match execute_tool(comp, &tool_name, &input).await {
                        Ok(result) => {
                            tool_results.push(ContentBlock::tool_result(tool_id, result));
                        }
                        Err(e) => {
                            let error_msg = format!("Error executing tool '{}': {}", tool_name, e);
                            debug!("{}", error_msg);
                            tool_results.push(ContentBlock::tool_result(tool_id, error_msg));
                        }
                    }
                }
                extra_messages.push((Role::User, tool_results));

                // Reconstruct full conversation: system + initial messages +
                // all tool_use/tool_result pairs accumulated so far.
                let mut new_builder = ChatRequestBuilder::new().system(system.clone());
                new_builder = new_builder.messages(initial_messages.clone());
                for (role, content) in &extra_messages {
                    new_builder = new_builder.message_with_content(role.clone(), content.clone());
                }

                if let Some(temp) = model.temperature {
                    new_builder = new_builder.temperature(temp);
                }
                let tools = convert_tools_to_anthropic(comp);
                if !tools.is_empty() {
                    new_builder = new_builder.tools(tools);
                }

                let new_request = new_builder.build();
                response = client
                    .execute_chat_with_model(anthropic_model.clone(), new_request)
                    .await
                    .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

                iteration += 1;
            }

            if iteration >= max_iterations {
                debug!("Warning: Tool use loop reached max iterations");
            }
        }
    }

    // Extract final text response
    let text = response
        .content
        .into_iter()
        .filter_map(|block| {
            if let ContentBlock::Text { text, .. } = block {
                Some(text)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("");

    debug!("Received response from Anthropic: {text}");
    Ok(text)
}

// ── LlmProvider implementation ──────────────────────────────────────────────

/// Anthropic provider implementation.
///
/// This struct implements the [`LlmProvider`] trait for Anthropic backends.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    api_key: String,
}

impl AnthropicProvider {
    /// Creates a new Anthropic provider with the given configuration.
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            api_key: config.api_key,
        }
    }

    /// Creates a new Anthropic provider with the given API key.
    pub fn with_api_key(api_key: String) -> Self {
        Self { api_key }
    }
}

/// Options for Anthropic requests.
///
/// Currently a placeholder - Anthropic doesn't have as many options as Ollama.
#[derive(Debug, Clone, Default)]
pub struct AnthropicOptions {
    // Reserved for future use
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    type Options = AnthropicOptions;

    async fn embed(&self, _model: &ModelConfig, _chunk: String) -> Result<Vec<f32>> {
        // Anthropic doesn't provide a native embeddings API
        Err(ErhLlmError::AnthropicError(
            "Embeddings not supported by Anthropic provider".to_string(),
        ))
    }

    async fn chat(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        _options: Self::Options,
        user_text: String,
        #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        anthropic_chat(
            &self.api_key,
            model,
            history,
            user_text,
            #[cfg(feature = "tools")]
            components,
        )
        .await
    }

    async fn chat_with_system(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        _options: Self::Options,
        system: String,
        user_query: String,
        #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        anthropic_chat_with_system(
            &self.api_key,
            model,
            history,
            system,
            user_query,
            #[cfg(feature = "tools")]
            components,
        )
        .await
    }
}

//! Native Anthropic API implementation using reqwest.
//!
//! Features:
//! - Chat (Messages API)
//! - Extended Thinking (claude-3-7-sonnet+)
//! - Prompt Caching
//! - Streaming (SSE)
//! - Vision / Multimodal (base64 & URL images)
//! - Token Counting
//! - Tool Use loop

use log::debug;
use serde::{Deserialize, Serialize};

use crate::errors::{ErhLlmError, Result};
use crate::provider::{AnthropicConfig, LlmProvider};
use crate::{ChatMessage as ErhChatMessage, ModelConfig};
#[cfg(feature = "tools")]
use crate::ComponentRegistry;

// ── Constants ────────────────────────────────────────────────────────────────

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const COUNT_TOKENS_URL: &str = "https://api.anthropic.com/v1/messages/count_tokens";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8192;

// ── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum AnthropicApiError {
    Http(reqwest::Error),
    Io(std::io::Error),
    Api { status: u16, body: String },
    Json(serde_json::Error),
}

impl std::fmt::Display for AnthropicApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(e) => write!(f, "HTTP error: {e}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Api { status, body } => write!(f, "Anthropic API error {status}: {body}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl From<reqwest::Error> for AnthropicApiError {
    fn from(e: reqwest::Error) -> Self { Self::Http(e) }
}
impl From<std::io::Error> for AnthropicApiError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}
impl From<serde_json::Error> for AnthropicApiError {
    fn from(e: serde_json::Error) -> Self { Self::Json(e) }
}
impl From<AnthropicApiError> for ErhLlmError {
    fn from(e: AnthropicApiError) -> Self {
        ErhLlmError::AnthropicError(e.to_string())
    }
}

// ── API types ────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Role { User, Assistant }

/// `cache_control: {"type":"ephemeral"}` — enables prompt caching on this block.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String, // "ephemeral"
}

impl CacheControl {
    pub fn ephemeral() -> Self { Self { kind: "ephemeral".into() } }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64 { media_type: String, data: String },
    /// Publicly accessible image URL.
    Url { url: String },
}

/// A content block in a message.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Returned by the model when it wants to call a tool.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Sent by the user to return a tool result.
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    /// Extended thinking block (claude-3-7-sonnet+).
    Thinking { thinking: String },
    /// Redacted thinking block (opaque).
    RedactedThinking { data: String },
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into(), cache_control: None }
    }
    pub fn text_cached(s: impl Into<String>) -> Self {
        Self::Text { text: s.into(), cache_control: Some(CacheControl::ephemeral()) }
    }
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::Base64 { media_type: media_type.into(), data: data.into() },
            cache_control: None,
        }
    }
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::Url { url: url.into() },
            cache_control: None,
        }
    }
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult { tool_use_id: tool_use_id.into(), content: content.into() }
    }
    /// Returns the text if this is a Text block.
    pub fn as_text(&self) -> Option<&str> {
        if let Self::Text { text, .. } = self { Some(text) } else { None }
    }
    /// Returns the thinking if this is a Thinking block.
    pub fn as_thinking(&self) -> Option<&str> {
        if let Self::Thinking { thinking } = self { Some(thinking) } else { None }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ApiMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl ApiMessage {
    pub fn user(content: Vec<ContentBlock>) -> Self { Self { role: Role::User, content } }
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::text(text)])
    }
    pub fn assistant(content: Vec<ContentBlock>) -> Self { Self { role: Role::Assistant, content } }
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::assistant(vec![ContentBlock::text(text)])
    }
}

/// A system prompt block — supports prompt caching via `cache_control`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: String, // "text"
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self { kind: "text".into(), text: text.into(), cache_control: None }
    }
    pub fn cached(text: impl Into<String>) -> Self {
        Self { kind: "text".into(), text: text.into(), cache_control: Some(CacheControl::ephemeral()) }
    }
}

/// Tool definition sent to the API.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl ToolDef {
    /// Simple single-string-input tool definition.
    pub fn simple(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "Input for this tool" }
                },
                "required": ["input"]
            }),
            cache_control: None,
        }
    }
}

/// Extended thinking configuration.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: String, // "enabled"
    pub budget_tokens: u32,
}

impl ThinkingConfig {
    pub fn enabled(budget_tokens: u32) -> Self {
        Self { kind: "enabled".into(), budget_tokens }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Option<Usage>,
}

impl ChatResponse {
    /// Concatenates all Text blocks into a single string.
    pub fn text(&self) -> String {
        self.content.iter().filter_map(|b| b.as_text()).collect::<Vec<_>>().join("")
    }
    /// Concatenates all Thinking blocks into a single string.
    pub fn thinking(&self) -> String {
        self.content.iter().filter_map(|b| b.as_thinking()).collect::<Vec<_>>().join("")
    }
    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content.iter().filter_map(|b| {
            if let ContentBlock::ToolUse { id, name, input } = b {
                Some((id.as_str(), name.as_str(), input))
            } else { None }
        }).collect()
    }
}

#[derive(Deserialize, Debug, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_creation_input_tokens: Option<u32>,
    pub cache_read_input_tokens: Option<u32>,
}

/// Returned by [`AnthropicClient::stream_chat`].
pub struct StreamResult {
    /// The full concatenated text response.
    pub text: String,
    /// Extended thinking content (if any).
    pub thinking: String,
    pub usage: Option<Usage>,
}

// ── SSE event types (internal) ───────────────────────────────────────────────

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SseEvent {
    Ping,
    MessageStart { message: SseMessageStart },
    ContentBlockStart { index: u32, content_block: SseContentBlockStart },
    ContentBlockDelta { index: u32, delta: SseDelta },
    ContentBlockStop { index: u32 },
    MessageDelta { delta: SseMessageDelta, usage: Option<SseDeltaUsage> },
    MessageStop,
    Error { error: SseError },
}

#[derive(Deserialize, Debug)]
struct SseMessageStart {
    usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SseContentBlockStart {
    Text { text: String },
    Thinking { thinking: String },
    ToolUse { id: String, name: String },
    RedactedThinking { data: String },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SseDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    InputJsonDelta { partial_json: String },
    SignatureDelta { signature: String },
}

#[derive(Deserialize, Debug)]
struct SseMessageDelta {
    stop_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct SseDeltaUsage {
    output_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct SseError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

// ── AnthropicClient ──────────────────────────────────────────────────────────

pub struct AnthropicClient {
    api_key: String,
    client: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { api_key: api_key.into(), client: reqwest::Client::new() }
    }

    fn base_request(&self, url: &str) -> reqwest::RequestBuilder {
        self.client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("anthropic-beta", "interleaved-thinking-2025-05-14")
    }

    /// Non-streaming chat request.
    pub async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, AnthropicApiError> {
        let resp = self.base_request(MESSAGES_URL).json(request).send().await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(AnthropicApiError::Api { status, body });
        }
        Ok(resp.json::<ChatResponse>().await?)
    }

    /// Streaming chat. Calls `on_chunk` for each text delta as it arrives.
    /// Returns the complete result when the stream ends.
    pub async fn stream_chat(
        &self,
        request: &ChatRequest,
        mut on_chunk: impl FnMut(String) + Send,
    ) -> Result<StreamResult, AnthropicApiError> {
        use tokio::io::AsyncBufReadExt;
        use tokio_util::io::StreamReader;
        use futures::TryStreamExt;

        let mut req = request.clone();
        req.stream = Some(true);

        let resp = self.base_request(MESSAGES_URL).json(&req).send().await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(AnthropicApiError::Api { status, body });
        }

        let byte_stream = resp
            .bytes_stream()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
        let reader = StreamReader::new(byte_stream);
        let mut lines = tokio::io::BufReader::new(reader).lines();

        let mut text = String::new();
        let mut thinking = String::new();
        let mut usage: Option<Usage> = None;

        while let Some(line) = lines.next_line().await? {
            let line = line.trim().to_string();
            if !line.starts_with("data: ") { continue; }
            let data = &line[6..];
            if data == "[DONE]" { break; }

            match serde_json::from_str::<SseEvent>(data) {
                Ok(SseEvent::ContentBlockDelta { delta, .. }) => match delta {
                    SseDelta::TextDelta { text: chunk } => {
                        text.push_str(&chunk);
                        on_chunk(chunk);
                    }
                    SseDelta::ThinkingDelta { thinking: chunk } => {
                        thinking.push_str(&chunk);
                    }
                    _ => {}
                },
                Ok(SseEvent::MessageStart { message }) => {
                    usage = message.usage;
                }
                Ok(SseEvent::Error { error }) => {
                    return Err(AnthropicApiError::Api {
                        status: 0,
                        body: format!("{}: {}", error.kind, error.message),
                    });
                }
                Ok(SseEvent::MessageStop) => break,
                _ => {}
            }
        }

        Ok(StreamResult { text, thinking, usage })
    }

    /// Count tokens for a request without sending it.
    pub async fn count_tokens(&self, request: &ChatRequest) -> Result<u32, AnthropicApiError> {
        #[derive(Deserialize)]
        struct CountResponse { input_tokens: u32 }

        let resp = self.base_request(COUNT_TOKENS_URL).json(request).send().await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(AnthropicApiError::Api { status, body });
        }
        Ok(resp.json::<CountResponse>().await?.input_tokens)
    }
}

// ── Tool conversion helpers ──────────────────────────────────────────────────

#[cfg(feature = "tools")]
fn registry_to_tool_defs(components: &ComponentRegistry) -> Vec<ToolDef> {
    let mut tools = Vec::new();
    for component in &components.components {
        for t in &component.tools {
            tools.push(ToolDef::simple(&t.name, &t.description));
        }
        for r in &component.resources {
            tools.push(ToolDef::simple(&r.name, &r.description));
        }
    }
    tools
}

#[cfg(feature = "tools")]
async fn execute_tool_from_registry(
    components: &ComponentRegistry,
    tool_name: &str,
    input: &serde_json::Value,
) -> Result<String> {
    let param = match input {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => map
            .get("input")
            .or_else(|| map.get("param"))
            .and_then(|v| v.as_str())
            .or_else(|| map.values().find_map(|v| v.as_str()))
            .unwrap_or("")
            .to_string(),
        _ => serde_json::to_string(input).unwrap_or_default(),
    };

    for component in &components.components {
        for t in &component.tools {
            if t.name == tool_name {
                return t.execute(&param).await
                    .ok_or_else(|| ErhLlmError::AnthropicError(format!("Tool '{tool_name}' returned None")));
            }
        }
        for r in &component.resources {
            if r.name == tool_name {
                return r.execute(&param).await
                    .ok_or_else(|| ErhLlmError::AnthropicError(format!("Resource '{tool_name}' returned None")));
            }
        }
    }
    Err(ErhLlmError::AnthropicError(format!("Tool '{tool_name}' not found")))
}

// ── Core chat helpers ─────────────────────────────────────────────────────────

/// Build history messages, skipping turns with empty bot responses
/// (which indicate a tool-use turn that should not be replayed as plain text).
fn build_history_messages(history: Vec<ErhChatMessage>) -> Vec<ApiMessage> {
    let mut msgs = Vec::new();
    for msg in history {
        if msg.bot_response.trim().is_empty() {
            debug!("Skipping history turn with empty bot_response");
            continue;
        }
        msgs.push(ApiMessage::user_text(msg.user_message));
        msgs.push(ApiMessage::assistant_text(msg.bot_response));
    }
    msgs
}

/// Run the tool-use agentic loop. `initial_messages` is everything up to and
/// including the first user turn; `response` is the first assistant response.
/// Returns the final text response.
#[cfg(feature = "tools")]
async fn run_tool_loop(
    client: &AnthropicClient,
    base_request: &ChatRequest,
    initial_messages: Vec<ApiMessage>,
    mut response: ChatResponse,
    components: &ComponentRegistry,
) -> Result<ChatResponse> {
    const MAX_ITER: u32 = 10;
    let mut extra: Vec<ApiMessage> = Vec::new();

    for _ in 0..MAX_ITER {
        let tool_uses: Vec<_> = response
            .content
            .iter()
            .filter_map(|b| {
                if let ContentBlock::ToolUse { id, name, input } = b {
                    Some((id.clone(), name.clone(), input.clone()))
                } else { None }
            })
            .collect();

        if tool_uses.is_empty() { break; }

        debug!("Tool loop: {} tool call(s) requested", tool_uses.len());

        // Append the assistant turn (contains ToolUse blocks).
        extra.push(ApiMessage::assistant(response.content.clone()));

        // Execute tools and collect results.
        let mut results = Vec::new();
        for (id, name, input) in tool_uses {
            debug!("Executing tool '{}' (id={})", name, id);
            let result = match execute_tool_from_registry(components, &name, &input).await {
                Ok(r) => r,
                Err(e) => format!("Error: {e}"),
            };
            results.push(ContentBlock::tool_result(&id, result));
        }
        extra.push(ApiMessage::user(results));

        // Rebuild full message list and send.
        let mut messages = initial_messages.clone();
        messages.extend(extra.clone());

        let mut new_req = base_request.clone();
        new_req.messages = messages;

        response = client.send(&new_req).await.map_err(ErhLlmError::from)?;
    }

    Ok(response)
}

// ── Public chat functions ────────────────────────────────────────────────────

pub async fn anthropic_chat(
    api_key: &str,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    user_text: String,
    thinking_budget: Option<u32>,
    #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
) -> Result<String> {
    let client = AnthropicClient::new(api_key);

    let mut messages = build_history_messages(history);
    messages.push(ApiMessage::user_text(&user_text));

    let mut request = ChatRequest {
        model: model.model.clone(),
        max_tokens: DEFAULT_MAX_TOKENS,
        messages: messages.clone(),
        system: None,
        tools: None,
        thinking: thinking_budget.map(ThinkingConfig::enabled),
        temperature: model.temperature,
        stream: None,
    };

    #[cfg(feature = "tools")]
    if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            let defs = registry_to_tool_defs(comp);
            if !defs.is_empty() {
                request.tools = Some(defs);
            }
        }
    }

    debug!("Sending to Anthropic (model={}): {user_text}", model.model);
    let response = client.send(&request).await.map_err(ErhLlmError::from)?;

    #[cfg(feature = "tools")]
    let response = if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            run_tool_loop(&client, &request, messages, response, comp).await?
        } else { response }
    } else { response };

    let text = response.text();
    debug!("Anthropic response: {text}");
    Ok(text)
}

pub async fn anthropic_chat_with_system(
    api_key: &str,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    system: String,
    user_query: String,
    thinking_budget: Option<u32>,
    cache_system: bool,
    #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
) -> Result<String> {
    let client = AnthropicClient::new(api_key);

    let system_block = if cache_system {
        SystemBlock::cached(&system)
    } else {
        SystemBlock::new(&system)
    };

    let mut messages = build_history_messages(history);
    messages.push(ApiMessage::user_text(&user_query));

    let mut request = ChatRequest {
        model: model.model.clone(),
        max_tokens: DEFAULT_MAX_TOKENS,
        messages: messages.clone(),
        system: Some(vec![system_block]),
        tools: None,
        thinking: thinking_budget.map(ThinkingConfig::enabled),
        temperature: model.temperature,
        stream: None,
    };

    #[cfg(feature = "tools")]
    if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            let defs = registry_to_tool_defs(comp);
            if !defs.is_empty() {
                request.tools = Some(defs);
            }
        }
    }

    debug!("Sending to Anthropic (model={}, system={}...): {user_query}", model.model, &system[..system.len().min(40)]);
    let response = client.send(&request).await.map_err(ErhLlmError::from)?;

    #[cfg(feature = "tools")]
    let response = if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            run_tool_loop(&client, &request, messages, response, comp).await?
        } else { response }
    } else { response };

    let text = response.text();
    debug!("Anthropic response: {text}");
    Ok(text)
}

// ── LlmProvider implementation ───────────────────────────────────────────────

/// Options specific to the Anthropic provider.
#[derive(Debug, Clone, Default)]
pub struct AnthropicOptions {
    /// Enable extended thinking. `Some(N)` sets the token budget.
    /// Requires claude-3-7-sonnet-20250219 or newer.
    pub thinking_budget: Option<u32>,
    /// Cache the system prompt using prompt caching.
    pub cache_system: bool,
}

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self { Self { api_key: config.api_key } }
    pub fn with_api_key(api_key: impl Into<String>) -> Self { Self { api_key: api_key.into() } }

    /// Expose the underlying client for advanced use (streaming, token counting, etc.).
    pub fn client(&self) -> AnthropicClient { AnthropicClient::new(&self.api_key) }
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    type Options = AnthropicOptions;

    async fn embed(&self, _model: &ModelConfig, _chunk: String) -> Result<Vec<f32>> {
        Err(ErhLlmError::AnthropicError(
            "Anthropic does not provide an embeddings API".to_string(),
        ))
    }

    async fn chat(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
        user_text: String,
        #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        anthropic_chat(
            &self.api_key,
            model,
            history,
            user_text,
            options.thinking_budget,
            #[cfg(feature = "tools")]
            components,
        )
        .await
    }

    async fn chat_with_system(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
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
            options.thinking_budget,
            options.cache_system,
            #[cfg(feature = "tools")]
            components,
        )
        .await
    }
}

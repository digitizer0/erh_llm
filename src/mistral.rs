//! MistralAI backend — local REST client + [`LlmProvider`] implementation.
//! API docs: <https://docs.mistral.ai/api/>

#![allow(dead_code)]

use futures::Stream;
use log::debug;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::errors::{ErhLlmError, Result};
use crate::provider::{LlmProvider, MistralConfig};
use crate::{ChatMessage as ErhChatMessage, ModelConfig};
#[cfg(feature = "tools")]
use crate::ComponentRegistry;

// ── Client error ─────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum MistralError {
    #[error("HTTP error {status}: {message}")]
    Api { status: u16, message: String },
    #[error("reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Stream ended unexpectedly")]
    StreamEnded,
}

// ── Message types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role { System, User, Assistant, Tool }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent { Text(String), Parts(Vec<ContentPart>) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")] pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    fn mk(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: MessageContent::Text(content.into()), name: None, tool_call_id: None, tool_calls: None }
    }
    pub fn system(c: impl Into<String>) -> Self { Self::mk(Role::System, c) }
    pub fn user(c: impl Into<String>) -> Self { Self::mk(Role::User, c) }
    pub fn assistant(c: impl Into<String>) -> Self { Self::mk(Role::Assistant, c) }
    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self { role: Role::User, content: MessageContent::Parts(parts), name: None, tool_call_id: None, tool_calls: None }
    }
    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: Role::Tool, content: MessageContent::Text(content.into()), name: None, tool_call_id: Some(id.into()), tool_calls: None }
    }
}

// ── Tool / function calling ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")] pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")] pub tool_type: String,
    pub function: FunctionDef,
}

impl Tool {
    pub fn function(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self { tool_type: "function".into(), function: FunctionDef { name: name.into(), description: Some(description.into()), parameters } }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto, None, Any,
    #[serde(untagged)]
    Specific { #[serde(rename = "type")] tool_type: String, function: FunctionName },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionName { pub name: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")] pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl FunctionCall {
    pub fn parse_arguments(&self) -> std::result::Result<serde_json::Value, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

// ── Response format ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: serde_json::Value },
}

// ── Chat request / response ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")] pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")] pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")] pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")] pub min_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")] pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")] pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")] pub random_seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")] pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")] pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")] pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")] pub safe_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")] pub prediction: Option<Prediction>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self { model: model.into(), messages, ..Default::default() }
    }
    pub fn temperature(mut self, t: f32) -> Self { self.temperature = Some(t); self }
    pub fn max_tokens(mut self, n: u32) -> Self { self.max_tokens = Some(n); self }
    pub fn tools(mut self, tools: Vec<Tool>) -> Self { self.tools = Some(tools); self }
    pub fn tool_choice(mut self, tc: ToolChoice) -> Self { self.tool_choice = Some(tc); self }
    pub fn json_response(mut self) -> Self { self.response_format = Some(ResponseFormat::JsonObject); self }
    pub fn seed(mut self, s: u64) -> Self { self.random_seed = Some(s); self }
    pub fn safe_prompt(mut self) -> Self { self.safe_prompt = Some(true); self }
    pub fn predict(mut self, prefix: impl Into<String>) -> Self {
        self.prediction = Some(Prediction { pred_type: "content".into(), content: prefix.into() });
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    #[serde(rename = "type")] pub pred_type: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl ChatResponse {
    /// Returns the text of the first choice, if any.
    pub fn first_content(&self) -> Option<&str> {
        self.choices.first()?.message.content.as_deref()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ── Embeddings ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub encoding_format: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingResponse {
    pub id: String,
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

// ── FIM (Codestral) ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct FimRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")] pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")] pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")] pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")] pub stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FimResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<FimChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FimChoice {
    pub index: u32,
    pub message: FimMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FimMessage { pub content: String }

// ── Models ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: Option<u64>,
    pub owned_by: Option<String>,
    pub capabilities: Option<serde_json::Value>,
    pub description: Option<String>,
    pub max_context_length: Option<u32>,
}

// ── REST client ───────────────────────────────────────────────────────────────

const BASE_URL: &str = "https://api.mistral.ai/v1";

#[derive(Clone)]
pub struct MistralClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl MistralClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { client: Client::new(), api_key: api_key.into(), base_url: BASE_URL.into() }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self { self.base_url = url.into(); self }

    fn auth(&self, rb: RequestBuilder) -> RequestBuilder { rb.bearer_auth(&self.api_key) }

    async fn check(resp: reqwest::Response) -> std::result::Result<reqwest::Response, MistralError> {
        let status = resp.status();
        if status.is_success() { return Ok(resp); }
        let msg = resp.text().await.unwrap_or_default();
        Err(MistralError::Api { status: status.as_u16(), message: msg })
    }

    pub async fn chat(&self, req: ChatRequest) -> std::result::Result<ChatResponse, MistralError> {
        let resp = self.auth(self.client.post(format!("{}/chat/completions", self.base_url)))
            .json(&req).send().await?;
        Ok(Self::check(resp).await?.json().await?)
    }

    pub async fn chat_stream(
        &self, mut req: ChatRequest,
    ) -> std::result::Result<Pin<Box<dyn Stream<Item = std::result::Result<StreamChunk, MistralError>> + Send>>, MistralError> {
        req.stream = Some(true);
        let resp = self.auth(self.client.post(format!("{}/chat/completions", self.base_url)))
            .json(&req).send().await?;
        let resp = Self::check(resp).await?;

        use futures::StreamExt;
        let stream = resp.bytes_stream().flat_map(|res| {
            let items: Vec<_> = match res {
                Err(e) => vec![Err(MistralError::Reqwest(e))],
                Ok(bytes) => String::from_utf8_lossy(&bytes).lines()
                    .filter(|l| l.starts_with("data: ") && *l != "data: [DONE]")
                    .map(|l| serde_json::from_str::<StreamChunk>(&l["data: ".len()..]).map_err(MistralError::Json))
                    .collect(),
            };
            futures::stream::iter(items)
        });
        Ok(Box::pin(stream))
    }

    pub async fn embeddings(&self, req: EmbeddingRequest) -> std::result::Result<EmbeddingResponse, MistralError> {
        let resp = self.auth(self.client.post(format!("{}/embeddings", self.base_url)))
            .json(&req).send().await?;
        Ok(Self::check(resp).await?.json().await?)
    }

    pub async fn fim(&self, req: FimRequest) -> std::result::Result<FimResponse, MistralError> {
        let resp = self.auth(self.client.post(format!("{}/fim/completions", self.base_url)))
            .json(&req).send().await?;
        Ok(Self::check(resp).await?.json().await?)
    }

    pub async fn list_models(&self) -> std::result::Result<ModelList, MistralError> {
        let resp = self.auth(self.client.get(format!("{}/models", self.base_url)))
            .send().await?;
        Ok(Self::check(resp).await?.json().await?)
    }
}

// ── LlmProvider ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MistralProvider {
    api_key: String,
    /// Default model ID, e.g. `"mistral-medium-latest"`.
    default_model: String,
}

#[derive(Debug, Clone, Default)]
pub struct MistralOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub safe_prompt: bool,
}

impl MistralProvider {
    pub fn new(config: MistralConfig) -> Self {
        Self { api_key: config.api_key, default_model: "mistral-medium-latest".into() }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self { api_key, default_model: "mistral-medium-latest".into() }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into(); self
    }

    fn client(&self) -> MistralClient { MistralClient::new(&self.api_key) }

    fn model_id<'a>(&'a self, cfg: &'a ModelConfig) -> &'a str {
        if cfg.model.is_empty() { &self.default_model } else { &cfg.model }
    }
}

#[async_trait::async_trait]
impl LlmProvider for MistralProvider {
    type Options = MistralOptions;

    async fn embed(&self, model: &ModelConfig, chunk: String) -> Result<Vec<f32>> {
        let req = EmbeddingRequest {
            model: self.model_id(model).to_string(),
            input: vec![chunk],
            encoding_format: None,
        };
        let resp = self.client().embeddings(req).await
            .map_err(|e| ErhLlmError::MistralError(e.to_string()))?;
        resp.data.into_iter().next()
            .map(|d| d.embedding)
            .ok_or_else(|| ErhLlmError::MistralError("Empty embedding response".into()))
    }

    async fn chat(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
        user_text: String,
        #[cfg(feature = "tools")] _components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        let mut messages: Vec<Message> = history.iter().flat_map(|m| {
            [Message::user(&m.user_message), Message::assistant(&m.bot_response)]
        }).collect();
        messages.push(Message::user(&user_text));

        let mut req = ChatRequest::new(self.model_id(model), messages);
        if let Some(t) = options.temperature { req = req.temperature(t); }
        if let Some(n) = options.max_tokens  { req = req.max_tokens(n); }
        if options.safe_prompt               { req = req.safe_prompt(); }

        debug!("Mistral chat → model={}", req.model);
        let resp = self.client().chat(req).await
            .map_err(|e| ErhLlmError::MistralError(e.to_string()))?;
        debug!("Mistral response id={}", resp.id);

        resp.first_content()
            .map(str::to_owned)
            .ok_or_else(|| ErhLlmError::MistralError("Empty response from Mistral".into()))
    }

    async fn chat_with_system(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
        system: String,
        user_query: String,
        #[cfg(feature = "tools")] _components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        let mut messages = vec![Message::system(&system)];
        for m in &history {
            messages.push(Message::user(&m.user_message));
            messages.push(Message::assistant(&m.bot_response));
        }
        messages.push(Message::user(&user_query));

        let mut req = ChatRequest::new(self.model_id(model), messages);
        if let Some(t) = options.temperature { req = req.temperature(t); }
        if let Some(n) = options.max_tokens  { req = req.max_tokens(n); }
        if options.safe_prompt               { req = req.safe_prompt(); }

        debug!("Mistral chat_with_system → model={}", req.model);
        let resp = self.client().chat(req).await
            .map_err(|e| ErhLlmError::MistralError(e.to_string()))?;

        resp.first_content()
            .map(str::to_owned)
            .ok_or_else(|| ErhLlmError::MistralError("Empty response from Mistral".into()))
    }
}

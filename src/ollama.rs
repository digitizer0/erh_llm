//! Ollama backend implementation for erh_llm.
//!
//! Uses the Ollama REST API directly via `reqwest`.
//! Provides [`ollama_embed`], [`ollama_chat`], and [`ollama_chat_with_system`]
//! as well as [`OllamaProvider`] which implements [`crate::provider::LlmProvider`].

use log::debug;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{ErhLlmError, Result};
use crate::provider::{LlmProvider, OllamaConfig};
use crate::{ModelConfig, ChatMessage as ErhChatMessage};
#[cfg(feature = "tools")]
use crate::ComponentRegistry;

// ── Public options type ───────────────────────────────────────────────────────

/// Generation options forwarded to the Ollama API.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ModelOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
}

impl ModelOptions {
    pub fn num_ctx(mut self, ctx: u64) -> Self { self.num_ctx = Some(ctx); self }
    pub fn num_predict(mut self, n: i64) -> Self { self.num_predict = Some(n); self }
    pub fn temperature(mut self, t: f32) -> Self { self.temperature = Some(t); self }
    pub fn top_p(mut self, p: f32) -> Self { self.top_p = Some(p); self }
    pub fn top_k(mut self, k: i64) -> Self { self.top_k = Some(k); self }
}

// ── Ollama API wire types ─────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ModelOptions>,
    /// Set to false to suppress chain-of-thought / thinking output (qwen3, deepseek-r1, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    #[cfg(feature = "tools")]
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDef>>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Message,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct StreamChunk {
    message: Message,
    #[serde(default)]
    done: bool,
}

#[cfg(feature = "tools")]
#[derive(Serialize, Clone)]
struct ToolDef {
    #[serde(rename = "type")]
    tool_type: String,
    function: ToolFunction,
}

#[cfg(feature = "tools")]
#[derive(Serialize, Clone)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Serialize, Deserialize, Clone)]
struct ToolCall {
    function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone)]
struct ToolCallFunction {
    name: String,
    arguments: Value,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ModelOptions>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Generates a vector embedding via the Ollama `/api/embed` endpoint.
pub async fn ollama_embed(
    host: &str,
    port: u16,
    model: &ModelConfig,
    chunk: String,
) -> Result<Vec<f32>> {
    let url = format!("{host}:{port}/api/embed");
    let options = ModelOptions::default().num_ctx(model.context_size.unwrap_or(2048) as u64);

    let body = EmbedRequest {
        model: model.model.clone(),
        input: chunk,
        options: Some(options),
    };

    let client = Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| ErhLlmError::EmbeddingError(e.to_string()))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(ErhLlmError::EmbeddingError(format!(
            "Ollama embed HTTP {status}: {text}"
        )));
    }

    let parsed: EmbedResponse = resp
        .json()
        .await
        .map_err(|e| ErhLlmError::EmbeddingError(e.to_string()))?;

    if parsed.embeddings.is_empty() {
        return Err(ErhLlmError::EmbeddingError(
            "Ollama returned empty embeddings".to_string(),
        ));
    }
    debug!("VectorCount: {}", parsed.embeddings[0].len());
    Ok(parsed.embeddings[0].clone())
}

/// Sends a user message to Ollama's `/api/chat` endpoint with optional history and tools.
pub async fn ollama_chat(
    host: &str,
    port: u16,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    base_options: ModelOptions,
    user_text: String,
    #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
) -> Result<String> {
    let options = build_options(base_options, model);
    let mut messages = build_messages(history);
    messages.push(Message { role: "user".to_string(), content: user_text, tool_calls: None });

    #[cfg(feature = "tools")]
    let tools = components.map(build_tool_defs);

    run_chat_loop(
        host, port, model, options, messages,
        #[cfg(feature = "tools")] tools,
        #[cfg(feature = "tools")] components,
    ).await
}

/// Sends a system + user message to Ollama's `/api/chat` endpoint.
#[allow(clippy::too_many_arguments)]
pub async fn ollama_chat_with_system(
    host: &str,
    port: u16,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    base_options: ModelOptions,
    system: String,
    user_query: String,
    #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
) -> Result<String> {
    let options = build_options(base_options, model);
    let mut messages = build_messages(history);
    messages.push(Message { role: "system".to_string(), content: system, tool_calls: None });
    messages.push(Message { role: "user".to_string(), content: user_query, tool_calls: None });

    #[cfg(feature = "tools")]
    let tools = components.map(build_tool_defs);

    run_chat_loop(
        host, port, model, options, messages,
        #[cfg(feature = "tools")] tools,
        #[cfg(feature = "tools")] components,
    ).await
}

/// Streaming variant: sends `system` + `user_query` to Ollama with `stream: true`
/// and calls `on_chunk` for every text delta received via NDJSON lines.
/// Returns the full concatenated response.
#[allow(clippy::too_many_arguments)]
pub async fn ollama_stream_chat_with_system(
    host: &str,
    port: u16,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    base_options: ModelOptions,
    system: String,
    user_query: String,
    on_chunk: &mut (dyn FnMut(String) + Send),
    #[cfg(feature = "tools")] _components: Option<&ComponentRegistry>,
) -> Result<String> {
    use tokio::io::AsyncBufReadExt;
    use tokio_util::io::StreamReader;
    use futures::TryStreamExt;

    let url = format!("{host}:{port}/api/chat");
    let options = build_options(base_options, model);
    let mut messages = build_messages(history);
    messages.push(Message { role: "system".to_string(), content: system, tool_calls: None });
    messages.push(Message { role: "user".to_string(), content: user_query, tool_calls: None });

    let req = ChatRequest {
        model: model.model.clone(),
        messages,
        stream: true,
        options: Some(options),
        think: Some(false),
        #[cfg(feature = "tools")]
        tools: None,
    };

    let client = Client::new();
    let resp = client
        .post(&url)
        .json(&req)
        .send()
        .await
        .map_err(|e| ErhLlmError::OllamaError(e.to_string()))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(ErhLlmError::OllamaError(format!("Ollama stream HTTP {status}: {text}")));
    }

    let byte_stream = resp
        .bytes_stream()
        .map_err(std::io::Error::other);
    let reader = StreamReader::new(byte_stream);
    let mut lines = tokio::io::BufReader::new(reader).lines();

    // Accumulate the raw response. Tags arrive token-by-token so we buffer
    // in `pending` and scan for complete <think> / </think> markers.
    // All content is forwarded via on_chunk; think content is prefixed with
    // the sentinel "\x00THINK\x00" so callers can distinguish it from the
    // real response without a second channel.
    let mut raw = String::new();
    let mut pending = String::new();
    let mut in_think = false;
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    const THINK_PREFIX: &str = "__THINK__:";  // ASCII-safe sentinel, stripped by api.js

    while let Some(line) = lines.next_line().await.map_err(|e| ErhLlmError::OllamaError(e.to_string()))? {
        let line = line.trim().to_string();
        if line.is_empty() { continue; }
        match serde_json::from_str::<StreamChunk>(&line) {
            Ok(chunk) => {
                let delta = chunk.message.content;
                if !delta.is_empty() {
                    raw.push_str(&delta);
                    pending.push_str(&delta);

                    loop {
                        if in_think {
                            if let Some(pos) = pending.find(CLOSE) {
                                // Emit think content before the closing tag
                                let think_content = pending[..pos].to_string();
                                pending = pending[pos + CLOSE.len()..].to_string();
                                in_think = false;
                                if !think_content.is_empty() {
                                    on_chunk(format!("{THINK_PREFIX}{}", think_content.replace('\n', "\\n")));
                                }
                                // continue loop — may be visible content after </think>
                            } else {
                                // Still accumulating think — emit all now, no tail needed
                                // since </think> arrives as its own discrete chunk(s)
                                if !pending.is_empty() {
                                    on_chunk(format!("{THINK_PREFIX}{}", pending.replace('\n', "\\n")));
                                    pending.clear();
                                }
                                break;
                            }
                        } else if let Some(pos) = pending.find(OPEN) {
                            // Emit visible content before the opening tag
                            let safe = pending[..pos].to_string();
                            pending = pending[pos + OPEN.len()..].to_string();
                            in_think = true;
                            if !safe.is_empty() {
                                on_chunk(safe);
                            }
                            // continue loop — handle think content
                        } else {
                            // No think tag — emit everything immediately.
                            // Do NOT hold back a tail: <think> tokens arrive as
                            // discrete chunks so the tag assembles across iterations.
                            if !pending.is_empty() {
                                on_chunk(pending.clone());
                                pending.clear();
                            }
                            break;
                        }
                    }
                }
                if chunk.done {
                    if !pending.is_empty() {
                        if in_think {
                            on_chunk(format!("{THINK_PREFIX}{}", pending.replace('\n', "\\n")));
                        } else {
                            on_chunk(pending.clone());
                        }
                    }
                    break;
                }
            }
            Err(e) => debug!("Ollama stream: skipping unparseable line ({e}): {line}"),
        }
    }

    Ok(strip_think_tags(&raw))
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn build_options(base: ModelOptions, model: &ModelConfig) -> ModelOptions {
    let num_ctx = base.num_ctx.or_else(|| model.context_size.map(|c| c as u64));
    ModelOptions {
        num_ctx,
        // Cap at 2048 tokens max output; -1 means unlimited which causes runaway loops.
        num_predict: base.num_predict.or(Some(2048)),
        temperature: base.temperature.or(model.temperature),
        ..base
    }
}

/// Keep only the most recent N conversation turns to bound context size.
const MAX_HISTORY_TURNS: usize = 5;

fn build_messages(history: Vec<ErhChatMessage>) -> Vec<Message> {
    // Take only the most recent turns to avoid unbounded context growth.
    let history = if history.len() > MAX_HISTORY_TURNS {
        history.into_iter().rev().take(MAX_HISTORY_TURNS).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>()
    } else {
        history
    };
    let mut out = Vec::with_capacity(history.len() * 2);
    for msg in history {
        out.push(Message { role: "user".to_string(), content: msg.user_message, tool_calls: None });
        // Strip think-block content from stored responses before feeding them
        // back as context — guards against rows stored before the filter existed.
        out.push(Message { role: "assistant".to_string(), content: strip_think_tags(&msg.bot_response), tool_calls: None });
    }
    out
}

#[cfg(feature = "tools")]
fn build_tool_defs(components: &ComponentRegistry) -> Vec<ToolDef> {
    components
        .get_tool_definitions()
        .into_iter()
        .filter_map(|def| {
            let name = def.get("name")?.as_str()?.to_string();
            let description = def.get("description")
                .and_then(|v| v.as_str()).unwrap_or("").to_string();
            let parameters = def.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({
                "type": "object",
                "properties": { "param": { "type": "string" } },
                "required": ["param"]
            }));
            Some(ToolDef {
                tool_type: "function".to_string(),
                function: ToolFunction { name, description, parameters },
            })
        })
        .collect()
}

/// Remove `<think>…</think>` blocks (chain-of-thought) from a model response.
/// Used for reasoning models like qwen3 and deepseek-r1 that emit scratchpads.
fn strip_think_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        match rest.find("<think>") {
            None => { out.push_str(rest); break; }
            Some(start) => {
                out.push_str(&rest[..start]);
                rest = &rest[start + "<think>".len()..];
                match rest.find("</think>") {
                    None => break, // unclosed tag — drop remainder
                    Some(end) => rest = &rest[end + "</think>".len()..],
                }
            }
        }
    }
    out.trim().to_string()
}

async fn run_chat_loop(
    host: &str,
    port: u16,
    model: &ModelConfig,
    options: ModelOptions,
    mut messages: Vec<Message>,
    #[cfg(feature = "tools")] tools: Option<Vec<ToolDef>>,
    #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
) -> Result<String> {
    let url = format!("{host}:{port}/api/chat");
    let client = Client::new();

    loop {
        let req = ChatRequest {
            model: model.model.clone(),
            messages: messages.clone(),
            stream: false,
            options: Some(options.clone()),
            think: Some(false),
            #[cfg(feature = "tools")]
            tools: if model.tool.unwrap_or(false) { tools.clone() } else { None },
        };

        debug!("Sending chat request to Ollama ({} messages)", req.messages.len());

        let http_resp = client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| ErhLlmError::OllamaError(e.to_string()))?;

        if !http_resp.status().is_success() {
            let status = http_resp.status();
            let text = http_resp.text().await.unwrap_or_default();
            return Err(ErhLlmError::OllamaError(format!("Ollama chat HTTP {status}: {text}")));
        }

        let parsed: ChatResponse = http_resp
            .json()
            .await
            .map_err(|e| ErhLlmError::OllamaError(e.to_string()))?;

        let assistant_msg = parsed.message;

        #[cfg(feature = "tools")]
        if model.tool.unwrap_or(false)
            && let Some(tool_calls) = &assistant_msg.tool_calls
            && !tool_calls.is_empty() {
                    debug!("Received {} tool call(s), executing...", tool_calls.len());
                    messages.push(assistant_msg.clone());
                    if let Some(comp) = components {
                        for tc in tool_calls {
                            let result = comp
                                .execute_tool(&tc.function.name, tc.function.arguments.clone())
                                .await
                                .unwrap_or_else(|| format!("Tool '{}' not found", tc.function.name));
                            debug!("Tool '{}' result: {}", tc.function.name, result);
                            messages.push(Message { role: "tool".to_string(), content: result, tool_calls: None });
                        }
                    }
                    continue;
                }

        let content = strip_think_tags(&assistant_msg.content);
        debug!("Received response: {}", content);
        return Ok(content);
    }
}

// ── LlmProvider implementation ────────────────────────────────────────────────

/// Ollama provider that calls the Ollama REST API directly.
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    host: String,
    port: u16,
}

impl OllamaProvider {
    pub fn new(config: OllamaConfig) -> Self {
        Self { host: config.host, port: config.port }
    }

    pub fn with_host_port(host: String, port: u16) -> Self {
        Self { host, port }
    }
}

impl Default for OllamaProvider {
    fn default() -> Self { Self::new(OllamaConfig::default()) }
}

#[async_trait::async_trait]
impl LlmProvider for OllamaProvider {
    type Options = ModelOptions;

    async fn embed(&self, model: &ModelConfig, chunk: String) -> Result<Vec<f32>> {
        ollama_embed(&self.host, self.port, model, chunk).await
    }

    async fn chat(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
        user_text: String,
        #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        ollama_chat(
            &self.host, self.port, model, history, options, user_text,
            #[cfg(feature = "tools")] components,
        ).await
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
        ollama_chat_with_system(
            &self.host, self.port, model, history, options, system, user_query,
            #[cfg(feature = "tools")] components,
        ).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream_chat_with_system(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
        system: String,
        user_query: String,
        on_chunk: &mut (dyn FnMut(String) + Send),
        #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        ollama_stream_chat_with_system(
            &self.host, self.port, model, history, options, system, user_query, on_chunk,
            #[cfg(feature = "tools")] components,
        ).await
    }
}

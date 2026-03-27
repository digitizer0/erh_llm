//! Ollama backend implementation for erh_llm.
//!
//! Provides [`ollama_embed`], [`ollama_chat`], and [`ollama_chat_with_system`]
//! as thin wrappers around `ollama-rs` so that [`crate::Query`] does not need
//! to import Ollama types directly.

use log::debug;
use ollama_rs::{
    coordinator::Coordinator,
    generation::{
        chat::{self, ChatMessage},
        embeddings::request::{self, EmbeddingsInput},
    },
};
pub use ollama_rs::models::ModelOptions;

use crate::errors::{ErhLlmError, Result};
use crate::{ModelConfig, ChatMessage as ErhChatMessage};
#[cfg(feature = "tools")]
use crate::ComponentRegistry;

/// Generates a vector embedding for `chunk` using an Ollama instance.
///
/// Returns an empty vector if the request fails.
pub async fn ollama_embed(
    host: &str,
    port: u16,
    model: &ModelConfig,
    chunk: String,
) -> Result<Vec<f32>> {
    let ollama = ollama_rs::Ollama::new(host, port);
    let input = EmbeddingsInput::Single(chunk);
    let options =
        ModelOptions::default().num_ctx(model.context_size.unwrap_or(2048) as u64);
    let response = ollama
        .generate_embeddings(
            request::GenerateEmbeddingsRequest::new(model.model.clone(), input)
                .options(options),
        )
        .await;

    match response {
        Ok(r) => {
            debug!("VectorCount: {:?}", r.embeddings[0].len());
            Ok(r.embeddings[0].clone())
        }
        Err(e) => {
            debug!("Error generating embeddings: {e:?}");
            Err(ErhLlmError::EmbeddingError(e.to_string()))
        }
    }
}

/// Sends a single user message to Ollama, optionally injecting chat history
/// and tool components.
///
/// Returns the raw text response. History is **not** persisted by this function.
pub async fn ollama_chat(
    host: &str,
    port: u16,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    base_options: ModelOptions,
    user_text: String,
    #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
) -> Result<String> {
    let ollama = ollama_rs::Ollama::new(host, port);
    let options = build_options(base_options, model);
    let chat_history = build_chat_history(history);

    let mut coordinator =
        Coordinator::new(ollama, model.model.clone(), chat_history).options(options);

    #[cfg(feature = "tools")]
    if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            debug!("Adding components/tools to Ollama coordinator");
            coordinator = comp.clone().add_tools(coordinator);
        }
    }

    let cm = ChatMessage::new(chat::MessageRole::User, user_text);
    debug!("Sending prompt to Ollama: {:?}", cm);

    let resp = coordinator.chat(vec![cm]).await;
    match resp {
        Ok(r) => {
            debug!("Received response: {}", r.message.content);
            Ok(r.message.content)
        }
        Err(e) => {
            debug!("Error communicating with Ollama: {e:?}");
            Err(ErhLlmError::OllamaError(e.to_string()))
        }
    }
}

/// Sends a system message followed by a user message to Ollama, optionally
/// injecting chat history and tool components.
///
/// Returns the raw text response. History is **not** persisted by this function.
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
    let ollama = ollama_rs::Ollama::new(host, port);
    let options = build_options(base_options, model);
    let chat_history = build_chat_history(history);

    let mut coordinator =
        Coordinator::new(ollama, model.model.clone(), chat_history).options(options);

    #[cfg(feature = "tools")]
    if model.tool.unwrap_or(false) {
        if let Some(comp) = components {
            debug!("Adding components/tools to Ollama coordinator");
            coordinator = comp.clone().add_tools(coordinator);
        }
    }

    let messages = vec![
        ChatMessage::new(chat::MessageRole::System, system),
        ChatMessage::new(chat::MessageRole::User, user_query),
    ];

    debug!("Sending composed prompt to Ollama");
    let resp = coordinator.chat(messages).await;
    match resp {
        Ok(r) => {
            debug!("Received response: {}", r.message.content);
            Ok(r.message.content)
        }
        Err(e) => {
            debug!("Error communicating with Ollama: {e:?}");
            Err(ErhLlmError::OllamaError(e.to_string()))
        }
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn build_options(base: ModelOptions, model: &ModelConfig) -> ModelOptions {
    let opts = if let Some(ctx) = model.context_size {
        base.num_ctx(ctx as u64)
    } else {
        base
    };
    opts.num_predict(-1)
}

fn build_chat_history(history: Vec<ErhChatMessage>) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(history.len() * 2);
    for msg in history {
        out.push(ChatMessage::new(
            chat::MessageRole::User,
            msg.user_message.clone(),
        ));
        out.push(ChatMessage::new(
            chat::MessageRole::Assistant,
            msg.bot_response.clone(),
        ));
    }
    out
}

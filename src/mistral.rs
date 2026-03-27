//! MistralAI backend implementation for erh_llm.
//!
//! Provides [`mistral_chat`] as a thin wrapper around `mistralai-client` so
//! that [`crate::Query`] does not need to import Mistral types directly.
//!
//! Also provides [`MistralProvider`] which implements the [`crate::provider::LlmProvider`] trait.

use log::debug;
use mistralai_client::v1::{
    chat::{ChatMessage, ChatMessageRole, ChatParams},
    client::Client as MistralClient,
    constants::Model,
};

use crate::errors::{ErhLlmError, Result};
use crate::provider::{LlmProvider, MistralConfig};
use crate::{ChatMessage as ErhChatMessage, ModelConfig};
#[cfg(feature = "tools")]
use crate::ComponentRegistry;

/// Sends `text` to the MistralAI cloud API using the provided `api_key`.
///
/// Returns the raw text response. History is **not** persisted by this function.
pub fn mistral_chat(
    api_key: &str,
    text: String,
) -> Result<String> {
    let client = MistralClient::new(Some(api_key.to_string()), None, None, None)
        .map_err(|e| ErhLlmError::MistralError(e.to_string()))?;
    let model = Model::MistralMediumLatest;
    let messages = vec![ChatMessage {
        role: ChatMessageRole::User,
        content: text.clone(),
        tool_calls: None,
    }];
    let options = Some(ChatParams {
        ..Default::default()
    });

    debug!("Sending prompt to MistralAI: {text}");
    let response = client
        .chat(model, messages, options)
        .map_err(|e| ErhLlmError::MistralError(e.to_string()))?;
    debug!("Received response: {}", response.object);
    Ok(response.object)
}

// ── LlmProvider implementation ──────────────────────────────────────────────

/// Mistral provider implementation.
///
/// This struct implements the [`LlmProvider`] trait for Mistral backends.
#[derive(Debug, Clone)]
pub struct MistralProvider {
    api_key: String,
}

impl MistralProvider {
    /// Creates a new Mistral provider with the given configuration.
    pub fn new(config: MistralConfig) -> Self {
        Self {
            api_key: config.api_key,
        }
    }

    /// Creates a new Mistral provider with the given API key.
    pub fn with_api_key(api_key: String) -> Self {
        Self { api_key }
    }
}

/// Options for Mistral requests.
///
/// Currently a placeholder - can be extended with Mistral-specific options.
#[derive(Debug, Clone, Default)]
pub struct MistralOptions {
    // Reserved for future use - e.g., temperature, top_p, etc.
}

#[async_trait::async_trait]
impl LlmProvider for MistralProvider {
    type Options = MistralOptions;

    async fn embed(&self, _model: &ModelConfig, _chunk: String) -> Result<Vec<f32>> {
        // Mistral embeddings would need to be implemented separately
        Err(ErhLlmError::MistralError(
            "Embeddings not yet implemented for Mistral provider".to_string(),
        ))
    }

    async fn chat(
        &self,
        _model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        _options: Self::Options,
        user_text: String,
        #[cfg(feature = "tools")] _components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        // Note: History and tools are not yet fully integrated for Mistral
        // This is a basic implementation that can be extended
        if !history.is_empty() {
            debug!("Warning: Mistral provider does not yet fully support history");
        }

        // The current mistral_chat is synchronous, but we're in an async context
        // We'll use tokio::task::spawn_blocking to run it
        let api_key = self.api_key.clone();
        tokio::task::spawn_blocking(move || {
            mistral_chat(&api_key, user_text)
        })
        .await
        .map_err(|e| ErhLlmError::MistralError(format!("Task join error: {}", e)))?
    }

    async fn chat_with_system(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
        _system: String,
        user_query: String,
        #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
    ) -> Result<String> {
        // Note: System messages are not yet implemented for Mistral
        // For now, we'll just use the regular chat
        debug!("Warning: Mistral provider does not yet support system messages");
        self.chat(
            model,
            history,
            options,
            user_query,
            #[cfg(feature = "tools")]
            components,
        )
        .await
    }
}

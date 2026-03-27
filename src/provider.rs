//! LLM Provider trait for abstracting different LLM backends.
//!
//! This module defines the [`LlmProvider`] trait which provides a common interface
//! for interacting with different LLM providers (Ollama, Anthropic, Mistral, etc.).

use crate::errors::Result;
use crate::{ChatMessage as ErhChatMessage, ModelConfig};
#[cfg(feature = "tools")]
use crate::ComponentRegistry;

/// Trait for LLM provider implementations.
///
/// This trait abstracts the common operations needed for LLM interactions:
/// - Generating embeddings
/// - Chat completions
/// - Chat completions with system messages
///
/// Each provider implementation can define its own options type through
/// the associated `Options` type.
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Provider-specific options type (e.g., ModelOptions for Ollama).
    ///
    /// This allows each provider to have its own configuration while
    /// maintaining a common interface.
    type Options: Default + Clone + Send + Sync;

    /// Generates a vector embedding for the given text chunk.
    ///
    /// # Arguments
    ///
    /// * `model` - The model configuration to use
    /// * `chunk` - The text to generate embeddings for
    ///
    /// # Returns
    ///
    /// A vector of floating-point values representing the embedding,
    /// or an error if the operation fails.
    async fn embed(&self, model: &ModelConfig, chunk: String) -> Result<Vec<f32>>;

    /// Sends a user message to the LLM with optional chat history.
    ///
    /// # Arguments
    ///
    /// * `model` - The model configuration to use
    /// * `history` - Previous chat messages for context
    /// * `options` - Provider-specific options
    /// * `user_text` - The user's message
    /// * `components` - Optional tool/component registry (requires `tools` feature)
    ///
    /// # Returns
    ///
    /// The LLM's response as a string, or an error if the operation fails.
    async fn chat(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
        user_text: String,
        #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
    ) -> Result<String>;

    /// Sends a system message and user message to the LLM with optional chat history.
    ///
    /// # Arguments
    ///
    /// * `model` - The model configuration to use
    /// * `history` - Previous chat messages for context
    /// * `options` - Provider-specific options
    /// * `system` - The system message/prompt to set context
    /// * `user_query` - The user's message
    /// * `components` - Optional tool/component registry (requires `tools` feature)
    ///
    /// # Returns
    ///
    /// The LLM's response as a string, or an error if the operation fails.
    async fn chat_with_system(
        &self,
        model: &ModelConfig,
        history: Vec<ErhChatMessage>,
        options: Self::Options,
        system: String,
        user_query: String,
        #[cfg(feature = "tools")] components: Option<&ComponentRegistry>,
    ) -> Result<String>;
}

/// Configuration for creating an Ollama provider instance.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub host: String,
    pub port: u16,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost".to_string(),
            port: 11434,
        }
    }
}

/// Configuration for creating an Anthropic provider instance.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: Option<String>,
}

impl AnthropicConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: None,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }
}

/// Configuration for creating a Mistral provider instance.
#[derive(Debug, Clone)]
pub struct MistralConfig {
    pub api_key: String,
    pub base_url: Option<String>,
}

impl MistralConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: None,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }
}

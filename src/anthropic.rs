//! Anthropic backend implementation for erh_llm.
//!
//! Provides [`anthropic_chat`] and [`anthropic_chat_with_system`] as thin
//! wrappers around `anthropic_rust` so that [`crate::Query`] does not need to
//! import Anthropic types directly.

use anthropic_rust::{
    ClientBuilder,
    types::{ChatRequestBuilder, ContentBlock, Role},
};
use log::debug;

use crate::errors::{ErhLlmError, Result};
use crate::{ChatMessage as ErhChatMessage, ModelConfig};

/// Sends a single user message to the Anthropic API, optionally injecting
/// chat history as alternating user/assistant turns.
///
/// Returns the raw text response. History is **not** persisted by this function.
pub async fn anthropic_chat(
    api_key: &str,
    model: &ModelConfig,
    history: Vec<ErhChatMessage>,
    user_text: String,
) -> Result<String> {
    let client = ClientBuilder::new()
        .api_key(api_key)
        .build()
        .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

    let mut builder = ChatRequestBuilder::new();

    // Inject prior turns as alternating user/assistant messages.
    for msg in history {
        builder = builder
            .message(Role::User, ContentBlock::text(msg.user_message))
            .message(Role::Assistant, ContentBlock::text(msg.bot_response));
    }

    builder = builder.user_message(ContentBlock::text(user_text.clone()));

    if let Some(temp) = model.temperature {
        builder = builder.temperature(temp);
    }

    let request = builder.build();
    debug!("Sending prompt to Anthropic: {user_text}");

    let response: anthropic_rust::types::Message = client
        .execute_chat(request)
        .await
        .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

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
) -> Result<String> {
    let client = ClientBuilder::new()
        .api_key(api_key)
        .build()
        .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

    let mut builder = ChatRequestBuilder::new().system(system);

    for msg in history {
        builder = builder
            .message(Role::User, ContentBlock::text(msg.user_message))
            .message(Role::Assistant, ContentBlock::text(msg.bot_response));
    }

    builder = builder.user_message(ContentBlock::text(user_query.clone()));

    if let Some(temp) = model.temperature {
        builder = builder.temperature(temp);
    }

    let request = builder.build();
    debug!("Sending prompt to Anthropic (with system): {user_query}");

    let response = client
        .execute_chat(request)
        .await
        .map_err(|e| ErhLlmError::AnthropicError(e.to_string()))?;

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

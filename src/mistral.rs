//! MistralAI backend implementation for erh_llm.
//!
//! Provides [`mistral_chat`] as a thin wrapper around `mistralai-client` so
//! that [`crate::Query`] does not need to import Mistral types directly.

use log::debug;
use mistralai_client::v1::{
    chat::{ChatMessage, ChatMessageRole, ChatParams},
    client::Client as MistralClient,
    constants::Model,
};

/// Sends `text` to the MistralAI cloud API using the provided `api_key`.
///
/// Returns the raw text response. History is **not** persisted by this function.
pub fn mistral_chat(
    api_key: &str,
    text: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = MistralClient::new(Some(api_key.to_string()), None, None, None)?;
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
    let response = client.chat(model, messages, options)?;
    debug!("Received response: {}", response.object);
    Ok(response.object)
}

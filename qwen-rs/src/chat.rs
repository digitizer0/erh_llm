use serde::{Deserialize, Serialize};

#[cfg(feature = "tools")]
use crate::tool::{ToolCall, ToolResult};

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    /// A tool result returned after the model issued a `ToolCall`.
    Tool,
}

impl std::fmt::Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::System => write!(f, "system"),
            ChatRole::User => write!(f, "user"),
            ChatRole::Assistant => write!(f, "assistant"),
            ChatRole::Tool => write!(f, "tool"),
        }
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,

    /// The visible text of the message.
    pub content: String,

    /// Tool invocations requested by the model (only present on `Assistant` messages).
    #[cfg(feature = "tools")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// The raw `<think>…</think>` reasoning produced by the model (stripped from `content`).
    #[cfg(feature = "reasoning")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl ChatMessage {
    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage {
            role: ChatRole::User,
            content: content.into(),
            #[cfg(feature = "tools")]
            tool_calls: None,
            #[cfg(feature = "reasoning")]
            reasoning: None,
        }
    }

    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage {
            role: ChatRole::System,
            content: content.into(),
            #[cfg(feature = "tools")]
            tool_calls: None,
            #[cfg(feature = "reasoning")]
            reasoning: None,
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage {
            role: ChatRole::Assistant,
            content: content.into(),
            #[cfg(feature = "tools")]
            tool_calls: None,
            #[cfg(feature = "reasoning")]
            reasoning: None,
        }
    }

    /// Create a tool-result message.
    #[cfg(feature = "tools")]
    pub fn tool_result(result: &ToolResult) -> Self {
        ChatMessage {
            role: ChatRole::Tool,
            content: result.content.clone(),
            tool_calls: None,
            #[cfg(feature = "reasoning")]
            reasoning: None,
        }
    }
}

/// An in-memory conversation session that accumulates `ChatMessage` turns.
///
/// Typically you will:
/// 1. Create a `ChatSession` with an optional system prompt.
/// 2. Call `push_user` with each user turn.
/// 3. Pass `session.messages()` to [`crate::model::QwenModel::chat`].
/// 4. Call `push_assistant` with the response so future turns see the history.
#[derive(Debug, Clone, Default)]
pub struct ChatSession {
    messages: Vec<ChatMessage>,
}

impl ChatSession {
    /// Create an empty session.
    pub fn new() -> Self {
        ChatSession::default()
    }

    /// Create a session pre-loaded with a system prompt.
    pub fn with_system(system_prompt: impl Into<String>) -> Self {
        let mut s = ChatSession::new();
        s.messages.push(ChatMessage::system(system_prompt));
        s
    }

    /// Append a user message.
    pub fn push_user(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::user(content));
    }

    /// Append an assistant message.
    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::assistant(content));
    }

    /// Append a tool-result message.
    #[cfg(feature = "tools")]
    pub fn push_tool_result(&mut self, result: &ToolResult) {
        self.messages.push(ChatMessage::tool_result(result));
    }

    /// Append an arbitrary message.
    pub fn push(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    /// Read access to the message history.
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Number of turns in the session.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns `true` if the session contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_with_system() {
        let session = ChatSession::with_system("You are a helpful assistant.");
        assert_eq!(session.len(), 1);
        assert_eq!(session.messages()[0].role, ChatRole::System);
    }

    #[test]
    fn test_session_push_user_assistant() {
        let mut session = ChatSession::new();
        session.push_user("Hello");
        session.push_assistant("Hi there!");
        assert_eq!(session.len(), 2);
        assert_eq!(session.messages()[0].role, ChatRole::User);
        assert_eq!(session.messages()[1].role, ChatRole::Assistant);
    }
}

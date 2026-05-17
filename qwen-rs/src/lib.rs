//! # qwen-rs
//!
//! A Rust library for running [Qwen 3.5](https://huggingface.co/Qwen) models locally
//! via [llama.cpp](https://github.com/ggerganov/llama.cpp), with first-class support for:
//!
//! - **Text generation** – single-turn and multi-turn chat
//! - **Tool / function calling** – model-driven tool invocation (feature `tools`, enabled by default)
//! - **Reasoning mode** – Qwen3 `/think` thinking tokens (feature `reasoning`, enabled by default)
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use qwen_rs::{
//!     model::{QwenConfig, QwenModel},
//!     chat::ChatSession,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = QwenConfig::new("/path/to/qwen3.5.Q4_K_M.gguf");
//! let model = QwenModel::load(config)?;
//!
//! let mut session = ChatSession::with_system("You are a helpful assistant.");
//! session.push_user("What is the capital of France?");
//!
//! let response = model.chat(
//!     session.messages(),
//!     &[], // no tools for this call
//! )?;
//! println!("{}", response.content);
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature flags
//!
//! | Feature    | Default | Description |
//! |------------|---------|-------------|
//! | `tools`    | ✅      | Tool / function calling support |
//! | `reasoning`| ✅      | Qwen3 thinking-token support (`<think>…</think>`) |
//! | `native`   | ❌      | Link against the native llama.cpp library (requires llama.cpp installed) |

pub mod chat;
pub mod model;
#[cfg(feature = "reasoning")]
pub mod reasoning;
#[cfg(feature = "tools")]
pub mod tool;

// Convenience re-exports for the most commonly used types.
pub use chat::{ChatMessage, ChatRole, ChatSession};
pub use model::{QwenConfig, QwenModel};
#[cfg(feature = "reasoning")]
pub use reasoning::ReasoningMode;
#[cfg(feature = "tools")]
pub use tool::{Tool, ToolCall, ToolResult};

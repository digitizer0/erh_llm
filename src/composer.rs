//! Prompt composition utilities following Ollama best practices.
//!
//! Ollama (and the underlying models) respond best when:
//! - A clear **system** message establishes the assistant's role and constraints
//!   — sent once as a `system` role message, not embedded inside the user turn.
//! - **RAG context** is injected into the system message so the model can
//!   cleanly distinguish grounding material from the question.
//! - The **user turn** contains only the bare question / task.
//! - Section markers use a consistent, model-agnostic format (`### Section`)
//!   that most instruction-tuned models recognise as structural dividers.
//!
//! The [`PromptComposer`] `constraint` field doubles as the role/system
//! instruction. There is no separate `role` field — a constraint such as
//! "You are a helpful assistant. Answer only in Norwegian." is both a role
//! definition and a behavioural constraint, so merging them avoids redundancy.
//!
//! Use [`PromptComposer::build`] to get a [`ComposedPrompt`] with a ready-made
//! `system` string and a bare `user` string.

/// The two parts of a composed prompt ready to be sent to Ollama.
#[derive(Debug, Clone, Default)]
pub struct ComposedPrompt {
    /// Content for the `system` role message.
    pub system: String,
    /// Content for the `user` role message.
    pub user: String,
}

/// Builder that assembles Ollama prompts from structured parts.
///
/// The `constraint` field acts as the system/role instruction. Set it to
/// describe the assistant's persona and any behavioural rules.
///
/// # Example
/// ```rust
/// let composed = PromptComposer::new()
///     .constraint("You are a helpful assistant. Answer only with information found in the context.")
///     .context("The document states that the deadline is 2026-06-01.")
///     .style("formal")
///     .build("When is the deadline?");
///
/// // composed.system → system message with constraint + context + style
/// // composed.user   → "When is the deadline?"
/// ```
#[derive(Debug, Clone, Default)]
pub struct PromptComposer {
    /// Optional RAG / retrieved context to ground the answer.
    context: Option<String>,
    /// System instruction: role definition and/or behavioural constraint.
    /// Defaults to `"You are a helpful assistant."` when not set.
    constraint: Option<String>,
    /// Optional tone / style instruction (e.g. `"formal"`, `"concise"`).
    style: Option<String>,
}

impl PromptComposer {
    /// Creates a new, empty [`PromptComposer`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets retrieved / RAG context that the model should use to answer.
    pub fn context(mut self, ctx: impl Into<String>) -> Self {
        let s = ctx.into();
        if !s.is_empty() {
            self.context = Some(s);
        }
        self
    }

    /// Sets the system instruction used as both the role definition and
    /// behavioural constraint (e.g. `"You are a support agent. Answer only in
    /// English and only about billing topics."`).
    ///
    /// If not set, `"You are a helpful assistant."` is used as a fallback.
    pub fn constraint(mut self, c: impl Into<String>) -> Self {
        let s = c.into();
        if !s.is_empty() {
            self.constraint = Some(s);
        }
        self
    }

    /// Sets the desired output style (e.g. `"formal"`, `"bullet points"`).
    pub fn style(mut self, s: impl Into<String>) -> Self {
        let st = s.into();
        if !st.is_empty() {
            self.style = Some(st);
        }
        self
    }

    /// Builds a [`ComposedPrompt`] from the configured parts and the given user query.
    ///
    /// The system message is structured as:
    /// ```text
    /// ### Instructions
    /// <constraint>          (or default role)
    ///
    /// ### Context           (omitted when empty)
    /// <rag context>
    ///
    /// ### Style             (omitted when empty)
    /// <style>
    /// ```
    ///
    /// The user message is simply the raw `query` string.
    pub fn build(self, query: impl Into<String>) -> ComposedPrompt {
        let mut system = String::new();

        // Constraint doubles as the role instruction.
        let instructions = self
            .constraint
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());
        system.push_str("### Instructions\n");
        system.push_str(&instructions);
        system.push('\n');

        // RAG context — placed in the system message so it is clearly separated
        // from the user turn and not replayed in conversational history.
        if let Some(ctx) = self.context {
            system.push_str("\n### Context\n");
            system.push_str(&ctx);
            system.push('\n');
        }

        // Style
        if let Some(style) = self.style {
            system.push_str("\n### Style\n");
            system.push_str(&style);
            system.push('\n');
        }

        ComposedPrompt {
            system,
            user: query.into(),
        }
    }
}


/// Controls whether the Qwen3 model uses its internal reasoning/thinking process.
///
/// Qwen3 models support a built-in reasoning mode that produces a `<think>…</think>` block
/// before the final answer. This can improve response quality at the cost of extra tokens.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ReasoningMode {
    /// Force reasoning on: the model MUST produce a `<think>` block.
    /// Maps to the `/think` system-level switch in Qwen3.
    Enabled,

    /// Force reasoning off: the model skips the thinking step entirely.
    /// Maps to the `/no_think` system-level switch in Qwen3.
    Disabled,

    /// Let the model decide per-turn whether to reason (default).
    #[default]
    Auto,
}

impl ReasoningMode {
    /// Returns the system-prompt snippet that activates this reasoning mode.
    pub fn system_hint(&self) -> Option<&'static str> {
        match self {
            ReasoningMode::Enabled => Some("/think"),
            ReasoningMode::Disabled => Some("/no_think"),
            ReasoningMode::Auto => None,
        }
    }
}

/// Strips the `<think>…</think>` block from a model response and returns both parts.
///
/// # Returns
/// `(thinking, answer)` where `thinking` is the reasoning content (if present)
/// and `answer` is the final response text.
pub fn split_thinking(response: &str) -> (Option<String>, String) {
    let think_start = "<think>";
    let think_end = "</think>";

    if let Some(start) = response.find(think_start) {
        if let Some(end) = response.find(think_end) {
            let thinking = response[start + think_start.len()..end].trim().to_string();
            let answer = response[end + think_end.len()..].trim().to_string();
            return (Some(thinking), answer);
        }
    }

    // No thinking block – the full response is the answer.
    (None, response.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_thinking_present() {
        let resp = "<think>Let me reason...</think>Here is the answer.";
        let (thinking, answer) = split_thinking(resp);
        assert_eq!(thinking, Some("Let me reason...".to_string()));
        assert_eq!(answer, "Here is the answer.");
    }

    #[test]
    fn test_split_thinking_absent() {
        let resp = "Just an answer.";
        let (thinking, answer) = split_thinking(resp);
        assert_eq!(thinking, None);
        assert_eq!(answer, "Just an answer.");
    }

    #[test]
    fn test_reasoning_mode_hints() {
        assert_eq!(ReasoningMode::Enabled.system_hint(), Some("/think"));
        assert_eq!(ReasoningMode::Disabled.system_hint(), Some("/no_think"));
        assert_eq!(ReasoningMode::Auto.system_hint(), None);
    }
}

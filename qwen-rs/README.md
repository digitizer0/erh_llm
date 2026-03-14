# qwen-rs

A Rust library for running **Qwen 3.5** models locally via **llama.cpp**, providing a clean, library-first API for:

- **Text-only chat** – single-turn and multi-turn conversations
- **Tool / function calling** – let the model invoke your Rust functions
- **Reasoning mode** – leverage Qwen3's built-in `/think` chain-of-thought

This library works natively with llama.cpp (no Ollama required), making it a lighter and more embeddable option for application developers.

---

## Features

| Feature     | Default | Description |
|-------------|---------|-------------|
| `tools`     | ✅       | Enable tool / function calling support |
| `reasoning` | ✅       | Enable Qwen3 thinking-token support (`<think>…</think>`) |
| `native`    | ❌       | Link against the native llama.cpp library |

The `native` feature requires llama.cpp to be installed on your system (see [Prerequisites](#prerequisites)).

---

## Prerequisites

To use the `native` feature you need:

1. A Qwen 3.5 GGUF model — download from [Hugging Face](https://huggingface.co/Qwen):
   ```
   huggingface-cli download Qwen/Qwen3-1.7B-GGUF qwen3-1.7b-q4_k_m.gguf
   ```
2. llama.cpp installed and its shared library (`libllama`) visible on your `LD_LIBRARY_PATH`/`LIBRARY_PATH`.

---

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
qwen-rs = { path = "qwen-rs" }                    # without native inference
# qwen-rs = { path = "qwen-rs", features = ["native"] }  # with native inference
```

### Text-only chat

```rust
use qwen_rs::{
    model::{QwenConfig, QwenModel},
    chat::ChatSession,
};

fn main() -> anyhow::Result<()> {
    let config = QwenConfig::new("/path/to/qwen3.5.Q4_K_M.gguf");
    let model = QwenModel::load(config)?;

    let mut session = ChatSession::with_system("You are a helpful assistant.");
    session.push_user("Explain the Rust borrow checker in one paragraph.");

    let response = model.chat(session.messages(), &[])?;
    println!("{}", response.content);

    // If reasoning is enabled, print the thinking chain too:
    #[cfg(feature = "reasoning")]
    if let Some(thinking) = response.reasoning {
        eprintln!("[thinking]\n{thinking}");
    }

    Ok(())
}
```

### Tool calling

```rust
use qwen_rs::{
    model::{QwenConfig, QwenModel},
    chat::ChatSession,
    tool::{Tool, ToolCall, ToolResult},
};

fn get_weather(location: &str) -> String {
    format!("The weather in {location} is sunny, 22 °C.")
}

fn main() -> anyhow::Result<()> {
    let weather_tool = Tool::new(
        "get_weather",
        "Return the current weather for a given city.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City name" }
            },
            "required": ["location"]
        }),
    );

    let config = QwenConfig::new("/path/to/qwen3.5.Q4_K_M.gguf");
    let model = QwenModel::load(config)?;

    let mut session = ChatSession::with_system("You are a helpful assistant.");
    session.push_user("What's the weather in Tokyo?");

    let response = model.chat(session.messages(), &[weather_tool])?;

    // Handle tool calls in the response
    #[cfg(feature = "tools")]
    if let Some(calls) = response.tool_calls {
        for call in &calls {
            if call.name == "get_weather" {
                let location = call.arguments["location"].as_str().unwrap_or("unknown");
                let result = ToolResult::new(&call.id, get_weather(location));
                session.push_tool_result(&result);
            }
        }
        // Send the tool results back to get the final answer
        let final_response = model.chat(session.messages(), &[])?;
        println!("{}", final_response.content);
    } else {
        println!("{}", response.content);
    }

    Ok(())
}
```

### Reasoning mode

```rust
use qwen_rs::{
    model::{QwenConfig, QwenModel},
    chat::ChatSession,
    ReasoningMode,
};

fn main() -> anyhow::Result<()> {
    let config = QwenConfig::new("/path/to/qwen3.5.Q4_K_M.gguf")
        .with_reasoning(ReasoningMode::Enabled); // force /think mode

    let model = QwenModel::load(config)?;
    let mut session = ChatSession::new();
    session.push_user("Solve: if 2x + 3 = 11, what is x?");

    let response = model.chat(session.messages(), &[])?;

    if let Some(thinking) = response.reasoning {
        println!("[reasoning]\n{thinking}\n");
    }
    println!("[answer]\n{}", response.content);

    Ok(())
}
```

---

## Architecture

```
qwen-rs/
├── Cargo.toml
└── src/
    ├── lib.rs          ← public API & re-exports
    ├── model.rs        ← QwenConfig, QwenModel (model loading & inference)
    ├── chat.rs         ← ChatMessage, ChatRole, ChatSession
    ├── tool.rs         ← Tool, ToolCall, ToolResult  (feature: tools)
    └── reasoning.rs    ← ReasoningMode, split_thinking  (feature: reasoning)
```

---

## License

MIT

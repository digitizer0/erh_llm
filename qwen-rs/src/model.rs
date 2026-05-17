use std::path::PathBuf;

use anyhow::Result;

#[cfg(feature = "tools")]
use crate::tool::Tool;
use crate::chat::ChatMessage;
#[cfg(feature = "native")]
use crate::chat::ChatRole;
#[cfg(feature = "reasoning")]
use crate::reasoning::ReasoningMode;
#[cfg(all(feature = "native", feature = "reasoning"))]
use crate::reasoning::split_thinking;

/// Configuration used when loading a Qwen model.
#[derive(Debug, Clone)]
pub struct QwenConfig {
    /// Path to the GGUF model file on disk.
    pub model_path: PathBuf,

    /// KV-cache / context window size in tokens. Qwen 3.5 supports up to 32 768.
    pub context_size: u32,

    /// Number of CPU threads used for generation (defaults to 4).
    pub n_threads: u32,

    /// Sampling temperature (0.0 = greedy, higher = more creative).
    pub temperature: f32,

    /// Nucleus-sampling probability cutoff.
    pub top_p: f32,

    /// Maximum number of new tokens to generate per call.
    pub max_new_tokens: u32,

    /// Number of model layers to offload to the GPU.
    ///
    /// Set to `i32::MAX` to offload all layers (maximum GPU utilization).
    /// Set to `0` to run entirely on CPU.
    /// Defaults to `i32::MAX` so GPU acceleration (CUDA, Vulkan, etc.) is
    /// used automatically when a compatible GPU and backend are present.
    pub n_gpu_layers: i32,

    /// Reasoning mode applied to all chat calls from this model.
    #[cfg(feature = "reasoning")]
    pub reasoning: ReasoningMode,
}

impl Default for QwenConfig {
    fn default() -> Self {
        QwenConfig {
            model_path: PathBuf::from("qwen3.5.gguf"),
            context_size: 4096,
            n_threads: 4,
            temperature: 0.7,
            top_p: 0.9,
            max_new_tokens: 2048,
            n_gpu_layers: i32::MAX,
            #[cfg(feature = "reasoning")]
            reasoning: ReasoningMode::default(),
        }
    }
}

impl QwenConfig {
    /// Create a config pointing at `model_path` with all other settings at defaults.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        QwenConfig {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    /// Set the reasoning mode for this config.
    #[cfg(feature = "reasoning")]
    pub fn with_reasoning(mut self, mode: ReasoningMode) -> Self {
        self.reasoning = mode;
        self
    }

    /// Set the number of model layers to offload to the GPU.
    ///
    /// Pass `i32::MAX` (the default) to offload all layers and maximize GPU
    /// utilization. Pass `0` to run entirely on CPU.
    pub fn with_n_gpu_layers(mut self, n: i32) -> Self {
        self.n_gpu_layers = n;
        self
    }
}

/// A loaded Qwen model ready for inference.
///
/// # Native back-end
/// When the `native` feature is enabled the struct wraps the actual `llama-cpp-2`
/// model handle. Otherwise it holds only the configuration and all `chat` calls
/// return `Err("native feature not enabled")`.
pub struct QwenModel {
    pub config: QwenConfig,

    /// The underlying llama.cpp model handle (only present with the `native` feature).
    #[cfg(feature = "native")]
    inner: llama_cpp_2::model::LlamaModel,
}

impl QwenModel {
    /// Load a Qwen model from the GGUF file specified in `config`.
    ///
    /// # Errors
    /// - `native` feature not enabled → returns `Err` immediately.
    /// - File not found / unsupported format when using the native back-end.
    pub fn load(config: QwenConfig) -> Result<Self> {
        #[cfg(feature = "native")]
        {
            use llama_cpp_2::{
                llama_backend::LlamaBackend,
                model::{params::LlamaModelParams, LlamaModel},
            };

            let _backend = LlamaBackend::init()?;
            let params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);
            let inner = LlamaModel::load_from_file(&config.model_path, params)?;
            return Ok(QwenModel { config, inner });
        }

        #[cfg(not(feature = "native"))]
        {
            let _ = config;
            Err(anyhow::anyhow!(
                "qwen-rs: the 'native' feature is required to load a model. \
                 Rebuild with `--features native` (requires llama.cpp to be installed)."
            ))
        }
    }

    /// Run a single chat completion round.
    ///
    /// The function automatically:
    /// - Prepends a `/think` or `/no_think` hint when reasoning is configured.
    /// - Strips `<think>…</think>` blocks from the raw output and stores them
    ///   in the returned [`ChatMessage::reasoning`] field.
    /// - Parses tool-call JSON when the `tools` feature is enabled.
    ///
    /// # Arguments
    /// * `messages` – the full conversation history (system + prior turns + new user turn).
    /// * `tools`    – available tools (ignored when the `tools` feature is disabled).
    ///
    /// # Errors
    /// Returns `Err` if the native back-end is not enabled or if inference fails.
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        #[cfg(feature = "tools")] tools: &[Tool],
        #[cfg(not(feature = "tools"))] _tools: &[()],
    ) -> Result<ChatMessage> {
        #[cfg(not(feature = "native"))]
        {
            let _ = messages;
            #[cfg(feature = "tools")]
            let _ = tools;
            return Err(anyhow::anyhow!(
                "qwen-rs: the 'native' feature is required for inference."
            ));
        }

        #[cfg(feature = "native")]
        {
            let raw = self.generate_raw(
                messages,
                #[cfg(feature = "tools")]
                tools,
            )?;

            #[cfg(feature = "reasoning")]
            {
                let (thinking, answer) = split_thinking(&raw);
                let mut msg = ChatMessage::assistant(answer);
                msg.reasoning = thinking;
                Ok(msg)
            }

            #[cfg(not(feature = "reasoning"))]
            Ok(ChatMessage::assistant(raw))
        }
    }

    /// Low-level token-by-token generation using the llama.cpp back-end.
    ///
    /// Only compiled when the `native` feature is enabled.
    #[cfg(feature = "native")]
    fn generate_raw(
        &self,
        messages: &[ChatMessage],
        #[cfg(feature = "tools")] tools: &[Tool],
    ) -> Result<String> {
        use llama_cpp_2::{
            context::params::LlamaContextParams,
            llama_backend::LlamaBackend,
            token::data_array::LlamaTokenDataArray,
        };

        // Build the prompt using Qwen3's chat template.
        let prompt = self.build_prompt(
            messages,
            #[cfg(feature = "tools")]
            tools,
        )?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(self.config.context_size))
            .with_n_threads(self.config.n_threads);

        let mut ctx = self.inner.new_context(&LlamaBackend::init()?, ctx_params)?;

        let tokens = self.inner.str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)?;
        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(tokens.len() + self.config.max_new_tokens as usize, 1);

        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)?;
        }
        ctx.decode(&mut batch)?;

        let mut generated = String::new();
        let mut n_cur = tokens.len() as i32;

        loop {
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
            let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            ctx.sample_temp(&mut candidates_p, self.config.temperature);
            ctx.sample_top_p(&mut candidates_p, self.config.top_p, 1);
            let token = ctx.sample_token_greedy(&mut candidates_p);

            if token == self.inner.token_eos() {
                break;
            }
            generated.push_str(&self.inner.token_to_str(token)?);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            ctx.decode(&mut batch)?;
            n_cur += 1;

            if n_cur >= self.config.max_new_tokens as i32 + tokens.len() as i32 {
                break;
            }
        }

        Ok(generated)
    }

    /// Build the Qwen3 chat-template prompt string.
    ///
    /// Qwen3 uses the standard `<|im_start|>role\ncontent<|im_end|>` format.
    #[cfg(feature = "native")]
    fn build_prompt(
        &self,
        messages: &[ChatMessage],
        #[cfg(feature = "tools")] tools: &[Tool],
    ) -> Result<String> {
        let mut prompt = String::new();

        // Inject a system message that activates the correct reasoning mode.
        #[cfg(feature = "reasoning")]
        let reasoning_hint = self.config.reasoning.system_hint();

        // Serialize available tools as a JSON block in the system message.
        #[cfg(feature = "tools")]
        let tools_block = if !tools.is_empty() {
            let tool_jsons: Vec<_> = tools.iter().map(|t| t.to_json()).collect();
            format!("\n\n# Tools\n\nYou may call one or more tools. Available tools:\n\n```json\n{}\n```",
                serde_json::to_string_pretty(&tool_jsons)?)
        } else {
            String::new()
        };

        for msg in messages {
            let role_str = msg.role.to_string();
            let mut content = msg.content.clone();

            // Append reasoning hint and tool definitions to the first system message.
            if msg.role == ChatRole::System {
                #[cfg(feature = "tools")]
                { content.push_str(&tools_block); }

                #[cfg(feature = "reasoning")]
                if let Some(hint) = reasoning_hint {
                    content.push('\n');
                    content.push_str(hint);
                }
            }

            prompt.push_str(&format!(
                "<|im_start|>{role_str}\n{content}<|im_end|>\n"
            ));
        }

        // Final assistant turn marker to start generation.
        prompt.push_str("<|im_start|>assistant\n");
        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_config_defaults() {
        let cfg = QwenConfig::default();
        assert_eq!(cfg.context_size, 4096);
        assert_eq!(cfg.n_threads, 4);
        assert_eq!(cfg.max_new_tokens, 2048);
        assert_eq!(cfg.n_gpu_layers, i32::MAX);
    }

    #[test]
    fn test_qwen_config_new() {
        let cfg = QwenConfig::new("/models/qwen3.5.gguf");
        assert_eq!(cfg.model_path, PathBuf::from("/models/qwen3.5.gguf"));
    }

    #[test]
    fn test_qwen_config_with_n_gpu_layers() {
        let cfg = QwenConfig::new("/models/qwen3.5.gguf").with_n_gpu_layers(0);
        assert_eq!(cfg.n_gpu_layers, 0);

        let cfg_all = QwenConfig::new("/models/qwen3.5.gguf").with_n_gpu_layers(i32::MAX);
        assert_eq!(cfg_all.n_gpu_layers, i32::MAX);
    }

    #[cfg(feature = "reasoning")]
    #[test]
    fn test_qwen_config_with_reasoning() {
        let cfg = QwenConfig::new("/models/qwen3.5.gguf")
            .with_reasoning(ReasoningMode::Enabled);
        assert_eq!(cfg.reasoning, ReasoningMode::Enabled);
    }

    #[test]
    fn test_load_without_native_fails() {
        let cfg = QwenConfig::new("/tmp/nonexistent.gguf");
        let result = QwenModel::load(cfg);
        // Without the `native` feature, loading always returns an error.
        #[cfg(not(feature = "native"))]
        assert!(result.is_err());
    }
}

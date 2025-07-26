#[derive(Debug,Clone,Default,PartialEq)]
pub struct OllamaConfig {
    pub url: String, // Ollama server URL
    pub port: u16, // Ollama server port
}


#[derive(Debug,Clone,Default,PartialEq)]
pub struct Config {
    pub ollama: Option<OllamaConfig>,
}

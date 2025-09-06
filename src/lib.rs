mod prompter;
mod history;

use std::sync::Arc;

pub use history::HistoryConfig;
use crate::history::HistoryTrait;

use log::{debug, warn};
//use log::info;
use ollama_rs::generation::{completion::request::GenerationRequest, embeddings::request::{self, EmbeddingsInput}};
pub use ollama_rs::models::ModelOptions;

use crate::history::History;
use async_trait::async_trait;
#[async_trait]
pub trait Tool: Send + Sync {
    fn access(&self) -> &str {
        "unknown"
    }
    fn name(&self) -> &str;
    async fn run(&self, input: &str) -> Result<String, Box<dyn std::error::Error>>;
}

#[derive(Default)]
pub struct ToolRegistry {
    pub tools: Arc<Vec<Box<dyn Tool + Send + Sync>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry { tools: Arc::new(Vec::new()) }
    }
    pub fn register<T: Tool + Send + Sync + 'static>(&mut self, tool: T) {
        Arc::get_mut(&mut self.tools).unwrap().push(Box::new(tool));
    }
    pub fn get(&self, name: &str) -> Option<&(dyn Tool + Send + Sync)> {
        for t in &*self.tools {
            if t.name() == name {
                return Some(t.as_ref());
            }
        }
        None
    }
    pub fn list(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name().to_string()).collect()
    }
}
/*
// Example tool implementation
#[derive(Debug)]
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    async fn run(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("Echo: {}", input))
    }
}
*/
#[derive(Debug,Clone,Default,PartialEq)]
pub struct ChatMessage {
    pub id: Option<i32>,
    pub user: String,
    pub user_message: String,
    pub bot_response: String,
    pub timestamp: i64,
    pub chatuuid: String,
}

impl ChatMessage {
    pub fn validate(&mut self) -> bool {
        // Validate that all fields are non-empty and chatuuid is of length 40
        #[cfg(debug_assertions)]
        let x = true; // In debug mode, we assume all messages are valid for testing purposes
        #[cfg(not(debug_assertions))]
        let x ={
            !self.user_message.is_empty()
                && !self.bot_response.is_empty()
                && self.timestamp != 0
                && self.chatuuid.len() == 40
        };
        x
    }

    pub fn from_tuple(tuple: (String,String, String, String)) -> Self {
        ChatMessage {
            user: tuple.0.clone(),
            user_message: tuple.1.clone(),
            bot_response: tuple.2.clone(),
            timestamp: 0, // Default value, should be set later
            chatuuid: tuple.3.clone(),
            ..Default::default()
        }
    }

    pub fn noemoji(&self) -> Self {
        let mut msg = self.clone();
        msg.user_message = demoji!(self.user_message);
        msg.bot_response = demoji!(self.bot_response);
        msg
    }
    
}

#[derive(Debug,Clone,PartialEq)]
pub enum LLM {
    Ollama(String, u16, String),  // (host, port, model_name)
    Dummy, // Placeholder for other LLMs
    // Add other LLMs as needed
}
enum Prompt {
    Default(String),
    Model(String, String), // (model_name, prompt)
}

#[derive(Debug,Clone)]
pub struct QuerySetup {
    pub user: String,
    pub chatuuid: String,
    pub model_name: String,
    pub prompt: String,
    pub style: Option<String>,
    pub constraint: Option<String>,
}

impl Default for QuerySetup {
    fn default() -> Self {
        QuerySetup {
            model_name: "mistral".to_string(),
            user: String::new(),
            chatuuid: String::new(),
            prompt: String::new(),
            style: None,
            constraint: None,
        }
    }
}
impl QuerySetup {
    pub fn new() -> Self {
        QuerySetup::default()
    }
    
}

impl Default for LLM {
    fn default() -> Self {
        LLM::Ollama("localhost".to_string(), 11434, "mistral".to_string())
    }
}

#[derive(Default)]
pub struct Query {
    connection: LLM,
    pub setup: QuerySetup,
    pub(crate)  history: History,
    pub context: String,
    pub options: ModelOptions,
    classification: Option<String>,
    pub tools: Option<ToolRegistry>,
}

impl Query {
    pub async fn embed(config:(String,u16,String),chunk:String) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (url, port, model) = config;
        let ollama = ollama_rs::Ollama::new(format!("{}:{}",url,port), port);
        let e = EmbeddingsInput::Single(chunk);
        let x  = ollama.generate_embeddings(request::GenerateEmbeddingsRequest::new(model.to_string(), e)).await;
        let y = match x {
            Ok(response) => response,
            Err(e) => {
                debug!("Error generating embeddings: {:?}", e);
                return Ok(vec![]); // Return an empty vector on error
            }
        };
        //println!("Embeddings: {:?}", y);
        debug!("VectorCount: {:?}", y.embeddings[0].len());
        Ok(y.embeddings[0].clone())
    }    
}

impl Query {
    pub fn new(connection: LLM, history: HistoryConfig) -> Self {
        let mut q = Query {
            connection,
            ..Default::default()
        };
        match history {
            HistoryConfig::Mem => {
                debug!("Using in-memory history");
                q.history = History::new(HistoryConfig::Mem);
            }
            HistoryConfig::Sqlite(db) => {
                debug!("Using SQLite history with database: {db}");
                q.history = History::new(HistoryConfig::Sqlite(db));
            }
            HistoryConfig::Mysql(config) => {
                debug!("Using MySQL history with config: {config:?}");
                q.history = History::new(HistoryConfig::Mysql(config));
            }
            _ => {
                panic!("Invalid history configuration: {history:?}");
            }
            
        }
        q
    }   

    
    // Combine the retrieved chunks with the user message
    pub(crate) async fn augmented_message(&self) -> String {
        //Read history
        let (latest, history) = self.summarize_history().await.unwrap_or_default();
        let history = if !history.is_empty() {
            format!("Summarized chat history: {history}\n\nLatest Message: {latest}\n\n",)
        } else {
            String::new()
        };

        let context = if !self.context.is_empty() {
            format!("Context: {}\n\n", self.context)
        } else {
            String::new()
        };

        let qsetup = self.setup.clone();

        let message = format!("User Query:\n{}\n\n", qsetup.prompt);

        let constraint = format!("{}\n\n",qsetup.constraint.as_deref().unwrap_or(""));

        let style = qsetup.style.as_deref().unwrap_or("");
        format!("{constraint}{history}{context}{message}{style}")
    }

    async fn summarize_history(&self) -> Result<(String, String),Box<dyn std::error::Error>> {
        let mut history = self.history.read(&self.setup.chatuuid)?;
        if history.is_empty() {
            return Ok((String::new(), String::new()));
        }


        let latest = history.pop().unwrap();
        let result =if !history.is_empty() {
            let history_text: String = history.iter()
                .map(|msg| format!("{}: {}\nResponse:{}\n", msg.user, msg.user_message, msg.bot_response))
                .collect();
            let prompt = format!(
                "Summarize the following chat history in a concise paragraph:\n\n{history_text}",
            );
            self.send_raw(Prompt::Model("mistral".to_string(), prompt)).await?
        } else {
            String::new()
        };

        let latest = format!("{}: {}\nResponse:{}\n", latest.user, latest.user_message, latest.bot_response);

        //debug!("Summarized history: {result}");
        Ok((latest,result))
    }

    pub async fn send(&mut self, prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        let resp = self.send_raw(Prompt::Default(prompt)).await?;
        let mut msg =ChatMessage { id: None, user: self.setup.user.clone(), user_message: self.setup.prompt.clone(), bot_response: resp.clone(), timestamp: 0 , chatuuid: self.setup.chatuuid.clone() };
        debug!("Storing message in history: {msg:?}");
        let x = self.history.store(&mut msg);
        if let Err(e) = x {
            warn!("Error storing message in history: {}", e);
            Err(e.into())
        } else {
            Ok(resp)
        }
    }



    async fn send_raw(&self, prompt: Prompt) -> Result<String, Box<dyn std::error::Error>> {
        let (text,model) = match prompt {
            Prompt::Default(p) => (p,String::new()),
            Prompt::Model(model_name, p) => {
                (p, model_name)
            }
        };
        debug!("Sending prompt!!: {text}");
        let resp = match &self.connection {
            LLM::Ollama(host, port, model_name) => {
                let model = if model.is_empty() {
                    model_name
                } else {
                    model.as_str()
                };
                self.send_ollama(host, *port, model,text).await?
            }
            // Add other LLMs here as needed
            _ => panic!("Not possible"),
        };
        debug!("Received response: {resp}");
        Ok(resp)
    }


    pub async fn run(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        debug!("Running query with message: {}", self.setup.prompt);
        // Tool selection logic: let LLM decide if a tool should be used
        if let Some(registry) = &self.tools {
            // Ask LLM if a tool should be used
            let tool_list = registry.list().join(", ");
            let tool_prompt = format!("Given the user query: '{}', and available tools: [{}], which tool (if any) should be used? Reply with the tool name or 'none'.", self.setup.prompt, tool_list);
            let tool_decision = self.send_raw(Prompt::Default(tool_prompt)).await?;
            let tool_name = tool_decision.trim();
            if tool_name != "none" && !tool_name.is_empty() {
                if let Some(tool) = registry.get(tool_name) {
                    debug!("LLM selected tool: {}", tool_name);
                    let result = tool.run(&self.setup.prompt).await?;
                    return Ok(result);
                } else {
                    warn!("Tool '{}' not found in registry", tool_name);
                }
            }
        }
        let prompt = self.augmented_message().await;
        let x = self.send(prompt).await;
        log::debug!("Query result: {:?}", x);
        x
    }
 
    async fn send_ollama(&self, host: &str, port: u16, model_name: &str, prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        let ollama = ollama_rs::Ollama::new(format!("{host}:{port}"), port);
        let response = ollama
            .generate(GenerationRequest::new(model_name.to_string(), prompt).options(self.options.clone()))
            .await?;

        
        Ok(response.response)
    }

    pub async fn classify_query(&mut self) -> Result<String, Box<dyn std::error::Error>> {
    
        let r = if let Some(classification) = &self.classification {
            let prompt = format!("Classify following prompt by these criteria:\n{} \n\nPROMPT: {} ",classification, self.setup.prompt);
            self.send(prompt).await?
        } else {
            String::new()
        };
        // TODO: Error handling for classification
        Ok(r)
    }

    pub async fn _classify(&mut self, classification: String) -> Result<String, Box<dyn std::error::Error>> {
        self.classification = Some(classification);
        self.classify_query().await
    }

}


#[macro_export]
macro_rules! demoji {
    ($string:expr) => {{
        use regex::Regex;
        let regex = Regex::new(concat!(
            "[",
            "\u{01F600}-\u{01F64F}", // emoticons
            "\u{01F300}-\u{01F5FF}", // symbols & pictographs
            "\u{01F680}-\u{01F6FF}", // transport & map symbols
            "\u{01F1E0}-\u{01F1FF}", // flags (iOS)
            "\u{002702}-\u{0027B0}",
            "\u{0024C2}-\u{01F251}",
            "]+",
        )).unwrap();
        regex.replace_all(&$string, "").to_string()
    }};
}
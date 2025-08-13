mod prompter;
mod history;

pub use history::HistoryConfig;
use crate::history::HistoryTrait;

use log::{debug, warn};
//use log::info;
use ollama_rs::generation::{completion::request::GenerationRequest, embeddings::request::{self, EmbeddingsInput}};
pub use ollama_rs::models::ModelOptions;

use crate::history::History;

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

    pub fn from_tuple(tuple: (String, String, String)) -> Self {
        ChatMessage {
            user_message: tuple.0,
            bot_response: tuple.1,
            timestamp: 0, // Default value, should be set later
            chatuuid: tuple.2,
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
    // Add other LLMs as needed
}

enum Prompt {
    Default(String),
    Model(String, String), // (model_name, prompt)
}

impl Default for LLM {
    fn default() -> Self {
        LLM::Ollama("localhost".to_string(), 11434, "mistral".to_string())
    }
}

#[derive(Debug, Default)]
pub struct Query {
    connection: LLM,
    pub(crate)  history: History,
    pub classification: Option<String>, // Optional classification of the query
    pub constraint: Option<String>, // Optional constraint for the query
    pub style: Option<String>, // Optional style for the query
    pub context: String,
    pub chatuuid: String, // UUID for the chat session
    pub user: String, // User identifier
    pub message: String,
    pub model_name: String,
    pub options: ModelOptions,
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
        let h = self.summarize_history().await.unwrap_or_default();
        let history = if !h.is_empty() {
            format!("Summarized chat history: {h}\n\n",)
        } else {
            String::new()
        };

        let context = if !self.context.is_empty() {
            format!("Context: {}\n\n", self.context)
        } else {
            String::new()
        };

        let message = format!("User Query:\n{}\n\n", self.message);

        let constraint = format!("{}\n\n",self.constraint.as_deref().unwrap_or(""));

        let style = self.style.as_deref().unwrap_or("");
        format!("{constraint}{history}{context}{message}{style}")
    }

    async fn summarize_history(&self) -> Result<String,Box<dyn std::error::Error>> {
        let h = self.history.read(&self.chatuuid)?;
        if h.is_empty() {
            return Ok(String::new());
        }
        let history_text: String = h.iter()
            .map(|msg| format!("{}: {}\nResponse:{}\n", msg.user, msg.user_message, msg.bot_response))
            .collect();
        let prompt = format!(
            "Summarize the following chat history in a concise paragraph:\n\n{history_text}",
            
        );
        let result = self.send_raw(Prompt::Default(prompt)).await?;
        debug!("Summarized history: {result}");
        Ok(result)
    }

    pub async fn send(&mut self, prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        let resp = self.send_raw(Prompt::Default(prompt)).await?;
        let mut msg =ChatMessage { id: None, user: self.user.clone(), user_message: self.message.clone(), bot_response: resp.clone(), timestamp: 0 , chatuuid: self.chatuuid.clone() };
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
        debug!("Sending prompt: {text}");
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
            //_ => panic!("Not possible"),
            
        };
        debug!("Received response: {resp}");
        Ok(resp)
    }


    pub async fn run(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        debug!("Running query with message: {}", self.message);
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
            let prompt = format!("Classify following prompt by these criteria:\n{} \n\nPROMPT: {} ",classification, self.message);
            self.send(prompt).await?
        } else {
            String::new()
        };
        // TODO: Error handling for classification
        Ok(r)
    }

    pub async fn classify(&mut self, classification: String) -> Result<String, Box<dyn std::error::Error>> {
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
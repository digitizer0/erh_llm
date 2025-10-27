mod history;
#[cfg(feature="tools")]
mod components;

//use std::collections::HashMap;

pub use history::HistoryConfig;
use crate::history::HistoryTrait;

use log::{debug, warn};
use ollama_rs::{coordinator::Coordinator, generation::{chat, embeddings::request::{self, EmbeddingsInput}}};
pub use ollama_rs::models::ModelOptions;

use crate::history::History;
#[cfg(feature="tools")]
pub use crate::components::{ComponentRegistry, Component, ComponentSource, tools::Tool as Tool, prompt::Prompt as Prompt, resource::Resource, sampling::Sampling};


#[derive(Debug,Clone,Default)]
pub struct ChatMessage {
    pub ollama: Option<chat::ChatMessage>,
    pub id: Option<i32>,
    pub user: String,
    pub user_message: String,
    pub bot_response: String,
    pub timestamp: i64,
    pub chatuuid: String,
}

impl ChatMessage {
    pub fn validate(&mut self) -> bool {
        {
            !self.user_message.is_empty()
                && !self.bot_response.is_empty()
                //&& self.timestamp != 0
                //&& self.chatuuid.len() == 40 
        }
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
pub enum UserPrompt {
    Default(String),
    Model(String, String), // (model_name, prompt)
}

#[derive(Clone)]
pub struct QuerySetup {
    pub user: String,
    pub chatuuid: String,
    pub model_name: String,
    pub prompt: String,
#[cfg(feature="tools")]
    pub components: Option<ComponentRegistry>,
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
            #[cfg(feature="tools")]
            components: None,
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
    pub(crate)  history: Option<History>,
    pub context: String,
    pub options: ModelOptions,
    classification: Option<String>,
    #[cfg(feature="tools")]
    pub components: Option<ComponentRegistry>,
}

impl Query {
    pub async fn embed(config:(String,u16,String),chunk:String) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (url, port, model) = config;
        let ollama = ollama_rs::Ollama::new(format!("{url}:{port}"), port);
        let e = EmbeddingsInput::Single(chunk);
        let x  = ollama.generate_embeddings(request::GenerateEmbeddingsRequest::new(model.to_string(), e)).await;
        let y = match x {
            Ok(response) => response,
            Err(e) => {
                debug!("Error generating embeddings: {e:?}");
                return Ok(vec![]); // Return an empty vector on error
            }
        };
        //println!("Embeddings: {:?}", y);
        debug!("VectorCount: {:?}", y.embeddings[0].len());
        Ok(y.embeddings[0].clone())
    }

    pub async fn get_history(uuid: &str, history: HistoryConfig) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
        let h = History::new(history);
        let msgs = h.read(uuid)?;
        Ok(msgs)
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
                q.history = Some(History::new(HistoryConfig::Mem));
            }
            HistoryConfig::Sqlite(db) => {
                debug!("Using SQLite history with database: {db}");
                q.history = Some(History::new(HistoryConfig::Sqlite(db)));
            }
            HistoryConfig::Mysql(config) => {
                debug!("Using MySQL history with config: {config:?}");
                q.history = Some(History::new(HistoryConfig::Mysql(config)));
            }
            _ => {
                q.history = None;
            }
            
        }
        q
    }   

    
    // Combine the retrieved chunks with the user message
    pub async fn execute(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        debug!("Running query with message: {}", self.setup.prompt);
        //Read history
        let (latest, history) = self.summarize_history().await.unwrap_or_default();
        let history = if !history.is_empty() {
            format!("SUMMARIZED CHAT:\n{history}\n\nLATEST MESSAGE: {latest}\n\n",)
        } else {
            String::new()
        };

        let context = if !self.context.is_empty() {
            format!("CONTEXT: {}\n\n", self.context)
        } else {
            String::new()
        };

        let qsetup = self.setup.clone();

        let message = format!("USER QUERY: {}\n\n", qsetup.prompt);

        let constraint = format!("CONSTRAINT: {}\n\n",qsetup.constraint.as_deref().unwrap_or(""));

        let style = qsetup.style.as_deref().unwrap_or("");
        let p = format!("{constraint}{history}{context}{message}{style}");
        let x = self.send(p).await.unwrap_or_default();
        log::debug!("Query result: {x:?}");
        Ok(x)
    }

    async fn summarize_history(&self) -> Result<(String, String),Box<dyn std::error::Error>> {
        let mut history = if let Some(history) = &self.history {
            let history = history.read(&self.setup.chatuuid)?;
            if history.is_empty() {
                return Ok((String::new(), String::new()));
            }
            history
        } else {
            return Ok((String::new(), String::new()));
        };

        let latest = history.pop().unwrap();
        let result =if !history.is_empty() {
            let history_text: String = history.iter()
                .map(|msg| format!("USER: {}\nMESSAGE: {}\nRESPONSE: {}\n", msg.user, msg.user_message, msg.bot_response))
                .collect();
            let prompt = format!(
                "QUERY: Summarize the following chat history in a concise paragraph:\n\nCHAT_HISTORY: {history_text}\n",
            );
            self.send_raw(UserPrompt::Model("mistral".to_string(), prompt)).await?
        } else {
            String::new()
        };

        let latest = format!("USER: {}\nMESSAGE: {}\nRESPONSE: {}\n", latest.user, latest.user_message, latest.bot_response);

        //debug!("Summarized history: {result}");
        Ok((latest,result))
    }

    pub async fn send(&mut self, prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        let resp = self.send_raw(UserPrompt::Default(prompt)).await?;
        let mut msg =ChatMessage { id: None, user: self.setup.user.clone(), user_message: self.setup.prompt.clone(), bot_response: resp.clone(), timestamp: 0 , chatuuid: self.setup.chatuuid.clone(),..Default::default() };
        debug!("Storing message in history: {msg:?}");
        let x = if let Some(history) = &mut self.history {
            history.store(&mut msg)
        } else {
            Ok(())
        };
        if let Err(e) = x {
            warn!("Error storing message in history: {e}");
            Err(e)
        } else {
            Ok(resp)
        }
    }



    pub async fn send_raw(&self, prompt: UserPrompt) -> Result<String, Box<dyn std::error::Error>> {
        let (text,model) = match prompt {
            UserPrompt::Default(p) => (p,String::new()),
            UserPrompt::Model(model_name, p) => {
                (p, model_name)
            }
        };
        //debug!("Sending prompt!!: {text}");
        let resp = match &self.connection {
            LLM::Ollama(host, port, model_name) => {
                let model = if model.is_empty() {
                    model_name
                } else {
                    model.as_str()
                };
                let history = vec![];
                let ollama = ollama_rs::Ollama::new(format!("{host}:{port}"), *port);
                let mut x = Coordinator::new(ollama, model.to_string(), history)
                    .options(self.options.clone());

                let cm = chat::ChatMessage::new(chat::MessageRole::Assistant, text);
                let resp = x.chat(vec![cm]).await?;
                resp.message.content
            }
            // Add other LLMs here as needed
            _ => panic!("Not possible"),
        };
        debug!("Received response: {resp}");
        Ok(resp)
    }

    pub async fn classify_query(&mut self) -> Result<String, Box<dyn std::error::Error>> {
    
        let r = if let Some(classification) = &self.classification {
            let prompt = format!("QUERY: Classify following prompt by these criteria: {}\n\nPROMPT: {}", classification, self.setup.prompt);
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
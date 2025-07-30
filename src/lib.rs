mod prompter;
mod config;
mod history;

use crate::history::HistoryTrait;

//use log::info;
use ollama_rs::generation::{completion::request::GenerationRequest};
pub use ollama_rs::models::ModelOptions;

use crate::history::History;

#[derive(Debug,Clone,Default,PartialEq)]
pub struct ChatMessage {
    pub id: Option<i32>,
    pub user: String,
    pub message: String,
    pub timestamp: i64,
}

#[derive(Debug,Clone,PartialEq)]
pub enum LLM {
    Ollama(String, u16, String),  // (host, port, model_name)
    // Add other LLMs as needed
}

impl Default for LLM {
    fn default() -> Self {
        LLM::Ollama("localhost".to_string(), 11434, "mistral".to_string())
    }
}

#[derive(Debug,Clone, Default)]
pub struct Query {
    connection: LLM,
    pub(crate) history: History,
    pub classification: Option<String>, // Optional classification of the query
    pub constraint: Option<String>, // Optional constraint for the query
    pub style: Option<String>, // Optional style for the query
    pub context: String,
    pub message: String,
    pub model_name: String,
    pub options: ModelOptions,
}

impl Query {
    pub fn new(connection: LLM) -> Self {
        Query {
            connection,
            ..Default::default()
        }
    }   

    
    // Combine the retrieved chunks with the user message
    pub(crate) fn augmented_message(&self) -> String {
        //Read history
        
        let constraint = self.constraint.as_deref().unwrap_or("");
        let style = self.style.as_deref().unwrap_or("");
        format!("{}\n\nContext:{}\n\nUser Query:\n{}\n{}",constraint,self.context,self.message,style)
    }

    async fn summarize_history(&self) -> Result<String,Box<dyn std::error::Error>> {
        self.history.read()?;

        let prompt = String::from("");
        let result = self.send_raw(prompt).await?; 
        Ok(result)
    }

    pub async fn send(&mut self, prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        let resp = self.send_raw(prompt).await?;
        let msg =ChatMessage { id: None, user: "test".to_string(), message: self.message.clone(), timestamp: 0 };
        self.history.store(&msg)?;
        Ok(resp)
    }

    async fn send_raw(&self, prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        let resp = match &self.connection {
            LLM::Ollama(host, port, model_name) => {
                self.send_ollama(host, *port, model_name, prompt).await?
            }
            // Add other LLMs here as needed
            //_ => panic!("Not possible"),
            
        };
        Ok(resp)
    }


    pub async fn run(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let prompt = self.augmented_message();
        self.send(prompt).await
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

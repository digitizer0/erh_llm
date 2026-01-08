mod history;
#[cfg(feature="tools")]
mod components;

pub use history::HistoryConfig;
use mistralai_client::v1::{chat::{ChatMessage as MistralChatMessage, ChatParams}, client::Client as MistralClient, constants::Model};
use serde::{Deserialize, Serialize};

use crate::history::HistoryTrait;

use log::{debug, warn};
use ollama_rs::{coordinator::Coordinator, generation::{chat, embeddings::request::{self, EmbeddingsInput}}};
pub use ollama_rs::models::ModelOptions;

use crate::history::History;
#[cfg(feature="tools")]
pub use crate::components::{ComponentRegistry, Component, tools::Tool as Tool, prompt::Prompt as Prompt, resource::Resource, sampling::Sampling};


#[derive(Debug, Clone, Serialize, Deserialize, Default,PartialEq)]
pub struct ModelConfig {
    pub model: String,
    pub short: Option<String>,
    pub tool: Option<bool>,
    pub temperature: Option<f32>,
    pub context_size: Option<u32>,
}

impl ModelConfig {
    pub fn new(model: &str) -> Self {
        ModelConfig {
            model: model.to_string(),
            ..Default::default()
        }
    }
}

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
    Ollama(String, u16, ModelConfig),  // (host, port, model_name)
    MistralAI(String),
    Dummy, // Placeholder for other LLMs
    // Add other LLMs as needed
}
pub enum UserPrompt {
    Default(String),
    Model(ModelConfig, String), // (model_name, prompt)
}

#[derive(Clone)]
pub struct QuerySetup {
    pub user: String,
    pub chatuuid: String,
    pub model : ModelConfig,
    pub prompt: String,
#[cfg(feature="tools")]
    pub components: Option<ComponentRegistry>,
    pub style: Option<String>,
    pub constraint: Option<String>,
}

impl Default for QuerySetup {
    fn default() -> Self {
        QuerySetup {
            model: ModelConfig::default(),
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
        LLM::Ollama("localhost".to_string(), 11434, ModelConfig::new("mistral"))
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
    pub async fn embed(config:(String,u16,ModelConfig),chunk:String) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (url, port, model) = config;
        let ollama = ollama_rs::Ollama::new(format!("{url}:{port}"), port);
        let e = EmbeddingsInput::Single(chunk);
        let options = ModelOptions::default();
        options.num_ctx(model.context_size.unwrap_or(2048) as u64);
        let x  = ollama.generate_embeddings(request::GenerateEmbeddingsRequest::new(model.model.clone(), e)).await;
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
            HistoryConfig::Sqlite(db) => {
                debug!("Using SQLite history with database: {db}");
                q.history = Some(History::new(HistoryConfig::Sqlite(db)));
            }
            HistoryConfig::Mysql(config) => {
                debug!("Using MySQL history with config: {config:?}");
                q.history = Some(History::new(HistoryConfig::Mysql(config)));
            }
            HistoryConfig::MsSql(config) => {
                debug!("Using MSSQL history with config: {config:?}");
                q.history = Some(History::new(HistoryConfig::MsSql(config)));
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
        debug!("ComponentRegistry: {:?}", self.components.as_ref().map(|c| c.components.len()));
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
            self.send_raw(UserPrompt::Model(ModelConfig::new("mistral"), prompt)).await?
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
        let (text,_model) = match prompt {
            UserPrompt::Default(p) => (p,ModelConfig::default()),
            UserPrompt::Model(model, p) => {
                (p, model)
            }
        };
        //debug!("Sending prompt!!: {text}");
        let resp = match &self.connection {
            LLM::Ollama(host, port, model) => {
                let history = vec![];
                let ollama = ollama_rs::Ollama::new(format!("{host}:{port}"), *port);
                let mut coordinator = Coordinator::new(ollama, model.model.to_string(), history)
                    .options(self.options.clone());

                let cm = chat::ChatMessage::new(chat::MessageRole::User, text);

                #[cfg(feature="tools")]
                if model.tool.unwrap_or(false){
                    if let Some(components) = &self.components {
                        debug!("Adding components/tools to Ollama coordinator");
                        coordinator = components.clone().add_tools(coordinator);
                    }
                }
                
                debug!("Sending prompt to Ollama: {:?}", cm);
                let resp = coordinator.chat(vec![cm]).await;
                match resp {
                    Ok(response) => response.message.content,
                    Err(e) => {
                        debug!("Error communicating with Ollama: {e:?}");
                        return Err(Box::new(e));
                    }
                }
            }
            LLM::MistralAI(apikey) => {
                let client = MistralClient::new(Some(apikey.clone()), None, None, None).unwrap();
                let model = Model::MistralMediumLatest;
                let messages = vec![
                    MistralChatMessage {
                        role:mistralai_client::v1::chat::ChatMessageRole::User,
                        content:text.clone(),
                        tool_calls: None, }
                ];
                let options = Some(ChatParams {
                    ..Default::default()
                });

                /* TODO: Add tool support for MistralAI                 
                #[cfg(feature = "tools")]
                {
                    match &self.components {
                        Some(comp) => {
                            use mistralai_client::v1::{client, tool::{self, Tool}};

                            debug!("Using tools/components in query");
                            options
                        },
                        None =>  {
                            debug!("No tools/components in query");
                        }
                    }
                } */

                debug!("Sending prompt to MistralAI: {text}");
                let response = client.chat(model, messages, options)?;
                response.object
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
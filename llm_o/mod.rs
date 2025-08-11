mod prompter;
mod documentation;
use std::sync::Arc;


use log::info;
use ollama_rs::{generation::{completion::request::GenerationRequest}, models::ModelOptions};

use crate::{config::Config, llm::documentation::retrieve_relevant_chunks, chat::history::ChatMessage};

#[derive(Debug,Clone,Default,PartialEq)]
pub(crate) enum QueryType {
    #[default]
    Auto,
    General,
    Documentation,
}

#[derive(Debug,Clone, Default)]
pub(crate) struct Query {
    pub history: Vec<ChatMessage>,
    pub constraint: Option<String>, // Optional constraint for the query
    pub style: Option<String>, // Optional style for the query
    pub context: String,
    pub message: String,
    pub model_name: String,
    pub query_type: QueryType,
    pub options: ModelOptions,
}

impl Query {
    pub fn new() -> Self {
        Default::default()
    }   

    // Combine the retrieved chunks with the user message
    pub(crate) fn augmented_message(&self) -> String {
        let constraint = self.constraint.as_deref().unwrap_or("");
        let style = self.style.as_deref().unwrap_or("");
        format!("{}\n\nContext:{}\n\nUser Query:\n{}\n{}",constraint,self.context,self.message,style)
    }
}


pub(crate) async fn send_local(q: Query) -> Result<String, Box<dyn std::error::Error>> {
    let ollama = ollama_rs::Ollama::new("http://localhost:11434", 11434);
    let response = ollama
        .generate(GenerationRequest::new(q.model_name, q.message).options(q.options))
        .await?;

    Ok(response.response) // No documents for general queries

}

// Function to classify the query
async fn classify_query(config: &Arc<Config>, query: &str, _model_name: &str) -> Result<QueryType, Box<dyn std::error::Error>> {
    let model_name = "mistral"; // Use a specific model for classification, e.g., "mistral" //TODO: Make this configurable
    let classification_prompt = format!(
        "Context: Okuma is a machine, Any query mentioning 'Okuma' is classified as 'machine'. Classify the following query as 'general' or 'machine'\n\nQuery: {}\n\nAnswer with only 'general' or 'machine'.",
        query
    );

    let ollama = ollama_rs::Ollama::new(format!("{}:{}", config.ollama.url.clone(), config.ollama.port), config.ollama.port);
    let response = ollama
        .generate(GenerationRequest::new(model_name.to_string(), classification_prompt))
        .await?;

    let classification = response.response.trim().to_lowercase();
    let classification = match classification.as_str() {
        "general" => QueryType::General,
        "machine" => QueryType::Documentation,
        _ => QueryType::General, // Default to general if classification is unknown
        
    };
    Ok(classification)
}

pub(crate) async fn prepare_prompt(config:&Arc<Config>, mut query: Query) -> Result<(String, Option<Vec<String>>), Box<dyn std::error::Error>> {
    //let qtype = classify_query(&message, &model_name).await?;
    if query.query_type == QueryType::Auto { // TODO: Implement this inside Query struct
        query.query_type = classify_query(config,&query.message, &query.model_name).await?;
    };
    match query.query_type {
        QueryType::Documentation => {
            info!("Classified as machine query");
            query.options = ModelOptions::default() //TODO: Make options configurable
            .temperature(0.3)
            .repeat_penalty(1.4)
            .top_k(20)
            .top_p(0.4);
            // Step 2: Retrieve relevant chunks from Qdrant
            let retrieved_chunks = retrieve_relevant_chunks(config,&query.message).await?;
            // Step 3: Summarize the retrieved chunks
            //let summarized_chunks = summarize_each_chunk(retrieved_chunks.0.clone(), &model_name).await?;
            let summarized_chunks = retrieved_chunks.0.clone();
            let chunks = (summarized_chunks, retrieved_chunks.1);
            //let augmented_message = prompter::create_prompt_summarized(&message, &summarized_chunks);
            prompter::create_prompt(&mut query, &chunks);
            // Step 4: Send the augmented message to Ollama
            //println!("Augmented message: {}", augmented_message);
            let response = send_local(query).await?;

            info!("We responded to a machine query");
            Ok((response, Some(chunks.1)))
        }
        QueryType::General => {
            info!("Classified as general query");
            // Handle general queries directly
            let r = general_prompt(query).await;
            if let Ok((response, _)) = r {
                info!("We responded to a general query");
                Ok((response, None))
            } else {
                Err(r.unwrap_err())
            }
        }
        _ => {
            panic!("Unknown query type encountered during classification");
        }
        
    }

}

// Function to send a message to the Ollama server with RAG
async fn general_prompt(mut query: Query) -> Result<(String,Vec<String>), Box<dyn std::error::Error>> {
        query.options = ModelOptions::default()
            .temperature(0.7)
            .repeat_penalty(1.2)
            .top_k(40)
            .top_p(0.9);

        let response = send_local(query).await?;

        Ok((response, vec![])) // No documents for general queries
}
use std::{collections::HashSet, sync::Arc};

use log::{debug, info};
use ollama_rs::{generation::completion::request::GenerationRequest, models::ModelOptions};
use qdrant_client::{qdrant::{Query, QueryPointsBuilder, SearchParamsBuilder}, Qdrant};

use crate::{config::Config, embed};


// Function to retrieve relevant chunks from Qdrant
pub(crate) async fn retrieve_relevant_chunks(config: &Arc<Config>,query: &str) -> Result<(Vec<String>,Vec<String>), Box<dyn std::error::Error>> {
    debug!("Retrieving relevant chunks for query");
    let url = format!("{}:{}", config.qdrant.url.clone(), config.qdrant.port);
    let qd = Qdrant::from_url(&url).build()?;
    let collection_name = config.qdrant.collection_name.clone();         

    // Convert the query into an embedding (you may need to implement this)
    let query_embedding = embed::embed(config,query.to_string()).await;

    let x = qd.query(
        QueryPointsBuilder::new(collection_name)
            .query(Query::new_nearest( query_embedding))
            .limit(10)
            .with_payload(true)
            //.score_threshold(0.50)
            .params(SearchParamsBuilder::default().hnsw_ef(128).exact(false)),
    )
    .await?;

    // Extract the relevant chunks from the query result
        let chunks: Vec<String> = x.result.clone().into_iter()
        .filter_map(|point| point.payload.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()))
        .filter(|chunk| {
            // Example: Filter out chunks with less than 20 words
            chunk.split_whitespace().count() >= 20
        })
        .collect();

    // Deduplicate chunks
    let chunks = deduplicate_chunks(chunks).await;
    
    info!("Extracted {} chunks from database", chunks.len());

    // Extract and deduplicate document filenames
    let docs: Vec<String> = x.result.into_iter()
        .filter_map(|point| point.payload.get("filename").and_then(|v| v.as_str()).map(|s| s.to_string()))
        .collect::<HashSet<_>>() // Remove duplicates by converting to a HashSet
        .into_iter()
        .collect();

    Ok((chunks, docs))
}

async fn deduplicate_chunks(chunks: Vec<String>) -> Vec<String> {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    chunks
        .into_iter()
        .filter(|chunk| seen.insert(chunk.clone())) // Retain only unique chunks
        .collect()
}

pub(crate) async fn _summarize_each_chunk(chunks: Vec<String>, model_name: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let ollama = ollama_rs::Ollama::new("http://localhost:11434", 11434);
    let mut summaries = Vec::new();

    for chunk in chunks {
        let summarization_prompt = format!(
            "Summarize the following text:\n\n{}\n\nProvide a concise summary.",
            chunk
            
        );
        

        let options = ModelOptions::default()
            .temperature(0.5)
            .repeat_penalty(1.2)
            .top_k(30)
            .top_p(0.8);

        let response = ollama
            .generate(GenerationRequest::new(model_name.to_string(), summarization_prompt).options(options))
            .await?;

        println!("Summarized chunk: {}", response.response);
        summaries.push(response.response);
    }

    Ok(summaries)
}
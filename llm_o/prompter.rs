use crate::llm::Query;


pub(crate) fn create_prompt(
    query: &mut Query,
    retrieved_chunks: &(Vec<String>, Vec<String>),
) -> () {
    query.constraint = Some("You are a helpful assistant. You have access to the following context:".to_string());
    query.style = Some("If you are unsure about a question, make sure you ask for clarification before moving forward".to_string());
    query.context = retrieved_chunks.0.clone()
        .iter()
        .map(|chunk| format!("{}\n", chunk))
        .collect::<String>();
}


#[allow(unused)]  
pub(crate) fn create_prompt_summarized(
    message: &str,
    retrieved_chunks: &String,
) -> String {
    let constraint = "You are a helpful assistant. You have access to the following context:";
    let style = "If you are unsure about a question, make sure you ask for clarification before moving forward";
    let context = retrieved_chunks;
    // Combine the retrieved chunks with the user message
    let augmented_message = format!("{constraint}\n\nContext:{}\n\nUser Query:\n{}\n{style}",
        context,       
        message
    );
    augmented_message
}
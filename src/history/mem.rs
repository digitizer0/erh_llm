use log::debug;
use once_cell::sync::Lazy;
use std::sync::Mutex;

use crate::{history::HistoryTrait, ChatMessage};

#[derive(Debug,Clone, Default)]
pub(crate) struct MemHistory {
    //pub(crate) messages: Vec<ChatMessage>
}

// Global, thread-safe, lazily-initialized message database
pub static MESSAGE_DB: Lazy<Mutex<Vec<ChatMessage>>> = Lazy::new(|| Mutex::new(Vec::new()));

impl MemHistory {
    pub fn new() -> Self {
        MemHistory {
        }
    }

    pub fn len(&self) -> usize {
        // Return the length of the global message database
        let db = MESSAGE_DB.lock().unwrap();
        db.len()
    }
    
}
impl HistoryTrait for MemHistory {
    fn store(&mut self,msg: &ChatMessage) -> Result<(), Box<dyn std::error::Error>> {
        // Store in the global message database
        {
            let mut db = MESSAGE_DB.lock().unwrap();
            db.insert(0, msg.clone());
        }
        debug!("Stored message in memory history: {msg:?}");
        Ok(())
    }    
    fn read(&self) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
        // Read from the global message database
        let db = MESSAGE_DB.lock().unwrap();
        debug!("Reading {} messages from memory history", db.len());

        Ok(db.clone())
    }
}

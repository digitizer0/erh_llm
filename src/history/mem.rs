use crate::{history::HistoryTrait, ChatMessage};

#[derive(Debug,Clone, Default)]
pub(crate) struct MemHistory {
    pub(crate) messages: Vec<ChatMessage>
}

impl MemHistory {
    pub fn new() -> Self {
        MemHistory {
            messages : vec![]
        }
    }
    
}
impl HistoryTrait for MemHistory {
    fn store(&mut self,msg: &ChatMessage) -> Result<(), Box<dyn std::error::Error>> {
        self.messages.insert(0,msg.clone());
        Ok(())
    }    
    fn read(&self) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
        todo!()
    }
}

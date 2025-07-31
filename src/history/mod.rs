use crate::ChatMessage;
use crate::history::mem::MemHistory;

mod mem;
#[cfg(feature="sqlite_hist")]
mod sqlite;


pub(crate) trait HistoryTrait {
    fn store(&mut self, msg: &ChatMessage) -> Result<(), Box<dyn std::error::Error>>;
    fn read(&self) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>>;
}

#[derive(Debug,Clone, Default)]
pub(crate)  struct History {
    mem: Option<MemHistory>,
#[cfg(feature="sqlite_hist")]
    sqlite: Option<SqliteHistory>
}

/*
impl History {
    pub(crate) fn new() -> Self {
        History {
            mem: Some(MemHistory::new()),
        }
        
    }
}
*/

impl HistoryTrait for History {
    fn store(&mut self, msg: &ChatMessage) -> Result<(), Box<dyn std::error::Error>> {
        {
            if let Some(x) = &mut self.mem {
                x.store(msg)?;
            } else {
                self.mem = Some(MemHistory::new());
                self.mem.as_mut().unwrap().store(msg)?;
            }
        }

#[cfg(feature="sqlite_hist")]
        sqlite::store(msg)?;
        Ok(())

    }
    fn read(&self) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
        let m = if let Some(x) = &self.mem {
            x.read()?
        } else {
            vec![]
        };
#[cfg(feature="sqlite_hist")]
        todo!();

        Ok(m)
    }
    
}


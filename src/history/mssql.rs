// Allow dead code in this module //FIXME remove later
#![allow(unused)]
use tiberius::{AuthMethod, Client, Config as MsConfig};
use tokio_util::compat::{TokioAsyncReadCompatExt};
use log::debug;

use crate::ChatMessage;
use crate::history::HistoryTrait;

#[derive(Debug)]
pub struct MsSqlHistory {

}

impl MsSqlHistory {
    pub fn new(config: String) -> Self {
        MsSqlHistory {}
   }


}

impl HistoryTrait for MsSqlHistory {
    fn store(&mut self, msg: &mut ChatMessage) -> std::result::Result<(), Box<dyn std::error::Error>> {
        if !msg.validate() {
            return Err(anyhow::anyhow!("Invalid chat message data").into());
        }
        let _msg = msg.noemoji();

        Ok(())
    }

    fn read(&self, chatuuid: &str) -> std::result::Result<Vec<crate::ChatMessage>, Box<dyn std::error::Error>> {
        let result = vec![];
        Ok(result)
    }
}
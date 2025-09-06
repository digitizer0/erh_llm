use mysql::*;
use mysql::prelude::*;
use log::debug;

use crate::ChatMessage;
use crate::history::HistoryTrait;

#[derive(Debug)]
pub struct MysqlHistory {
    pool: Pool,
}

impl MysqlHistory {
    pub fn new(config: String) -> Self {
        let opts = Opts::from_url(&config).unwrap();
        let pool = Pool::new(opts).unwrap();
        MysqlHistory { pool }
    }

    pub fn get_connection(&self) -> Result<PooledConn, mysql::Error> {
        let mut conn = self.pool.get_conn()?;

        conn.query_drop(
            r#"CREATE TABLE IF NOT EXISTS chat_history (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(60),
                chatuuid VARCHAR(40) NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"#
        )?;
        debug!("Database initialized successfully.");
        Ok(conn)
    }

}

impl HistoryTrait for MysqlHistory {
    fn store(&mut self, msg: &mut ChatMessage) -> std::result::Result<(), Box<dyn std::error::Error>> {
        if !msg.validate() {
            return Err(anyhow::anyhow!("Invalid chat message data").into());
        }
        let msg = msg.noemoji();
        let mut conn = self.get_connection()?;
        let params = (msg.user, msg.chatuuid, msg.user_message, msg.bot_response);
        conn.exec_drop(
            "INSERT INTO chat_history (username, chatuuid, user_message, bot_response) VALUES (?, ?, ?, ?)",
            params,
        )?; 
        Ok(())
    }

    fn read(&self, chatuuid: &str) -> std::result::Result<Vec<crate::ChatMessage>, Box<dyn std::error::Error>> {
        let mut conn = self.get_connection()?;
        let result: Vec<(String, String, String, String)> = conn.exec(
            "SELECT username, chatuuid, user_message, bot_response FROM chat_history WHERE chatuuid = ?",
            (chatuuid,),
        )?;
        let result: Vec<ChatMessage> = result.into_iter()
            .map(ChatMessage::from_tuple)
            .collect();
        Ok(result)
    }
}
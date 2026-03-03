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
    /// Creates a new [`MysqlHistory`] by parsing `config` as a MySQL connection
    /// URL (e.g. `mysql://user:pass@host:3306/db`) and establishing a
    /// connection pool.
    ///
    /// # Panics
    /// Panics if the URL is invalid or if the pool cannot be created.
    pub fn new(config: String) -> Self {
        let opts = Opts::from_url(&config).unwrap();
        let pool = Pool::new(opts).unwrap();
        MysqlHistory { pool }
    }

    /// Acquires a pooled connection and ensures the `chat_history` table exists.
    ///
    /// The `CREATE TABLE IF NOT EXISTS` statement is executed on every call so
    /// that the schema is always present without requiring a separate migration
    /// step.
    ///
    /// # Errors
    /// Returns a [`mysql::Error`] if a connection cannot be obtained from the
    /// pool or if the DDL statement fails.
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
    /// Validates and inserts a [`ChatMessage`] into the `chat_history` table.
    ///
    /// The message is validated via [`ChatMessage::validate`] and sanitised
    /// with [`ChatMessage::noemoji`] before insertion.  The INSERT uses a
    /// parameterised query to prevent SQL injection.
    ///
    /// # Errors
    /// Returns an error if validation fails, if a connection cannot be
    /// obtained, or if the INSERT statement fails.
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

    /// Retrieves all [`ChatMessage`]s for the given `chatuuid` from MySQL.
    ///
    /// Rows are fetched with a parameterised SELECT and mapped to
    /// [`ChatMessage`] via [`ChatMessage::from_tuple`].
    ///
    /// # Errors
    /// Returns an error if a connection cannot be obtained or if the SELECT
    /// query fails.
    fn read(&self, chatuuid: &str) -> std::result::Result<Vec<crate::ChatMessage>, Box<dyn std::error::Error>> {
        let mut conn = self.get_connection()?;
        let result: Vec<(String, String, String, String)> = conn.exec(
            "SELECT username, user_message, bot_response, chatuuid FROM chat_history WHERE chatuuid = ?",
            (chatuuid,),
        )?;
        let result: Vec<ChatMessage> = result.into_iter()
            .map(ChatMessage::from_tuple)
            .collect();
        Ok(result)
    }
}
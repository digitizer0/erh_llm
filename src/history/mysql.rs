use mysql::*;
use mysql::prelude::*;
use log::debug;

use crate::errors::ErhLlmError;
use crate::ChatMessage;
use crate::history::HistoryTrait;

/// Remove `<think>…</think>` blocks from a stored response.
fn strip_think(s: String) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s.as_str();
    loop {
        match rest.find("<think>") {
            None => { out.push_str(rest); break; }
            Some(start) => {
                out.push_str(&rest[..start]);
                rest = &rest[start + "<think>".len()..];
                match rest.find("</think>") {
                    None => break,
                    Some(end) => rest = &rest[end + "</think>".len()..],
                }
            }
        }
    }
    out.trim().to_string()
}

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
                chatuuid VARCHAR(64) NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                feedback CHAR(1) DEFAULT NULL,
                timestamp DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6)
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
    fn store(&mut self, msg: &mut ChatMessage) -> Result<(), ErhLlmError> {
        if !msg.validate() {
            return Err(ErhLlmError::InvalidMessage("empty user or bot content".into()));
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
    fn read(&self, chatuuid: &str) -> Result<Vec<crate::ChatMessage>, ErhLlmError> {
        let mut conn = self.get_connection()?;
        // ORDER BY ensures chronological context; LIMIT caps token usage.
        let result: Vec<(String, String, String, String)> = conn.exec(
            "SELECT username, user_message, bot_response, chatuuid \
             FROM chat_history WHERE chatuuid = ? \
             ORDER BY timestamp ASC LIMIT 40",
            (chatuuid,),
        )?;
        let result: Vec<ChatMessage> = result.into_iter()
            .map(|mut row| {
                // Strip think-block content that may have been stored before
                // the streaming filter was in place.
                row.2 = strip_think(row.2);
                ChatMessage::from_tuple(row)
            })
            .collect();
        Ok(result)
    }

    /// Sets the feedback value (`"U"` or `"D"`) for the row identified by `message_id`.
    ///
    /// # Errors
    /// Returns [`ErhLlmError::InvalidFeedback`] if the value is not `"U"` or `"D"`,
    /// or a [`ErhLlmError::MysqlError`] if the update fails.
    fn set_feedback(&mut self, message_id: i64, feedback: &str) -> Result<(), ErhLlmError> {
        if feedback != "U" && feedback != "D" {
            return Err(ErhLlmError::InvalidFeedback(feedback.to_string()));
        }
        let mut conn = self.get_connection()?;
        conn.exec_drop(
            "UPDATE chat_history SET feedback = ? WHERE id = ?",
            (feedback, message_id),
        )?;
        Ok(())
    }
}
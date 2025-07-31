use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, Result};
use crate::{history::HistoryTrait, ChatMessage};

#[derive(Debug)]
pub struct SqliteHistory {
    pool: Pool<SqliteConnectionManager>,
}

impl SqliteHistory {
    pub fn new(database: String) -> Self {
        let manager = SqliteConnectionManager::file(database.clone());
        let pool = Pool::new(manager).expect("Failed to create pool");
        SqliteHistory {
            pool,
        }
    }

    pub fn init_db(&self) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.pool.get()?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                response TEXT
            )",
            [],
        )?;
        Ok(())
    }
}

impl HistoryTrait for SqliteHistory {
    fn store(&mut self, msg: &ChatMessage) -> Result<(), Box<dyn std::error::Error>> {
        let user = &msg.user;
        let message = &msg.message;
        let timestamp = msg.timestamp;
        let response = &msg.response;
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO chat_history (user, message, response, timestamp) VALUES (?1, ?2, ?3, ?4)",
            params![user, message, response, timestamp],
        )?;

        Ok(())
    }

    fn read(&self) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
        let limit = 100; // Default limit for the number of messages to read
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, user, message, timestamp, response FROM chat_history ORDER BY timestamp DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(ChatMessage {
                id: row.get(0)?,
                user: row.get(1)?,
                message: row.get(2)?,
                timestamp: row.get(3)?,
                response: row.get(4).unwrap_or_default(),
            })
        })?;
        let mut messages = Vec::new();
        for msg in rows {
            messages.push(msg?);
        }
        Ok(messages)
    }
}


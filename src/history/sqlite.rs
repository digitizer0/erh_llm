use log::debug;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, Result};
use crate::{history::HistoryTrait, ChatMessage};
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub struct SqliteHistory {
    pool: Pool<SqliteConnectionManager>,
}

impl SqliteHistory {
    /// Checks if the SQLite database file exists, and creates it if it does not.
    pub fn ensure_db_file_exists(database: &str) -> std::io::Result<()> {
        let path = Path::new(database);
        if !path.exists() {
            fs::File::create(path)?;
        }
        Ok(())
    }

    /// Creates a new [`SqliteHistory`] backed by the SQLite file at `database`.
    ///
    /// This method:
    /// 1. Calls [`Self::ensure_db_file_exists`] to create the file if absent.
    /// 2. Builds an r2d2 connection pool over the file.
    /// 3. Executes `CREATE TABLE IF NOT EXISTS chat_history …` to initialise
    ///    the schema.
    ///
    /// # Panics
    /// Panics if the database file cannot be created, if the pool cannot be
    /// built, or if the schema initialisation query fails.
    pub fn new(database: String) -> Self {
        debug!("Initializing SqliteHistory with database: {database}");
        Self::ensure_db_file_exists(&database).expect("Failed to ensure db file exists");
        let manager = SqliteConnectionManager::file(database.clone());
        let pool = Pool::new(manager).expect("Failed to create pool");

        let conn = pool.get().expect("Failed to get connection from pool");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                chatuuid TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).expect("Failed to create table");

        SqliteHistory {
            pool,
        }
    }

    pub fn get_connection(&self) -> std::result::Result<r2d2::PooledConnection<SqliteConnectionManager>, r2d2::Error> {
        self.pool.get()
    }
}

impl HistoryTrait for SqliteHistory {
    /// Inserts a [`ChatMessage`] into the `chat_history` table.
    ///
    /// All fields of `msg` (`user`, `chatuuid`, `user_message`, `bot_response`,
    /// `timestamp`) are written via a parameterised INSERT statement.
    ///
    /// # Errors
    /// Returns an error if a connection cannot be obtained from the pool or
    /// if the INSERT statement fails.
    fn store(&mut self, msg: &mut ChatMessage) -> Result<(), Box<dyn std::error::Error>> {
        if !msg.validate() {
            return Err(anyhow::anyhow!("Invalid chat message data").into());
        }
        let msg = msg.noemoji();
        let conn = self.get_connection()?;
        conn.execute(
            "INSERT INTO chat_history (username, chatuuid, user_message, bot_response) VALUES (?1, ?2, ?3, ?4)",
            params![msg.user, msg.chatuuid, msg.user_message, msg.bot_response],
        )?;

        Ok(())
    }

    /// Retrieves up to 100 [`ChatMessage`]s for the given `chatuuid`.
    ///
    /// Rows are ordered by `timestamp DESC` and mapped from the raw SQLite
    /// columns (`id`, `user`, `message`, `timestamp`, `response`) into
    /// [`ChatMessage`] structs.
    ///
    /// # Errors
    /// Returns an error if a connection cannot be obtained from the pool,
    /// if statement preparation fails, or if row mapping fails.
    fn read(&self, chatuuid: &str) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT username, user_message, bot_response, chatuuid FROM chat_history WHERE chatuuid = ?1"
        )?;
        let rows = stmt.query_map(params![chatuuid], |row| {
            Ok(ChatMessage {
                id: None,
                user: row.get(0)?,
                user_message: row.get(1)?,
                bot_response: row.get(2)?,
                chatuuid: row.get(3)?,
                timestamp: 0,
                ollama: None,
            })
        })?;
        let mut messages = Vec::new();
        for msg in rows {
            messages.push(msg?);
        }
        Ok(messages)
    }
}


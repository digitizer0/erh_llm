use rusqlite::{params, Connection, Result};

use crate::ChatMessage;

pub fn init_db(conn: &Connection) -> Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp INTEGER NOT NULL
        )",
        [],
    )?;
    Ok(())
}

pub fn save_message(conn: &Connection, user: &str, message: &str, timestamp: i64) -> Result<usize> {
    conn.execute(
        "INSERT INTO chat_history (user, message, timestamp) VALUES (?1, ?2, ?3)",
        params![user, message, timestamp],
    )
}

pub fn get_history(conn: &Connection, limit: usize) -> Result<Vec<ChatMessage>> {
    let mut stmt = conn.prepare(
        "SELECT id, user, message, timestamp FROM chat_history ORDER BY timestamp DESC LIMIT ?1"
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        Ok(ChatMessage {
            id: row.get(0)?,
            user: row.get(1)?,
            message: row.get(2)?,
            timestamp: row.get(3)?,
        })
    })?;
    let mut messages = Vec::new();
    for msg in rows {
        messages.push(msg?);
    }
    Ok(messages)
}

pub fn create_connection(db_path: &str) -> Result<Connection> {
    Connection::open(db_path)
}

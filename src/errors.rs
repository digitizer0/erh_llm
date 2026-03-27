use thiserror::Error;

#[derive(Error, Debug)]
pub enum ErhLlmError {
    // ── LLM backend errors ─────────────────────────────────────────────────
    /// An error returned by the Ollama backend.
    #[error("Ollama error: {0}")]
    OllamaError(String),

    /// An error returned by the Anthropic API.
    #[error("Anthropic error: {0}")]
    AnthropicError(String),

    /// An error returned by the MistralAI API.
    #[error("MistralError error: {0}")]
    MistralError(String),

    // ── History / database errors ──────────────────────────────────────────
    /// A SQLite database error.
    #[cfg(feature = "sqlite_hist")]
    #[error("SQLite error: {0}")]
    SqliteError(#[from] rusqlite::Error),

    /// An r2d2 connection-pool error (SQLite).
    #[cfg(feature = "sqlite_hist")]
    #[error("SQLite pool error: {0}")]
    SqlitePoolError(#[from] r2d2::Error),

    /// A MySQL database error.
    #[cfg(feature = "mysql_hist")]
    #[error("MySQL error: {0}")]
    MysqlError(#[from] mysql::Error),

    /// A Tiberius (MSSQL) database error.
    #[cfg(feature = "mssql_hist")]
    #[error("MSSQL error: {0}")]
    MsSqlError(#[from] tiberius::error::Error),

    /// No history backend was compiled in or configured.
    #[error("No history backend configured")]
    NoHistoryBackend,

    // ── Validation / input errors ──────────────────────────────────────────
    /// A chat message failed validation (e.g. empty user or bot content).
    #[error("Invalid chat message: {0}")]
    InvalidMessage(String),

    /// A configuration value is missing or malformed (e.g. bad connection string).
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// The feedback value was not one of the accepted values (`"U"` / `"D"`).
    #[error("Invalid feedback value '{0}': must be 'U' or 'D'")]
    InvalidFeedback(String),

    // ── Embedding errors ───────────────────────────────────────────────────
    /// Embedding generation returned no vectors.
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    // ── I/O errors ─────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    // ── Generic catch-all ──────────────────────────────────────────────────
    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, ErhLlmError>;

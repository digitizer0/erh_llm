#[allow(unused)]
use std::sync::Arc;

use log::debug;

use crate::ChatMessage;

#[cfg(feature="mysql_hist")]
mod mysql;
#[cfg(feature="mysql_hist")]
use crate::history::mysql::MysqlHistory;
#[cfg(feature="sqlite_hist")]
mod sqlite;
#[cfg(feature="sqlite_hist")]
use crate::history::sqlite::SqliteHistory;
#[cfg(feature="mssql_hist")]
mod mssql;
#[cfg(feature="mssql_hist")]
use crate::history::mssql::MsSqlHistory;

pub(crate) trait HistoryTrait {
    /// Persists a [`ChatMessage`] to the backing history store.
    ///
    /// Implementations are expected to assign any backend-generated fields
    /// (e.g. a primary-key `id`) back onto `msg` before returning.
    ///
    /// # Errors
    /// Returns an error if the underlying database operation fails or if
    /// `msg` fails validation.
    fn store(&mut self, msg: &mut ChatMessage) -> Result<(), Box<dyn std::error::Error>>;

    /// Retrieves all [`ChatMessage`]s associated with the given `chatuuid`.
    ///
    /// Messages are returned in the order defined by the backend
    /// (typically ascending by timestamp).
    ///
    /// # Errors
    /// Returns an error if the underlying database query fails.
    fn read(&self, chatuuid: &str) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>>;
}

#[derive(Debug, Default,Clone, PartialEq, Eq)]
#[allow(unused)]
pub enum HistoryConfig {
    Sqlite(String),
    Mysql(String),
    MsSql(String),
    None,
    #[default]
    Unknown,
}

#[derive(Debug, Default)]
#[allow(unused)]
pub(crate)  struct History {
#[cfg_attr(not(any(feature="sqlite_hist",feature="mysql_hist")),allow(dead_code))]
    database: String,
#[cfg(feature="sqlite_hist")]
    sqlite: Option<SqliteHistory>,
#[cfg(feature="mysql_hist")]
    mysql: Option<MysqlHistory>,
#[cfg(feature="mssql_hist")]
    mssql: Option<MsSqlHistory>,
}

impl History {
    /// Creates a new [`History`] instance configured by `cfg`.
    ///
    /// The appropriate backend (SQLite, MySQL, or MSSQL) is initialised
    /// based on the variant of [`HistoryConfig`] supplied.  All other
    /// backend fields are set to `None`.
    ///
    /// # Panics
    /// Panics if `cfg` is [`HistoryConfig::None`] or
    /// [`HistoryConfig::Unknown`], as those variants carry no connection
    /// information.
    pub(crate) fn new(cfg: HistoryConfig) -> Self {
        match cfg {
            HistoryConfig::Sqlite(db) => {
                History {
                    database: db.clone(),
#[cfg(feature="sqlite_hist")]
                    sqlite: Some(SqliteHistory::new(db)),
#[cfg(feature="mysql_hist")]
                    mysql: None,
#[cfg(feature="mssql_hist")]
                    mssql: None,
                }
            }
            HistoryConfig::Mysql(config) => {
                History {
                    database: config.clone(),
#[cfg(feature="sqlite_hist")]
                    sqlite: None,
#[cfg(feature="mysql_hist")]
                    mysql: Some(MysqlHistory::new(config)),
#[cfg(feature="mssql_hist")]
                    mssql: None,
                }
            }
            HistoryConfig::MsSql(config) => {
                History {
                    database: config.clone(),
#[cfg(feature="sqlite_hist")]
                    sqlite: None,
#[cfg(feature="mysql_hist")]
                    mysql: None,
#[cfg(feature="mssql_hist")]
                    mssql: Some(MsSqlHistory::new(config)),
                }
            }
            _ => {
                panic!("Invalid configuration for History: {cfg:?}");
            }
        }
    }
}

impl HistoryTrait for History {
    /// Delegates to the active backend's [`HistoryTrait::store`] implementation.
    ///
    /// Iterates through each conditionally-compiled backend in priority order
    /// (SQLite → MySQL → MSSQL) and forwards the call to the first one that
    /// is present.
    ///
    /// # Errors
    /// Returns `"No history backend configured"` if no backend is enabled at
    /// compile time, or propagates the backend-specific error otherwise.
    fn store(&mut self, msg: &mut ChatMessage) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Storing message in history: {msg:?}");

#[cfg(feature="sqlite_hist")]
        if let Some(x) = &mut self.sqlite {
            debug!("Using sqlite history");
            return x.store(msg);
        }

#[cfg(feature="mysql_hist")]
        if let Some(x) = &mut self.mysql {
            debug!("Using mysql history");
            return x.store(msg);
        }

#[cfg(feature="mssql_hist")]
        if let Some(x) = &mut self.mssql {
            debug!("Using mssql history");
            return x.store(msg);
        }

        Err("No history backend configured".into())
    }

    /// Delegates to the active backend's [`HistoryTrait::read`] implementation.
    ///
    /// Iterates through each conditionally-compiled backend in priority order
    /// (SQLite → MySQL → MSSQL) and forwards the call to the first one that
    /// is present.  Returns an empty vector when no backend is compiled in.
    ///
    /// # Errors
    /// Propagates any error returned by the active backend.
    fn read(&self, _chatuuid: &str) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
#[cfg(feature="sqlite_hist")]
        if let Some(x) = &self.sqlite {
            debug!("Reading sqlite history");
            return Ok(x.read(chatuuid)?);
        }

#[cfg(feature="mysql_hist")]
        if let Some(x) = &self.mysql {
            debug!("Reading mysql history");
            return Ok(x.read(_chatuuid)?);
        }

#[cfg(feature="mssql_hist")]
        if let Some(x) = &self.mssql {
            debug!("Reading mssql history");
            return Ok(x.read(_chatuuid)?);
        }

        debug!("No history backend configured, returning empty vector.");
        Ok(vec![])
    }
    
}


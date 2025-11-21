#[allow(unused)]
use std::sync::Arc;

use log::debug;

use crate::ChatMessage;

#[cfg(feature="mysql_hist")]
mod mysql;
#[cfg(feature="mysql_hist")]
use crate::history::mysql::MysqlHistory;
#[cfg(feature="mem_hist")]
use crate::history::mem::MemHistory;
#[cfg(feature="mem_hist")]
mod mem;
#[cfg(feature="sqlite_hist")]
mod sqlite;
#[cfg(feature="sqlite_hist")]
use crate::history::sqlite::SqliteHistory;
#[cfg(feature="mssql_hist")]
mod mssql;
#[cfg(feature="mssql_hist")]
use crate::history::mssql::MsSqlHistory;

pub(crate) trait HistoryTrait {
    fn store(&mut self, msg: &mut ChatMessage) -> Result<(), Box<dyn std::error::Error>>;
    fn read(&self, chatuuid: &str) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>>;
}

#[derive(Debug, Default,Clone, PartialEq, Eq)]
#[allow(unused)]
pub enum HistoryConfig {
    Mem,
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
#[cfg_attr(not(any(feature="mem_hist",feature="sqlite_hist",feature="mysql_hist")),allow(dead_code))]
    database: String,
#[cfg(feature="mem_hist")]
    mem: Option<MemHistory>,
#[cfg(feature="sqlite_hist")]
    sqlite: Option<SqliteHistory>,
#[cfg(feature="mysql_hist")]
    mysql: Option<MysqlHistory>,
#[cfg(feature="mssql_hist")]
    mssql: Option<MsSqlHistory>,
}

impl History {
    pub(crate) fn new(cfg: HistoryConfig) -> Self {
        match cfg {
            HistoryConfig::Mem => {
                History {
                    database: String::new(),
#[cfg(feature="mem_hist")] 
                    mem: Some(MemHistory::new()),
#[cfg(feature="sqlite_hist")]
                    sqlite: None,
#[cfg(feature="mysql_hist")]
                    mysql: None,
#[cfg(feature="mssql_hist")]
                    mssql: None,
                }
            }
            HistoryConfig::Sqlite(db) => {
                History {
                    database: db.clone(),
#[cfg(feature="mem_hist")] 
                    mem: None,
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
#[cfg(feature="mem_hist")] 
                    mem: None,
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
#[cfg(feature="mem_hist")] 
                    mem: None,
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
    fn store(&mut self, msg: &mut ChatMessage) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Storing message in history: {msg:?}");
        
#[cfg(feature="mem_hist")]
        if let Some(x) = &mut self.mem {
            debug!("Using memory history");
            return x.store(msg);
        }

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
    #[allow(unused)]
    fn read(&self, chatuuid: &str) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
#[cfg(feature="mem_hist")]
        if let Some(x) = &self.mem {
            debug!("Reading memory history with {} messages.", x.len());
            return Ok(x.read()?);
        }

#[cfg(feature="sqlite_hist")]
        if let Some(x) = &self.sqlite {
            debug!("Reading sqlite history");
            return Ok(x.read(chatuuid)?);
        }

#[cfg(feature="mysql_hist")]
        if let Some(x) = &self.mysql {
            debug!("Reading mysql history");
            return Ok(x.read(chatuuid)?);
        }

#[cfg(feature="mssql_hist")]
        if let Some(x) = &self.mssql {
            debug!("Reading mssql history");
            return Ok(x.read(chatuuid)?);
        }

        debug!("No history backend configured, returning empty vector.");
        Ok(vec![])
    }
    
}


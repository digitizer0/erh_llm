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
    None,
    #[default]
    Unknown,
}

#[derive(Debug, Default)]
pub(crate)  struct History {
#[cfg_attr(not(any(feature="mem_hist",feature="sqlite_hist",feature="mysql_hist")),allow(dead_code))]
    database: String,
#[cfg(feature="mem_hist")]
    mem: Option<MemHistory>,
#[cfg(feature="sqlite_hist")]
    sqlite: Option<SqliteHistory>,
#[cfg(feature="mysql_hist")]
    mysql: Option<MysqlHistory>,
}

impl History {
    pub(crate) fn new(cfg: HistoryConfig) -> Self {
        let cfgstr = if let HistoryConfig::Sqlite(a) = cfg.clone() {
            a
        } else if let HistoryConfig::Mysql(a) = cfg.clone() {
            a
        } else {
            panic!("Invalid configuration for History: {cfg:?}");
        };
        History {
            database: cfgstr.clone(),
#[cfg(feature="mem_hist")] 
            mem: Some(MemHistory::new()),
#[cfg(feature="sqlite_hist")]
            sqlite: Some(SqliteHistory::new(cfgstr.clone())),
#[cfg(feature="mysql_hist")]
            mysql: Some(MysqlHistory::new(cfgstr)),
        }
    }
}

impl HistoryTrait for History {
    fn store(&mut self, msg: &mut ChatMessage) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Storing message in history: {msg:?}");
#[cfg(feature="mem_hist")]
        {
            if let Some(x) = &mut self.mem {
                debug!("Found existing memory history with {} messages.", x.len());
                x.store(msg)?;
            } else {
                debug!("No memory history found, creating a new one.");
                self.mem = Some(MemHistory::new());
                self.mem.as_mut().unwrap().store(msg)?;
            }
        }

#[cfg(feature="sqlite_hist")]
        {
            if let Some(x) = &mut self.sqlite {
                debug!("Found existing memory history with");
                x.store(msg)?;
            } else {
                debug!("No memory history found, creating a new one.");
                self.sqlite = Some(SqliteHistory::new(self.database.clone()));
                self.sqlite.as_mut().unwrap().store(msg)?;
            }
        }
#[cfg(feature="mysql_hist")]
        {
            if let Some(x) = &mut self.mysql {
                debug!("Found existing mysql history with");
                x.store(msg)?;
            } else {
                debug!("No mysql history found, creating a new one.");
                self.mysql = Some(MysqlHistory::new(self.database.clone()));
                self.mysql.as_mut().unwrap().store(msg)?;
            }
        }
        Ok(())

    }
    fn read(&self, chatuuid: &str) -> Result<Vec<ChatMessage>, Box<dyn std::error::Error>> {
#[cfg(feature="mem_hist")]
        let m = if let Some(x) = &self.mem {
            debug!("Reading memory history with {} messages.", x.len());
            x.read()?
        } else {
            debug!("No memory history found, returning empty vector.");
            vec![]
        };
#[cfg(feature="sqlite_hist")]
        let m = if let Some(x) = &self.sqlite {
            debug!("Reading sqlite history");
            x.read(chatuuid)?
        } else {
            debug!("No sqlite history found, returning empty vector.");
            vec![]
        };
#[cfg(feature="mysql_hist")]
        let m = if let Some(x) = &self.mysql {
            debug!("Reading mysql history");
            x.read(chatuuid)?
        } else {
            debug!("No mysql history found, returning empty vector.");
            vec![]
        };
#[cfg(not(any(feature="mem_hist",feature="sqlite_hist",feature="mysql_hist")))]
        let m = {
            _ = chatuuid;
            debug!("No history feature enabled, returning empty vector.");
            vec![]
        };
        Ok(m)
    }
    
}


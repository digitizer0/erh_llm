use tiberius::{AuthMethod, Client, Config as MsConfig};
use tokio::net::TcpStream;
use log::debug;
use futures::stream::TryStreamExt;

use crate::ChatMessage;
use crate::history::HistoryTrait;

#[derive(Debug)]
pub struct MsSqlHistory {
    config_string: String,
}

impl MsSqlHistory {
    pub fn new(config: String) -> Self {
        MsSqlHistory {
            config_string: config,
        }
    }

    fn get_config(&self) -> Result<(MsConfig,String), Box<dyn std::error::Error + Send + Sync>> {
        // Check if it's URI format: "mssql://{user}:{password}@{host}:{port}/{db_name}"
        let cfg=  if self.config_string.starts_with("mssql://") {
            let without_scheme = self.config_string.strip_prefix("mssql://").unwrap();
            
            // Split on '@' to separate credentials from host
            let parts: Vec<&str> = without_scheme.split('@').collect();
            if parts.len() != 2 {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid URI format: missing @ separator")));
            }
            
            // Parse credentials (user:password)
            let credentials: Vec<&str> = parts[0].split(':').collect();
            if credentials.len() != 2 {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid URI format: invalid credentials")));
            }
            let user = credentials[0].to_string();
            let password = credentials[1].to_string();
            
            // Parse host:port/database
            let host_db_parts: Vec<&str> = parts[1].split('/').collect();
            if host_db_parts.len() != 2 {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid URI format: missing database")));
            }
            
            let host_port: Vec<&str> = host_db_parts[0].split(':').collect();
            let server =  host_port[0].to_string();
            let database = host_db_parts[1].to_string();
            
            debug!("Parsed MSSQL URI - Server: {}, Database: {}, User: {}", server, database, user);
            (server, database, user, password)
        } else {
            let mut server = String::new();
            let mut database = String::new();
            let mut user = String::new();
            let mut password = String::new();

            for part in self.config_string.split(';') {
                let part = part.trim();
                if part.starts_with("Server=") {
                    server = part.strip_prefix("Server=").unwrap_or("").to_string();
                } else if part.starts_with("Database=") {
                    database = part.strip_prefix("Database=").unwrap_or("").to_string();
                } else if part.starts_with("User Id=") {
                    user = part.strip_prefix("User Id=").unwrap_or("").to_string();
                } else if part.starts_with("Password=") {
                    password = part.strip_prefix("Password=").unwrap_or("").to_string();
                }
            }
            (server, database, user, password)
        };

        if cfg.0.is_empty() || cfg.1.is_empty() || cfg.2.is_empty() || cfg.3.is_empty() {
            return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid connection string format")));
        }

        let mut config = MsConfig::new();
        config.host(&cfg.0);
        config.port(1433);
        config.database(&cfg.1);
        config.authentication(AuthMethod::sql_server(&cfg.2, &cfg.3));
        config.trust_cert();

        Ok((config,cfg.0))
    }

    async fn get_client(&self) -> Result<Client<tokio_util::compat::Compat<TcpStream>>, Box<dyn std::error::Error + Send + Sync>> {
        use tokio_util::compat::TokioAsyncWriteCompatExt;
        let (config, host) = self.get_config()?;
        // Add connection timeout
        debug!("MSSQL connecting to {}:{}", host, 1433);
        let tcp = match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio::net::TcpStream::connect(format!("{}:{}", host, 1433))
        ).await {
            Ok(Ok(stream)) => {
                debug!("TCP connection successful to {}:{}", host, 1433);
                stream
            },
            Ok(Err(e)) => {
                debug!("TCP connection failed to {}:{}: {}", host, 1433, e);
                return Err(Box::new(e));
            },
            Err(_) => {
                debug!("TCP connection timeout to {}:{}", host, 1433);
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, "Connection timeout")));
            }
        };
        debug!("MSSQL TCP connection established to {}:{}", host, 1433);
        tcp.set_nodelay(true)?;
        debug!("TCP connection established to {}:{}", host, 1433);
        
        // Add timeout for client connection
        let mut client = match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            Client::connect(config, tcp.compat_write())
        ).await {
            Ok(Ok(client)) => {
                debug!("MSSQL client connection successful");
                client
            },
            Ok(Err(e)) => {
                debug!("MSSQL client connection failed: {}", e);
                return Err(Box::new(e));
            },
            Err(_) => {
                debug!("MSSQL client connection timeout");
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, "Client connection timeout")));
            }
        };

        debug!("MSSQL client connected successfully.");

        // Create table if it doesn't exist
        let create_table_sql = r#"
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='chat_history' AND xtype='U')
            CREATE TABLE chat_history (
                id BIGINT IDENTITY(1,1) PRIMARY KEY,
                username NVARCHAR(60),
                chatuuid NVARCHAR(40) NOT NULL,
                user_message NTEXT NOT NULL,
                bot_response NTEXT NOT NULL,
                timestamp DATETIME DEFAULT GETDATE()
            )
        "#;

        let r = client.execute(create_table_sql, &[]).await;
        if let Err(e) = &r {
            return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to create table: {}", e))));
        }
        debug!("Database initialized successfully.");
        
        Ok(client)
    }

    fn execute_with_runtime<F, R>(&self, f: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: futures::Future<Output = Result<R, Box<dyn std::error::Error + Send + Sync>>> + Send + 'static,
        R: Send + 'static,
    {
        // Always use spawn_blocking to avoid runtime nesting issues
        // This creates a new thread that can safely create its own runtime
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { 
                Box::new(e)
            })?;
            rt.block_on(f)
        }).join().map_err(|_| -> Box<dyn std::error::Error + Send + Sync> { 
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Thread panicked"))
        })?
    }
}

impl HistoryTrait for MsSqlHistory {
    fn store(&mut self, msg: &mut ChatMessage) -> std::result::Result<(), Box<dyn std::error::Error>> {
        if !msg.validate() {
            return Err(anyhow::anyhow!("Invalid chat message data").into());
        }
        let msg = msg.noemoji();
        let config_string = self.config_string.clone();
        
        self.execute_with_runtime(async move {
            let history = MsSqlHistory::new(config_string);
            let mut client = history.get_client().await?;
            
            let insert_sql = "INSERT INTO chat_history (username, chatuuid, user_message, bot_response) VALUES (@P1, @P2, @P3, @P4)";
            
            client.execute(
                insert_sql,
                &[&msg.user, &msg.chatuuid, &msg.user_message, &msg.bot_response],
            ).await?;
            
            Ok(())
        }).map_err(|e| -> Box<dyn std::error::Error> { 
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("MSSQL store error: {}", e)))
        })
    }

    fn read(&self, chatuuid: &str) -> std::result::Result<Vec<crate::ChatMessage>, Box<dyn std::error::Error>> {
        let config_string = self.config_string.clone();
        let chatuuid = chatuuid.to_string();
        
        self.execute_with_runtime(async move {
            let history = MsSqlHistory::new(config_string);
            let mut client = history.get_client().await?;
            
            let select_sql = "SELECT username, user_message, bot_response, chatuuid FROM chat_history WHERE chatuuid = @P1 ORDER BY timestamp ASC";
            
            let mut stream = client.query(select_sql, &[&chatuuid]).await?;
            let mut messages = Vec::new();
            
            while let Some(item) = stream.try_next().await? {
                if let tiberius::QueryItem::Row(row) = item {
                    let username: Option<&str> = row.get(0);
                    let user_message: Option<&str> = row.get(1);
                    let bot_response: Option<&str> = row.get(2);
                    let chat_uuid: Option<&str> = row.get(3);
                    
                    let message = ChatMessage::from_tuple((
                        username.unwrap_or("").to_string(),
                        user_message.unwrap_or("").to_string(),
                        bot_response.unwrap_or("").to_string(),
                        chat_uuid.unwrap_or("").to_string(),
                    ));
                    messages.push(message);
                }
            }
            
            Ok(messages)
        }).map_err(|e| -> Box<dyn std::error::Error> { 
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("MSSQL read error: {}", e)))
        })
    }
}
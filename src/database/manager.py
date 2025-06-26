"""
Enhanced Database Manager for Autonomous Trading System - CORRECTED ASYNC POOL FIX
Handles database connections with proper async session management
"""
import asyncio
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text, Engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
import time

load_dotenv(override=True)

# Add project root to sys.path to allow running this script directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.logging_config import logger
from src.database.models import Base


class AsyncDatabaseManager:
    """Thread-safe database manager with proper async session handling"""

    def __init__(self):
        self.engine: Optional[Engine] = None
        self.async_engine: Optional[AsyncEngine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self.AsyncSessionLocal: Optional[async_sessionmaker] = None
        self._initialized = False
        self._async_initialized = False
        self._lock = threading.Lock()
        self._session_pool = {}  # Track active sessions per thread

    def initialize_sync_db(self):
        """Initialize synchronous database connection"""
        with self._lock:
            if self._initialized:
                return
                
            try:
                database_url = os.getenv("DATABASE_URL")
                logger.info(f"Database URL: {database_url}")
                
                if database_url:
                    # Use QueuePool for sync connections with pool parameters
                    self.engine = create_engine(
                        database_url,
                        poolclass=QueuePool,
                        pool_size=10,
                        max_overflow=20,
                        pool_pre_ping=True,
                        pool_recycle=3600,
                        echo=False  # Set to True for SQL debugging
                    )
                else:
                    # Fallback to SQLite for easy development
                    db_path = Path("data") / "trading_system.db"
                    db_path.parent.mkdir(exist_ok=True)
                    sqlite_url = f"sqlite:///{db_path}"
                    self.engine = create_engine(
                        sqlite_url,
                        poolclass=StaticPool,
                        pool_pre_ping=True,
                        echo=False
                    )
                
                # Create session factory
                self.SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine
                )
                
                # Create tables if they don't exist
                Base.metadata.create_all(bind=self.engine)
                
                self._initialized = True
                logger.info("✅ Sync database initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize sync database: {e}")
                raise

    async def initialize_async_db(self):
        """Initialize asynchronous database connection with proper pool"""
        if self._async_initialized:
            return
            
        try:
            database_url = os.getenv("DATABASE_URL")
            
            if database_url:
                # Convert PostgreSQL URL to async format
                if database_url.startswith("postgresql://"):
                    async_database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
                elif database_url.startswith("postgresql+psycopg2://"):
                    async_database_url = database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
                else:
                    async_database_url = database_url
                
                # Use NullPool for async connections - NO pool parameters!
                self.async_engine = create_async_engine(
                    async_database_url,
                    poolclass=NullPool,  # NullPool doesn't use pool_size/max_overflow
                    echo=False,  # Set to True for SQL debugging
                    future=True
                )
            else:
                # Fallback to async SQLite
                db_path = Path("data") / "trading_system.db"
                db_path.parent.mkdir(exist_ok=True)
                sqlite_url = f"sqlite+aiosqlite:///{db_path}"
                
                self.async_engine = create_async_engine(
                    sqlite_url,
                    poolclass=StaticPool,
                    echo=False,
                    future=True
                )
            
            # Create async session factory
            self.AsyncSessionLocal = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables if they don't exist (async)
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._async_initialized = True
            logger.info("✅ Async database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize async database: {e}")
            raise

    def test_connection(self) -> bool:
        """Test synchronous database connection"""
        try:
            if not self._initialized or self.engine is None:
                return False
                
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
                
            logger.info("✅ Database connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

    async def test_async_connection(self) -> bool:
        """Test asynchronous database connection"""
        try:
            if not self._async_initialized or self.async_engine is None:
                return False
                
            async with self.async_engine.connect() as connection:
                result = await connection.execute(text("SELECT 1"))
                result.fetchone()
                
            logger.info("✅ Async database connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Async database connection test failed: {e}")
            return False

    def get_session(self) -> Session:
        """Get database session (synchronous)"""
        if not self._initialized:
            self.initialize_sync_db()
        if self.SessionLocal is None:
            raise RuntimeError("Session factory not initialized")
        return self.SessionLocal()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with proper cleanup"""
        if not self._async_initialized:
            await self.initialize_async_db()
            
        if self.AsyncSessionLocal is None:
            raise RuntimeError("Async session factory not initialized")
            
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def safe_db_operation(self, operation_func, *args, **kwargs):
        """Execute database operation with proper error handling"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with self.get_async_session() as session:
                    result = await operation_func(session, *args, **kwargs)
                    await session.commit()
                    return result
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database operation failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                    raise

    def get_table_info(self) -> dict:
        """Get information about database tables"""
        try:
            if not self._initialized:
                return {}
                
            inspector = inspect(self.engine)
        except Exception as e:
            logger.error("❌ Failed to create database inspector")
            return {}
            
        if inspector is None:
            logger.error("❌ Database inspector is None")
            return {}
            
        tables = inspector.get_table_names()
        
        table_info = {}
        for table_name in tables:
            columns = inspector.get_columns(table_name)
            table_info[table_name] = {
                'columns': len(columns),
                'column_names': [col['name'] for col in columns]
            }
        
        return table_info

    async def cleanup(self):
        """Cleanup database connections"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
                logger.info("✅ Async database connections cleaned up")
            
            if self.engine:
                self.engine.dispose()
                logger.info("✅ Sync database connections cleaned up")
                
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")


# Global database manager instance
db_manager = AsyncDatabaseManager()


# Enhanced convenience functions with better error handling
def get_db_session():
    """Get database session (convenience function)"""
    return db_manager.get_session()


async def get_async_db_session():
    """Get async database session (convenience function)"""
    async with db_manager.get_async_session() as session:
        yield session


def init_database():
    """Initialize database (convenience function)"""
    db_manager.initialize_sync_db()
    return db_manager.test_connection()


async def init_async_database():
    """Initialize async database (convenience function)"""
    await db_manager.initialize_async_db()
    return await db_manager.test_async_connection()


# Safe database operation helpers
async def safe_log_event(event_data: dict):
    """Safely log event to database without blocking other operations"""
    async def _log_operation(session, data):
        from src.database.models import EventLog, LogLevel
        
        event = EventLog(
            level=LogLevel(data.get('level', 'INFO')),
            agent_name=data.get('agent_name', 'Unknown'),
            event_type=data.get('event_type', 'GENERAL'),
            message=data.get('message', ''),
            context=data.get('context', {}),
            session_id=data.get('session_id')
        )
        session.add(event)
        return event
    
    try:
        await db_manager.safe_db_operation(_log_operation, event_data)
        logger.debug("Event logged to database successfully")
    except Exception as e:
        logger.error(f"Failed to log event to database: {e}")


async def safe_log_agent_action(action_data: dict):
    """Safely log agent action to database"""
    async def _log_action(session, data):
        from src.database.models import AgentAction
        
        action = AgentAction(
            agent_name=data.get('agent_name', 'Unknown'),
            action_type=data.get('action_type', 'UNKNOWN'),
            input_data=data.get('input_data', {}),
            output_data=data.get('output_data', {}),
            confidence_score=data.get('confidence_score', 0),
            execution_time_ms=data.get('execution_time_ms', 0),
            session_id=data.get('session_id')
        )
        session.add(action)
        return action
    
    try:
        await db_manager.safe_db_operation(_log_action, action_data)
        logger.debug("Agent action logged to database successfully")
    except Exception as e:
        logger.error(f"Failed to log agent action to database: {e}")


# Database initialization script
async def main():
    """Main database setup script"""
    logger.info("🗄️ Starting database initialization...")
    
    # Initialize sync database
    logger.info("📊 Setting up synchronous database...")
    success = init_database()
    
    if not success:
        logger.error("❌ Failed to initialize synchronous database")
        sys.exit(1)
    
    # Initialize async database
    logger.info("⚡ Setting up asynchronous database...")
    async_success = await init_async_database()
    
    if not async_success:
        logger.error("❌ Failed to initialize asynchronous database")
        sys.exit(1)
    
    # Show table information
    table_info = db_manager.get_table_info()
    logger.info("📋 Database schema created", tables=list(table_info.keys()))

    for table_name, info in table_info.items():
        logger.info(
            f"📊 Table: {table_name}",
            columns=info["columns"],
            column_names=info["column_names"][:5],
        )

    logger.info("🎉 Database initialization complete!")
    logger.info("🔗 Ready for MCP servers and agents!")

if __name__ == "__main__":
    asyncio.run(main())
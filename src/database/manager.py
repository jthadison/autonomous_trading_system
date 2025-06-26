"""
Enhanced Database Manager for Autonomous Trading System - CONCURRENCY FIXED
Handles database connections with proper async session management
"""
import asyncio
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool
import time
from typing import AsyncGenerator

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
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
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
                    self.engine = create_engine(
                        database_url,
                        poolclass=QueuePool,
                        pool_size=10,
                        max_overflow=20,
                        pool_pre_ping=True,
                        pool_recycle=3600
                    )
                else:
                    # Fallback to SQLite for easy development
                    db_path = Path("data") / "trading_system.db"
                    db_path.parent.mkdir(exist_ok=True)
                    sqlite_url = f"sqlite:///{db_path}"
                    self.engine = create_engine(
                        sqlite_url, 
                        echo=False,  # Reduce log noise
                        poolclass=QueuePool,
                        pool_size=10,
                        max_overflow=20
                    )
                    logger.info("Using SQLite database for development", path=str(db_path))
                
                self.SessionLocal = sessionmaker(
                    autocommit=False, 
                    autoflush=False, 
                    bind=self.engine
                )
                
                # Create all tables
                Base.metadata.create_all(bind=self.engine)
                
                self._initialized = True
                logger.info("✅ Sync database initialized successfully")

            except Exception as e:
                logger.error("❌ Failed to initialize database", error=str(e))
                raise

    async def initialize_async_db(self):
        """Initialize asynchronous database connection with proper pooling"""
        if self._async_initialized:
            return
            
        try:
            database_url = os.getenv("DATABASE_URL")

            if database_url:
                async_url = database_url.replace(
                    "postgresql://", "postgresql+asyncpg://"
                )
                self.async_engine = create_async_engine(
                    async_url,
                    poolclass=QueuePool,
                    pool_size=15,  # Larger pool for async operations
                    max_overflow=25,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False  # Reduce log noise
                )
            else:
                # SQLite async
                db_path = Path("data") / "trading_system.db"
                db_path.parent.mkdir(exist_ok=True)
                async_url = f"sqlite+aiosqlite:///{db_path}"
                self.async_engine = create_async_engine(
                    async_url, 
                    echo=False,
                    poolclass=QueuePool,
                    pool_size=15,
                    max_overflow=25
                )

            self.AsyncSessionLocal = async_sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False  # Important for async operations
            )

            # Create all tables
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._async_initialized = True
            logger.info("✅ Async database initialized successfully")
            
        except Exception as e:
            logger.error("❌ Failed to initialize async database", error=str(e))
            raise

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with proper cleanup and error handling"""
        if not self._async_initialized:
            await self.initialize_async_db()
        
        if not self.AsyncSessionLocal:
            raise RuntimeError("AsyncSessionLocal not initialized")
        
        # Create a new session for each request
        session = self.AsyncSessionLocal()
        
        try:
            yield session
            # Only commit if no exception occurred
            await session.commit()
            logger.debug("Database session committed successfully")
            
        except Exception as e:
            # Rollback on any exception
            await session.rollback()
            logger.error(f"Database session rolled back due to error: {e}")
            raise
            
        finally:
            # Always close the session
            await session.close()
            logger.debug("Database session closed")

    async def safe_db_operation(self, operation_func, *args, **kwargs):
        """Execute database operation with retry logic and proper error handling"""
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                async with self.get_async_session() as session:
                    result = await operation_func(session, *args, **kwargs)
                    return result
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database operation failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                    raise

    def get_session(self) -> Session:
        """Get synchronous database session"""
        if not self._initialized:
            self.initialize_sync_db()
        
        if not self.SessionLocal:
            raise RuntimeError("SessionLocal not initialized")
            
        return self.SessionLocal()

    async def test_async_connection(self) -> bool:
        """Test async database connection"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(text("SELECT 1"))
                row = result.fetchone()
                if row and row[0] == 1:
                    logger.info("✅ Async database connection test successful")
                    return True
                else:
                    logger.error("❌ Async database connection test failed")
                    return False
        except Exception as e:
            logger.error("❌ Async database connection test failed", error=str(e))
            return False

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if not self._initialized:
                self.initialize_sync_db()
                
            with self.get_session() as session:
                result = session.execute(text("SELECT 1"))
                row = result.fetchone()
                if row and row[0] == 1:
                    logger.info("✅ Database connection test successful")
                    return True
                else:
                    logger.error("❌ Database connection test failed")
                    return False
        except Exception as e:
            logger.error("❌ Database connection test failed", error=str(e))
            return False

    def get_table_info(self) -> dict:
        """Get database table information"""
        if not self._initialized:
            return {}
            
        try:
            inspector = inspect(self.engine)
            if inspector is None:
                logger.error("❌ Database inspector is None")
                return {}
        except Exception as e:
            logger.error("❌ Failed to create database inspector")
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
"""
Database Manager for Autonomous Trading System
Handles database connections, initialization, and operations
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker

load_dotenv(override=True)

# Add project root to sys.path to allow running this script directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# from src.config.settings import settings
from src.config.logging_config import logger
from src.database.models import Base


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self._initialized = False

    def initialize_sync_db(self):
        """Initialize synchronous database connection"""
        try:
            # For development, we'll use SQLite if PostgreSQL isn't available
            database_url = os.getenv("DATABASE_URL")
            print(f"databse url:  {database_url}")
            if database_url:
                self.engine = create_engine(database_url)
            else:
                # Fallback to SQLite for easy development
                db_path = Path("data") / "trading_system.db"
                db_path.parent.mkdir(exist_ok=True)
                sqlite_url = f"sqlite:///{db_path}"
                self.engine = create_engine(sqlite_url, echo=True)
                logger.info(
                    "Using SQLite database for development", path=str(db_path)
                )
            
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            self._initialized = True
            logger.info("✅ Database initialized successfully")

        except Exception as e:
            logger.error("❌ Failed to initialize database", error=str(e))
            raise

    async def initialize_async_db(self):
        """Initialize asynchronous database connection"""
        try:
            database_url = os.getenv("DATABASE_URL")

            if database_url:
                async_url = database_url.replace(
                    "postgresql://", "postgresql+asyncpg://"
                )
                self.async_engine = create_async_engine(async_url)
            else:
                # SQLite async
                db_path = Path("data") / "trading_system.db"
                db_path.parent.mkdir(exist_ok=True)
                async_url = f"sqlite+aiosqlite:///{db_path}"
                self.async_engine = create_async_engine(async_url, echo=True)

            self.AsyncSessionLocal = async_sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.async_engine
            )

            # Create all tables
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("✅ Async database initialized successfully")
            
        except Exception as e:
            logger.error("❌ Failed to initialize async database", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session"""
        if not self.AsyncSessionLocal:
            await self.initialize_async_db()
        
        if not self.AsyncSessionLocal:
            raise RuntimeError("AsyncSessionLocal not initialized")
            
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_session(self) -> Session:
        """Get synchronous database session"""
        if not self._initialized:
            self.initialize_sync_db()
        
        if not self.SessionLocal:
            raise RuntimeError("SessionLocal not initialized")
            
        return self.SessionLocal()
    
    @asynccontextmanager
    async def get_session_context(self):
        """Get synchronous database session as context manager"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                logger.info("✅ Database connection test successful")
                return True
        except Exception as e:
            logger.error("❌ Database connection test failed", error=str(e))
            return False
    
    async def test_async_connection(self) -> bool:
        """Test async database connection"""
        try:
            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                logger.info("✅ Async database connection test successful")
                return True
        except Exception as e:
            logger.error(
                "❌ Async database connection test failed", error=str(e)
            )
            return False
    
    def drop_all_tables(self):
        """Drop all tables (for development/testing)"""
        if not self._initialized:
            self.initialize_sync_db()
        
        logger.warning("🗑️ Dropping all database tables")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("✅ All tables dropped")
    
    def create_all_tables(self):
        """Create all tables"""
        if not self._initialized:
            self.initialize_sync_db()
        
        logger.info("🏗️ Creating all database tables")
        Base.metadata.create_all(bind=self.engine)
        logger.info("✅ All tables created")
    
    def get_table_info(self):
        """Get information about all tables"""
        if not self._initialized:
            self.initialize_sync_db()
        inspector = inspect(self.engine)
        if inspector is None:
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


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
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
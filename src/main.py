#!/usr/bin/env python3
"""
Autonomous Trading System - Main Entry Point (Windows Compatible)
"""

import asyncio
import sys
from pathlib import Path
import os

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent))

from config.logging_config import logger

async def health_check():
    """Basic health check to verify environment setup"""
    
    logger.info("🚀 Starting Autonomous Trading System Health Check (Windows)")
    
    # Check configuration
    logger.info("📋 Configuration loaded", 
                environment=os.getenv("OANDA_ENVIRONMENT"),
                database_host=os.getenv("DATABASE_HOST"),
                log_level=os.getenv("LOG_LEVEL"))
    
    # Test imports
    try:
        import crewai
        import fastapi
        import pandas as pd
        import oandapyV20
        import streamlit
        logger.info("✅ All core dependencies imported successfully")
    except ImportError as e:
        logger.error("❌ Failed to import dependencies", error=str(e))
        return False
    
    # Test environment variables (only check if they exist, not values)
    required_env_vars = [
        'OANDA_ENVIRONMENT',
        'DATABASE_HOST'
    ]
    
    optional_env_vars = [
        'OANDA_ACCESS_TOKEN',
        'OANDA_ACCOUNT_ID', 
        'DATABASE_URL',
        'OPENAI_API_KEY'
    ]
    print(os.getenv("OANDA_ENVIRONMENT"))
    
    # Check required vars
    for var in required_env_vars:
        if os.getenv(var) == None:
            logger.error("❌ Missing required environment variable", variable=var)
            return False
    
    logger.info("✅ Required environment variables configured")
    
    # Check optional vars (warn if missing)
    missing_optional = []
    for var in optional_env_vars:
        attr_name = var.lower()
        if os.getenv(var) == None:
            missing_optional.append(var)
    
    if missing_optional:
        logger.warning("⚠️ Optional environment variables not set (needed for full functionality)", 
                      missing=missing_optional)
    
    # Test directory creation
    try:
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        logger.info("✅ Directory structure verified")
    except Exception as e:
        logger.error("❌ Failed to create directories", error=str(e))
        return False
    
    logger.info("🎉 Health check completed successfully!")
    logger.info("📊 Ready to build MCP servers and agents")
    logger.info("🔗 Next steps: Set up Oanda demo account and database")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(health_check())
        if not success:
            print("\n❌ Health check failed. Please fix the issues above.")
            sys.exit(1)
        else:
            print("\n✅ Environment setup complete! Ready for next steps.")
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user")
        sys.exit(1)
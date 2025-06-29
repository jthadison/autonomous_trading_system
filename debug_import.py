#!/usr/bin/env python3
"""
Debug script to test imports step by step
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print("Testing imports step by step...")

try:
    print("1. Testing config.logging_config import...")
    from config.logging_config import logger
    print("   ✅ config.logging_config import successful")
except Exception as e:
    print(f"   ❌ config.logging_config import failed: {e}")

try:
    print("2. Testing crewai.tools import...")
    from crewai.tools import tool
    print("   ✅ crewai.tools import successful")
except Exception as e:
    print(f"   ❌ crewai.tools import failed: {e}")

try:
    print("3. Testing backtesting_simulation_tools module import...")
    import src.autonomous_trading_system.tools.backtesting_simulation_tools as bst
    print("   ✅ Module import successful")
    print(f"   Available attributes: {[attr for attr in dir(bst) if not attr.startswith('_')]}")
except Exception as e:
    print(f"   ❌ Module import failed: {e}")

try:
    print("4. Testing specific function import...")
    from src.autonomous_trading_system.tools.backtesting_simulation_tools import simulate_historical_market_context
    print("   ✅ Function import successful")
except Exception as e:
    print(f"   ❌ Function import failed: {e}")

print("Debug complete.") 
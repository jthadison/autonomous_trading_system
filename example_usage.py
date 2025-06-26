#!/usr/bin/env python
"""
Usage Example: Autonomous Trading System with Full Execution Capabilities

This example shows how to run your enhanced trading system that can now:
1. Analyze markets using Wyckoff methodology
2. Calculate proper risk management
3. Make trading decisions  
4. EXECUTE actual trades automatically

Run this after implementing the execution tools.
"""

import sys
import warnings
import asyncio
from datetime import datetime
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Import your updated trading system
from autonomous_trading_system.crew import AutonomousTradingSystem
from src.config.logging_config import logger

# Import the new execution tools for manual testing (use sync versions for direct calling)
from src.autonomous_trading_system.tools.trading_execution_tools import get_portfolio_status_sync, get_open_positions_sync, get_pending_orders_sync
    # Note: For execute_market_trade, execute_limit_trade, etc., you'll need to create sync versions
    # or use the tool objects differently


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run_autonomous_trading_session():
    """
    Run a complete autonomous trading session with execution capabilities
    """
    print("🚀 Starting Autonomous Trading System with Execution")
    print("=" * 60)
    
    # Initialize the trading system
    trading_system = AutonomousTradingSystem()
    
    # Set trading parameters
    symbol_name = "EUR_USD"  # Change to your preferred instrument
    
    print(f"📊 Analyzing {symbol_name} using Wyckoff methodology...")
    print(f"⏰ Session started at: {datetime.now().isoformat()}")
    print()
    
    try:
        # Execute the full autonomous trading workflow
        # This will now analyze, assess risk, make decisions, AND execute trades
        result = trading_system.execute_trading_session(symbol_name)
        
        print("✅ Trading session completed successfully!")
        print(f"📋 Session result: {result}")
        
        # Get final portfolio status
        print("\n📈 Final Portfolio Status:")
        portfolio_status = get_portfolio_status_sync()
        if portfolio_status.get("success", False):
            print(f"   Total Positions: {portfolio_status.get('total_positions', 0)}")
            print(f"   Pending Orders: {portfolio_status.get('total_pending_orders', 0)}")
            print(f"   Total Exposure: ${portfolio_status.get('total_exposure', 0):,.2f}")
            print(f"   Exposure %: {portfolio_status.get('exposure_pct', 0):.2f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ Trading session failed: {e}")
        logger.error("Autonomous trading session failed", error=str(e))
        return None

def test_execution_tools_manually():
    """
    Test the execution tools manually before running autonomous session
    """
    print("🔧 Testing Execution Tools Manually")
    print("=" * 40)
    
    # Test 1: Get portfolio status
    print("1. Testing portfolio status...")
    portfolio = get_portfolio_status_sync()
    print(f"   Portfolio status: {'✅ Success' if portfolio.get('success') else '❌ Failed'}")
    
    # Test 2: Get current positions
    print("2. Testing position retrieval...")
    positions = get_open_positions_sync()
    print(f"   Positions: {'✅ Success' if positions.get('success') else '❌ Failed'}")
    print(f"   Open positions: {positions.get('position_count', 0)}")
    
    # Test 3: Get pending orders
    print("3. Testing order retrieval...")
    orders = get_pending_orders_sync()
    print(f"   Orders: {'✅ Success' if orders.get('success') else '❌ Failed'}")
    print(f"   Pending orders: {orders.get('order_count', 0)}")
    
    print("\n🔧 Basic testing completed!")
    print("⚠️  Note: Full execution testing (placing/canceling orders) requires")
    print("   implementing sync versions of execute_market_trade, etc.")
    return True

def monitor_live_trading():
    """
    Monitor live trading positions and provide updates
    """
    print("👀 Starting Live Trading Monitor")
    print("=" * 35)
    
    try:
        while True:
            # Get current status
            portfolio = get_portfolio_status_sync()
            
            if portfolio.get("success"):
                positions = portfolio.get("total_positions", 0)
                orders = portfolio.get("total_pending_orders", 0)
                exposure = portfolio.get("total_exposure", 0)
                
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | "
                      f"Positions: {positions} | "
                      f"Orders: {orders} | "
                      f"Exposure: ${exposure:,.0f}")
                
                # Show position details if any exist
                if positions > 0:
                    pos_data = get_open_positions_sync()
                    if pos_data.get("success"):
                        for pos in pos_data.get("positions", []):
                            instrument = pos.get("instrument", "Unknown")
                            units = pos.get("units", 0)
                            current_price = pos.get("current_price", 0)
                            print(f"   📊 {instrument}: {units} units @ ${current_price}")
            
            # Wait 30 seconds before next update
            import time
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n⏹️  Trading monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

def emergency_close_all_positions():
    """
    Emergency function to close all open positions
    """
    print("🚨 EMERGENCY: Closing all positions!")
    
    positions = get_open_positions_sync()
    if not positions.get("success"):
        print("❌ Failed to get positions")
        return
    
    print("⚠️  Note: Emergency close functionality requires implementing")
    print("   sync version of close_position function.")
    print(f"   Found {len(positions.get('positions', []))} positions to close:")
    
    for pos in positions.get("positions", []):
        instrument = pos.get("instrument", "Unknown")
        units = pos.get("units", 0)
        print(f"   📊 {instrument}: {units} units")
    
    print("   Use the agent-based system or implement close_position_sync() to execute closures.")

if __name__ == "__main__":
    print("🎯 Autonomous Trading System - Execution Enabled")
    print("=" * 55)
    
    # Show available options
    print("Choose an option:")
    print("1. Test execution tools manually")
    print("2. Run full autonomous trading session")
    print("3. Monitor live trading")
    print("4. Emergency close all positions")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        test_execution_tools_manually()
    elif choice == "2":
        run_autonomous_trading_session()
    elif choice == "3":
        monitor_live_trading()
    elif choice == "4":
        confirm = input("⚠️  Are you sure you want to close ALL positions? (yes/no): ")
        if confirm.lower() == "yes":
            emergency_close_all_positions()
        else:
            print("❌ Emergency close cancelled")
    elif choice == "5":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")
        
    print("\n🏁 Program completed")
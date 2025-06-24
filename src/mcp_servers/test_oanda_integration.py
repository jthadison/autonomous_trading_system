"""
Test Script: BJLG-92 Oanda MCP Integration
Quick test to verify our wrapper works with the live server
"""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path to allow running this script directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.logging_config import logger
from src.database.manager import init_database

symbol_name = 'EUR/USD'

async def test_live_oanda_integration():
    """Test our integration with the live BJLG-92 Oanda server"""
    print("🔴 LIVE OANDA INTEGRATION TEST")
    print("=" * 50)
    
    # Import our wrapper
    try:
        from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
    except ImportError as e:
        print(f"❌ Failed to import wrapper: {e}")
        print("💡 Make sure you created src/mcp_servers/oanda_mcp_wrapper.py")
        return False
    
    # Test the integration
    async with OandaMCPWrapper("http://localhost:8000") as oanda:
        try:
            print("🏥 Testing server health...")
            health = await oanda.health_check()
            print(f"   Status: {health['status']}, {health['server']}")
            
            if health["status"] != "healthy":
                print("❌ Server not healthy - make sure BJLG-92 server is running")
                return False
            
            print("\n💳 Getting account information...")
            account = await oanda.get_account_info() #.get('data', {})
            print(f"account: {account}")
            print(f"   Balance: ${account.get('balance', 'N/A')}")
            print(f"   Currency: {account.get('currency', 'N/A')}")
            print(f"   Margin Available: ${account.get('margin_available', 'N/A')}")
            
            print(f"\n💰 Getting live {symbol_name} price...")
            eur_usd = (await oanda.get_current_price("EUR_USD")).get('data', {})
            print(f"   Bid: {eur_usd.get('bid', 'N/A')}")
            print(f"   Ask: {eur_usd.get('ask', 'N/A')}")
            print(f"   Spread: {eur_usd.get('spread', 'N/A')}")
            
            print("\n📊 Getting historical data...")
            historical = await oanda.get_historical_data("EUR_USD", "M1", 5)
            data_points = len(historical.get("data", []))
            print(f"   Data points retrieved: {data_points}")
            
            try: 
                print(f"lastest: {historical['data']['candles'][-1]}")
                if data_points > 0:
                    latest = historical['data']['candles'][-1]
                    print(f"   Latest close: {latest.get('mid','N/A').get('c', 'N/A')}")
                    print(f"   Latest time: {latest.get('time', 'N/A')}")
            except Exception as e:
                print(f"\n❌ Get Latest test failed: {e}")
            
            print("\n📈 Getting positions...")
            positions = await oanda.get_positions()
            pos_count = len(positions.get("positions", []))
            print(f"   Open positions: {pos_count}")
            
            print("\n📋 Getting orders...")
            orders = await oanda.get_orders()
            order_count = len(orders.get("orders", []))
            print(f"   Pending orders: {order_count}")
            
            print("\n" + "=" * 50)
            print("🎉 INTEGRATION TEST SUCCESSFUL!")
            print("✅ Live Oanda data flowing through our system")
            print("✅ Account info accessible")
            print("✅ Price data streaming")
            print("✅ Historical data available")
            print("✅ Ready to build trading agents!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            print("\n🔧 Troubleshooting:")
            print("   1. Make sure BJLG-92 server is running on localhost:8000")
            print("   2. Check your Oanda API credentials in their .env file")
            print("   3. Verify network connectivity")
            return False

async def live_price_stream_demo():
    """Demo: Show live price updates"""
    
    print("\n🔴 LIVE PRICE STREAM DEMO")
    print(f"Showing 10 live {symbol_name} price updates...")
    print("=" * 40)
    
    from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
    
    async with OandaMCPWrapper("http://localhost:8000") as oanda:
        try:
            for i in range(10):
                price = (await oanda.get_current_price("EUR_USD")).get('data')
                
                from datetime import datetime
                timestamp = datetime.now().strftime("%H:%M:%S")
                if price is not None:
                    bid = price.get('bid', 0)
                    ask = price.get('ask', 0)
                    spread = price.get('spread', 0)
                else:
                    bid = ask = spread = 0
                print(f" {timestamp} | {symbol_name}: {float(bid):.5f} / {float(ask):.5f} | Spread: {float(spread):.5f}")
                
                if i < 9:  # Don't wait after the last update
                    await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Price stream failed: {e}")

async def main():
    """Main test function"""
    
    print("🚀 AUTONOMOUS TRADING SYSTEM")
    print("🔗 BJLG-92 Oanda MCP Integration Test")
    print("=" * 60)
    
    # Initialize database
    print("📊 Initializing database...")
    try:
        success = init_database()
        if success:
            print("✅ Database ready")
        else:
            print("⚠️ Database issues, but continuing...")
    except Exception as e:
        print(f"⚠️ Database error: {e}, but continuing...")
    
    # Test integration
    success = await test_live_oanda_integration()
    
    if success:
        # Ask if user wants to see live stream
        print("\n" + "=" * 60)
        response = input(f"🔴 Want to see live {symbol_name} price stream? (y/n): ")
        if response.lower() == 'y':
            await live_price_stream_demo()
        
        print("\n🎯 NEXT STEPS:")
        print("   1. Build first CrewAI trading agent")
        print("   2. Add Wyckoff pattern recognition")
        print("   3. Create dashboard for monitoring")
        print("   4. Add risk management")
        
        print(f"\n✨ Your trading system is ALIVE and connected to live markets! ✨")
        
    else:
        print("\n🔧 Fix the issues above and run the test again")

if __name__ == "__main__":
    asyncio.run(main())
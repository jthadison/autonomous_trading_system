# Save this as test_oanda_api.py and run it to test your API directly

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
    print("✅ OandaMCPWrapper imported successfully")
except ImportError as e:
    print(f"❌ Failed to import OandaMCPWrapper: {e}")
    sys.exit(1)

async def test_oanda_connection():
    """Test Oanda connection and data retrieval"""
    
    print("🔍 Testing Oanda MCP Connection...")
    print("=" * 50)
    
    try:
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            print("✅ Connection established")
            
            # Test 1: Health check
            print("\n📋 Testing health check...")
            health = await oanda.health_check()
            print(f"Health status: {health}")
            
            # Test 2: Current price
            print("\n💰 Testing current price...")
            symbol = "EUR_USD"
            price = await oanda.get_current_price(symbol)
            print(f"Current price for {symbol}: {price}")
            
            # Test 3: Historical data (the problematic one)
            print(f"\n📊 Testing historical data for {symbol}...")
            print("Requesting: M5, 10 candles")
            
            historical = await oanda.get_historical_data(symbol, "M5", 10)
            
            print(f"Response type: {type(historical)}")
            
            if isinstance(historical, dict):
                print(f"Response keys: {list(historical.keys())}")
                print(f"Full response:")
                import json
                print(json.dumps(historical, indent=2)[:1000] + "..." if len(str(historical)) > 1000 else json.dumps(historical, indent=2))
            elif isinstance(historical, list):
                print(f"Response is list with {len(historical)} items")
                if len(historical) > 0:
                    print(f"First item: {historical[0]}")
            else:
                print(f"Unexpected response type: {type(historical)}")
                print(f"Response: {historical}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 OANDA API DIRECT TEST")
    print("This will help debug the historical data issue")
    print()
    
    try:
        asyncio.run(test_oanda_connection())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
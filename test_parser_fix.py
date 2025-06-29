"""
Test the fixed Oanda data parser with the actual data structure
"""

import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.logging_config import logger

async def test_fixed_parser():
    """Test the fixed parser with real Oanda data"""
    
    logger.info("🧪 TESTING FIXED OANDA PARSER")
    logger.info("=" * 50)
    
    try:
        from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            
            # Test different timeframes with the fixed parser
            timeframes_to_test = ["M15", "H1", "M5"]
            
            for timeframe in timeframes_to_test:
                logger.info(f"\n🔍 Testing {timeframe} with fixed parser...")
                
                try:
                    # Get raw data
                    historical_result = await oanda.get_historical_data("EUR_USD", timeframe, 5)
                    
                    # Apply the FIXED parsing logic
                    data = None
                    
                    if isinstance(historical_result, dict):
                        if historical_result.get('success') and 'data' in historical_result:
                            oanda_data = historical_result['data']
                            
                            if isinstance(oanda_data, dict):
                                actual_granularity = oanda_data.get('granularity', 'Unknown')
                                logger.info(f"📊 Requested: {timeframe}, Got: {actual_granularity}")
                                
                                if 'candles' in oanda_data:
                                    data = oanda_data['candles']
                                    logger.info(f"✅ Found {len(data)} candles")
                                else:
                                    logger.error(f"❌ No 'candles' in data: {list(oanda_data.keys())}")
                                    continue
                            else:
                                logger.error(f"❌ Data section is not dict: {type(oanda_data)}")
                                continue
                        else:
                            logger.error(f"❌ Invalid response structure")
                            continue
                    else:
                        logger.error(f"❌ Response is not dict: {type(historical_result)}")
                        continue
                    
                    # Test candle parsing
                    if data and len(data) > 0:
                        test_candle = data[0]
                        logger.info(f"🔍 Testing first candle: {test_candle}")
                        
                        # Extract fields using fixed logic
                        timestamp = test_candle.get('time')
                        mid_data = test_candle.get('mid', {})
                        volume = test_candle.get('volume', 1000)
                        
                        if mid_data and isinstance(mid_data, dict):
                            try:
                                open_price = float(mid_data.get('o', 0))
                                high_price = float(mid_data.get('h', 0))
                                low_price = float(mid_data.get('l', 0))
                                close_price = float(mid_data.get('c', 0))
                                
                                logger.info(f"✅ Parsed successfully:")
                                logger.info(f"   Timestamp: {timestamp}")
                                logger.info(f"   OHLC: {open_price:.5f}/{high_price:.5f}/{low_price:.5f}/{close_price:.5f}")
                                logger.info(f"   Volume: {volume}")
                                
                                # Validate OHLC relationships
                                if (high_price >= max(open_price, close_price) and 
                                    low_price <= min(open_price, close_price)):
                                    logger.info(f"✅ OHLC relationships valid")
                                else:
                                    logger.warning(f"⚠️ OHLC relationships invalid")
                                
                            except (ValueError, TypeError) as e:
                                logger.error(f"❌ Failed to parse OHLC: {e}")
                        else:
                            logger.error(f"❌ No 'mid' data found: {test_candle}")
                    
                    logger.info(f"✅ {timeframe} parsing test complete")
                    
                except Exception as e:
                    logger.error(f"❌ {timeframe} test failed: {e}")
        
        logger.info("\n🎉 PARSER TESTING COMPLETE!")
        logger.info("Apply the fixed parser to real_backtest_runner.py")
        
    except Exception as e:
        logger.error(f"❌ Parser test failed: {e}")

async def test_complete_workflow():
    """Test the complete workflow with fixed parser"""
    
    logger.info("\n🚀 TESTING COMPLETE WORKFLOW")
    logger.info("=" * 50)
    
    try:
        # Simulate the fixed get_real_historical_data function
        from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # Get data using fixed logic
            historical_result = await oanda.get_historical_data("EUR_USD", "M15", 10)
            
            # Process with fixed parser
            formatted_data = []
            
            if (isinstance(historical_result, dict) and 
                historical_result.get('success') and 
                'data' in historical_result):
                
                oanda_data = historical_result['data']
                candles = oanda_data.get('candles', [])
                
                logger.info(f"📊 Processing {len(candles)} candles...")
                
                for i, candle in enumerate(candles):
                    try:
                        timestamp = candle.get('time')
                        mid_data = candle.get('mid', {})
                        volume = candle.get('volume', 1000)
                        
                        if mid_data:
                            open_price = float(mid_data.get('o', 0))
                            high_price = float(mid_data.get('h', 0))
                            low_price = float(mid_data.get('l', 0))
                            close_price = float(mid_data.get('c', 0))
                            
                            formatted_bar = {
                                'timestamp': timestamp,
                                'open': round(open_price, 5),
                                'high': round(high_price, 5), 
                                'low': round(low_price, 5),
                                'close': round(close_price, 5),
                                'volume': int(volume)
                            }
                            formatted_data.append(formatted_bar)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing candle {i}: {e}")
                
                if formatted_data:
                    logger.info(f"✅ Successfully formatted {len(formatted_data)} bars")
                    logger.info(f"📊 First bar: {formatted_data[0]}")
                    logger.info(f"📊 Last bar: {formatted_data[-1]}")
                    
                    # Calculate movement
                    if len(formatted_data) > 1:
                        start_price = formatted_data[0]['close']
                        end_price = formatted_data[-1]['close']
                        movement = end_price - start_price
                        movement_pct = (movement / start_price) * 100
                        
                        logger.info(f"📈 Price movement: {start_price:.5f} → {end_price:.5f}")
                        logger.info(f"📈 Change: {movement:+.5f} ({movement_pct:+.2f}%)")
                    
                    logger.info("🎉 COMPLETE WORKFLOW SUCCESS!")
                    logger.info("The fixed parser is ready for production use!")
                else:
                    logger.error("❌ No bars could be formatted")
            else:
                logger.error("❌ Invalid response structure")
                
    except Exception as e:
        logger.error(f"❌ Workflow test failed: {e}")

async def main():
    """Run all parser tests"""
    
    logger.info("🔧 OANDA PARSER FIX VALIDATION")
    logger.info("=" * 60)
    
    # Test 1: Individual parsing components
    await test_fixed_parser()
    
    # Test 2: Complete workflow
    await test_complete_workflow()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 NEXT STEPS:")
    logger.info("1. Replace get_real_historical_data() in real_backtest_runner.py")
    logger.info("2. Run python real_backtest_runner.py")
    logger.info("3. Enjoy real H1 data analysis by your Wyckoff agents!")

if __name__ == "__main__":
    asyncio.run(main())
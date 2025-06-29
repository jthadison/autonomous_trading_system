"""
Diagnostic tool to understand Oanda data formats
Run this to see exactly what data formats you're receiving
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

async def diagnose_oanda_data_formats():
    """Diagnose what data formats Oanda returns for different timeframes"""
    
    logger.info("🔍 OANDA DATA FORMAT DIAGNOSTIC")
    logger.info("=" * 50)
    
    try:
        from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
        
        # Test different timeframes
        timeframes_to_test = [
            ("M1", 5),
            ("M5", 5), 
            ("M15", 5),
            ("H1", 5),
        ]
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            
            for timeframe, bars in timeframes_to_test:
                logger.info(f"\n🔍 Testing {timeframe} timeframe...")
                
                try:
                    # Get raw data
                    result = await oanda.get_historical_data("EUR_USD", timeframe, bars)
                    
                    # Analyze the response structure
                    logger.info(f"📊 Response type: {type(result)}")
                    logger.info(f"📊 Response keys (if dict): {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    
                    if isinstance(result, dict):
                        for key, value in result.items():
                            logger.info(f"   {key}: {type(value)} - {str(value)[:100]}...")
                            
                            # If this is the data array, analyze first bar
                            if key == 'data' and isinstance(value, list) and value:
                                first_bar = value[0]
                                logger.info(f"   First bar type: {type(first_bar)}")
                                logger.info(f"   First bar: {first_bar}")
                                
                                if isinstance(first_bar, dict):
                                    logger.info(f"   Bar fields: {list(first_bar.keys())}")
                                elif isinstance(first_bar, str):
                                    logger.info(f"   Bar string content: {first_bar}")
                                    try:
                                        parsed_bar = json.loads(first_bar)
                                        logger.info(f"   Parsed bar: {parsed_bar}")
                                        logger.info(f"   Parsed bar fields: {list(parsed_bar.keys())}")
                                    except:
                                        logger.info("   Could not parse as JSON")
                    
                    elif isinstance(result, list):
                        logger.info(f"📊 List length: {len(result)}")
                        if result:
                            logger.info(f"📊 First item: {result[0]}")
                            logger.info(f"📊 First item type: {type(result[0])}")
                    
                    elif isinstance(result, str):
                        logger.info(f"📊 String content: {result[:200]}...")
                        try:
                            parsed = json.loads(result)
                            logger.info(f"📊 Parsed JSON: {type(parsed)}")
                            logger.info(f"📊 Parsed content: {str(parsed)[:200]}...")
                        except:
                            logger.info("📊 Not valid JSON")
                    
                    logger.info(f"✅ {timeframe} diagnostic complete")
                    
                except Exception as e:
                    logger.error(f"❌ {timeframe} failed: {e}")
                    
        logger.info("\n🔍 DIAGNOSTIC COMPLETE")
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")

async def test_data_parsing_fix():
    """Test the new enhanced data parsing"""
    
    logger.info("\n🧪 TESTING ENHANCED DATA PARSING")
    logger.info("=" * 50)
    
    try:
        from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            
            # Test H1 data specifically (the problematic timeframe)
            logger.info("🔍 Testing H1 data parsing...")
            
            raw_result = await oanda.get_historical_data("EUR_USD", "H1", 10)
            logger.info(f"📊 Raw result: {raw_result}")
            
            # Apply the enhanced parsing logic
            data = None
            
            if isinstance(raw_result, dict):
                if 'data' in raw_result:
                    data = raw_result['data']
                elif 'result' in raw_result:
                    data = raw_result['result']
                else:
                    data = raw_result
                    
            elif isinstance(raw_result, list):
                data = raw_result
                
            elif isinstance(raw_result, str):
                try:
                    parsed_result = json.loads(raw_result)
                    if isinstance(parsed_result, dict) and 'data' in parsed_result:
                        data = parsed_result['data']
                    elif isinstance(parsed_result, list):
                        data = parsed_result
                    else:
                        data = parsed_result
                except json.JSONDecodeError:
                    logger.error(f"❌ Failed to parse JSON: {raw_result[:100]}...")
                    data = None
            
            if data and isinstance(data, list) and len(data) > 0:
                logger.info(f"✅ Successfully extracted data: {len(data)} bars")
                logger.info(f"📊 First bar: {data[0]}")
                
                # Test bar parsing
                bar = data[0]
                logger.info(f"🔍 Testing bar parsing for: {bar}")
                
                if isinstance(bar, str):
                    try:
                        bar = json.loads(bar)
                        logger.info(f"✅ Parsed string bar to: {bar}")
                    except:
                        logger.error("❌ Could not parse string bar as JSON")
                
                if isinstance(bar, dict):
                    # Try to extract fields
                    timestamp = None
                    for time_field in ['time', 'timestamp', 't', 'datetime']:
                        if time_field in bar:
                            timestamp = bar[time_field]
                            break
                    
                    ohlc = {}
                    for price_type in ['open', 'high', 'low', 'close']:
                        for field in [price_type[0], price_type, price_type.title()]:
                            if field in bar:
                                try:
                                    ohlc[price_type] = float(bar[field])
                                    break
                                except:
                                    continue
                    
                    logger.info(f"✅ Extracted timestamp: {timestamp}")
                    logger.info(f"✅ Extracted OHLC: {ohlc}")
                    
                    if timestamp and len(ohlc) == 4:
                        logger.info("🎉 Bar parsing would succeed!")
                    else:
                        logger.warning("⚠️ Bar parsing would fail")
            else:
                logger.error("❌ Could not extract data from response or data is not a list")
                
    except Exception as e:
        logger.error(f"❌ Parsing test failed: {e}")

async def main():
    """Run all diagnostics"""
    
    logger.info("🔍 COMPREHENSIVE OANDA DATA DIAGNOSTIC")
    logger.info("=" * 60)
    
    # Diagnostic 1: Understand data formats
    await diagnose_oanda_data_formats()
    
    # Diagnostic 2: Test parsing fix
    await test_data_parsing_fix()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 DIAGNOSTIC SUMMARY")
    logger.info("Use this information to understand your Oanda data format")
    logger.info("Then apply the enhanced parsing fix to real_backtest_runner.py")

if __name__ == "__main__":
    asyncio.run(main())
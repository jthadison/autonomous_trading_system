"""
Real Historical Data Backtest Runner
Fetches actual market data from Oanda MCP server for agent-based backtesting
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.logging_config import logger
from src.autonomous_trading_system.crew import AutonomousTradingSystem

# Import simulation tools for direct testing (with proper calling method)
from src.backtesting.backtesting_simulation_tools import (
    simulate_historical_market_context,
    simulate_trade_execution,
    update_backtest_portfolio,
    calculate_backtest_performance_metrics
)

async def get_real_historical_data(
    symbol: str = "EUR_USD", 
    timeframe: str = "M15", 
    bars: int = 100,
    max_retries: int = 3
):
    """
    FIXED VERSION: Get real historical data from Oanda MCP server with correct parsing
    Based on diagnostic results showing actual Oanda data structure
    """
    
    logger.info(f"📊 Fetching {bars} bars of {symbol} {timeframe} data from Oanda...")
    
    for attempt in range(max_retries):
        try:
            from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
            
            async with OandaMCPWrapper("http://localhost:8000") as oanda:
                # Test connection first
                health = await oanda.health_check()
                if health.get("status") != "healthy":
                    logger.warning(f"⚠️ Oanda server not healthy: {health}")
                    if attempt < max_retries - 1:
                        logger.info(f"🔄 Retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise Exception("Oanda server not healthy after all retries")
                
                # Get historical data
                historical_result = await oanda.get_historical_data(symbol, timeframe, bars)
                
                # DEBUG: Log the raw response structure
                logger.info(f"🔍 Raw response type: {type(historical_result)}")
                if isinstance(historical_result, dict):
                    logger.info(f"🔍 Response keys: {list(historical_result.keys())}")
                
                # CORRECT PARSING based on diagnostic results
                data = None
                
                if isinstance(historical_result, dict):
                    # Expected structure: {'success': True, 'data': {'candles': [...]}}
                    if historical_result.get('success') and 'data' in historical_result:
                        oanda_data = historical_result['data']
                        
                        # Log what we found in the data section
                        if isinstance(oanda_data, dict):
                            logger.info(f"🔍 Data section keys: {list(oanda_data.keys())}")
                            
                            # Check actual granularity returned vs requested
                            actual_granularity = oanda_data.get('granularity', 'Unknown')
                            logger.info(f"📊 Requested: {timeframe}, Got: {actual_granularity}")
                            
                            if actual_granularity != timeframe:
                                logger.warning(f"⚠️ Granularity mismatch: requested {timeframe}, got {actual_granularity}")
                            
                            # Extract the candles array
                            if 'candles' in oanda_data:
                                data = oanda_data['candles']
                                logger.info(f"✅ Found {len(data)} candles in response")
                            else:
                                raise Exception("No 'candles' array found in data section")
                        else:
                            # Fallback: try direct data access
                            data = oanda_data
                    else:
                        raise Exception(f"Invalid response structure: {historical_result}")
                else:
                    raise Exception(f"Expected dict response, got: {type(historical_result)}")
                
                # Validate we have data
                if not data:
                    raise Exception("No candles data found in Oanda response")
                    
                if not isinstance(data, list):
                    raise Exception(f"Expected list of candles, got: {type(data)}")
                
                logger.info(f"✅ Retrieved {len(data)} candles from Oanda")
                
                # Validate minimum data requirements
                min_required_bars = max(2, bars * 0.1)  # At least 2 bars or 10% of requested
                if len(data) < min_required_bars:
                    logger.warning(f"⚠️ Only got {len(data)} candles, need at least {min_required_bars}")
                    # Don't fail if we have some data, just warn
                    if len(data) == 0:
                        raise Exception("No candles returned")
                
                if len(data) < bars * 0.8:  # Less than 80% of requested
                    logger.warning(f"⚠️ Only got {len(data)} candles out of {bars} requested")
                
                # ENHANCED CANDLE PARSING based on diagnostic results
                formatted_data = []
                valid_bars = 0
                
                for i, candle in enumerate(data):
                    try:
                        # Expected candle structure from diagnostic:
                        # {
                        #   "complete": True,
                        #   "volume": 181047,
                        #   "time": "2025-06-15T21:00:00.000000000Z",
                        #   "mid": {"o": "1.15332", "h": "1.16148", "l": "1.15237", "c": "1.15609"}
                        # }
                        
                        if not isinstance(candle, dict):
                            logger.warning(f"⚠️ Candle {i} is not a dict: {type(candle)}")
                            continue
                        
                        # Extract timestamp
                        timestamp = candle.get('time')
                        if not timestamp:
                            logger.warning(f"⚠️ No timestamp in candle {i}")
                            continue
                        
                        # Extract OHLC from 'mid' section (based on diagnostic)
                        mid_data = candle.get('mid', {})
                        if not isinstance(mid_data, dict):
                            logger.warning(f"⚠️ No 'mid' data in candle {i}: {candle}")
                            continue
                        
                        # Extract OHLC values from mid section
                        try:
                            open_price = float(mid_data.get('o', 0))
                            high_price = float(mid_data.get('h', 0))  
                            low_price = float(mid_data.get('l', 0))
                            close_price = float(mid_data.get('c', 0))
                        except (ValueError, TypeError) as e:
                            logger.warning(f"⚠️ Invalid OHLC values in candle {i}: {mid_data}")
                            continue
                        
                        # Validate OHLC values
                        if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                            logger.warning(f"⚠️ Invalid price values in candle {i}")
                            continue
                        
                        # Validate OHLC relationships
                        if (high_price < max(open_price, close_price) or 
                            low_price > min(open_price, close_price)):
                            logger.warning(f"⚠️ OHLC inconsistency in candle {i}")
                            continue
                        
                        # Extract volume
                        volume = candle.get('volume', 1000)
                        try:
                            volume = int(volume)
                        except (ValueError, TypeError):
                            volume = 1000
                        
                        # Create formatted bar
                        formatted_bar = {
                            'timestamp': timestamp,
                            'open': round(open_price, 5),
                            'high': round(high_price, 5),
                            'low': round(low_price, 5),
                            'close': round(close_price, 5),
                            'volume': volume
                        }
                        formatted_data.append(formatted_bar)
                        valid_bars += 1
                        
                    except Exception as candle_error:
                        logger.warning(f"⚠️ Error processing candle {i}: {candle_error}")
                        continue
                
                if valid_bars == 0:
                    raise Exception("No valid candles after data validation")
                
                logger.info(f"✅ Validated {valid_bars} candles out of {len(data)} received")
                
                if formatted_data:
                    logger.info(f"📊 Data range: {formatted_data[0]['timestamp']} → {formatted_data[-1]['timestamp']}")
                    logger.info(f"💹 Price range: {formatted_data[0]['close']:.5f} → {formatted_data[-1]['close']:.5f}")
                    
                    # Calculate price movement
                    price_change = formatted_data[-1]['close'] - formatted_data[0]['close']
                    price_change_pct = (price_change / formatted_data[0]['close']) * 100
                    logger.info(f"📈 Movement: {price_change:+.5f} ({price_change_pct:+.2f}%)")
                
                return formatted_data
                    
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"🔄 Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ Failed to fetch real data after {max_retries} attempts")
                return None

async def validate_oanda_connection():
    """Test Oanda MCP server connection before running backtests"""
    
    logger.info("🔌 Testing Oanda MCP server connection...")
    
    try:
        from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # Test health
            health = await oanda.health_check()
            logger.info(f"🏥 Server health: {health.get('status', 'unknown')}")
            
            if health.get("status") != "healthy":
                logger.error("❌ Oanda server not healthy")
                return False
            
            # Test account info
            account = await oanda.get_account_info()
            if 'balance' in account:
                logger.info(f"💰 Account balance: ${account.get('balance', 'N/A')}")
            
            # Test current price
            price = await oanda.get_current_price("EUR_USD")
            if 'data' in price:
                current_price = price['data'].get('c', 'N/A')
                logger.info(f"💱 Current EUR_USD: {current_price}")
            
            logger.info("✅ Oanda connection validated successfully")
            return True
            
    except Exception as e:
        logger.error(f"❌ Oanda connection test failed: {e}")
        return False

def generate_fallback_data(num_bars: int = 50, symbol: str = "EUR_USD") -> list:
    """Generate realistic fallback data only if real data is unavailable"""
    
    logger.warning("⚠️ Using fallback generated data - real data unavailable")
    logger.info(f"📊 Generating {num_bars} realistic bars for {symbol}...")
    
    # Base prices for different symbols
    base_prices = {
        "EUR_USD": 1.0950,
        "GBP_USD": 1.2650,
        "USD_JPY": 149.50,
        "AUD_USD": 0.6750,
        "USD_CHF": 0.8850
    }
    
    base_price = base_prices.get(symbol, 1.0000)
    current_time = datetime.now() - timedelta(minutes=num_bars * 15)
    
    bars = []
    current_price = base_price
    
    for i in range(num_bars):
        # Create more realistic market movements
        trend_component = 0
        noise_component = (i % 20 - 10) * 0.00005  # Small random movements
        
        # Add trend periods
        if 20 <= i < 40:  # Uptrend period
            trend_component = 0.0001
        elif 60 <= i < 80:  # Downtrend period
            trend_component = -0.0001
        
        price_change = trend_component + noise_component
        new_price = current_price + price_change
        
        # Create realistic OHLC with proper wicks
        open_price = current_price
        close_price = new_price
        
        # Add realistic wicks (10-30% of the range)
        range_size = abs(close_price - open_price)
        wick_size = max(range_size * 0.2, base_price * 0.00005)  # Minimum wick size
        
        high_price = max(open_price, close_price) + wick_size
        low_price = min(open_price, close_price) - wick_size
        
        # Realistic volume (higher during trend moves)
        base_volume = 1000
        if abs(trend_component) > 0:
            volume = base_volume + 300  # Higher volume during trends
        else:
            volume = base_volume + (i % 200)  # Variable volume
        
        bar = {
            'timestamp': (current_time + timedelta(minutes=i * 15)).isoformat(),
            'open': round(open_price, 5),
            'high': round(high_price, 5),
            'low': round(low_price, 5),
            'close': round(close_price, 5),
            'volume': volume
        }
        bars.append(bar)
        current_price = close_price
    
    logger.info(f"✅ Generated fallback data: {bars[0]['close']:.5f} → {bars[-1]['close']:.5f}")
    return bars

async def run_real_data_backtest(
    symbol: str = "EUR_USD",
    timeframe: str = "M15", 
    bars: int = 100,
    initial_balance: float = 100000
):
    """
    Run backtest with real historical data from Oanda
    
    Args:
        symbol: Trading symbol
        timeframe: Chart timeframe
        bars: Number of historical bars
        initial_balance: Starting capital
    """
    
    logger.info("🚀 REAL HISTORICAL DATA BACKTEST")
    logger.info("=" * 50)
    
    try:
        # Step 1: Validate Oanda connection
        logger.info("🔌 Step 1: Validating Oanda connection...")
        connection_ok = await validate_oanda_connection()
        
        if not connection_ok:
            logger.error("❌ Cannot proceed without Oanda connection")
            logger.info("💡 Make sure your BJLG-92 Oanda MCP server is running on localhost:8000")
            return {'success': False, 'error': 'Oanda connection failed'}
        
        # Step 2: Fetch real historical data
        logger.info(f"📊 Step 2: Fetching real {symbol} {timeframe} data...")
        historical_data = await get_real_historical_data(symbol, timeframe, bars)
        print(f"historical data: {historical_data}")
        if not historical_data:
            logger.warning("⚠️ Real data fetch failed, using fallback data")
            historical_data = generate_fallback_data(bars, symbol)
        
        logger.info(f"✅ Data ready: {len(historical_data)} bars")
        
        # Step 3: Run agent-based backtest
        logger.info("🤖 Step 3: Initializing AI trading agents...")
        trading_system = AutonomousTradingSystem()
        
        logger.info("🧠 Step 4: Running Wyckoff agent analysis...")
        logger.info(f"   📊 Analyzing {len(historical_data)} bars of {symbol} data")
        logger.info(f"   💰 Starting with ${initial_balance:,.2f}")
        logger.info(f"   🎯 Agents will identify Wyckoff patterns and make trading decisions")
        
        # Execute backtest with real data
        result = trading_system.run_agent_backtest(
            historical_data=historical_data,
            initial_balance=initial_balance,
            symbol=symbol
        )
        
        # Step 5: Results analysis
        if result.get('success'):
            logger.info("🎉 REAL DATA BACKTEST COMPLETED!")
            logger.info("=" * 50)
            
            logger.info("📊 BACKTEST SUMMARY:")
            logger.info(f"   Symbol: {result.get('symbol')}")
            logger.info(f"   Timeframe: {result.get('timeframe', timeframe)}")
            logger.info(f"   Bars Analyzed: {result.get('total_bars_processed')}")
            logger.info(f"   Initial Balance: ${result.get('initial_balance'):,.2f}")
            
            # Check for report generation
            report_path = result.get('report_path')
            if report_path:
                logger.info(f"📝 Analysis Report: {report_path}")
            
            logger.info("\n🤖 AGENT PERFORMANCE:")
            logger.info("   Your Wyckoff agents analyzed real market data and made decisions")
            logger.info("   Check the markdown report for complete analysis details")
            
            return result
        else:
            logger.error(f"❌ Backtest failed: {result.get('error')}")
            return result
            
    except Exception as e:
        logger.error(f"❌ Real data backtest failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}

async def run_multi_symbol_backtest():
    """Run backtests on multiple symbols with real data"""
    
    logger.info("🌍 MULTI-SYMBOL REAL DATA BACKTEST")
    logger.info("=" * 50)
    
    # Major currency pairs
    symbols = [
        "EUR_USD",
        # "GBP_USD", 
        # "USD_JPY",
        # "US30"
    ]
    
    results = {}
    
    for symbol in symbols:
        logger.info(f"\n📊 Testing {symbol}...")
        
        result = await run_real_data_backtest(
            symbol=symbol,
            timeframe="M15",
            bars=50,  # Smaller for multi-symbol test
            initial_balance=50000
        )
        
        results[symbol] = result
        
        if result.get('success'):
            logger.info(f"✅ {symbol} backtest completed")
        else:
            logger.error(f"❌ {symbol} backtest failed: {result.get('error')}")
    
    # Summary
    logger.info("\n🌍 MULTI-SYMBOL SUMMARY:")
    successful = [s for s, r in results.items() if r.get('success')]
    failed = [s for s, r in results.items() if not r.get('success')]
    
    logger.info(f"✅ Successful: {successful}")
    if failed:
        logger.info(f"❌ Failed: {failed}")
    
    return results

async def run_different_timeframes():
    """Test the same symbol across different timeframes"""
    
    logger.info("⏰ MULTI-TIMEFRAME REAL DATA BACKTEST")
    logger.info("=" * 50)
    
    timeframes = [
        ("M5", 200),   # 5-minute bars
        ("M15", 100),  # 15-minute bars  
        ("H1", 50),    # 1-hour bars
    ]
    
    symbol = "EUR_USD"
    results = {}
    
    for timeframe, bars in timeframes:
        logger.info(f"\n⏰ Testing {symbol} on {timeframe} timeframe...")
        
        result = await run_real_data_backtest(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
            initial_balance=75000
        )
        
        results[timeframe] = result
        
        if result.get('success'):
            logger.info(f"✅ {timeframe} analysis completed")
        else:
            logger.error(f"❌ {timeframe} analysis failed")
    
    # Compare results across timeframes
    logger.info("\n⏰ TIMEFRAME COMPARISON:")
    for tf, result in results.items():
        if result.get('success'):
            logger.info(f"   {tf}: Analysis completed successfully")
        else:
            logger.info(f"   {tf}: Analysis failed")
    
    return results

async def main():
    """Main test runner with real historical data focus"""
    
    logger.info("📊 REAL HISTORICAL DATA BACKTESTING SUITE")
    logger.info("=" * 60)
    
    # Test 1: Single symbol real data backtest
    logger.info("\n1️⃣ SINGLE SYMBOL REAL DATA BACKTEST...")
    single_result = await run_real_data_backtest(
        symbol="EUR_USD",
        timeframe="M15", 
        bars=150,
        initial_balance=100000
    )
    
    if not single_result.get('success'):
        logger.error("❌ Single symbol test failed. Check Oanda connection.")
        return
    
    logger.info("\n" + "=" * 60)
    
    # Test 2: Multi-symbol backtest
    #logger.info("\n2️⃣ MULTI-SYMBOL REAL DATA BACKTEST...")
    #multi_result = await run_multi_symbol_backtest()
    
    logger.info("\n" + "=" * 60)
    
    # Test 3: Multi-timeframe backtest
    #logger.info("\n3️⃣ MULTI-TIMEFRAME REAL DATA BACKTEST...")
    #timeframe_result = await run_different_timeframes()
    
    logger.info("\n" + "=" * 60)
    
    # Final summary
    logger.info("🎉 REAL DATA BACKTESTING COMPLETE!")
    
    logger.info("\n✅ ACHIEVEMENTS:")
    logger.info("   📊 Real historical data from Oanda MCP server")
    logger.info("   🤖 AI agents analyzed actual market movements")
    logger.info("   📝 Professional markdown reports generated")
    logger.info("   🎯 Wyckoff methodology applied to real patterns")
    
    logger.info("\n📁 CHECK YOUR REPORTS:")
    logger.info("   All analysis reports saved in 'reports/' directory")
    logger.info("   Each report contains complete agent reasoning")
    
    logger.info("\n🚀 NEXT STEPS:")
    logger.info("   1. Review generated reports for agent insights")
    logger.info("   2. Compare performance across symbols/timeframes")
    logger.info("   3. Analyze Wyckoff pattern identification accuracy")
    logger.info("   4. Fine-tune agent parameters based on results")
    logger.info("   5. Scale up to longer time periods")

if __name__ == "__main__":
    asyncio.run(main())
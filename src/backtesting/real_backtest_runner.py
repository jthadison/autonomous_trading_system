"""
Real Historical Data Backtest Runner
Fetches actual market data from Oanda Direct API for agent-based backtesting
"""

from ast import Dict
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

from backtesting.enhanced_agent_backtest import EnhancedAgentBacktester

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
from src.backtesting.enhanced_backtest_metrics import (
    EnhancedMetricsCalculator, 
    EnhancedBacktestResults,
    enhance_existing_backtest_results
)

from src.config.logging_config import logger
from src.autonomous_trading_system.crew import AutonomousTradingSystem

async def get_real_historical_data(
    symbol, 
    timeframe, 
    bars,
    max_retries: int = 3
):
    """
    UPDATED VERSION: Get real historical data from Oanda Direct API with correct parsing
    """
    
    logger.info(f"📊 Fetching {bars} bars of {symbol} {timeframe} data from Oanda Direct API...")
    
    for attempt in range(max_retries):
        try:
            # UPDATED: Import OandaDirectAPI instead of OandaMCPWrapper
            from src.mcp_servers.oanda_direct_api import OandaDirectAPI
            
            # UPDATED: No URL needed for direct API
            async with OandaDirectAPI() as oanda:
                # Test connection first
                health = await oanda.health_check()
                if health.get("status") != "healthy":
                    logger.warning(f"⚠️ Oanda API not healthy: {health}")
                    if attempt < max_retries - 1:
                        logger.info(f"🔄 Retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise Exception("Oanda API not healthy after all retries")
                
                # UPDATED: Map timeframe for Oanda API (M15 -> M15, etc.)
                granularity_map = {
                    "M1": "M1",
                    "M5": "M5", 
                    "M15": "M15",
                    "M30": "M30",
                    "H1": "H1",
                    "H4": "H4",
                    "D": "D"
                }
                oanda_granularity = granularity_map.get(timeframe, timeframe)
                
                # Get historical data
                # Ensure oanda_granularity is not None before passing to API
                if oanda_granularity is None:
                    raise ValueError(f"Invalid timeframe '{timeframe}' - could not map to Oanda granularity")
                
                historical_result = await oanda.get_historical_data(
                    instrument=symbol,
                    granularity=oanda_granularity,
                    count=bars
                )
                
                # DEBUG: Log the raw response structure
                logger.info(f"🔍 Raw response type: {type(historical_result)}")
                if isinstance(historical_result, dict):
                    logger.info(f"🔍 Response keys: {list(historical_result.keys())}")
                
                # UPDATED: Parse OandaDirectAPI response format
                data = None
                
                if isinstance(historical_result, dict):
                    # Expected structure from OandaDirectAPI: 
                    # {'success': True, 'data': {'candles': [...]}, 'instrument': '...', 'granularity': '...'}
                    if historical_result.get('success') and 'data' in historical_result:
                        oanda_data = historical_result['data']
                        
                        # Log what we found in the data section
                        if isinstance(oanda_data, dict):
                            logger.info(f"🔍 Data section keys: {list(oanda_data.keys())}")
                            
                            # Check actual granularity returned vs requested
                            actual_granularity = historical_result.get('granularity', 'Unknown')
                            logger.info(f"📊 Requested: {timeframe}, Got: {actual_granularity}")
                            
                            if actual_granularity != oanda_granularity:
                                logger.warning(f"⚠️ Granularity mismatch: requested {oanda_granularity}, got {actual_granularity}")
                            
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
                        error_msg = historical_result.get('error', 'Unknown error')
                        raise Exception(f"API request failed: {error_msg}")
                else:
                    raise Exception(f"Expected dict response, got: {type(historical_result)}")
                
                # Validate we have data
                if not data:
                    raise Exception("No candles data found in Oanda response")
                    
                if not isinstance(data, list):
                    raise Exception(f"Expected list of candles, got: {type(data)}")
                
                logger.info(f"✅ Retrieved {len(data)} candles from Oanda Direct API")
                
                # Validate minimum data requirements
                min_required_bars = max(2, bars * 0.1)  # At least 2 bars or 10% of requested
                if len(data) < min_required_bars:
                    logger.warning(f"⚠️ Only got {len(data)} candles, need at least {min_required_bars}")
                    # Don't fail if we have some data, just warn
                    if len(data) == 0:
                        raise Exception("No candles returned")
                
                if len(data) < bars * 0.8:  # Less than 80% of requested
                    logger.warning(f"⚠️ Only got {len(data)} candles out of {bars} requested")
                
                # UPDATED: Enhanced candle parsing for OandaDirectAPI format
                formatted_data = []
                valid_bars = 0
                
                for i, candle in enumerate(data):
                    try:
                        # Expected candle structure from OandaDirectAPI:
                        # {
                        #   "time": "2025-06-15T21:00:00.000000000Z",
                        #   "volume": 181047,
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
                        
                        # Extract OHLC from 'mid' section
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
    """Test Oanda Direct API connection before running backtests"""
    
    logger.info("🔌 Testing Oanda Direct API connection...")
    
    try:
        # UPDATED: Import OandaDirectAPI
        from src.mcp_servers.oanda_direct_api import OandaDirectAPI
        
        # UPDATED: No URL parameter needed
        async with OandaDirectAPI() as oanda:
            # Test health
            health = await oanda.health_check()
            logger.info(f"🏥 API health: {health.get('status', 'unknown')}")
            
            if health.get("status") != "healthy":
                logger.error("❌ Oanda Direct API not healthy")
                return False
            
            # Test account info
            account = await oanda.get_account_info()
            if account.get('success') and 'balance' in account:
                logger.info(f"💰 Account balance: ${account.get('balance', 'N/A'):,.2f}")
                logger.info(f"💱 Account currency: {account.get('currency', 'N/A')}")
                logger.info(f"📊 Environment: {account.get('environment', 'N/A')}")
            else:
                logger.warning(f"⚠️ Account info error: {account.get('error', 'Unknown')}")
            
            # Test current price
            price = await oanda.get_current_price("EUR_USD")
            if price.get('success'):
                current_price = price.get('price', 'N/A')
                bid = price.get('bid', 'N/A')
                ask = price.get('ask', 'N/A')
                spread = price.get('spread', 'N/A')
                logger.info(f"💱 Current EUR_USD: {current_price} (Bid: {bid}, Ask: {ask}, Spread: {spread})")
            else:
                logger.warning(f"⚠️ Price data error: {price.get('error', 'Unknown')}")
            
            logger.info("✅ Oanda Direct API connection validated successfully")
            return True
            
    except Exception as e:
        logger.error(f"❌ Oanda Direct API connection test failed: {e}")
        logger.error("💡 Make sure you have set OANDA_API_KEY, OANDA_ACCOUNT_ID, and OANDA_ENVIRONMENT in your .env file")
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
    symbol,
    timeframe, 
    bars,
    initial_balance
):
    """
    Run backtest with real historical data from Oanda Direct API
    
    Args:
        symbol: Trading symbol
        timeframe: Chart timeframe
        bars: Number of historical bars
        initial_balance: Starting capital
    """
    
    logger.info("🚀 REAL HISTORICAL DATA BACKTEST (Oanda Direct API)")
    logger.info("=" * 50)
    
    try:
        # Step 1: Validate Oanda connection
        logger.info("🔌 Step 1: Validating Oanda Direct API connection...")
        connection_ok = await validate_oanda_connection()
        
        if not connection_ok:
            logger.error("❌ Cannot proceed without Oanda Direct API connection")
            logger.info("💡 Check your .env file for OANDA_API_KEY, OANDA_ACCOUNT_ID, and OANDA_ENVIRONMENT")
            return {'success': False, 'error': 'Oanda Direct API connection failed'}
        
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
    
    logger.info("🌍 MULTI-SYMBOL REAL DATA BACKTEST (Oanda Direct API)")
    logger.info("=" * 50)
    
    # Major currency pairs
    symbols = [
        "EUR_USD",
        # "GBP_USD", 
        # "USD_JPY",
        # "US30"  # Note: Check if this symbol is available in your Oanda account
    ]
    
    results = {}
    
    for symbol in symbols:
        logger.info(f"\n📊 Testing {symbol}...")
        
        result = await run_real_data_backtest(
            symbol=symbol,
            timeframe="M5",
            bars=200,  # Smaller for multi-symbol test
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
    
    logger.info("⏰ MULTI-TIMEFRAME REAL DATA BACKTEST (Oanda Direct API)")
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

async def run_comprehensive_backtest(
        symbol: str = "EUR_USD",
        timeframe: str = "M15",
        bars: int = 100,
        initial_balance: float = 100000,
        use_real_data: bool = True
    ) -> dict[str, Any]:
    """
    SINGLE ENTRY POINT: Run comprehensive backtest
    
    Features:
    ✅ Real Oanda data (with fallback)
    ✅ Enhanced metrics (optional)
    ✅ Windows compatibility
    ✅ Comprehensive reporting
    ✅ Error handling
    """
    
    logger.info("🚀 COMPREHENSIVE BACKTEST RUNNER")
    
    try:
        # Step 1: Get data (real or fallback)
        historical_data = None
        
        if use_real_data:
            historical_data = await get_real_historical_data(symbol, timeframe, bars)
        
        if not historical_data:
            logger.info("📊 Using fallback data")
            historical_data = generate_fallback_data(bars, symbol)
        
        # Step 2: Run backtest (single class handles everything)
        backtester = EnhancedAgentBacktester(initial_balance)
        results = await backtester.run_agent_backtest(
            historical_data, initial_balance, symbol
        )
        
        # Step 3: Return results
        return results
        
    except Exception as e:
        logger.error(f"❌ Comprehensive backtest failed: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Run enhanced testing suite with Oanda Direct API"""
    
    logger.info("🚀 Starting enhanced testing with Oanda Direct API...")
    
    result = await run_comprehensive_backtest(
        symbol="EUR_USD",
        timeframe="M1", 
        bars=200,
        initial_balance=100000,
        use_real_data=True
    )
    
    if result.get('success'):
        print("✅ Backtest completed!")
        print(f"📊 Return: {result.get('total_return_pct', 0):+.2f}%")
        print(f"📝 Report: {result.get('report_path')}")
    else:
        print(f"❌ Backtest failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())
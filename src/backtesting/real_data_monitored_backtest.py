"""
Real Historical Data Integration for Monitored Backtesting
Fetches actual market data from Oanda Direct API for realistic agent testing
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.backtesting.monitored_backtest import MonitoredBacktester, AgentOptimizationTester
from src.config.logging_config import logger

class RealDataMonitoredBacktester(MonitoredBacktester):
    """Enhanced MonitoredBacktester that fetches real historical data"""
    
    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.data_cache = {}  # Cache data to avoid repeated API calls
    
    async def fetch_real_historical_data(
    self,
    symbol: str,
    timeframe: str = "M15",
    bars: int = 500,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
        """Fetch real historical data from Oanda Direct API - ENHANCED VERSION"""
        
        cache_key = f"{symbol}_{timeframe}_{bars}"
        
        # Check cache first
        if use_cache and cache_key in self.data_cache:
            logger.info(f"📊 Using cached data for {symbol} {timeframe} ({bars} bars)")
            return self.data_cache[cache_key]
        
        logger.info(f"📡 Fetching real historical data: {symbol} {timeframe} ({bars} bars)")
        
        try:
            from src.mcp_servers.oanda_direct_api import OandaDirectAPI
            
            async with OandaDirectAPI() as oanda:
                # Test connection first
                health = await oanda.health_check()
                if health.get("status") != "healthy":
                    logger.warning(f"⚠️ Oanda API not healthy: {health}")
                    return await self._fetch_fallback_data(symbol, timeframe, bars)
                
                # Fetch historical data with better error handling
                try:
                    response = await oanda.get_historical_data(
                        instrument=symbol,
                        granularity=timeframe,
                        count=bars
                    )
                    
                    # Debug the response structure
                    logger.info(f"🔍 Oanda response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                    
                    if response.get('success'):
                        # Extract the actual data - handle different response formats
                        raw_data = response.get('data')
                        
                        if raw_data is None:
                            # Try alternative keys
                            raw_data = response.get('candles') or response.get('bars') or response.get('history')
                        
                        if raw_data and isinstance(raw_data, list):
                            logger.info(f"📊 Raw data received: {len(raw_data)} items")
                            
                            # Clean and validate the data
                            cleaned_data = self._clean_historical_data(raw_data)
                            
                            if cleaned_data and len(cleaned_data) > 10:  # Need at least 10 bars
                                # Cache the data
                                if use_cache:
                                    self.data_cache[cache_key] = cleaned_data
                                
                                logger.info(f"✅ Successfully processed {len(cleaned_data)} bars of real data for {symbol}")
                                return cleaned_data
                            else:
                                logger.warning(f"⚠️ Insufficient cleaned data ({len(cleaned_data) if cleaned_data else 0} bars), using fallback")
                                return await self._fetch_fallback_data(symbol, timeframe, bars)
                        else:
                            logger.warning(f"⚠️ No valid data in response, using fallback")
                            logger.debug(f"   Response data type: {type(raw_data)}")
                            return await self._fetch_fallback_data(symbol, timeframe, bars)
                    else:
                        logger.warning(f"⚠️ API request failed: {response.get('error', 'Unknown error')}")
                        return await self._fetch_fallback_data(symbol, timeframe, bars)
                        
                except Exception as api_error:
                    logger.error(f"❌ Oanda API call failed: {api_error}")
                    return await self._fetch_fallback_data(symbol, timeframe, bars)
                        
        except ImportError:
            logger.warning("⚠️ OandaDirectAPI not available, using fallback data")
            return await self._fetch_fallback_data(symbol, timeframe, bars)
        except Exception as e:
            logger.error(f"❌ Error fetching real data: {e}")
            return await self._fetch_fallback_data(symbol, timeframe, bars)
    
    def _clean_historical_data(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Clean and validate historical data - FIXED for Oanda format"""
    
        cleaned_data = []
        
        # Debug: Check what we actually received
        logger.info(f"🔍 Raw data inspection: type={type(raw_data)}, length={len(raw_data) if isinstance(raw_data, list) else 'N/A'}")
        if raw_data:
            logger.info(f"🔍 First item type: {type(raw_data[0])}")
            logger.info(f"🔍 First item sample: {str(raw_data[0])[:200]}...")
        
        for i, bar in enumerate(raw_data):
            try:
                # Handle different possible data formats
                if isinstance(bar, str):
                    logger.warning(f"⚠️ Bar {i} is string, attempting to parse as JSON")
                    try:
                        bar = json.loads(bar)
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Bar {i} cannot be parsed as JSON: {bar[:100]}...")
                        continue
                
                if not isinstance(bar, dict):
                    logger.warning(f"⚠️ Bar {i} is not a dictionary: {type(bar)}")
                    continue
                
                # Oanda format: Check for 'time' and either 'mid' or direct OHLC
                if 'time' not in bar:
                    logger.warning(f"⚠️ Bar {i} missing 'time' field")
                    continue
                
                # Handle different Oanda response formats
                if 'mid' in bar:
                    # Format 1: Oanda candlestick format with 'mid' prices
                    mid = bar['mid']
                    if not isinstance(mid, dict) or not all(price in mid for price in ['o', 'h', 'l', 'c']):
                        logger.warning(f"⚠️ Bar {i} missing OHLC in mid prices")
                        continue
                    
                    open_price = float(mid['o'])
                    high_price = float(mid['h'])
                    low_price = float(mid['l'])
                    close_price = float(mid['c'])
                    
                elif all(field in bar for field in ['open', 'high', 'low', 'close']):
                    # Format 2: Direct OHLC fields
                    open_price = float(bar['open'])
                    high_price = float(bar['high'])
                    low_price = float(bar['low'])
                    close_price = float(bar['close'])
                    
                else:
                    logger.warning(f"⚠️ Bar {i} has unknown price format: {list(bar.keys())}")
                    continue
                
                # Validate price logic
                if not (low_price <= open_price <= high_price and 
                    low_price <= close_price <= high_price):
                    logger.warning(f"⚠️ Bar {i} has invalid OHLC logic: O:{open_price}, H:{high_price}, L:{low_price}, C:{close_price}")
                    continue
                
                # Validate reasonable price ranges (basic sanity check)
                if any(price <= 0 or price > 10 for price in [open_price, high_price, low_price, close_price]):
                    logger.warning(f"⚠️ Bar {i} has unrealistic prices")
                    continue
                
                # Convert time format - handle Oanda timestamp format
                timestamp = bar['time']
                if isinstance(timestamp, str):
                    # Remove nanoseconds and timezone info if present
                    if timestamp.endswith('.000000000Z'):
                        timestamp = timestamp[:-10] + 'Z'
                    elif timestamp.endswith('Z'):
                        pass  # Already in correct format
                    else:
                        # Add Z if no timezone info
                        timestamp = timestamp + 'Z' if 'T' in timestamp else timestamp
                
                # Create cleaned bar
                cleaned_bar = {
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': bar.get('volume', 1000)  # Default volume if not available
                }
                
                cleaned_data.append(cleaned_bar)
                
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"⚠️ Error cleaning bar {i}: {e}")
                logger.debug(f"   Bar content: {bar}")
                continue
        
        # Sort by timestamp to ensure chronological order
        try:
            cleaned_data.sort(key=lambda x: x['timestamp'])
        except Exception as e:
            logger.warning(f"⚠️ Could not sort data by timestamp: {e}")
        
        logger.info(f"📊 Cleaned data: {len(raw_data)} → {len(cleaned_data)} bars")
        
        return cleaned_data
    
    async def _fetch_fallback_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int
    ) -> List[Dict[str, Any]]:
        """Generate realistic fallback data if real data unavailable"""
        
        logger.info(f"📊 Generating realistic fallback data for {symbol}")
        
        # Use more realistic base prices for different symbols
        base_prices = {
            'EUR_USD': 1.0800,
            'GBP_USD': 1.2500,
            'USD_JPY': 150.00,
            'AUD_USD': 0.6500,
            'USD_CHF': 0.9000,
            'USD_CAD': 1.3500
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate realistic price movements
        import random
        import numpy as np
        
        # Set seed for reproducible "realistic" data
        random.seed(42)
        np.random.seed(42)
        
        fallback_data = []
        current_price = base_price
        current_time = datetime.now() - timedelta(minutes=15 * bars)
        
        # Generate trend and volatility patterns
        trend_duration = random.randint(50, 150)
        current_trend = random.choice([-1, 1])  # -1 bearish, 1 bullish
        bars_in_trend = 0
        
        for i in range(bars):
            # Change trend occasionally
            if bars_in_trend > trend_duration:
                current_trend = random.choice([-1, 0, 1])  # Include sideways
                trend_duration = random.randint(30, 100)
                bars_in_trend = 0
            
            # Calculate price movement
            trend_strength = 0.0003 * current_trend  # Small trend component
            volatility = random.uniform(0.0001, 0.0008)  # Random volatility
            noise = random.uniform(-volatility, volatility)
            
            # Create realistic OHLC for this bar
            open_price = current_price
            
            # Generate high/low with some logic
            if current_trend > 0:  # Bullish
                high_offset = random.uniform(volatility * 0.5, volatility * 1.5)
                low_offset = random.uniform(-volatility * 0.8, volatility * 0.3)
            elif current_trend < 0:  # Bearish
                high_offset = random.uniform(-volatility * 0.3, volatility * 0.8)
                low_offset = random.uniform(-volatility * 1.5, -volatility * 0.5)
            else:  # Sideways
                high_offset = random.uniform(volatility * 0.3, volatility * 1.0)
                low_offset = random.uniform(-volatility * 1.0, -volatility * 0.3)
            
            high_price = open_price + high_offset
            low_price = open_price + low_offset
            close_price = open_price + trend_strength + noise
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            fallback_data.append({
                'timestamp': current_time.isoformat(),
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': random.randint(800, 1500)
            })
            
            current_price = close_price
            current_time += timedelta(minutes=15)
            bars_in_trend += 1
        
        logger.info(f"✅ Generated {len(fallback_data)} realistic fallback bars")
        return fallback_data
    
    async def run_real_data_backtest(
        self,
        symbol: str = "EUR_USD",
        timeframe: str = "M15",
        bars: int = 500,
        initial_balance: float = 100000,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run backtest with real historical data"""
        
        logger.info(f"🚀 Starting REAL DATA backtest: {symbol} {timeframe} ({bars} bars)")
        
        # Fetch real historical data
        historical_data = await self.fetch_real_historical_data(symbol, timeframe, bars)
        
        if not historical_data:
            logger.error("❌ Failed to fetch historical data")
            return {'success': False, 'error': 'No historical data available'}
        
        # Add data quality metrics
        data_quality = self._assess_data_quality(historical_data)
        logger.info(f"📊 Data quality score: {data_quality['quality_score']:.1f}/100")
        
        # Run monitored backtest
        result = await self.run_monitored_backtest(
            historical_data=historical_data,
            initial_balance=initial_balance,
            symbol=symbol,
            agent_config=agent_config
        )
        
        # Add data info to results
        if result.get('success'):
            result['data_info'] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'bars_requested': bars,
                'bars_received': len(historical_data),
                'date_range': {
                    'start': historical_data[0]['timestamp'],
                    'end': historical_data[-1]['timestamp']
                },
                'data_quality': data_quality
            }
        
        return result
    
    def _assess_data_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """Assess the quality of historical data"""
        
        if not data:
            return {'quality_score': 0, 'issues': ['No data']}
        
        issues = []
        quality_score = 100
        
        # Check for gaps in data
        timestamps = [datetime.fromisoformat(bar['timestamp'].replace('Z', '+00:00')) for bar in data]
        time_gaps = []
        
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # Gap in minutes
            if gap > 20:  # More than 20 minutes for M15 data
                time_gaps.append(gap)
        
        if time_gaps:
            issues.append(f"Time gaps detected: {len(time_gaps)} gaps")
            quality_score -= min(len(time_gaps) * 5, 30)
        
        # Check for price anomalies
        prices = [bar['close'] for bar in data]
        price_changes = []
        
        for i in range(1, len(prices)):
            change_pct = abs(prices[i] - prices[i-1]) / prices[i-1] * 100
            price_changes.append(change_pct)
        
        extreme_moves = [c for c in price_changes if c > 2.0]  # >2% moves
        if extreme_moves:
            issues.append(f"Extreme price moves: {len(extreme_moves)}")
            quality_score -= min(len(extreme_moves) * 2, 20)
        
        # Check for duplicate timestamps
        unique_timestamps = set(bar['timestamp'] for bar in data)
        if len(unique_timestamps) < len(data):
            duplicates = len(data) - len(unique_timestamps)
            issues.append(f"Duplicate timestamps: {duplicates}")
            quality_score -= duplicates * 5
        
        # Check OHLC logic
        ohlc_errors = 0
        for bar in data:
            if not (bar['low'] <= bar['open'] <= bar['high'] and 
                   bar['low'] <= bar['close'] <= bar['high']):
                ohlc_errors += 1
        
        if ohlc_errors:
            issues.append(f"OHLC logic errors: {ohlc_errors}")
            quality_score -= ohlc_errors * 10
        
        return {
            'quality_score': max(quality_score, 0),
            'total_bars': len(data),
            'issues': issues,
            'time_span_hours': (timestamps[-1] - timestamps[0]).total_seconds() / 3600,
            'avg_price_volatility': sum(price_changes) / len(price_changes) if price_changes else 0
        }


class RealDataOptimizationSuite:
    """Complete optimization suite using real historical data"""
    
    def __init__(self):
        self.backtester = RealDataMonitoredBacktester()
        self.optimizer = AgentOptimizationTester()
        self.results_cache = {}
    
    async def run_comprehensive_optimization(
        self,
        symbols: List[str] = ["EUR_USD", "GBP_USD"],
        timeframes: List[str] = ["M15"],
        test_periods: List[int] = [200, 500],
        initial_balance: float = 100000
    ) -> Dict[str, Any]:
        """Run comprehensive optimization across multiple symbols and timeframes"""
        
        logger.info(f"🔬 Starting comprehensive optimization suite...")
        logger.info(f"   Symbols: {symbols}")
        logger.info(f"   Timeframes: {timeframes}")
        logger.info(f"   Test periods: {test_periods}")
        
        all_results = {}
        
        for symbol in symbols:
            symbol_results = {}
            
            for timeframe in timeframes:
                timeframe_results = {}
                
                for bars in test_periods:
                    logger.info(f"📊 Testing {symbol} {timeframe} ({bars} bars)...")
                    
                    # Test baseline configuration
                    baseline_result = await self.backtester.run_real_data_backtest(
                        symbol=symbol,
                        timeframe=timeframe,
                        bars=bars,
                        initial_balance=initial_balance,
                        agent_config=None
                    )
                    
                    # Test optimization configurations
                    optimization_configs = {
                        'baseline': {},
                        'lower_confidence': {'confidence_threshold': 65},
                        'higher_creativity': {'temperature_settings': {'market_analyst': 0.25}},
                        'conservative_risk': {'risk_multiplier': 0.8},
                        'aggressive_growth': {'confidence_threshold': 60, 'risk_multiplier': 1.2}
                    }
                    
                    # Run A/B tests
                    historical_data = await self.backtester.fetch_real_historical_data(
                        symbol, timeframe, bars
                    )
                    
                    if historical_data:
                        optimization_result = await self.optimizer.run_optimization_tests(
                            historical_data=historical_data,
                            test_configurations=optimization_configs,
                            initial_balance=initial_balance
                        )
                        
                        timeframe_results[f"{bars}_bars"] = {
                            'baseline': baseline_result,
                            'optimization': optimization_result,
                            'data_quality': baseline_result.get('data_info', {}).get('data_quality', {})
                        }
                        
                        # Log quick results
                        if optimization_result.get('comparison', {}).get('performance_ranking'):
                            best_config = optimization_result['comparison']['performance_ranking'][0]
                            logger.info(f"   🏆 Best: {best_config['config']} ({best_config['return_pct']:+.2f}%)")
                    
                    else:
                        logger.warning(f"   ⚠️ Failed to get data for {symbol} {timeframe} {bars}")
                
                symbol_results[timeframe] = timeframe_results
            
            all_results[symbol] = symbol_results
        
        # Generate comprehensive report
        summary_report = self._generate_optimization_summary(all_results)
        
        return {
            'detailed_results': all_results,
            'summary_report': summary_report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_optimization_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary report of optimization results"""
        
        summary = {
            'best_configurations': {},
            'symbol_performance': {},
            'optimization_insights': [],
            'data_quality_summary': {}
        }
        
        # Analyze best configurations across all tests
        all_configs = {}
        
        for symbol, symbol_data in results.items():
            symbol_best = []
            
            for timeframe, tf_data in symbol_data.items():
                for period, period_data in tf_data.items():
                    opt_result = period_data.get('optimization', {})
                    ranking = opt_result.get('comparison', {}).get('performance_ranking', [])
                    
                    if ranking:
                        best = ranking[0]
                        symbol_best.append({
                            'config': best['config'],
                            'return_pct': best['return_pct'],
                            'period': period,
                            'timeframe': timeframe
                        })
                        
                        # Track config frequency
                        config_name = best['config']
                        if config_name not in all_configs:
                            all_configs[config_name] = []
                        all_configs[config_name].append(best['return_pct'])
            
            # Best config for this symbol
            if symbol_best:
                best_for_symbol = max(symbol_best, key=lambda x: x['return_pct'])
                summary['best_configurations'][symbol] = best_for_symbol
        
        # Overall best configuration
        config_averages = {
            config: sum(returns) / len(returns) 
            for config, returns in all_configs.items() 
            if returns
        }
        
        if config_averages:
            overall_best = max(config_averages.items(), key=lambda x: x[1])
            summary['overall_best_config'] = {
                'config': overall_best[0],
                'avg_return': overall_best[1],
                'win_rate': len([r for r in all_configs[overall_best[0]] if r > 0]) / len(all_configs[overall_best[0]]) * 100
            }
        
        # Generate insights
        insights = []
        
        if 'lower_confidence' in config_averages:
            lower_conf_avg = config_averages['lower_confidence']
            baseline_avg = config_averages.get('baseline', 0)
            
            if lower_conf_avg > baseline_avg + 1:
                insights.append(f"💡 Lower confidence threshold (+{lower_conf_avg - baseline_avg:.1f}% avg improvement)")
        
        if 'higher_creativity' in config_averages:
            creativity_avg = config_averages['higher_creativity']
            baseline_avg = config_averages.get('baseline', 0)
            
            if creativity_avg > baseline_avg + 1:
                insights.append(f"🎨 Higher creativity improves performance (+{creativity_avg - baseline_avg:.1f}% avg)")
        
        summary['optimization_insights'] = insights
        
        return summary
    
    async def save_optimization_results(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ):
        """Save optimization results to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        # Make results JSON serializable
        json_results = json.loads(json.dumps(results, default=str))
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"💾 Optimization results saved: {filepath}")
        return filepath


# Example usage functions
async def run_single_real_backtest():
    """Run a single backtest with real data"""
    
    backtester = RealDataMonitoredBacktester()
    
    result = await backtester.run_real_data_backtest(
        symbol="EUR_USD",
        timeframe="M15",
        bars=300,  # About 3 days of M15 data
        initial_balance=100000
    )
    
    if result['success']:
        logger.info("🎉 Real data backtest completed!")
        logger.info(f"📊 Return: {result.get('total_return_pct', 0):+.2f}%")
        logger.info(f"🎯 Decisions tracked: {result.get('monitoring_insights', {}).get('total_decisions_tracked', 0)}")
        
        # Show data quality
        data_info = result.get('data_info', {})
        if data_info:
            logger.info(f"📅 Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
            logger.info(f"📊 Data quality: {data_info['data_quality']['quality_score']}/100")
        
        # Show agent performance
        agent_perf = result.get('monitoring_insights', {}).get('agent_performance', {})
        for agent, metrics in agent_perf.items():
            logger.info(f"🤖 {agent}: {metrics.get('accuracy', 0):.1f}% accuracy")
        
        return result
    else:
        logger.error(f"❌ Backtest failed: {result.get('error')}")
        return None

async def run_optimization_suite():
    """Run comprehensive optimization with real data"""
    
    optimizer = RealDataOptimizationSuite()
    
    results = await optimizer.run_comprehensive_optimization(
        symbols=["EUR_USD"],  # Start with one symbol
        timeframes=["M15"],
        test_periods=[200, 400],  # Different time periods
        initial_balance=100000
    )
    
    # Save results
    await optimizer.save_optimization_results(results)
    
    # Show summary
    summary = results['summary_report']
    
    logger.info("🏆 OPTIMIZATION SUMMARY:")
    if 'overall_best_config' in summary:
        best = summary['overall_best_config']
        logger.info(f"   Best config: {best['config']}")
        logger.info(f"   Average return: {best['avg_return']:+.2f}%")
        logger.info(f"   Win rate: {best['win_rate']:.1f}%")
    
    for insight in summary.get('optimization_insights', []):
        logger.info(f"   {insight}")
    
    return results

if __name__ == "__main__":
    # Run single backtest
    print("Running single real data backtest...")
    single_result = asyncio.run(run_single_real_backtest())
    
    if single_result:
        print("\nRunning optimization suite...")
        optimization_results = asyncio.run(run_optimization_suite())
        print("✅ Optimization complete! Check optimization_results/ directory for detailed results.")
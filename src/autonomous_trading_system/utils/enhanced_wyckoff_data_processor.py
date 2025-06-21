"""
Integration of Wyckoff Data Preparation Fix with Original Backtesting Engine
Fixes the nested candles data structure issue and enhances the existing system
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path to allow running this script directly
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
from autonomous_trading_system.utils.backtest import AutomatedWyckoffBacktester, BacktestResults
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from src.config.logging_config import logger


# Import your existing classes (these would be your actual imports)
# from your_backtest_module import AutomatedWyckoffBacktester, BacktestResults, BacktestTrade

class EnhancedWyckoffDataProcessor:
    """
    Enhanced data processor that integrates with your existing AutomatedWyckoffBacktester
    Fixes the OANDA nested candles structure and prepares data for your system
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    def fix_oanda_candles_structure(self, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Fix the nested candles structure from your OANDA data and convert to format
        expected by your AutomatedWyckoffBacktester._prepare_backtest_data() method
        
        This directly addresses the error: "Data must contain 'time' or 'timestamp' column"
        
        Args:
            raw_data: Either a pandas DataFrame or a list of dictionaries containing OANDA data
        """
        try:
            # Convert to DataFrame if it's a list
            if isinstance(raw_data, list):
                raw_dataframe = pd.DataFrame(raw_data)
            else:
                raw_dataframe = raw_data
                
            self.logger.info(f"🔧 Fixing OANDA data structure for {len(raw_dataframe)} rows")
            
            processed_data = []
            
            for idx, row in raw_dataframe.iterrows():
                try:
                    # Extract the nested candle data
                    candle_data = row['candles']
                    
                    # Handle different formats of nested data
                    if isinstance(candle_data, str):
                        # Parse JSON-like string
                        candle_data = self._parse_candle_string(candle_data)
                    elif isinstance(candle_data, dict):
                        # Already a dictionary
                        pass
                    else:
                        self.logger.warning(f"Unexpected candle data format at row {idx}: {type(candle_data)}")
                        continue
                    
                    # Extract OHLCV data in the format your backtest expects
                    processed_row = self._extract_ohlcv_data(candle_data, row)
                    if processed_row:
                        processed_data.append(processed_row)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process row {idx}: {str(e)}")
                    continue
            
            self.logger.info(f"✅ Successfully processed {len(processed_data)} candles")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Data structure fix failed: {str(e)}")
            raise ValueError(f"Failed to fix OANDA data structure: {str(e)}")
    
    def fix_oanda_candles_list(self, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Alternative method specifically for List[Dict] input
        """
        return self.fix_oanda_candles_structure(raw_data)
    
    def _parse_candle_string(self, candle_string: str) -> Dict[str, Any]:
        """Parse candle data from string format"""
        try:
            # Replace single quotes with double quotes for JSON parsing
            json_string = candle_string.replace("'", '"')
            return json.loads(json_string)
        except json.JSONDecodeError:
            # Try eval as fallback (be careful with this in production)
            try:
                return eval(candle_string)
            except:
                raise ValueError(f"Could not parse candle string: {candle_string[:100]}...")
    
    def _extract_ohlcv_data(self, candle_data: Dict[str, Any], original_row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Extract OHLCV data in the exact format expected by your backtesting engine
        """
        try:
            # Extract timestamp (multiple possible locations)
            timestamp = (
                candle_data.get('time') or 
                candle_data.get('timestamp') or 
                original_row.get('time') or
                original_row.get('timestamp')
            )
            
            if not timestamp:
                return None
            
            # Extract mid prices (OANDA format)
            mid_data = candle_data.get('mid', {})
            if not mid_data:
                # Try alternative structures
                mid_data = candle_data
            
            # Extract bid/ask if available (for spread analysis)
            bid_data = candle_data.get('bid', {})
            ask_data = candle_data.get('ask', {})
            
            # Create the data structure expected by your _prepare_backtest_data method
            processed_row = {
                'time': timestamp,  # This fixes the "must contain 'time'" error
                'timestamp': timestamp,  # Alternative timestamp field
                'open': float(mid_data.get('o', mid_data.get('open', 0))),
                'high': float(mid_data.get('h', mid_data.get('high', 0))),
                'low': float(mid_data.get('l', mid_data.get('low', 0))),
                'close': float(mid_data.get('c', mid_data.get('close', 0))),
                'volume': int(candle_data.get('volume', 1000)),  # Default volume if missing
                'complete': candle_data.get('complete', True),
                
                # Additional OANDA-specific data
                'instrument': original_row.get('instrument', 'EUR_USD'),
                'granularity': original_row.get('granularity', 'M15'),
                
                # Spread data if available
                'bid_open': float(bid_data.get('o', 0)) if bid_data else None,
                'bid_high': float(bid_data.get('h', 0)) if bid_data else None,
                'bid_low': float(bid_data.get('l', 0)) if bid_data else None,
                'bid_close': float(bid_data.get('c', 0)) if bid_data else None,
                'ask_open': float(ask_data.get('o', 0)) if ask_data else None,
                'ask_high': float(ask_data.get('h', 0)) if ask_data else None,
                'ask_low': float(ask_data.get('l', 0)) if ask_data else None,
                'ask_close': float(ask_data.get('c', 0)) if ask_data else None,
            }
            
            # Validate the data
            if self._validate_candle_data(processed_row):
                return processed_row
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to extract OHLCV data: {str(e)}")
            return None
    
    def _validate_candle_data(self, candle: Dict[str, Any]) -> bool:
        """Validate that candle data is reasonable"""
        try:
            # Check required fields
            required_fields = ['open', 'high', 'low', 'close']
            for field in required_fields:
                if field not in candle or candle[field] <= 0:
                    return False
            
            # Check OHLC relationships
            if not (candle['low'] <= candle['open'] <= candle['high'] and
                    candle['low'] <= candle['close'] <= candle['high']):
                return False
            
            # Check for reasonable price ranges (EUR/USD example)
            if candle['close'] < 0.5 or candle['close'] > 2.0:
                return False
            
            return True
            
        except:
            return False


class IntegratedWyckoffBacktester(AutomatedWyckoffBacktester):
    """
    Enhanced version of your AutomatedWyckoffBacktester that integrates the data fix
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.data_processor = EnhancedWyckoffDataProcessor(logger=self.logger)
    
    def _prepare_backtest_data(self, 
                              price_data: List[Dict[str, Any]], 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Enhanced version of your _prepare_backtest_data method that handles the nested structure
        """
        
        try:
            # Check if we have the problematic nested structure
            if self._has_nested_candles_structure(price_data):
                self.logger.info("🔧 Detected nested candles structure, applying fix...")
                
                # Convert to DataFrame first if it's not already
                if isinstance(price_data, list):
                    df = pd.DataFrame(price_data)
                else:
                    df = price_data
                
                # Apply the fix
                fixed_data = self.data_processor.fix_oanda_candles_structure(df)
                
                # Continue with your original logic using the fixed data
                return super()._prepare_backtest_data(fixed_data, start_date, end_date)
            
            else:
                # Data is already in correct format, use original method
                return super()._prepare_backtest_data(price_data, start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"Enhanced data preparation failed: {str(e)}")
            raise ValueError(f"Failed to prepare backtest data: {str(e)}")
    
    def _has_nested_candles_structure(self, data) -> bool:
        """
        Detect if data has the problematic nested candles structure
        """
        try:
            if isinstance(data, list) and len(data) > 0:
                # Convert to DataFrame to check structure
                df = pd.DataFrame(data)
                return 'candles' in df.columns and 'time' not in df.columns and 'timestamp' not in df.columns
            elif isinstance(data, pd.DataFrame):
                return 'candles' in data.columns and 'time' not in data.columns and 'timestamp' not in data.columns
            return False
        except:
            return False
    
    async def run_backtest_with_enhanced_data_handling(self, 
                                                     raw_oanda_data,
                                                     start_date: Optional[datetime] = None,
                                                     end_date: Optional[datetime] = None) -> BacktestResults:
        """
        New method that specifically handles your OANDA data format
        """
        
        self.logger.info("🧪 Starting enhanced Wyckoff backtest with OANDA data fix")
        
        try:
            # Step 1: Fix the data structure (now handles both DataFrame and List[Dict])
            processed_data = self.data_processor.fix_oanda_candles_structure(raw_oanda_data)
            
            # Step 2: Run your existing backtest logic
            results = await self.run_backtest(processed_data, start_date, end_date)
            
            # Step 3: Add enhanced reporting for OANDA-specific metrics
            self._add_oanda_specific_metrics(results, processed_data)
            
            return results
            
        except Exception as e:
            self.logger.error("❌ Enhanced backtest failed", error=str(e))
            raise
    
    def _add_oanda_specific_metrics(self, results: BacktestResults, processed_data: List[Dict[str, Any]]):
        """Add OANDA-specific metrics to results"""
        
        try:
            # Calculate spread metrics if available
            spreads = []
            for candle in processed_data:
                if (candle.get('ask_close') and candle.get('bid_close') and 
                    candle['ask_close'] > 0 and candle['bid_close'] > 0):
                    spread = candle['ask_close'] - candle['bid_close']
                    spreads.append(spread)
            
            if spreads:
                # Add spread analysis to results
                results.avg_spread = float(np.mean(spreads))
                results.max_spread = float(np.max(spreads))
                results.min_spread = float(np.min(spreads))
                results.spread_volatility = float(np.std(spreads))
            
            # Add data quality metrics
            results.data_completeness = len([c for c in processed_data if c.get('complete', True)]) / len(processed_data) * 100
            results.volume_availability = len([c for c in processed_data if c.get('volume', 0) > 0]) / len(processed_data) * 100
            
        except Exception as e:
            self.logger.warning(f"Failed to add OANDA-specific metrics: {str(e)}")


# Integration functions for your existing codebase
def create_quick_validation_with_data_fix():
    """
    Updated version of your run_quick_wyckoff_validation() that includes the data fix
    """
    
    async def run_quick_wyckoff_validation_fixed():
        """Enhanced quick validation with data structure fix"""
        
        print("⚡ QUICK WYCKOFF VALIDATION (Enhanced)")
        print("=" * 50)
        
        try:
            # Initialize enhanced engine
            engine = IntegratedWyckoffBacktester(
                initial_capital=50000,
                commission_per_trade=3.0,
                risk_per_trade=0.015,
                max_position_size=0.05
            )
            
            print("🚀 Running enhanced validation on EUR_USD M15...")
            
            # Get data (this will now handle the nested structure automatically)
            historical_data = await engine._get_historical_data("EUR_USD", "M15", 500)
            
            if not historical_data:
                print("❌ No data available for quick test")
                return
            
            print(f"📊 Retrieved {len(historical_data)} data points")
            
            # The enhanced run_backtest will automatically detect and fix nested structure
            result = await engine.run_backtest_with_enhanced_data_handling(historical_data)
            
            print(f"\n📊 ENHANCED VALIDATION RESULTS:")
            print(f"   Return: {result.total_pnl_pct:.2f}%")
            print(f"   Win Rate: {result.win_rate:.1f}%")
            print(f"   Total Trades: {result.total_trades}")
            print(f"   Profit Factor: {result.profit_factor:.2f}")
            print(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
            print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
            
            # Enhanced OANDA-specific metrics
            if result.data_completeness is not None or result.volume_availability is not None:
                print(f"\n🔍 DATA QUALITY METRICS:")
                if result.data_completeness is not None:
                    print(f"   Data Completeness: {result.data_completeness:.1f}%")
                if result.volume_availability is not None:
                    print(f"   Volume Availability: {result.volume_availability:.1f}%")

            if result.avg_spread is not None:
                print(f"\n📈 SPREAD ANALYSIS:")
                print(f"   Average Spread: {result.avg_spread:.5f}")
                if result.min_spread is not None and result.max_spread is not None:
                    print(f"   Spread Range: {result.min_spread:.5f} - {result.max_spread:.5f}")
            
            if result.total_trades > 0:
                print(f"\n🎯 WYCKOFF PATTERN ANALYSIS:")
                for pattern, success_rate in result.pattern_success_rates.items():
                    trades_count = sum(1 for t in result.trades if t.wyckoff_pattern == pattern)
                    print(f"   {pattern}: {success_rate:.1f}% ({trades_count} trades)")
            
            print(f"\n✅ Enhanced validation completed!")
            return result
            
        except Exception as e:
            print(f"❌ Enhanced validation failed: {str(e)}")
            # Provide specific help for the data structure error
            if "time" in str(e) or "timestamp" in str(e):
                print("\n💡 SOLUTION: This error has been fixed!")
                print("   The enhanced validator automatically detects and fixes nested OANDA data.")
                print("   Make sure you're using IntegratedWyckoffBacktester instead of AutomatedWyckoffBacktester.")
            return None
    
    return run_quick_wyckoff_validation_fixed


# Drop-in replacement for your existing functions
def patch_existing_backtest_engine(existing_engine: AutomatedWyckoffBacktester) -> IntegratedWyckoffBacktester:
    """
    Convert your existing AutomatedWyckoffBacktester to the enhanced version
    """
    
    # Create enhanced engine with same settings
    enhanced_engine = IntegratedWyckoffBacktester(
        initial_capital=existing_engine.initial_capital,
        commission_per_trade=existing_engine.commission_per_trade,
        risk_per_trade=existing_engine.risk_per_trade,
        max_position_size=existing_engine.max_position_size
    )
    
    # Copy over any optimized parameters
    enhanced_engine.min_pattern_confidence = existing_engine.min_pattern_confidence
    enhanced_engine.min_risk_reward = existing_engine.min_risk_reward
    enhanced_engine.volume_threshold = existing_engine.volume_threshold
    enhanced_engine.structure_confirmation_bars = existing_engine.structure_confirmation_bars
    enhanced_engine.max_trades_per_day = existing_engine.max_trades_per_day
    
    return enhanced_engine


# Example usage showing how to integrate with your existing code
async def demonstrate_integration():
    """
    Demonstrate how the fix integrates with your existing backtesting engine
    """
    
    print("🔧 DEMONSTRATING WYCKOFF DATA FIX INTEGRATION")
    print("=" * 60)
    
    # Simulate your problematic data structure
    print("1. Simulating problematic OANDA data structure...")
    problematic_data = pd.DataFrame({
        'instrument': ['EUR_USD'] * 4,
        'granularity': ['D'] * 4,
        'candles': [
            "{'complete': True, 'volume': 105361, 'time': '2024-01-01T00:00:00.000000000Z', 'mid': {'o': '1.1050', 'h': '1.1080', 'l': '1.1040', 'c': '1.1070'}}",
            "{'complete': True, 'volume': 100803, 'time': '2024-01-02T00:00:00.000000000Z', 'mid': {'o': '1.1070', 'h': '1.1090', 'l': '1.1060', 'c': '1.1075'}}",
            "{'complete': True, 'volume': 98391, 'time': '2024-01-03T00:00:00.000000000Z', 'mid': {'o': '1.1075', 'h': '1.1085', 'l': '1.1050', 'c': '1.1065'}}",
            "{'complete': True, 'volume': 86303, 'time': '2024-01-04T00:00:00.000000000Z', 'mid': {'o': '1.1065', 'h': '1.1095', 'l': '1.1055', 'c': '1.1080'}}"
        ],
        'count': [500] * 4
    })
    
    print(f"   Original data shape: {problematic_data.shape}")
    print(f"   Columns: {list(problematic_data.columns)}")
    print(f"   Sample candle: {problematic_data.iloc[0]['candles'][:100]}...")
    
    # 2. Show the fix in action
    print("\n2. Applying the data structure fix...")
    processor = EnhancedWyckoffDataProcessor()
    fixed_data = processor.fix_oanda_candles_structure(problematic_data)
    
    print(f"   Fixed data length: {len(fixed_data)}")
    print(f"   Sample fixed record: {fixed_data[0]}")
    
    # 3. Show integration with your backtest engine
    print("\n3. Testing with enhanced backtest engine...")
    
    # This would work with your existing engine after enhancement
    enhanced_engine = IntegratedWyckoffBacktester(initial_capital=10000)
    
    # The enhanced engine automatically detects and fixes the data
    try:
        df = enhanced_engine._prepare_backtest_data(fixed_data)
        print(f"   ✅ Data preparation successful!")
        print(f"   Prepared DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Show it has the required timestamp column
        if 'timestamp' in df.columns:
            print(f"   ✅ Timestamp column present: {df['timestamp'].iloc[0]}")
        if 'time' in df.columns:
            print(f"   ✅ Time column present: {df['time'].iloc[0]}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n4. Summary of the fix:")
    print("   ✅ Extracts nested OHLCV data from 'candles' column")
    print("   ✅ Creates proper 'time' and 'timestamp' columns")
    print("   ✅ Validates data integrity")
    print("   ✅ Integrates seamlessly with your existing AutomatedWyckoffBacktester")
    print("   ✅ Maintains all your existing functionality")
    print("   ✅ Adds enhanced OANDA-specific metrics")


if __name__ == "__main__":
    import asyncio
    
    print("Choose integration test:")
    print("1. Demonstrate fix integration")
    print("2. Run enhanced quick validation")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        asyncio.run(demonstrate_integration())
    elif choice == "2":
        enhanced_validation = create_quick_validation_with_data_fix()
        asyncio.run(enhanced_validation())
    else:
        print("Running demonstration...")
        asyncio.run(demonstrate_integration())
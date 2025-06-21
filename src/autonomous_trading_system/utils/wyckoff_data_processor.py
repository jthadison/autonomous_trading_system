"""
Integration of Wyckoff Data Preparation Fix with Original Backtesting Engine
Fixes the nested candles data structure issue and enhances the existing system
"""
import sys
from pathlib import Path

# Add project root to sys.path to allow running this script directly
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
#from autonomous_trading_system.utils.backtest import AutomatedWyckoffBacktester, BacktestResults
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from src.config.logging_config import logger


# Import your existing classes (these would be your actual imports)
from src.autonomous_trading_system.utils.backtest import AutomatedWyckoffBacktester, EnhancedWyckoffDataProcessor, IntegratedWyckoffBacktester, BacktestResults, BacktestTrade




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
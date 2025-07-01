# Save this as test_all_fixes.py and run it to verify everything works

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

async def test_all_fixes():
    """Test that all attribute and import errors are fixed"""
    
    print("🧪 TESTING ALL FIXES")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from src.backtesting.enhanced_agent_backtest import EnhancedAgentBacktester, BacktestTrade
        from src.config.logging_config import logger
        print("✅ Basic imports successful")
        
        # Test backtester creation
        print("🏗️ Creating backtester...")
        backtester = EnhancedAgentBacktester(initial_capital=50000)
        print("✅ Backtester created successfully")
        
        # Add sample data to test with
        print("📊 Adding sample data...")
        sample_trades = []
        for i in range(3):
            trade = BacktestTrade(
                id=f"test_{i}",
                timestamp=f"2024-01-0{i+1} 10:00:00",
                symbol="EUR_USD",
                action="buy" if i % 2 == 0 else "sell",
                entry_price=1.1000 + (i * 0.0010),
                quantity=10000,
                stop_loss=1.0950 + (i * 0.0010),
                take_profit=1.1050 + (i * 0.0010),
                confidence=70 + (i * 5),
                wyckoff_phase="accumulation" if i % 2 == 0 else "distribution",
                pattern_type="spring" if i % 2 == 0 else "upthrust",
                reasoning=f"Test trade {i}",
                agent_name="test_agent",
                exit_price=1.1025 + (i * 0.0010),
                exit_timestamp=f"2024-01-0{i+1} 12:00:00",
                exit_reason="take_profit",
                pnl=25.0 + (i * 10) if i % 2 == 0 else -(10 + i * 5),
                pnl_pct=0.25 + (i * 0.1) if i % 2 == 0 else -(0.1 + i * 0.05),
                duration_bars=12 + i,
                is_closed=True
            )
            sample_trades.append(trade)
        
        # Set up backtester
        backtester.trades = sample_trades
        backtester.current_historical_data = [
            {'open': 1.1000, 'high': 1.1050, 'low': 1.0950, 'close': 1.1025, 'volume': 1000}
            for _ in range(10)
        ]
        backtester.equity_curve = [50000 + i * 10 for i in range(10)]
        backtester.pattern_counts = {"spring": 2, "upthrust": 1}
        backtester.phase_counts = {"accumulation": 2, "distribution": 1}
        
        print("✅ Sample data added")
        
        # Test 1: Enhanced metrics availability check
        print("🔍 Testing enhanced metrics availability...")
        from src.backtesting.enhanced_agent_backtest import check_enhanced_metrics_availability
        enhanced_available = check_enhanced_metrics_availability()
        print(f"   Enhanced metrics available: {enhanced_available}")
        
        # Test 2: Metrics calculation (this was causing the attribute error)
        print("📊 Testing metrics calculation...")
        start_time = datetime.now()
        
        try:
            results = backtester._calculate_comprehensive_metrics("EUR_USD", start_time)
            print("✅ Metrics calculation successful!")
            print(f"   Type: {type(results).__name__}")
            print(f"   Has sortino_ratio: {hasattr(results, 'sortino_ratio')}")
            print(f"   Total Return: {getattr(results, 'total_return_pct', 'N/A'):.2f}%")
            print(f"   Win Rate: {getattr(results, 'win_rate', 'N/A'):.1f}%")
            
        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            return False
        
        # Test 3: Report generation (this was causing the sortino_ratio error)
        print("📝 Testing report generation...")
        
        try:
            report_path = await backtester._generate_enhanced_report(results, "EUR_USD")
            
            if report_path and Path(report_path).exists():
                print(f"✅ Report generated successfully!")
                print(f"   Location: {report_path}")
                
                # Check report content
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Verify no attribute errors in report
                if "sortino_ratio" in content or "Enhanced" in content:
                    print("✅ Report contains enhanced metrics info")
                elif "Basic" in content:
                    print("✅ Report correctly shows basic metrics mode")
                else:
                    print("⚠️ Report content unclear about metrics level")
                
                # Show preview
                lines = content.split('\n')[:8]
                print("📖 Report preview:")
                for line in lines:
                    print(f"   {line}")
                
            else:
                print("❌ Report generation failed - no file created")
                return False
                
        except Exception as e:
            print(f"❌ Report generation error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 4: Full backtest simulation
        print("🚀 Testing full backtest method...")
        
        try:
            # This should work without errors now
            result_dict = await backtester.run_agent_backtest(
                historical_data=backtester.current_historical_data,
                initial_balance=50000,
                symbol="EUR_USD"
            )
            
            if result_dict.get('success'):
                print("✅ Full backtest completed successfully!")
                print(f"   Enhanced available: {result_dict.get('enhanced_metrics_available', 'Unknown')}")
                print(f"   Report path: {result_dict.get('report_path', 'None')}")
            else:
                print(f"❌ Full backtest failed: {result_dict.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Full backtest error: {e}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Attribute errors fixed")
        print("✅ Import errors handled")
        print("✅ Report generation works")
        print("✅ Full backtest works")
        
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    print("🚀 Enhanced Metrics Integration - Final Test")
    print("=" * 60)
    
    success = await test_all_fixes()
    
    if success:
        print(f"\n🎉 SUCCESS! All fixes are working!")
        print(f"✅ Your enhanced metrics integration is ready!")
        print(f"\n📋 WHAT'S WORKING:")
        print(f"   • No more attribute errors")
        print(f"   • Robust import handling")
        print(f"   • Safe report generation")
        print(f"   • Enhanced metrics when available")
        print(f"   • Graceful fallback to basic metrics")
        print(f"\n🚀 You can now run your normal backtests!")
        
    else:
        print(f"\n⚠️ SOME ISSUES REMAIN")
        print(f"📋 NEXT STEPS:")
        print(f"   1. Check the error messages above")
        print(f"   2. Ensure all code changes are applied")
        print(f"   3. Verify file locations and imports")
        print(f"   4. Test step by step if needed")

if __name__ == "__main__":
    asyncio.run(main())
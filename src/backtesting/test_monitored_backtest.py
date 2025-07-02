# test_real_data_backtest.py
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.backtesting.real_data_monitored_backtest import RealDataMonitoredBacktester

async def test_real_data():
    """Test the FIXED real data integration"""
    
    print("🧪 Testing FIXED Real Data Integration...")
    
    from src.backtesting.real_data_monitored_backtest import RealDataMonitoredBacktester
    
    backtester = RealDataMonitoredBacktester()
    
    # Test with smaller dataset first
    result = await backtester.run_real_data_backtest(
        symbol="EUR_USD",
        timeframe="M15",
        bars=100,  # Smaller test first
        initial_balance=50000,  # Smaller balance for testing
    )
    
    if result['success']:
        print("✅ FIXED version working!")
        
        # Show improved metrics
        print(f"💰 Total Return: {result.get('total_return_pct', 0):+.2f}%")
        print(f"📊 Win Rate: {result.get('win_rate', 0):.1f}%")
        print(f"🎯 Total Trades: {result.get('total_trades', 0)}")
        
        # Check data quality
        data_info = result.get('data_info', {})
        if data_info:
            quality = data_info.get('data_quality', {})
            print(f"📊 Data Quality: {quality.get('quality_score', 0)}/100")
            print(f"📅 Bars Used: {data_info.get('bars_received', 0)}/{data_info.get('bars_requested', 0)}")
            
            issues = quality.get('issues', [])
            if issues:
                print(f"⚠️ Data Issues: {issues}")
        
        # Show FIXED agent performance
        monitoring = result.get('monitoring_insights', {})
        agent_perf = monitoring.get('agent_performance', {})
        
        print(f"\n🤖 FIXED Agent Performance:")
        for agent, metrics in agent_perf.items():
            accuracy = metrics.get('accuracy', 0)
            decisions = metrics.get('total_decisions', 0)
            print(f"   {agent}: {accuracy:.1f}% accuracy ({decisions} decisions)")
        
        # Show recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        return result
    else:
        print(f"❌ Still has issues: {result.get('error')}")
        return None

if __name__ == "__main__":
    asyncio.run(test_real_data())
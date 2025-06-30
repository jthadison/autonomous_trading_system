"""
Test script for the new Backtesting Orchestrator Agent
Run this to validate the agent setup and basic functionality
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

from config.logging_config import logger
from src.autonomous_trading_system.crew import AutonomousTradingSystem

# Import the new backtesting tools for testing
from src.autonomous_trading_system.tools.backtesting_simulation_tools import (
    simulate_historical_market_context,
    simulate_trade_execution,
    update_backtest_portfolio,
    calculate_backtest_performance_metrics
)

def generate_sample_historical_data(num_bars: int = 50) -> list:
    """Generate sample historical data for testing"""
    
    base_price = 1.1000
    current_time = datetime.now() - timedelta(hours=num_bars)
    
    bars = []
    for i in range(num_bars):
        # Simulate price movement
        price_change = (i % 10 - 5) * 0.0001  # Small random-like movements
        current_price = base_price + price_change
        
        bar = {
            'timestamp': (current_time + timedelta(hours=i)).isoformat(),
            'open': current_price - 0.0005,
            'high': current_price + 0.0010,
            'low': current_price - 0.0015,
            'close': current_price,
            'volume': 1000 + (i % 500)
        }
        bars.append(bar)
    
    return bars

async def test_backtesting_tools():
    """Test individual backtesting tools"""
    
    logger.info("🧪 Testing Backtesting Simulation Tools...")
    
    # Generate test data
    sample_bars = generate_sample_historical_data(20)
    logger.info(f"✅ Generated {len(sample_bars)} sample bars")
    
    # Test 1: Market Context Simulation
    logger.info("🔄 Testing market context simulation...")
    
    market_context = simulate_historical_market_context.func(
        historical_bars=json.dumps(sample_bars),
        current_bar_index=10,
        account_info=json.dumps({
            'balance': 100000,
            'equity': 100000,
            'margin_available': 50000
        })
    )
    
    context_data = json.loads(market_context)
    if 'error' in context_data:
        logger.error(f"❌ Market context simulation failed: {context_data['error']}")
        return False
    
    logger.info(f"✅ Market context created - Price: {context_data['current_price']:.5f}")
    logger.info(f"   Spread: {context_data['spread']:.5f}, Session: {context_data['market_hours']}")
    
    # Test 2: Trade Execution Simulation
    logger.info("🔄 Testing trade execution simulation...")
    
    execution_result = simulate_trade_execution.func(
        trade_decision=json.dumps({
            'side': 'buy',
            'quantity': 10000,
            'order_type': 'market',
            'symbol': 'EUR_USD',
            'price': context_data['current_price']
        }),
        market_context=market_context
    )
    
    execution_data = json.loads(execution_result)
    if 'error' in execution_data:
        logger.error(f"❌ Trade execution simulation failed: {execution_data['error']}")
        return False
    
    logger.info(f"✅ Trade executed - Order ID: {execution_data['order_id']}")
    logger.info(f"   Requested: {execution_data['requested_price']:.5f}, Executed: {execution_data['executed_price']:.5f}")
    logger.info(f"   Slippage: {execution_data['slippage']:.5f}, Commission: ${execution_data['commission']:.2f}")
    
    # Test 3: Portfolio Update
    logger.info("🔄 Testing portfolio update...")
    
    initial_portfolio = {
        'initial_balance': 100000,
        'current_balance': 100000,
        'equity': 100000,
        'unrealized_pnl': 0,
        'realized_pnl': 0,
        'open_positions': [],
        'closed_trades': [],
        'margin_used': 0,
        'free_margin': 100000
    }
    
    updated_portfolio = update_backtest_portfolio.func(
        portfolio_state=json.dumps(initial_portfolio),
        execution_result=execution_result,
        current_prices=json.dumps({'EUR_USD': context_data['current_price']})
    )
    
    portfolio_data = json.loads(updated_portfolio)
    if 'error' in portfolio_data:
        logger.error(f"❌ Portfolio update failed: {portfolio_data['error']}")
        return False
    
    logger.info(f"✅ Portfolio updated - Balance: ${portfolio_data['current_balance']:.2f}")
    logger.info(f"   Open positions: {len(portfolio_data['open_positions'])}")
    logger.info(f"   Equity: ${portfolio_data['equity']:.2f}")
    
    logger.info("🎉 All backtesting tools tests passed!")
    return True

async def test_backtesting_agent():
    """Test the backtesting orchestrator agent"""
    
    logger.info("🤖 Testing Backtesting Orchestrator Agent...")
    
    try:
        # Initialize the trading system
        trading_system = AutonomousTradingSystem()
        
        # Create the backtesting orchestrator agent
        logger.info("🔄 Creating backtesting orchestrator agent...")
        orchestrator = trading_system.backtesting_orchestrator()
        
        logger.info(f"✅ Backtesting orchestrator created: {orchestrator.role}")
        #logger.info(f"   Tools available: {len(orchestrator.tools)}")
        
        # Test the backtesting crew creation
        logger.info("🔄 Creating backtesting crew...")
        backtest_crew = trading_system.backtesting_crew()
        
        logger.info(f"✅ Backtesting crew created with {len(backtest_crew.agents)} agents:")
        for agent in backtest_crew.agents:
            logger.info(f"   - {agent.role}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtesting agent test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    logger.info("🚀 BACKTESTING ORCHESTRATOR AGENT - TEST SUITE")
    logger.info("=" * 60)
    
    # Test 1: Individual tools
    tools_success = await test_backtesting_tools()
    
    if not tools_success:
        logger.error("❌ Tools testing failed. Please check the implementation.")
        return
    
    logger.info("\n" + "=" * 60)
    
    # Test 2: Agent creation and integration
    agent_success = await test_backtesting_agent()
    
    if not agent_success:
        logger.error("❌ Agent testing failed. Please check the configuration.")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("✅ Backtesting Orchestrator Agent is ready for use")
    
    logger.info("\n🔗 Next Steps:")
    logger.info("   1. Run a full backtest with historical data")
    logger.info("   2. Compare results with current backtesting approach")
    logger.info("   3. Fine-tune simulation parameters")
    logger.info("   4. Integrate with your existing workflow")

if __name__ == "__main__":
    
    
    
    asyncio.run(main())
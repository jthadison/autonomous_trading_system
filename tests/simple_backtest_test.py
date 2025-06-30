"""
Simple Single-Bar Backtest Test
Test the orchestrator with just one bar to verify the workflow
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.logging_config import logger
from src.autonomous_trading_system.crew import AutonomousTradingSystem

# Import simulation tools directly for testing
from src.backtesting.backtesting_simulation_tools import (
    simulate_historical_market_context,
    simulate_trade_execution,
    update_backtest_portfolio
)

def test_single_bar_workflow():
    """Test processing a single bar through the entire workflow"""
    
    logger.info("🧪 SINGLE BAR WORKFLOW TEST")
    logger.info("=" * 40)
    
    # Create single sample bar
    sample_bar = {
        'timestamp': datetime.now().isoformat(),
        'open': 1.0950,
        'high': 1.0965,
        'low': 1.0945,
        'close': 1.0960,
        'volume': 1000
    }
    
    # Create sample account
    sample_account = {
        'balance': 10000,
        'equity': 10000,
        'margin_available': 5000
    }
    
    logger.info(f"📊 Test bar: {sample_bar['close']:.5f}")
    
    try:
        # Step 1: Test market context simulation
        logger.info("🔄 Step 1: Creating market context...")
        
        # Call the tool's underlying function directly
        market_context = simulate_historical_market_context._run(
            historical_bars=json.dumps([sample_bar]),
            current_bar_index=0,
            account_info=json.dumps(sample_account)
        )
        
        context_data = json.loads(market_context)
        if 'error' in context_data:
            logger.error(f"❌ Market context failed: {context_data['error']}")
            return False
        
        logger.info(f"✅ Market context: Price {context_data['current_price']:.5f}, Spread {context_data['spread']:.5f}")
        
        # Step 2: Test trade decision simulation
        logger.info("🔄 Step 2: Simulating trade decision...")
        
        # Simulate a simple buy decision
        sample_decision = {
            'side': 'buy',
            'quantity': 1000,
            'order_type': 'market',
            'symbol': 'EUR_USD',
            'price': context_data['current_price']
        }
        
        # Step 3: Test trade execution
        logger.info("🔄 Step 3: Simulating trade execution...")
        
        execution_result = simulate_trade_execution._run(
            trade_decision=json.dumps(sample_decision),
            market_context=market_context
        )
        
        execution_data = json.loads(execution_result)
        if 'error' in execution_data:
            logger.error(f"❌ Trade execution failed: {execution_data['error']}")
            return False
        
        logger.info(f"✅ Trade executed: {execution_data['executed_price']:.5f}")
        logger.info(f"   Order ID: {execution_data['order_id']}")
        logger.info(f"   Slippage: {execution_data['slippage']:.5f}")
        
        # Step 4: Test portfolio update
        logger.info("🔄 Step 4: Updating portfolio...")
        
        initial_portfolio = {
            'initial_balance': 10000,
            'current_balance': 10000,
            'equity': 10000,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'open_positions': [],
            'closed_trades': [],
            'margin_used': 0,
            'free_margin': 10000
        }
        
        updated_portfolio = update_backtest_portfolio._run(
            portfolio_state=json.dumps(initial_portfolio),
            execution_result=execution_result,
            current_prices=json.dumps({'EUR_USD': context_data['current_price']})
        )
        
        portfolio_data = json.loads(updated_portfolio)
        if 'error' in portfolio_data:
            logger.error(f"❌ Portfolio update failed: {portfolio_data['error']}")
            return False
        
        logger.info(f"✅ Portfolio updated:")
        logger.info(f"   Balance: ${portfolio_data['current_balance']:.2f}")
        logger.info(f"   Equity: ${portfolio_data['equity']:.2f}")
        logger.info(f"   Open positions: {len(portfolio_data['open_positions'])}")
        
        logger.info("🎉 Single bar workflow completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Single bar workflow failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_agent_creation():
    """Test that we can create the trading system and agents"""
    
    logger.info("🤖 AGENT CREATION TEST")
    logger.info("=" * 40)
    
    try:
        # Create trading system
        logger.info("🔄 Creating trading system...")
        trading_system = AutonomousTradingSystem()
        
        # Test agent creation
        logger.info("🔄 Creating orchestrator agent...")
        orchestrator = trading_system.backtesting_orchestrator()
        logger.info(f"✅ Orchestrator: {orchestrator.role}")
        
        # Test crew creation
        logger.info("🔄 Creating backtesting crew...")
        crew = trading_system.backtesting_crew()
        logger.info(f"✅ Crew: {len(crew.agents)} agents, {len(crew.tasks)} tasks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent creation failed: {e}")
        return False

def main():
    """Main test function"""
    
    logger.info("🎯 SIMPLE BACKTESTING TESTS")
    logger.info("=" * 50)
    
    # Test 1: Agent creation
    logger.info("\n1️⃣ Testing agent creation...")
    agent_success = test_agent_creation()
    
    if not agent_success:
        logger.error("❌ Agent creation failed. Fix before proceeding.")
        return
    
    logger.info("\n" + "=" * 50)
    
    # Test 2: Single bar workflow
    logger.info("\n2️⃣ Testing single bar workflow...")
    workflow_success = test_single_bar_workflow()
    
    logger.info("\n" + "=" * 50)
    
    if agent_success and workflow_success:
        logger.info("🎉 ALL SIMPLE TESTS PASSED!")
        logger.info("✅ Ready to run full backtests!")
        logger.info("\n🔗 Next: Run 'python real_backtest_runner.py'")
    else:
        logger.error("❌ Some tests failed. Check logs above.")

if __name__ == "__main__":
    main()
"""
Updated Autonomous Trading System Crew with Execution Capabilities
Integrates the new trading execution tools with existing Wyckoff analysis agents
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any

from crewai.utilities.events.third_party.agentops_listener import agentops
from mcp_servers.oanda_direct_api import OandaDirectAPI

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

#from autonomous_trading_system.trading_dashboard import get_historical_data
from src.autonomous_trading_system.utils.wyckoff_pattern_analyzer import wyckoff_analyzer
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from src.config.logging_config import logger
#from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
from src.database.manager import db_manager
from src.database.models import AgentAction, LogLevel
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from pydantic import PydanticDeprecatedSince20
from langchain_anthropic import ChatAnthropic
import warnings
import os

from src.backtesting.backtesting_simulation_tools import (
    simulate_historical_market_context,
    simulate_trade_execution,
    update_backtest_portfolio,
    calculate_backtest_performance_metrics
)

from dotenv import load_dotenv
load_dotenv(override=True)

# Import the new trading execution tools
from src.autonomous_trading_system.tools.trading_execution_tools_sync import (
    execute_market_trade,
    execute_limit_trade,
    cancel_pending_order,
    get_open_positions,
    get_pending_orders,
    close_position,
    get_portfolio_status
)

warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

# Existing analysis and data tools
@tool
def get_live_price(instrument: str) -> Dict[str, Any]:
    """Get live price for a forex instrument using Direct API"""
    async def _get_price():
        # UPDATED: Use Direct API instead of MCP wrapper
        async with OandaDirectAPI() as oanda:
            return await oanda.get_current_price(instrument)
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _get_price())
                return future.result()
        else:
            return asyncio.run(_get_price())
    except Exception as e:
        logger.error(f"Failed to get live price for {instrument}: {str(e)}")
        return {"error": str(e)}

@tool  
def get_historical_data(instrument: str, timeframe: str = "M15", count: int = 200) -> Dict[str, Any]:
    """Get historical price data using Direct API"""
    async def _get_historical():
        # UPDATED: Use Direct API instead of MCP wrapper
        async with OandaDirectAPI() as oanda:
            return await oanda.get_historical_data(instrument, timeframe, count)
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _get_historical())
                return future.result()
        else:
            return asyncio.run(_get_historical())
    except Exception as e:
        logger.error(f"Failed to get historical data for {instrument}: {str(e)}")
        return {"error": str(e)}

@tool
def get_account_info() -> Dict[str, Any]:
    """Get current account information using Direct API"""
    async def _get_account():
        # UPDATED: Use Direct API instead of MCP wrapper
        async with OandaDirectAPI() as oanda:
            return await oanda.get_account_info()
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _get_account())
                return future.result()
        else:
            return asyncio.run(_get_account())
    except Exception as e:
        logger.error("Failed to get account info", error=str(e))
        return {"error": str(e)}

@tool
def calculate_position_size(account_balance: float, risk_per_trade: float, stop_distance_pips: float, pip_value: float = 1.0) -> Dict[str, Any]:
    """Calculate position size based on Wyckoff levels and risk management"""
    try:
        # Risk amount in dollars
        risk_amount = account_balance * risk_per_trade
        
        # Position size calculation
        if stop_distance_pips > 0 and pip_value > 0:
            position_size = risk_amount / (stop_distance_pips * pip_value)
        else:
            position_size = 0
        
        return {
            "account_balance": account_balance,
            "risk_per_trade_pct": risk_per_trade * 100,
            "risk_amount": round(risk_amount, 2),
            "stop_distance_pips": stop_distance_pips,
            "position_size": round(position_size, 0),
            "pip_value": pip_value,
            "max_position_size": round(account_balance * 0.1, 0)  # Never risk more than 10% of account
        }
        
    except Exception as e:
        return {"error": str(e)}

async def log_wyckoff_analysis(analysis_data: Dict[str, Any]):
    """Log Wyckoff analysis to database"""
    try:
        async with db_manager.get_async_session() as session:
            action = AgentAction(
                agent_name="WyckoffAnalyzer",
                action_type="WYCKOFF_ANALYSIS",
                input_data={"timestamp": datetime.now().isoformat()},
                output_data=analysis_data,
                confidence_score=analysis_data.get("confidence_score", 0)
            )
            session.add(action)
            await session.commit()
    except Exception as e:
        logger.error("Failed to log Wyckoff analysis", error=str(e))

@CrewBase
class AutonomousTradingSystem():
    """AutonomousTradingSystem crew with full execution capabilities"""
    agentops.init(os.getenv("AGENTOP_API_KEY"), skip_auto_end_session=True)

    agents: List[BaseAgent]
    tasks: List[Task]
    
    agents_config = 'config/agents.yaml' 
    tasks_config = 'config/tasks.yaml' 
    
    # Global Oanda wrapper instance
    oanda_wrapper = None

    from src.autonomous_trading_system.utils.wyckoff_pattern_analyzer import wyckoff_analyzer
    
    @agent
    def wyckoff_market_analyst(self) -> Agent:
        """Agent responsible for Wyckoff market structure analysis"""
        return Agent(
            config=self.agents_config['wyckoff_market_analyst'], # type: ignore[index]
            tools=[
                get_live_price,
                get_historical_data
            ],
            verbose=True,
            max_iter=3
        )
        
    @agent
    def wyckoff_risk_manager(self) -> Agent:
        """Agent responsible for risk management and position sizing"""
        return Agent(
            config=self.agents_config['wyckoff_risk_manager'], # type: ignore[index]
            tools=[
                get_account_info, 
                calculate_position_size,
                get_portfolio_status,
                get_open_positions,
                get_pending_orders
            ],
            verbose=True,
            max_iter=3
        )
        
    @agent
    def wyckoff_trading_coordinator(self) -> Agent:
        """Agent responsible for trading decisions and execution"""
        return Agent(
            config=self.agents_config['wyckoff_trading_coordinator'], # type: ignore[index]
            tools=[
                # Market data tools
                get_live_price, 
                get_account_info,
                get_portfolio_status,
                
                # Position management tools
                get_open_positions,
                get_pending_orders,
                close_position,
                
                # Order execution tools - THE KEY ADDITION!
                execute_market_trade,
                execute_limit_trade,
                cancel_pending_order
            ],
            verbose=True,
            max_iter=3
        )
        
    @agent
    def backtesting_orchestrator(self) -> Agent:
        """Agent responsible for coordinating backtests and simulating market conditions"""
        return Agent(
            config=self.agents_config['backtesting_orchestrator'], # type: ignore[index]
            tools=[
                # Historical market simulation tools
                simulate_historical_market_context,
                simulate_trade_execution,
                update_backtest_portfolio,
                calculate_backtest_performance_metrics,
                
                # Access to existing tools for coordination
                get_historical_data,  # To load historical data
            ],
            verbose=True,
            max_iter=5
        )

    @task
    def wyckoff_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['wyckoff_analysis_task'], # type: ignore[index]
        )

    @task
    def wyckoff_risk_task(self) -> Task:
        return Task(
            config=self.tasks_config['wyckoff_risk_task'], # type: ignore[index]
        )
        
    @task
    def wyckoff_decision_task(self) -> Task:
        return Task(
            config=self.tasks_config['wyckoff_decision_task'], # type: ignore[index]
        )
        
    @task
    def backtesting_coordination_task(self) -> Task:
        return Task(
            config=self.tasks_config['backtesting_coordination_task'], # type: ignore[index]
        )

    @crew
    def backtesting_crew(self) -> Crew:
        """Creates a specialized backtesting crew that uses existing trading agents"""
        return Crew(
            agents=[
                # Backtesting coordinator (call the method to get Agent instance)
                self.backtesting_orchestrator(),
                
                # Existing trading agents for decision making (call methods)
                self.wyckoff_market_analyst(),
                self.wyckoff_risk_manager(), 
                self.wyckoff_trading_coordinator()
            ],
            tasks=[
                # Main backtesting task (call the method to get Task instance)
                self.backtesting_coordination_task(),
                
                # You can also include existing tasks if needed
                # self.wyckoff_analysis_task(),
                # self.wyckoff_risk_task(),
                # self.wyckoff_decision_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            step_callback=lambda step: time.sleep(3)
        )
        
    @crew
    def crew(self) -> Crew:
        """Creates the AutonomousTradingSystem crew with full execution capabilities"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            step_callback=lambda step: time.sleep(3)
        )
        
    

    # Optional: Add execution monitoring methods
    async def monitor_executions(self, check_interval_seconds: int = 30):
        """Monitor and log all trade executions"""
        while True:
            try:
                portfolio_status = get_portfolio_status()
                
                if portfolio_status.get("total_positions", 0) > 0:
                    logger.info(
                        "Portfolio monitoring",
                        total_positions=portfolio_status.get("total_positions"),
                        total_pending_orders=portfolio_status.get("total_pending_orders"),
                        total_exposure=portfolio_status.get("total_exposure", 0),
                        exposure_pct=portfolio_status.get("exposure_pct", 0)
                    )
                
                await asyncio.sleep(check_interval_seconds)
                
            except Exception as e:
                logger.error("Portfolio monitoring error", error=str(e))
                await asyncio.sleep(check_interval_seconds)

    def execute_trading_session(self, symbol_name: str = "EUR_USD"):
        """Execute a complete trading session with monitoring"""
        try:
            # Set up inputs for the crew
            inputs = {
                'topic': 'Wyckoff Trading Analysis',
                'symbol': symbol_name,
                'symbol_name': symbol_name,
                'current_year': str(datetime.now().year),
                'timestamp': str(datetime.now().timestamp()).replace('.','')
            }
            
            logger.info(
                "Starting autonomous trading session",
                symbol=symbol_name,
                timestamp=datetime.now().isoformat()
            )
            
            # Execute the crew workflow
            result = self.crew().kickoff(inputs=inputs)
            
            logger.info(
                "Trading session completed",
                symbol=symbol_name,
                result_summary=str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Trading session failed",
                symbol=symbol_name,
                error=str(e)
            )
            raise Exception(f"Trading session error: {e}")
        
    def run_agent_backtest(
        self, 
        historical_data: List[Dict], 
        initial_balance: float = 100000,
        symbol: str = "EUR_USD"
    ) -> Dict[str, Any]:
        """
        Run a complete backtest using the backtesting crew
        
        Args:
            historical_data: List of OHLC bars
            initial_balance: Starting capital
            symbol: Trading symbol
        
        Returns:
            Complete backtest results
        """
        try:
            logger.info(f"🤖 Starting agent-based backtest with {len(historical_data)} bars")
            
            # Initialize backtesting crew
            backtest_crew = self.backtesting_crew()
            
            # Prepare backtest inputs
            backtest_inputs = {
                'historical_data': json.dumps(historical_data),
                'initial_balance': initial_balance,
                'symbol': symbol,
                'symbol_name': symbol,
                'start_date': historical_data[0]['timestamp'] if historical_data else None,
                'end_date': historical_data[-1]['timestamp'] if historical_data else None,
                'total_bars': len(historical_data),
                'timestamp': str(datetime.now().timestamp()).replace('.','')
            }
            
            # Execute backtest - use correct CrewAI syntax
            logger.info("🚀 Executing backtesting crew...")
            result = backtest_crew.kickoff(inputs=backtest_inputs)
            
            # Parse and return results
            logger.info("✅ Backtest completed successfully")
            return {
                'success': True,
                'result': result,
                'total_bars_processed': len(historical_data),
                'symbol': symbol,
                'initial_balance': initial_balance
            }
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # Example usage method you can add to test the backtesting agent
    async def test_backtesting_orchestrator(self):
        """
        Test the backtesting orchestrator with sample data
        """
        # Sample historical data for testing
        sample_data = [
            {
                'timestamp': '2024-01-01T09:00:00',
                'open': 1.1000,
                'high': 1.1050,
                'low': 1.0990,
                'close': 1.1030,
                'volume': 1000
            },
            {
                'timestamp': '2024-01-01T09:15:00', 
                'open': 1.1030,
                'high': 1.1080,
                'low': 1.1020,
                'close': 1.1060,
                'volume': 1200
            },
            # Add more sample bars as needed...
        ]
        
        logger.info("🧪 Testing backtesting orchestrator agent...")
        
        # Test market context simulation
        market_context = simulate_historical_market_context(
            historical_bars=json.dumps(sample_data),
            current_bar_index=1,
            account_info=json.dumps({
                'balance': 100000,
                'equity': 100000,
                'margin_available': 50000
            })
        )
        
        logger.info("✅ Market context simulation successful")
        logger.info(f"Market context: {market_context}")
        
        # Test trade execution simulation
        execution_result = simulate_trade_execution(
            trade_decision=json.dumps({
                'side': 'buy',
                'quantity': 10000,
                'order_type': 'market',
                'symbol': 'EUR_USD'
            }),
            market_context=market_context
        )
        
        logger.info("✅ Trade execution simulation successful")
        logger.info(f"Execution result: {execution_result}")
        
        return True
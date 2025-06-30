"""
Updated Autonomous Trading System Crew with Execution Capabilities
Integrates the new trading execution tools with existing Wyckoff analysis agents
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any

from crewai.utilities.events.third_party.agentops_listener import agentops
from mcp_servers.oanda_direct_api import OandaDirectAPI

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

#from autonomous_trading_system.trading_dashboard import get_historical_data
from autonomous_trading_system.utils.wyckoff_pattern_analyzer import wyckoff_analyzer
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
                'symbol_name': symbol_name,
                'current_year': str(datetime.now().year)
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
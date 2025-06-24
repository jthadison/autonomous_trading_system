import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from crewai.utilities.events.third_party.agentops_listener import agentops


# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from autonomous_trading_system.trading_dashboard import get_historical_data
from autonomous_trading_system.utils.wyckoff_pattern_analyzer import wyckoff_analyzer
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from src.config.logging_config import logger
from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
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

warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
 
@tool
def get_live_price(instrument: str) -> Dict[str, Any]:
    """Get live price for a forex instrument"""
    async def _get_price():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
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
        logger.error(f"Failed to get live price for {instrument}", error=str(e))
        return {"error": str(e)}
    
@tool  
def get_historical_data(instrument: str, timeframe: str = "M15", count: int = 200) -> Dict[str, Any]:
    """Get historical price data for Wyckoff analysis"""
    async def _get_historical():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
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
        logger.error(f"Failed to get historical data for {instrument}", error=str(e))
        return {"error": str(e)}

@tool
async def analyze_wyckoff_patterns(instrument: str, timeframe: str = "M15") -> Dict[str, Any]:
    """Perform comprehensive Wyckoff pattern analysis"""
    try:
        # Get historical data for analysis
        historical_data = get_historical_data(instrument, timeframe, 200)
        
        if "error" in historical_data:
            return {"error": f"Failed to get data: {historical_data['error']}"}
        
        # Run Wyckoff analysis
        analysis_result = await wyckoff_analyzer.analyze_market_data(historical_data['data'], timeframe)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Wyckoff analysis failed for {instrument}", error=str(e))
        return {"error": str(e)}

@tool
def get_account_info() -> Dict[str, Any]:
    """Get current account information"""
    async def _get_account():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
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
    """AutonomousTradingSystem crew"""
    agentops.init(os.getenv("AGENTOP_API_KEY"), skip_auto_end_session=True)

    agents: List[BaseAgent]
    tasks: List[Task]
    
    agents_config = 'config/agents.yaml' 
    tasks_config = 'config/tasks.yaml' 
    
    ''' def __init__(self):
        server_params = StdioServerParameters(
            command= "uvx",            
            args= [
               "mcp_servers/server.py"],
                env={"UV_PYTHON": "3.12", **os.environ})
        
        with MCPServerAdapter(server_params) as mcp_tools:
            print(f"Available tools: {[tool.name for tool in mcp_tools]}") '''
        
    #self.twelveData_tools = None

        # Initialize Claude Sonnet 3.5 with conservative settings
    ''' self.claude_llm = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.1,  # Low for consistent outputs
        max_tokens=1500,  # Conservative token limit to avoid rate limits
        max_retries=3,
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        timeout=300,
        stop=4
    ) '''

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    
    # Global Oanda wrapper instance
    oanda_wrapper = None

    from src.autonomous_trading_system.utils.wyckoff_pattern_analyzer import wyckoff_analyzer

    
    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_engineer'], # type: ignore[index]
            tools=[get_historical_data],
            verbose=True
        )
        
    @agent
    def wyckoff_market_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['wyckoff_market_analyst'], # type: ignore[index]
            verbose=True
        )
        
    @agent
    def wyckoff_risk_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['wyckoff_risk_manager'], # type: ignore[index]
            tools=[get_account_info, calculate_position_size],
            verbose=True
        )
        
    @agent
    def wyckoff_trading_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['wyckoff_trading_coordinator'], # type: ignore[index]
            tools=[get_live_price, get_account_info],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    
    @task
    def data_engineer_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_engineer_task'], # type: ignore[index]
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
        """Creates the AutonomousTradingSystem crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
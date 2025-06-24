import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json

from crewai.utilities.events.third_party.agentops_listener import agentops

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
from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
from src.database.manager import db_manager
from src.database.models import AgentAction, LogLevel
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from pydantic import PydanticDeprecatedSince20, SecretStr
from langchain_anthropic import ChatAnthropic
from src.autonomous_trading_system.tools.data_verification_agent import data_quality_report_tool, data_verification_tool
import warnings
import os

from dotenv import load_dotenv
load_dotenv(override=True)

warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

# Import the data verification tools
try:
    from autonomous_trading_system.tools.data_verification_agent import (
        data_verification_tool, 
        data_quality_report_tool
    )
except ImportError:
    logger.warning("Could not import data verification tools")
    data_verification_tool = None
    data_quality_report_tool = None

@tool
def get_raw_market_data(instrument: str, timeframe: str = "M15", count: int = 200) -> str:
    """Get raw market data from OANDA for verification and cleaning"""
    async def _get_raw_data():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            return await oanda.get_historical_data(instrument, timeframe, count)
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _get_raw_data())
                result = future.result()
        else:
            result = asyncio.run(_get_raw_data())
        
        # Return raw data as JSON string for the verification agent to process
        if "error" in result:
            return f"Error retrieving data: {result['error']}"
        
        # Convert to JSON string (this is the potentially problematic data)
        raw_data = result.get("data", {}).get("candles", [])
        return json.dumps(raw_data)
        
    except Exception as e:
        logger.error(f"Failed to get raw market data for {instrument}", error=str(e))
        return f"Error: {str(e)}"

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
        logger.error("Failed to get live price", instrument=instrument, error=str(e))
        return {"error": str(e)}

@tool
async def analyze_wyckoff_patterns(instrument: str, timeframe: str = "M15") -> Dict[str, Any]:
    """Perform comprehensive Wyckoff pattern analysis on cleaned data"""
    try:
        # This will now work with cleaned data from the verification agent
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            return await oanda.get_historical_data(instrument, "M1", 100)
        
        if "error" in historical_data:
            return {"error": f"Failed to get data: {historical_data['error']}"}
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
def calculate_position_size(entry_price: float, stop_loss: float, risk_percent: float = 2.0) -> Dict[str, Any]:
    """Calculate position size based on risk management"""
    try:
        account = get_account_info()
        if "error" in account:
            return {"error": "Cannot get account info"}
        
        balance = float(account.get("balance", 0))
        if balance <= 0:
            return {"error": "Invalid account balance"}
        
        risk_amount = balance * (risk_percent / 100)
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return {"error": "Invalid stop loss level"}
        
        position_size = risk_amount / stop_distance
        
        return {
            "position_size": round(position_size, 0),
            "risk_amount": risk_amount,
            "stop_distance": stop_distance,
            "balance": balance
        }
    except Exception as e:
        logger.error("Failed to calculate position size", error=str(e))
        return {"error": str(e)}

@CrewBase
class AutonomousTradingSystem():
    """
    Enhanced AutonomousTradingSystem with 4-Agent Pipeline:
    1. Data Verification & Cleaning Agent
    2. Wyckoff Market Analyst  
    3. Wyckoff Risk Manager
    4. Wyckoff Trading Coordinator
    """
    
    def __init__(self):
        if os.getenv("AGENTOP_API_KEY"):
            agentops.init(os.getenv("AGENTOP_API_KEY"), skip_auto_end_session=True)
        
        self.claude_llm = self._init_claude_llm()
    
    def _init_claude_llm(self):
        """Initialize Claude LLM"""
        try:
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY not found")
            
            claude_llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                temperature=0.1,
                #max_tokens=2000,
                max_retries=3,
                api_key=SecretStr(os.getenv('ANTHROPIC_API_KEY') or ""),

                timeout=300,
                stop=None
            )
            
            logger.info("Claude LLM initialized successfully")
            return claude_llm
            
        except Exception as e:
            logger.error("Failed to initialize Claude LLM", error=str(e))
            raise

    agents: List[BaseAgent]
    tasks: List[Task]
    
    agents_config = 'config/agents.yaml' 
    tasks_config = 'config/tasks.yaml'

    @agent
    def data_verification_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['data_verification_agent'], # type: ignore[index]
            tools=[
                get_raw_market_data
            ] ,
            llm=self.claude_llm,
            verbose=True
        )

    @agent
    def wyckoff_market_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['wyckoff_market_analyst'], # type: ignore[index]
            tools=[analyze_wyckoff_patterns],
            llm=self.claude_llm,
            verbose=True
        )
        
    @agent
    def wyckoff_risk_manager(self) -> Agent:
        """Agent 3: Wyckoff Risk Manager"""
        return Agent(
            config=self.agents_config['wyckoff_risk_manager'], # type: ignore[index]
            tools=[get_account_info, calculate_position_size],
            llm=self.claude_llm,
            verbose=True
        )
        
    @agent
    def wyckoff_trading_coordinator(self) -> Agent:
        """Agent 4: Wyckoff Trading Coordinator"""
        return Agent(
            config=self.agents_config['wyckoff_trading_coordinator'], # type: ignore[index]
            tools=[get_live_price, get_account_info],
            llm=self.claude_llm,
            verbose=True
        )

    @task
    def data_verification_task(self) -> Task:
        """Task 2: Wyckoff Analysis (uses clean data)"""
        return Task(
            config=self.tasks_config['data_verification_task'], # type: ignore[index]  # Depends on clean data
        )        

    @task
    def wyckoff_analysis_task(self) -> Task:
        """Task 2: Wyckoff Analysis (uses clean data)"""
        return Task(
            config=self.tasks_config['wyckoff_analysis_task'], # type: ignore[index]
            context=[self.data_verification_task()]  # Depends on clean data
        )

    @task
    def wyckoff_risk_task(self) -> Task:
        """Task 3: Risk Assessment"""
        return Task(
            config=self.tasks_config['wyckoff_risk_task'], # type: ignore[index]
            context=[self.wyckoff_analysis_task()]  # Depends on market analysis
        )
        
    @task
    def wyckoff_decision_task(self) -> Task:
        """Task 4: Trading Decision"""
        return Task(
            config=self.tasks_config['wyckoff_decision_task'], # type: ignore[index]
            context=[self.wyckoff_analysis_task(), self.wyckoff_risk_task()]  # Depends on both
        )

    @crew
    def crew(self) -> Crew:
        """Creates the 4-Agent AutonomousTradingSystem crew with data verification"""
        
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            # Data flows: Raw Data -> Verified Data -> Analysis -> Risk -> Decision
        )
"""
Integration of Performance Monitoring with Existing CrewAI Trading System
This shows how to modify your crew.py to include comprehensive performance tracking
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from crewai.crew import BaseAgent
from pydantic import BaseModel

# Your existing imports
from crewai.utilities.events.third_party.agentops_listener import agentops
from crewai.project import CrewBase, agent, crew, task
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool

# Add the performance monitoring system
from src.monitoring.agent_performance_monitor import PerformanceMonitor, PerformanceDashboard

# Your existing imports (keeping them as they are)
from src.config.logging_config import logger
from src.database.manager import db_manager
from src.database.models import AgentAction, LogLevel

# Import your existing tools (you'll need to adjust these imports to match your actual structure)
# These are placeholder imports - replace with your actual tool imports
try:
    from src.autonomous_trading_system.tools.trading_execution_tools_sync import (
        execute_market_trade,
        execute_limit_trade,
        cancel_pending_order,
        get_open_positions,
        get_pending_orders,
        close_position,
        get_portfolio_status
    )
    #from src.autonomous_trading_system.trading_dashboard import get_historical_data
    # Add other tool imports as needed
    def get_live_price(*args, **kwargs):
        """Placeholder - replace with actual implementation"""
        pass
    
    def get_account_info(*args, **kwargs):
        """Placeholder - replace with actual implementation"""
        pass
    
    def calculate_position_size(*args, **kwargs):
        """Placeholder - replace with actual implementation"""
        pass
        
except ImportError as e:
    logger.warning(f"⚠️ Could not import some tools: {e}")
    # Define placeholder tools
    def get_live_price(*args, **kwargs): pass
    def get_historical_data(*args, **kwargs): pass
    def get_account_info(*args, **kwargs): pass
    def calculate_position_size(*args, **kwargs): pass
    def get_portfolio_status(*args, **kwargs): pass
    def get_open_positions(*args, **kwargs): pass
    def get_pending_orders(*args, **kwargs): pass
    def close_position(*args, **kwargs): pass
    def execute_market_trade(*args, **kwargs): pass
    def execute_limit_trade(*args, **kwargs): pass
    def cancel_pending_order(*args, **kwargs): pass

import os

# Enhanced tool wrapper for monitoring
class MonitoredTool(BaseTool):
    """Wrapper to add performance monitoring to CrewAI tools"""
    
    def __init__(self, original_tool: BaseTool, monitor: PerformanceMonitor, agent_name: str):
        # Get original tool properties with proper defaults
        tool_name = getattr(original_tool, 'name', str(original_tool))
        tool_description = getattr(original_tool, 'description', f"Monitored version of {original_tool}")
        
        # Handle args_schema properly - create a default if None
        original_args_schema = getattr(original_tool, 'args_schema', None)
        if original_args_schema is None:
            # Create a simple default BaseModel if no schema exists
            from pydantic import BaseModel
            
            class DefaultToolArgs(BaseModel):
                """Default arguments schema for monitored tools"""
                pass
            
            tool_args_schema = DefaultToolArgs
        else:
            tool_args_schema = original_args_schema
        
        # Initialize BaseTool with original tool's properties
        super().__init__(
            name=tool_name,
            description=tool_description,
            args_schema=tool_args_schema
        )
        
        self.original_tool = original_tool
        self.monitor = monitor
        self.agent_name = agent_name
        self.tool_name = self.name
    
    def _run(self, *args, **kwargs) -> Any:
        """Execute tool with monitoring (synchronous version)"""
        start_time = time.time()
        success = False
        result = None
        
        try:
            # Execute the original tool
            if hasattr(self.original_tool, '_run'):
                result = self.original_tool._run(*args, **kwargs)
            elif callable(self.original_tool):
                result = self.original_tool(*args, **kwargs)
            else:
                raise ValueError(f"Tool {self.tool_name} is not callable")
            
            success = True
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool {self.tool_name} failed: {e}")
            raise
        
        finally:
            # Track tool usage
            execution_time = (time.time() - start_time) * 1000
            
            # Run monitoring in background thread to avoid blocking
            import threading
            threading.Thread(
                target=self._track_tool_usage,
                args=(execution_time, success),
                daemon=True
            ).start()
    
    async def _arun(self, *args, **kwargs) -> Any:
        """Execute tool with monitoring (asynchronous version)"""
        start_time = time.time()
        success = False
        result = None
        
        try:
            # Try different execution methods based on what's available
            # if hasattr(self.original_tool, '_arun'):
            #     # If the original tool has async support, use it
            #     result = await self.original_tool._arun(*args, **kwargs)
            if hasattr(self.original_tool, '_run'):
                # Fall back to sync execution
                result = self.original_tool._run(*args, **kwargs)
            elif callable(self.original_tool):
                # Direct call if it's a simple function
                if asyncio.iscoroutinefunction(self.original_tool):
                    result = await self.original_tool(*args, **kwargs)
                else:
                    result = self.original_tool(*args, **kwargs)
            else:
                raise ValueError(f"Tool {self.tool_name} is not callable")
            
            success = True
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool {self.tool_name} failed: {e}")
            raise
        
        finally:
            # Track tool usage
            execution_time = (time.time() - start_time) * 1000
            await self._track_tool_usage_async(execution_time, success)
    
    def _track_tool_usage(self, execution_time: float, success: bool):
        """Track tool usage synchronously"""
        try:
            # Create async task for monitoring
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._track_tool_usage_async(execution_time, success))
            loop.close()
        except Exception as e:
            logger.error(f"❌ Failed to track tool usage: {e}")
    
    async def _track_tool_usage_async(self, execution_time: float, success: bool):
        """Track tool usage asynchronously"""
        try:
            # Update tool metrics in monitor
            await self.monitor._update_tool_metrics(
                self.agent_name, 
                [self.tool_name], 
                execution_time
            )
            
            # Log tool usage to database
            import sqlite3
            with sqlite3.connect(self.monitor.db_path) as conn:
                conn.execute("""
                    INSERT INTO tool_usage 
                    (tool_name, agent_name, timestamp, success, execution_time_ms, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.tool_name,
                    self.agent_name,
                    datetime.now().isoformat(),
                    success,
                    execution_time,
                    self.monitor.session_id
                ))
        except Exception as e:
            logger.error(f"❌ Failed to log tool usage: {e}")

# Enhanced Agent wrapper for monitoring
class MonitoredAgent:
    """Wrapper to add comprehensive monitoring to CrewAI agents"""
    
    def __init__(self, agent: Agent, monitor: PerformanceMonitor, agent_name: str):
        self.agent = agent
        self.monitor = monitor
        self.agent_name = agent_name
        self.decision_history = []
        
        # Wrap all agent tools with monitoring (safer approach)
        if hasattr(agent, 'tools') and agent.tools:
            try:
                monitored_tools = []
                for tool in agent.tools:
                    try:
                        if isinstance(tool, BaseTool):
                            monitored_tool = MonitoredTool(tool, monitor, agent_name)
                            monitored_tools.append(monitored_tool)
                            logger.debug(f"✅ Wrapped BaseTool: {getattr(tool, 'name', str(tool))}")
                        else:
                            # Keep non-BaseTool tools as-is
                            logger.debug(f"⚠️ Tool {tool} is not a BaseTool, keeping original")
                            monitored_tools.append(tool)
                    except Exception as tool_error:
                        logger.warning(f"⚠️ Failed to wrap individual tool {tool}: {tool_error}")
                        # Keep the original tool if wrapping fails
                        monitored_tools.append(tool)
                
                # Only assign if we successfully processed tools
                if monitored_tools:
                    self.agent.tools = monitored_tools
                    wrapped_count = sum(1 for tool in monitored_tools if isinstance(tool, MonitoredTool))
                    logger.info(f"✅ Wrapped {wrapped_count}/{len(monitored_tools)} tools for {agent_name}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not wrap tools for {agent_name}: {e}")
                # Continue without tool monitoring for this agent
    
    async def execute_task_with_monitoring(self, task, context: Dict[str, Any] = {}):
        """Execute agent task with comprehensive monitoring"""
        
        start_time = time.time()
        decision_id = None
        
        try:
            # Pre-execution: Track decision start
            input_data = {
                "task_description": getattr(task, 'description', str(task)),
                "task_type": getattr(task, 'expected_output', 'unknown'),
                "context": context or {}
            }
            
            # Extract market context for better tracking
            market_context = self._extract_market_context(context)
            
            # CrewAI agents don't have execute() method - we'll track the decision without execution
            logger.info(f"🤖 {self.agent_name} tracking task decision...")
            
            # For CrewAI, we'll simulate the decision tracking since actual execution happens through crew
            execution_time = (time.time() - start_time) * 1000
            
            # Create a simulated result for tracking purposes
            simulated_output = {
                "agent_name": self.agent_name,
                "task_description": input_data["task_description"],
                "execution_method": "crew_system",
                "timestamp": datetime.now().isoformat()
            }
            # Post-execution: Extract outputs and confidence (simulated)
            output_data = simulated_output
            confidence = 100.0  # Simulated confidence for CrewAI agent

            # Determine tools used (simplified - you might need to enhance this)
            tools_used = getattr(self.agent, "tools", [])
            # Track the decision
            decision_id = await self.monitor.track_agent_decision(
                agent_name=self.agent_name,
                decision_type=self._get_decision_type(task),
                input_data=input_data,
                output_data=output_data,
                confidence=confidence,
                tools_used=tools_used,
                execution_time_ms=execution_time,
                market_context=market_context
            )
            
            # Store for outcome tracking
            self.decision_history.append({
                'decision_id': decision_id,
                'timestamp': datetime.now(),
                'task_type': self._get_decision_type(task),
                'result': simulated_output
            })
            
            logger.info(f"✅ {self.agent_name} decision tracked (confidence: {confidence:.1f}%)")
            
            return simulated_output, decision_id
            
        except Exception as e:
            logger.error(f"❌ {self.agent_name} task tracking failed: {e}")
            
            # Track failed decision
            if decision_id:
                await self.monitor.update_decision_outcome(
                    decision_id=decision_id,
                    outcome_positive=False,
                    outcome_value=-1.0
                )
            
            raise
    
    def _extract_market_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market context from task context"""
        market_context = {}
        
        if context:
            market_context.update({
                'symbol': context.get('symbol_name', 'unknown'),
                'timeframe': context.get('timeframe', 'unknown'),
                'timestamp': context.get('current_time', datetime.now().isoformat()),
                'market_hours': self._determine_market_hours(),
                'volatility_regime': self._estimate_volatility(context)
            })
        
        return market_context
    
    def _parse_agent_output(self, result) -> Tuple[Dict[str, Any], float]:
        """Parse agent output to extract structured data and confidence"""
        
        output_data = {"raw_result": str(result)}
        confidence = 50.0  # Default confidence
        
        try:
            # Try to parse structured output
            if hasattr(result, '__dict__'):
                output_data.update(result.__dict__)
            elif isinstance(result, dict):
                output_data.update(result)
            elif isinstance(result, str):
                # Try to extract confidence from text
                confidence = self._extract_confidence_from_text(result)
                output_data["parsed_text"] = result
            
            # Look for explicit confidence
            if 'confidence' in output_data:
                # Safely extract confidence if present and valid
                try:
                    conf_val = output_data.get('confidence', None)
                    if conf_val is not None:
                        confidence = float(conf_val)
                    elif hasattr(result, 'confidence'):
                        confidence = float(getattr(result, 'confidence'))
                except Exception as conf_e:
                    logger.warning(f"⚠️ Could not extract confidence: {conf_e}")
        except Exception as e:
            logger.error(f"❌ {self.agent_name} task execution failed: {e}")
        return output_data, confidence
    
    def _extract_confidence_from_text(self, text: str) -> float:
        """Extract confidence score from text output"""
        import re
        
        # Look for patterns like "confidence: 85%", "85% confident", etc.
        confidence_patterns = [
            r'confidence[:\s]+(\d+)%?',
            r'(\d+)%\s+confident',
            r'confidence[:\s]+(\d+\.\d+)',
            r'certainty[:\s]+(\d+)%?'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    conf = float(match.group(1))
                    return min(100.0, max(0.0, conf))  # Clamp between 0-100
                except ValueError:
                    continue
        
        # Default confidence based on text analysis
        if any(word in text.lower() for word in ['strong', 'clear', 'definite', 'certain']):
            return 75.0
        elif any(word in text.lower() for word in ['weak', 'uncertain', 'maybe', 'possible']):
            return 35.0
        else:
            return 50.0
    
    def _get_tools_used(self) -> List[str]:
        """Get list of tools used by the agent"""
        if hasattr(self.agent, 'tools') and self.agent.tools:
            return [getattr(tool, 'name', str(tool)) for tool in self.agent.tools]
        return []
    
    def _get_decision_type(self, task) -> str:
        """Determine the type of decision based on task"""
        task_desc = getattr(task, 'description', str(task)).lower()
        
        if 'analysis' in task_desc or 'analyze' in task_desc:
            return 'market_analysis'
        elif 'risk' in task_desc:
            return 'risk_assessment'
        elif 'trade' in task_desc or 'decision' in task_desc:
            return 'trading_decision'
        elif 'backtest' in task_desc:
            return 'backtest_coordination'
        else:
            return 'general_task'
    
    def _determine_market_hours(self) -> str:
        """Determine current market session"""
        current_hour = datetime.now().hour
        
        if 0 <= current_hour < 6:
            return 'asia_session'
        elif 6 <= current_hour < 14:
            return 'london_session'
        elif 14 <= current_hour < 22:
            return 'ny_session'
        else:
            return 'after_hours'
    
    def _estimate_volatility(self, context: Dict[str, Any]) -> str:
        """Estimate volatility regime from context"""
        # This is simplified - you could enhance with actual volatility calculation
        if context and 'historical_data' in context:
            return 'medium'  # Placeholder
        return 'unknown'
    
    async def update_decision_outcome_from_trade(self, trade_result: Dict[str, Any]):
        """Update decision outcomes based on trade results"""
        
        # Find recent trading decisions to update
        recent_decisions = [
            d for d in self.decision_history 
            if d['task_type'] == 'trading_decision' and 
            (datetime.now() - d['timestamp']).total_seconds() < 3600  # Last hour
        ]
        
        for decision in recent_decisions:
            if trade_result.get('success', False):
                await self.monitor.update_decision_outcome(
                    decision_id=decision['decision_id'],
                    outcome_positive=trade_result.get('pnl', 0) > 0,
                    outcome_value=trade_result.get('pnl', 0)
                )

# Enhanced CrewAI Trading System with Monitoring
@CrewBase
class MonitoredAutonomousTradingSystem:
    """Enhanced AutonomousTradingSystem with comprehensive performance monitoring"""
    
    def __init__(self):
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.dashboard = PerformanceDashboard(self.performance_monitor)
        self.monitored_agents = {}
        self.monitoring_started = False
        
        # Start monitoring
        #asyncio.create_task(self.performance_monitor.start_monitoring())
        
        # Add performance alert handler
        self.performance_monitor.add_alert_callback(self._handle_performance_alert)
        
        agentops.init(os.getenv("AGENTOP_API_KEY"), skip_auto_end_session=True)
    
    async def _handle_performance_alert(self, message: str):
        """Handle performance alerts"""
        logger.warning(f"🚨 PERFORMANCE ALERT: {message}")
        
        # You could add additional alert handling here:
        # - Send notifications
        # - Automatically adjust parameters
        # - Trigger emergency stops
    
    agents: List[BaseAgent]
    tasks: List[Task]
    
    agents_config = 'config/agents.yaml' 
    tasks_config = 'config/tasks.yaml'
    
    @agent
    def wyckoff_market_analyst(self) -> Agent:
        """Enhanced market analyst with monitoring"""
        base_agent = Agent(
            config=self.agents_config['wyckoff_market_analyst'], # type: ignore[index]
            tools=[
                get_live_price,
                get_historical_data
            ],
            verbose=True,
            max_iter=3
        )
        
        # Wrap with monitoring
        monitored_agent = MonitoredAgent(base_agent, self.performance_monitor, "wyckoff_market_analyst")
        self.monitored_agents["wyckoff_market_analyst"] = monitored_agent
        
        return base_agent
    
    @agent
    def wyckoff_risk_manager(self) -> Agent:
        """Enhanced risk manager with monitoring"""
        base_agent = Agent(
            config=self.agents_config['wyckoff_risk_manager'],# type: ignore[index]
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
        
        # Wrap with monitoring
        monitored_agent = MonitoredAgent(base_agent, self.performance_monitor, "wyckoff_risk_manager")
        self.monitored_agents["wyckoff_risk_manager"] = monitored_agent
        
        return base_agent
    
    @agent
    def wyckoff_trading_coordinator(self) -> Agent:
        """Enhanced trading coordinator with monitoring"""
        base_agent = Agent(
            config=self.agents_config['wyckoff_trading_coordinator'],# type: ignore[index]
            tools=[
                get_live_price,
                get_account_info,
                get_portfolio_status,
                get_open_positions,
                get_pending_orders,
                close_position,
                execute_market_trade,
                execute_limit_trade,
                cancel_pending_order
            ],
            verbose=True,
            max_iter=3
        )
        
        # Wrap with monitoring
        monitored_agent = MonitoredAgent(base_agent, self.performance_monitor, "wyckoff_trading_coordinator")
        self.monitored_agents["wyckoff_trading_coordinator"] = monitored_agent
        
        return base_agent
    
    # Your existing task definitions remain the same
    @task
    def wyckoff_analysis_task(self) -> Task:
        return Task(config=self.tasks_config['wyckoff_analysis_task']# type: ignore[index]
                    )
    
    @task
    def wyckoff_risk_task(self) -> Task:
        return Task(config=self.tasks_config['wyckoff_risk_task']# type: ignore[index]
                    )
    
    @task
    def wyckoff_decision_task(self) -> Task:
        return Task(config=self.tasks_config['wyckoff_decision_task']# type: ignore[index]
                    )
    
    @crew
    def crew(self) -> Crew:
        """Enhanced crew with monitoring capabilities"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True
        )
    
    async def execute_monitored_crew(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute crew with comprehensive monitoring"""
        
        logger.info("🚀 Starting monitored crew execution...")
        
        await self._ensure_monitoring_started()
        
        symbol = inputs.get('symbol_name', inputs['symbol_name'])
        await self.update_market_data(symbol)
        
        start_time = time.time()
        
        try:
            # Pre-execution monitoring: Track crew start
            for agent_name in self.monitored_agents.keys():
                await self.performance_monitor.track_agent_decision(
                    agent_name=agent_name,
                    decision_type="crew_start",
                    input_data={"crew_inputs": inputs},
                    output_data={"status": "starting"},
                    confidence=50.0,
                    tools_used=[],
                    execution_time_ms=0.0,
                    market_context={"session_start": datetime.now().isoformat()}
                )
            
            # Execute crew normally - this is where actual agent work happens
            logger.info("⚙️ Executing CrewAI crew...")
            result = self.crew().kickoff(inputs=inputs)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Post-execution monitoring: Track crew completion
            await self._process_crew_execution_results(result, inputs, execution_time)
            
            return {
                'success': True,
                'result': result,
                'execution_time_ms': execution_time,
                'monitoring_data': self.performance_monitor.get_real_time_dashboard_data()
            }
            
        except Exception as e:
            logger.error(f"❌ Monitored crew execution failed: {e}")
            
            # Track failed execution for all agents
            for agent_name in self.monitored_agents.keys():
                await self.performance_monitor.track_agent_decision(
                    agent_name=agent_name,
                    decision_type="crew_error",
                    input_data={"error": str(e)},
                    output_data={"status": "failed"},
                    confidence=0.0,
                    tools_used=[],
                    execution_time_ms=(time.time() - start_time) * 1000,
                    market_context={"error_time": datetime.now().isoformat()}
                )
            
            return {
                'success': False,
                'error': str(e),
                'monitoring_data': self.performance_monitor.get_real_time_dashboard_data()
            }
    
    async def update_market_data(self, symbol: str, timeframe: str = "M15", bars: int = 100):
        """Update market data for enhanced regime detection - FIXED VERSION"""
        try:
            logger.info(f"📊 Attempting to update market data for {symbol}...")
            
            # Option 1: Try to use your existing Oanda Direct API
            try:
                from src.mcp_servers.oanda_direct_api import OandaDirectAPI
                
                async with OandaDirectAPI() as oanda:
                    # Get historical data using your existing API
                    response = await oanda.get_historical_data(
                        instrument=symbol,
                        granularity=timeframe,
                        count=bars
                    )
                    
                    if response.get('success') and response.get('data'):
                        price_data = response['data']
                        self.performance_monitor.update_price_data(price_data)
                        logger.info(f"✅ Updated market data: {len(price_data)} bars for {symbol}")
                        return True
                        
            except ImportError:
                logger.warning("⚠️ OandaDirectAPI not available, trying fallback...")
            except Exception as e:
                logger.warning(f"⚠️ Oanda API fetch failed: {e}")
            
            # Option 2: Generate minimal dummy data (fallback)
            logger.info("📊 Using fallback dummy data for testing...")
            current_time = datetime.now()
            dummy_price_data = []
            
            base_price = 1.1000 if 'EUR' in symbol else 1.0000
            
            for i in range(min(bars, 20)):  # Limit to 20 bars
                timestamp = current_time - timedelta(minutes=15 * (bars - i))
                
                import random
                price_change = random.uniform(-0.001, 0.001)
                
                open_price = base_price + price_change
                high_price = open_price + random.uniform(0, 0.0005)
                low_price = open_price - random.uniform(0, 0.0005)
                close_price = open_price + random.uniform(-0.0003, 0.0003)
                
                dummy_price_data.append({
                    'timestamp': timestamp.isoformat(),
                    'open': round(open_price, 5),
                    'high': round(high_price, 5),
                    'low': round(low_price, 5),
                    'close': round(close_price, 5),
                    'volume': random.randint(800, 1200)
                })
                
                base_price = close_price
            
            # Update with dummy data
            self.performance_monitor.update_price_data(dummy_price_data)
            logger.info(f"✅ Updated with fallback data: {len(dummy_price_data)} bars for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update market data: {e}")
            return False  # Don't fail the entire system
            
    async def _process_crew_execution_results(self, result, inputs: Dict[str, Any], execution_time: float):
        """Process crew execution results for monitoring"""
        
        try:
            # Extract outcomes from crew result
            if hasattr(result, 'tasks_output') and result.tasks_output:
                logger.info(f"📊 Processing {len(result.tasks_output)} task outputs for monitoring")
                
                for i, task_output in enumerate(result.tasks_output):
                    # Try to extract agent information from task output
                    agent_name = self._extract_agent_from_task_output(task_output, i)
                    
                    if agent_name in self.monitored_agents:
                        # Determine success based on task output
                        success = self._determine_task_success(task_output)
                        
                        # Extract confidence from output if possible
                        confidence = self._extract_confidence_from_task_output(task_output)
                        
                        # Track the actual agent decision
                        decision_id = await self.performance_monitor.track_agent_decision(
                            agent_name=agent_name,
                            decision_type="crew_task_completion",
                            input_data={"task_index": i, "inputs": inputs},
                            output_data={"task_output": str(task_output), "success": success},
                            confidence=confidence,
                            tools_used=self._get_agent_tools(agent_name),
                            execution_time_ms=execution_time / len(result.tasks_output),  # Distribute time
                            market_context={"completion_time": datetime.now().isoformat()}
                        )
                        
                        # Update decision outcome
                        await self.performance_monitor.update_decision_outcome(
                            decision_id=decision_id,
                            outcome_positive=success,
                            outcome_value=1.0 if success else -1.0
                        )
                        
                        logger.debug(f"✅ Tracked decision for {agent_name}: success={success}, confidence={confidence:.1f}%")
            
            else:
                # Fallback: track completion for all agents without specific task outputs
                logger.info("📊 No specific task outputs found, tracking general completion")
                for agent_name in self.monitored_agents.keys():
                    decision_id = await self.performance_monitor.track_agent_decision(
                        agent_name=agent_name,
                        decision_type="crew_completion",
                        input_data={"inputs": inputs},
                        output_data={"result": str(result)},
                        confidence=60.0,  # Default confidence for completion
                        tools_used=self._get_agent_tools(agent_name),
                        execution_time_ms=execution_time / len(self.monitored_agents),
                        market_context={"completion_time": datetime.now().isoformat()}
                    )
                    
                    # Assume success if no specific error
                    await self.performance_monitor.update_decision_outcome(
                        decision_id=decision_id,
                        outcome_positive=True,
                        outcome_value=1.0
                    )
        
        except Exception as e:
            logger.error(f"❌ Error processing crew execution results: {e}")
    
    def _extract_agent_from_task_output(self, task_output, task_index: int) -> str:
        """Extract agent name from task output"""
        
        # Try to get agent from task output
        if hasattr(task_output, 'agent'):
            return task_output.agent
        
        # Fallback: map task index to agent (assumes sequential execution)
        agent_names = list(self.monitored_agents.keys())
        if task_index < len(agent_names):
            return agent_names[task_index]
        
        # Default fallback
        return "unknown_agent"
    
    def _determine_task_success(self, task_output) -> bool:
        """Determine if a task was successful based on its output"""
        
        # Check for explicit error
        if hasattr(task_output, 'error') and task_output.error:
            return False
        
        # Check for empty or null output
        if not task_output or str(task_output).strip() == "":
            return False
        
        # Check for error keywords in output
        output_str = str(task_output).lower()
        error_keywords = ['error', 'failed', 'exception', 'cannot', 'unable']
        if any(keyword in output_str for keyword in error_keywords):
            return False
        
        # Default to success
        return True
    
    def _extract_confidence_from_task_output(self, task_output) -> float:
        """Extract confidence score from task output"""
        
        # Try to get explicit confidence
        if hasattr(task_output, 'confidence'):
            try:
                return float(task_output.confidence)
            except (ValueError, TypeError):
                pass
        
        # Try to extract from text using the existing method
        if hasattr(self, '_extract_confidence_from_text'):
            # Use first monitored agent's method (they all have the same implementation)
            first_agent = list(self.monitored_agents.values())[0]
            return first_agent._extract_confidence_from_text(str(task_output))
        
        # Default confidence
        return 60.0
    
    async def _ensure_monitoring_started(self):
        """Ensure monitoring is started (call this before any monitored operations)"""
        if not self.monitoring_started:
            await self.performance_monitor.start_monitoring()
            self.monitoring_started = True
            logger.info("✅ Performance monitoring started")
        
    def _get_agent_tools(self, agent_name: str) -> List[str]:
        """Get tools for a specific agent"""
        if agent_name in self.monitored_agents:
            return self.monitored_agents[agent_name]._get_tools_used()
        return []
    
    async def start_performance_dashboard(self):
        """Start the real-time performance dashboard"""
        await self._ensure_monitoring_started()  # Ensure monitoring is started first
        logger.info("📊 Starting performance dashboard...")
        await self.dashboard.start_dashboard(update_interval=30)
    
    def stop_performance_dashboard(self):
        """Stop the performance dashboard"""
        self.dashboard.stop_dashboard()
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get current performance report"""
        return self.performance_monitor.get_real_time_dashboard_data()
    
    async def shutdown_monitoring(self):
        """Properly shutdown monitoring system"""
        if self.monitoring_started:
            await self.performance_monitor.stop_monitoring()
            self.monitoring_started = False
            logger.info("✅ Performance monitoring stopped")

# Usage example
async def run_with_monitoring():
    """Example of running the system with monitoring"""
    # Initialize monitoring
    monitor = PerformanceMonitor()
    await monitor.start_monitoring()
    
    # Initialize the enhanced system
    trading_system = MonitoredAutonomousTradingSystem()
    
    try:
        # Start dashboard in background (optional)
        dashboard_task = asyncio.create_task(trading_system.start_performance_dashboard())
        
        # Execute trading crew with monitoring
        crew_inputs = {
            'symbol_name': 'EUR_USD',
            'current_year': '2025',
            'topic': 'Wyckoff Market Analysis'
        }
        
        result = await trading_system.execute_monitored_crew(crew_inputs)
        
        # Get insights
        dashboard_data = monitor.get_real_time_dashboard_data()
        print(f"Agent Accuracy: {dashboard_data['overall_metrics']['overall_accuracy']:.1f}%")
        
        # Get performance report
        performance_report = await trading_system.get_performance_report()
        
        logger.info("📊 Performance Summary:")
        logger.info(f"Overall Accuracy: {performance_report.get('overall_metrics', {}).get('overall_accuracy', 0):.1f}%")
        logger.info(f"Total Decisions: {performance_report.get('overall_metrics', {}).get('total_decisions', 0)}")
        
        return result, performance_report
        
    finally:
        # Cleanup
        trading_system.stop_performance_dashboard()
        await trading_system.shutdown_monitoring()

if __name__ == "__main__":
    asyncio.run(run_with_monitoring())
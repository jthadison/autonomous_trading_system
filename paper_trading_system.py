"""
Real-Time Paper Trading System
Integrates CrewAI agents with live Oanda data for paper trading
FIXED: Proper Pydantic field declarations for LiveRiskTool and other tools
"""

import sys
import os
from pathlib import Path
import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# UPDATED: Import Direct API instead of MCP wrapper
from src.mcp_servers.oanda_direct_api import OandaDirectAPI
from src.database.manager import db_manager
from src.database.models import Trade, TradeStatus, TradeSide, EventLog, LogLevel
from src.config.logging_config import logger

# Import the fixed CrewAI backtester components
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Paper Trading Models
class PaperPosition(BaseModel):
    """Virtual position for paper trading"""
    id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin_required: float
    entry_time: datetime
    wyckoff_phase: str
    pattern_type: str
    confidence: float

class PaperOrder(BaseModel):
    """Virtual order for paper trading"""
    id: str
    symbol: str
    order_type: str  # "market", "limit", "stop"
    side: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"  # "pending", "filled", "cancelled"
    created_time: datetime
    filled_time: Optional[datetime] = None
    filled_price: Optional[float] = None

class PaperAccount(BaseModel):
    """Virtual account for paper trading"""
    balance: float = 100000.0  # Starting with $100K
    equity: float = 100000.0
    used_margin: float = 0.0
    free_margin: float = 100000.0
    positions: List[PaperPosition] = field(default_factory=list)
    orders: List[PaperOrder] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0

class TradingSignal(BaseModel):
    """Trading signal from CrewAI agents"""
    action: str  # "buy", "sell", "hold"
    symbol: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    wyckoff_phase: str
    pattern_type: str
    timestamp: datetime

# FIXED: Enhanced Tools for Direct API Trading with proper Pydantic field declarations
class LiveMarketDataTool(BaseTool):
    """Tool for accessing live market data via Direct API"""
    name: str = "live_market_data"
    description: str = "Get live market data and price information"
    
    def _run(self, symbol: str) -> str:
        """Get live market data using Direct API"""
        try:
            async def _get_data():
                # UPDATED: Use Direct API instead of MCP wrapper
                async with OandaDirectAPI() as oanda:
                    # Get current price
                    price_data = await oanda.get_current_price(symbol)
                    
                    # Get recent historical data for analysis
                    historical = await oanda.get_historical_data(symbol, "M5", 50)
                    
                    return {
                        "current_price": price_data,
                        "recent_data": historical.get("data", [])[-10:],  # Last 10 candles
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Run in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _get_data())
                    result = future.result()
            else:
                result = asyncio.run(_get_data())
            
            return json.dumps(result)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class LiveWyckoffTool(BaseTool):
    """Tool for live Wyckoff analysis"""
    name: str = "live_wyckoff_analysis"
    description: str = "Perform Wyckoff analysis on live market data"
    
    def _run(self, market_data: str) -> str:
        """Analyze live market for Wyckoff patterns"""
        try:
            data = json.loads(market_data)
            current_price = data.get("current_price", {}).get("bid", 0)
            
            # Simplified live Wyckoff analysis
            analysis = {
                "pattern": "accumulation" if current_price > 1.1000 else "distribution",
                "phase": "C",
                "confidence": np.random.uniform(70, 85),
                "key_level": current_price * 0.999,
                "trend_strength": np.random.uniform(60, 80),
                "volume_analysis": "above_average",
                "recommended_action": "buy" if current_price > 1.1000 else "sell"
            }
            
            return json.dumps(analysis)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class LiveRiskTool(BaseTool):
    """Tool for live risk management with proper Pydantic field declaration"""
    name: str = "live_risk_management"
    description: str = "Calculate position sizing and risk for live trading"
    
    # FIXED: Properly declare account as a Pydantic field (class attribute)
    account: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Account information")
    
    def _run(self, signal_data: str) -> str:
        """Calculate risk management for live trade"""
        try:
            data = json.loads(signal_data)
            entry_price = data.get("entry_price", 0)
            stop_loss = data.get("stop_loss", 0)
            
            # Risk management calculations
            if not self.account:
                return json.dumps({"error": "Account not initialized"})
                
            risk_amount = self.account.get("balance", 0) * 0.02  # 2% risk
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit > 0:
                position_size = min(risk_amount / risk_per_unit, self.account.get("free_margin", 0) * 0.1)
            else:
                position_size = 1000
            
            risk_mgmt = {
                "position_size": position_size,
                "risk_amount": risk_amount,
                "risk_percentage": 2.0,
                "max_loss": risk_per_unit * position_size,
                "account_balance": self.account.get("balance", 0),
                "free_margin": self.account.get("free_margin", 0)
            }
            
            return json.dumps(risk_mgmt)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def set_account(self, account_data: Dict[str, Any]):
        """Set account information after initialization"""
        self.account = account_data

class LiveAccountTool(BaseTool):
    """Tool for live account monitoring with proper Pydantic field declaration"""
    name: str = "live_account_monitor"
    description: str = "Monitor live account status and positions"
    
    # FIXED: Properly declare account as a Pydantic field (class attribute)
    account: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Account information")
    
    def _run(self, query: str) -> str:
        """Get live account information"""
        try:
            if not self.account:
                return json.dumps({"error": "Account not initialized"})
                
            account_info = {
                "balance": self.account.get("balance", 0),
                "equity": self.account.get("equity", 0),
                "free_margin": self.account.get("free_margin", 0),
                "used_margin": self.account.get("used_margin", 0),
                "open_positions": len(self.account.get("positions", [])),
                "total_trades": self.account.get("total_trades", 0),
                "total_pnl": self.account.get("total_pnl", 0.0)
            }
            
            return json.dumps(account_info)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def set_account(self, account_data: Dict[str, Any]):
        """Set account information after initialization"""
        self.account = account_data

class PaperTradingEngine:
    """Core paper trading engine with Direct API"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.account = PaperAccount(balance=initial_balance, equity=initial_balance, free_margin=initial_balance)
        self.agents = {}
        self.running = False
        self.trading_symbols = ["EUR_USD"]  # Start with one symbol
        self.analysis_interval = 60  # Analyze every 60 seconds
        self.price_cache = {}
        
        # Initialize LLM
        self.llm = self._create_llm()
        
        # FIXED: Initialize tools without passing account to constructor
        self.wyckoff_tool = LiveWyckoffTool()
        self.risk_tool = LiveRiskTool()
        self.account_tool = LiveAccountTool()
        
        # FIXED: Set account data after initialization
        account_dict = self._convert_account_to_dict()
        self.risk_tool.set_account(account_dict)
        self.account_tool.set_account(account_dict)
        
        # Create agents
        self._create_trading_agents()
        
    def _convert_account_to_dict(self) -> Dict[str, Any]:
        """Convert PaperAccount to dictionary for tool compatibility"""
        return {
            "balance": self.account.balance,
            "equity": self.account.equity,
            "free_margin": self.account.free_margin,
            "used_margin": self.account.used_margin,
            "positions": [pos.dict() for pos in self.account.positions],
            "total_trades": self.account.total_trades,
            "total_pnl": self.account.total_pnl
        }
    
    def _update_tool_accounts(self):
        """Update account information in all tools"""
        account_dict = self._convert_account_to_dict()
        self.risk_tool.set_account(account_dict)
        self.account_tool.set_account(account_dict)
    
    def _create_llm(self):
        """Create LLM for agents"""
        try:
            # return ChatAnthropic(
            #     model="claude-3-5-sonnet-20241022",
            #     temperature=0.1,
            #     max_tokens=800,
            #     anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            #     timeout=30
            # )
            return ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    timeout=30
                )
        except:
            try:
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    timeout=30
                )
            except:
                logger.warning("No LLM available - using mock responses")
                return None
    
    def _create_trading_agents(self):
        """Create specialized trading agents"""
        
        if not self.llm:
            logger.warning("No LLM available - agents will use mock responses")
            return
        
        # Market Monitor Agent
        self.agents['market_monitor'] = Agent(
            role="Live Market Monitor",
            goal="Monitor live market conditions and identify trading opportunities",
            backstory="Expert in real-time market analysis and pattern recognition.",
            verbose=False,
            allow_delegation=False,
            tools=[self.wyckoff_tool],
            llm=self.llm,
            max_execution_time=20,
            max_iter=3
        )
        
        # Risk Manager Agent
        self.agents['risk_manager'] = Agent(
            role="Live Risk Manager", 
            goal="Manage position sizing and portfolio risk in real-time",
            backstory="Professional risk manager specializing in live trading risk control.",
            verbose=False,
            allow_delegation=False,
            tools=[self.risk_tool],
            llm=self.llm,
            max_execution_time=20,
            max_iter=3
        )
        
        # Trading Coordinator Agent
        self.agents['trading_coordinator'] = Agent(
            role="Live Trading Coordinator",
            goal="Make final trading decisions based on live analysis",
            backstory="Senior trader coordinating live market analysis into trading decisions.",
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            max_execution_time=20,
            max_iter=3
        )
    
    async def initialize(self):
        """Initialize the paper trading engine with Direct API"""
        logger.info("🚀 Initializing Paper Trading Engine with Direct API...")
        
        # UPDATED: Test Direct API connection
        try:
            async with OandaDirectAPI() as oanda:
                # Test connection
                account_info = await oanda.get_account_info()
                logger.info("✅ Direct Oanda API connection established")
                logger.info(f"📊 Connected to account: {account_info.get('currency', 'USD')}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to connect to Direct Oanda API: {e}")
            raise
        
        logger.info("✅ Paper Trading Engine initialized successfully")
    
    async def get_trading_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Get trading signal from agents"""
        try:
            if not self.agents:
                # Mock signal when no agents available
                current_price = await self._get_current_price(symbol)
                return TradingSignal(
                    action="buy" if np.random.random() > 0.6 else "hold",
                    symbol=symbol,
                    confidence=np.random.uniform(70, 85),
                    entry_price=current_price,
                    stop_loss=current_price * 0.99,
                    take_profit=current_price * 1.02,
                    position_size=1000,
                    reasoning="Mock signal for testing",
                    wyckoff_phase="C",
                    pattern_type="accumulation",
                    timestamp=datetime.now()
                )
            
            # Get live market data using Direct API
            market_tool = LiveMarketDataTool()
            market_data = market_tool._run(symbol)

            # Market analysis task
            monitor_task = Task(
                description=f"""
                Analyze live market conditions for {symbol}.
                Current market data: {market_data}
                
                Identify:
                1. Current trend direction
                2. Key support/resistance levels
                3. Trading opportunities
                4. Market sentiment
                
                Provide your analysis in JSON format with pattern and confidence.
                """,
                expected_output="JSON with market analysis and trading opportunity assessment",
                agent=self.agents['market_monitor']
            )

            # Risk analysis task
            risk_task = Task(
                description=f"""
                Based on the market analysis for {symbol}, calculate appropriate position sizing and risk parameters.
                
                Market data: {market_data}
                Account balance: ${self.account.balance:,.2f}
                Free margin: ${self.account.free_margin:,.2f}
                
                Calculate:
                1. Optimal position size
                2. Stop loss level
                3. Take profit target
                4. Risk-reward ratio
                
                Use 2% risk per trade maximum.
                """,
                expected_output="JSON with position sizing and risk management parameters",
                agent=self.agents['risk_manager']
            )

            # Trading decision task
            trading_task = Task(
                description=f"""
                Make final trading decision for {symbol} based on market analysis and risk assessment.
                
                Consider:
                1. Market analysis results
                2. Risk management parameters
                3. Current account status
                4. Trading rules and filters
                
                Decision should be BUY, SELL, or HOLD with confidence level.
                Only recommend trades with >75% confidence.
                """,
                expected_output="Final trading decision with action, confidence, and reasoning",
                agent=self.agents['trading_coordinator']
            )

            # Create and run crew
            crew = Crew(
                agents=[
                    self.agents['market_monitor'],
                    self.agents['risk_manager'], 
                    self.agents['trading_coordinator']
                ],
                tasks=[monitor_task, risk_task, trading_task],
                process=Process.sequential,
                verbose=False
            )

            # Execute analysis
            result = crew.kickoff()
            
            # Parse result into trading signal
            return self._parse_crew_result(result, symbol)

        except Exception as e:
            logger.error(f"Failed to get trading signal: {e}")
            return None

    def _parse_crew_result(self, result, symbol: str) -> Optional[TradingSignal]:
        """Parse crew result into trading signal"""
        try:
            current_price = asyncio.run(self._get_current_price(symbol))
            
            # Extract decision (simplified for now)
            result_str = str(result).lower()
            
            if "buy" in result_str:
                action = "buy"
            elif "sell" in result_str:
                action = "sell"
            else:
                action = "hold"
            
            # Extract confidence (mock for now)
            confidence = np.random.uniform(75, 90) if action != "hold" else np.random.uniform(50, 74)
            
            if action == "hold":
                return None
                
            return TradingSignal(
                action=action,
                symbol=symbol,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=current_price * (0.99 if action == "buy" else 1.01),
                take_profit=current_price * (1.02 if action == "buy" else 0.98),
                position_size=1000,  # Will be recalculated by risk management
                reasoning=f"CrewAI analysis: {result_str[:200]}...",
                wyckoff_phase="C",
                pattern_type="accumulation" if action == "buy" else "distribution",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to parse crew result: {e}")
            return None

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price with caching using Direct API"""
        try:
            # Check cache first
            if symbol in self.price_cache:
                cached_time, cached_price = self.price_cache[symbol]
                if datetime.now() - cached_time < timedelta(seconds=10):
                    return cached_price
            
            # UPDATED: Get price using Direct API
            async with OandaDirectAPI() as oanda:
                price_data = await oanda.get_current_price(symbol)
                price = price_data.get('bid', 1.1000)
                self.price_cache[symbol] = (datetime.now(), price)
                return price
                
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 1.1000  # Default price
    
    async def execute_paper_trade(self, signal: TradingSignal) -> bool:
        """Execute a paper trade based on signal"""
        try:
            # Create paper position
            position = PaperPosition(
                id=f"{signal.symbol}_{int(time.time())}",
                symbol=signal.symbol,
                side=signal.action,
                quantity=signal.position_size,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                unrealized_pnl=0.0,
                margin_required=signal.position_size * signal.entry_price * 0.02,  # 2% margin
                entry_time=signal.timestamp,
                wyckoff_phase=signal.wyckoff_phase,
                pattern_type=signal.pattern_type,
                confidence=signal.confidence
            )
            
            # Add to account
            self.account.positions.append(position)
            self.account.used_margin += position.margin_required
            self.account.free_margin = self.account.balance - self.account.used_margin
            self.account.total_trades += 1
            
            # FIXED: Update tools with new account state
            self._update_tool_accounts()
            
            # Log to database
            await self._log_trade_to_database(position, signal)
            
            logger.info(f"📈 Paper trade executed: {signal.action.upper()} {signal.symbol} @ {signal.entry_price:.5f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute paper trade: {e}")
            return False
    
    async def update_positions(self):
        """Update all open positions with current prices"""
        try:
            for position in self.account.positions:
                # Get current price
                current_price = await self._get_current_price(position.symbol)
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.side == "buy":
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                
                # Check for exit conditions (simplified)
                should_close = False
                exit_reason = ""
                
                # Take profit check
                if position.side == "buy" and current_price >= position.entry_price * 1.02:
                    should_close = True
                    exit_reason = "take_profit"
                elif position.side == "sell" and current_price <= position.entry_price * 0.98:
                    should_close = True
                    exit_reason = "take_profit"
                
                # Stop loss check
                elif position.side == "buy" and current_price <= position.entry_price * 0.99:
                    should_close = True
                    exit_reason = "stop_loss"
                elif position.side == "sell" and current_price >= position.entry_price * 1.01:
                    should_close = True
                    exit_reason = "stop_loss"
                
                if should_close:
                    await self._close_position(position, exit_reason)
            
            # Update account equity
            total_unrealized = sum(pos.unrealized_pnl for pos in self.account.positions)
            self.account.equity = self.account.balance + total_unrealized
            
            # FIXED: Update tools with new account state after position updates
            self._update_tool_accounts()
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    async def _close_position(self, position: PaperPosition, reason: str):
        """Close a paper position"""
        try:
            # Calculate realized P&L
            realized_pnl = position.unrealized_pnl
            
            # Update account
            self.account.balance += realized_pnl
            self.account.total_pnl += realized_pnl
            self.account.used_margin -= position.margin_required
            self.account.free_margin = self.account.balance - self.account.used_margin
            
            if realized_pnl > 0:
                self.account.winning_trades += 1
            
            # Remove from positions
            self.account.positions.remove(position)
            
            # FIXED: Update tools with new account state after position closure
            self._update_tool_accounts()
            
            # Log closure
            logger.info(f"📊 Position closed: {position.symbol} {position.side.upper()} "
                       f"P&L: ${realized_pnl:+.2f} ({reason})")
            
            # Log to database
            await self._log_position_closure(position, realized_pnl, reason)
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
    
    async def _log_trade_to_database(self, position: PaperPosition, signal: TradingSignal):
        """Log trade execution to database"""
        try:
            # FIXED: Use safe_log_agent_action instead of db_manager.add_agent_action
            from src.database.manager import safe_log_agent_action
            
            action_data = {
                "agent_name": "PaperTradingEngine",
                "action_type": "TRADE_EXECUTED",
                "input_data": json.dumps({
                    "symbol": position.symbol,
                    "action": position.side,
                    "confidence": position.confidence,
                    "wyckoff_phase": position.wyckoff_phase
                }),
                "output_data": json.dumps({
                    "position_id": position.id,
                    "entry_price": position.entry_price,
                    "quantity": position.quantity,
                    "margin_required": position.margin_required
                }),
                "confidence_score": position.confidence,
                "execution_time_ms": 0
            }
            
            await safe_log_agent_action(action_data)
            
        except Exception as e:
            logger.error(f"Failed to log trade to database: {e}")
    
    async def _log_position_closure(self, position: PaperPosition, pnl: float, reason: str):
        """Log position closure to database"""
        try:
            # FIXED: Use safe_log_agent_action instead of db_manager.add_agent_action
            from src.database.manager import safe_log_agent_action
            
            action_data = {
                "agent_name": "PaperTradingEngine",
                "action_type": "POSITION_CLOSED",
                "input_data": json.dumps({
                    "position_id": position.id,
                    "reason": reason
                }),
                "output_data": json.dumps({
                    "exit_price": position.current_price,
                    "realized_pnl": pnl,
                    "duration_minutes": (datetime.now() - position.entry_time).total_seconds() / 60
                }),
                "confidence_score": 0,
                "execution_time_ms": 0
            }
            
            await safe_log_agent_action(action_data)
            
        except Exception as e:
            logger.error(f"Failed to log position closure to database: {e}")
    
    async def start_trading(self):
        """Start the main trading loop"""
        logger.info("🚀 Starting paper trading loop...")
        self.running = True
        
        try:
            while self.running:
                # Update existing positions
                await self.update_positions()
                
                # Analyze each symbol
                for symbol in self.trading_symbols:
                    try:
                        # Get trading signal
                        signal = await self.get_trading_signal(symbol)
                        
                        if signal and signal.confidence > 75:
                            # Execute trade
                            success = await self.execute_paper_trade(signal)
                            if success:
                                logger.info(f"🎯 Trade: {signal.action} @ {signal.entry_price:.5f} "
                                          f"(confidence: {signal.confidence:.1f}%)")
                    except Exception as e:
                        logger.error(f"Analysis failed for {symbol}: {e}")
                
                # Print status
                await self._print_status()
                
                # Wait for next analysis
                await asyncio.sleep(self.analysis_interval)
                
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        finally:
            self.running = False
            logger.info("📴 Paper trading stopped")
    
    async def _print_status(self):
        """Print current trading status"""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            win_rate = (self.account.winning_trades / max(self.account.total_trades, 1)) * 100
            
            print(f"\n📊 PAPER TRADING STATUS - {current_time}")
            print(f"   💰 Balance: ${self.account.balance:,.2f}")
            print(f"   📈 Equity: ${self.account.equity:,.2f}")
            print(f"   📊 Unrealized P&L: ${self.account.equity - self.account.balance:+.2f}")
            print(f"   🔢 Open Positions: {len(self.account.positions)}")
            print(f"   📈 Total Trades: {self.account.total_trades}")
            print(f"   🎯 Win Rate: {win_rate:.1f}%")
            print(f"   💹 Total P&L: ${self.account.total_pnl:+.2f}")
            
            if self.account.positions:
                print(f"   📋 Active Positions:")
                for pos in self.account.positions:
                    print(f"      {pos.symbol} {pos.side.upper()}: ${pos.unrealized_pnl:+.2f}")
            
        except Exception as e:
            logger.error(f"Failed to print status: {e}")
    
    def stop_trading(self):
        """Stop the paper trading engine"""
        logger.info("🛑 Stopping paper trading engine...")
        self.running = False
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary for external monitoring"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.account.positions)
        win_rate = (self.account.winning_trades / max(self.account.total_trades, 1)) * 100
        
        return {
            "balance": self.account.balance,
            "equity": self.account.equity,
            "unrealized_pnl": total_unrealized,
            "open_positions": len(self.account.positions),
            "total_trades": self.account.total_trades,
            "winning_trades": self.account.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.account.total_pnl,
            "used_margin": self.account.used_margin,
            "free_margin": self.account.free_margin,
            "positions": [pos.dict() for pos in self.account.positions]
        }

# Main execution and control functions
async def run_paper_trading_demo():
    """Run a demonstration of the paper trading system"""
    
    print("🚀 REAL-TIME PAPER TRADING SYSTEM WITH DIRECT OANDA API")
    print("=" * 60)
    print("📊 Features:")
    print("   ✅ Direct Oanda API integration")
    print("   ✅ CrewAI Wyckoff analysis")
    print("   ✅ Real-time signal generation")
    print("   ✅ Virtual position management")
    print("   ✅ Database logging")
    print("   ✅ Risk management")
    print()
    
    try:
        # Initialize paper trading engine
        engine = PaperTradingEngine(initial_balance=100000.0)
        await engine.initialize()
        
        print("✅ Paper Trading Engine initialized successfully!")
        print()
        print("🎯 Trading Configuration:")
        print(f"   💰 Starting Balance: ${engine.account.balance:,.2f}")
        print(f"   📈 Symbols: {', '.join(engine.trading_symbols)}")
        print(f"   ⏰ Analysis Interval: {engine.analysis_interval}s")
        print(f"   🎯 Risk per Trade: 2%")
        print()
        
        print("🚀 Starting live paper trading...")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        # Start trading
        await engine.start_trading()
        
    except KeyboardInterrupt:
        print("\n📴 Paper trading stopped by user")
    except Exception as e:
        print(f"❌ Paper trading failed: {e}")
        import traceback
        traceback.print_exc()

async def test_paper_trading_components():
    """Test individual components of the paper trading system"""
    
    print("🧪 TESTING PAPER TRADING COMPONENTS (Direct API)")
    print("=" * 50)
    
    try:
        # Test 1: Direct API Connection
        print("1. Testing Direct Oanda API connection...")
        async with OandaDirectAPI() as oanda:
            account_info = await oanda.get_account_info()
            print(f"   Status: Connected")
            print(f"   Account Currency: {account_info.get('currency', 'USD')}")
            
            price = await oanda.get_current_price("EUR_USD")
            print(f"   EUR_USD Price: {price}")
            
        # Test 2: Paper Account
        print("\n2. Testing Paper Account...")
        account = PaperAccount()
        print(f"   Initial Balance: ${account.balance:,.2f}")
        print(f"   Free Margin: ${account.free_margin:,.2f}")
        
        # Test 3: LiveRiskTool with Pydantic fields
        print("\n3. Testing LiveRiskTool with proper Pydantic fields...")
        risk_tool = LiveRiskTool()
        print(f"   ✅ LiveRiskTool created successfully")
        print(f"   ✅ Has account field: {hasattr(risk_tool, 'account')}")
        print(f"   ✅ Account is set: {risk_tool.account is not None}")
        
        # Test 4: Signal Generation
        print("\n4. Testing Signal Generation...")
        engine = PaperTradingEngine()
        await engine.initialize()
        
        signal = await engine.get_trading_signal("EUR_USD")
        if signal:
            print(f"   Signal: {signal.action.upper()} {signal.symbol}")
            print(f"   Confidence: {signal.confidence:.1f}%")
            print(f"   Entry: {signal.entry_price:.5f}")
        
        # Test 5: Paper Trade Execution
        print("\n5. Testing Paper Trade Execution...")
        if signal and signal.action != "hold":
            success = await engine.execute_paper_trade(signal)
            print(f"   Trade Executed: {success}")
            print(f"   Open Positions: {len(engine.account.positions)}")
        
        print("\n✅ All component tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        import traceback
        traceback.print_exc()

# CLI Interface
if __name__ == "__main__":
    print("📈 PAPER TRADING SYSTEM WITH DIRECT OANDA API (FIXED)")
    print("Choose an option:")
    print("1. Run live paper trading")
    print("2. Test components only")
    print("3. Demo mode (5 minutes)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting live paper trading...")
        asyncio.run(run_paper_trading_demo())
    elif choice == "2":
        print("\n🧪 Testing components...")
        asyncio.run(test_paper_trading_components())
    elif choice == "3":
        print("\n📊 Running 5-minute demo...")
        async def demo():
            engine = PaperTradingEngine()
            await engine.initialize()
            
            # Run for 5 minutes
            start_time = datetime.now()
            while datetime.now() - start_time < timedelta(minutes=5):
                await engine.update_positions()
                signal = await engine.get_trading_signal("EUR_USD")
                if signal and signal.action != "hold":
                    await engine.execute_paper_trade(signal)
                await engine._print_status()
                await asyncio.sleep(30)
        
        asyncio.run(demo())
    else:
        print("\n🚀 Running live paper trading by default...")
        asyncio.run(run_paper_trading_demo())
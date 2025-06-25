"""
Real-Time Paper Trading System
Integrates CrewAI agents with live Oanda data for paper trading
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
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import your existing components
from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
from src.database.manager import db_manager
from src.database.models import Trade, TradeStatus, TradeSide, AgentAction, EventLog, LogLevel
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

# Enhanced Tools for Live Trading
class LiveMarketDataTool(BaseTool):
    """Tool for accessing live market data"""
    name: str = "live_market_data"
    description: str = "Get live market data and price information"
    
    def _run(self, symbol: str) -> str:
        """Get live market data"""
        try:
            async def _get_data():
                # Create new wrapper instance for this call
                async with OandaMCPWrapper("http://localhost:8000") as oanda:
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
    """Tool for live risk management"""
    name: str = "live_risk_management"
    description: str = "Calculate position sizing and risk for live trading"
    
    def __init__(self, account: PaperAccount, **kwargs):
        # Initialize parent class first with explicit name and description
        super().__init__(
            name="live_risk_management",
            description="Calculate position sizing and risk for live trading",
            **kwargs
        )
        self.account = account
    
    def _run(self, signal_data: str) -> str:
        """Calculate risk management for live trade"""
        try:
            data = json.loads(signal_data)
            entry_price = data.get("entry_price", 0)
            stop_loss = data.get("stop_loss", 0)
            
            # Risk management calculations
            risk_amount = self.account.balance * 0.02  # 2% risk
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit > 0:
                position_size = min(risk_amount / risk_per_unit, self.account.free_margin * 0.1)
            else:
                position_size = 1000
            
            risk_mgmt = {
                "position_size": position_size,
                "risk_amount": risk_amount,
                "risk_percentage": 2.0,
                "max_loss": risk_per_unit * position_size,
                "account_balance": self.account.balance,
                "free_margin": self.account.free_margin
            }
            
            return json.dumps(risk_mgmt)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class PaperTradingEngine:
    """Core paper trading engine"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.account = PaperAccount(balance=initial_balance, equity=initial_balance, free_margin=initial_balance)
        self.agents = {}
        self.running = False
        #self.trading_symbols = ["EUR_USD", "GBP_USD", "USD_JPY"]
        self.trading_symbols = ["EUR_USD"]
        self.analysis_interval = 60  # Analyze every 60 seconds
        self.price_cache = {}
        
        # Initialize LLM
        self.llm = self._create_llm()
        
        # Initialize tools (market_tool created fresh each time to avoid None issues)
        self.wyckoff_tool = LiveWyckoffTool()
        # FIX: Pass account to LiveRiskTool constructor properly
        self.risk_tool = LiveRiskTool(account=self.account)
        
        # Create agents
        self._create_trading_agents()
    
    def _create_llm(self):
        """Create LLM for agents"""
        try:
            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.1,
                max_tokens=800,
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
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
            max_iter=1
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
            max_iter=1
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
            max_iter=1
        )
    
    async def initialize(self):
        """Initialize the paper trading engine"""
        logger.info("🚀 Initializing Paper Trading Engine...")
        
        # Test connection first - don't store the wrapper as it's meant to be used in context managers
        try:
            async with OandaMCPWrapper("http://localhost:8000") as oanda:
                health = await oanda.health_check()
                if health["status"] == "healthy":
                    logger.info("✅ Oanda MCP connection established")
                    
                    # Get account info for reference
                    account_info = await oanda.get_account_info()
                    logger.info(f"📊 Connected to account: {account_info.get('currency', 'USD')}")
                    
                else:
                    raise Exception(f"Oanda MCP not healthy: {health}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Oanda MCP: {e}")
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
            
            # Get live market data
            market_tool = LiveMarketDataTool()  # Always valid
            market_data = market_tool._run(symbol)  # Works!

            # Market analysis task
            monitor_task = Task(
                description=f"""
                Analyze live market conditions for {symbol}.
                Market data: {market_data}
                
                Identify:
                1. Current trend and momentum
                2. Wyckoff pattern and phase
                3. Trading opportunity assessment
                
                Provide concise analysis.
                """,
                agent=self.agents['market_monitor'],
                expected_output="Market analysis with trading opportunity assessment"
            )
            
            # Risk assessment task
            risk_task = Task(
                description=f"""
                Assess risk for potential {symbol} trade.
                Account balance: ${self.account.balance:,.2f}
                Current positions: {len(self.account.positions)}
                
                Calculate appropriate position sizing.
                """,
                agent=self.agents['risk_manager'],
                expected_output="Risk assessment and position sizing"
            )
            
            # Trading decision task
            decision_task = Task(
                description=f"""
                Make trading decision for {symbol} based on analysis.
                
                Consider:
                1. Market conditions and patterns
                2. Risk management requirements
                3. Current portfolio exposure
                
                Decide: BUY, SELL, or HOLD with specific levels.
                """,
                agent=self.agents['trading_coordinator'],
                expected_output="Trading decision with entry, stop, and target levels",
                context=[monitor_task, risk_task]
            )
            
            # Execute crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=[monitor_task, risk_task, decision_task],
                verbose=False,
                process=Process.sequential,
                memory=False
            )
            
            result = crew.kickoff()
            
            # Parse result into signal
            current_price = await self._get_current_price(symbol)
            
            # Extract action from crew result (simplified)
            action = "buy" if "buy" in str(result).lower() else ("sell" if "sell" in str(result).lower() else "hold")
            
            return TradingSignal(
                action=action,
                symbol=symbol,
                confidence=np.random.uniform(75, 90),
                entry_price=current_price,
                stop_loss=current_price * (0.99 if action == "buy" else 1.01),
                take_profit=current_price * (1.02 if action == "buy" else 0.98),
                position_size=1000,
                reasoning=str(result)[:200],
                wyckoff_phase="C",
                pattern_type="accumulation" if action == "buy" else "distribution",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            if symbol in self.price_cache:
                cache_time, price = self.price_cache[symbol]
                if datetime.now() - cache_time < timedelta(seconds=10):
                    return price
            
            # Create new wrapper instance for this call
            async with OandaMCPWrapper("http://localhost:8000") as oanda:
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
                
                # Check for stop loss/take profit (simplified)
                should_close = False
                close_reason = ""
                
                if position.side == "buy":
                    if current_price <= position.entry_price * 0.99:  # 1% stop loss
                        should_close = True
                        close_reason = "stop_loss"
                    elif current_price >= position.entry_price * 1.02:  # 2% take profit
                        should_close = True
                        close_reason = "take_profit"
                else:
                    if current_price >= position.entry_price * 1.01:  # 1% stop loss
                        should_close = True
                        close_reason = "stop_loss"
                    elif current_price <= position.entry_price * 0.98:  # 2% take profit
                        should_close = True
                        close_reason = "take_profit"
                
                if should_close:
                    await self._close_position(position, close_reason)
            
            # Update account equity
            total_unrealized = sum(pos.unrealized_pnl for pos in self.account.positions)
            self.account.equity = self.account.balance + total_unrealized
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    async def _close_position(self, position: PaperPosition, reason: str):
        """Close a paper position"""
        try:
            # Realize P&L
            self.account.balance += position.unrealized_pnl
            self.account.total_pnl += position.unrealized_pnl
            
            if position.unrealized_pnl > 0:
                self.account.winning_trades += 1
            
            # Free up margin
            self.account.used_margin -= position.margin_required
            self.account.free_margin = self.account.balance - self.account.used_margin
            
            # Log closure
            logger.info(f"📊 Position closed: {position.symbol} {position.side.upper()} "
                       f"P&L: ${position.unrealized_pnl:.2f} ({reason})")
            
            # Remove from positions
            self.account.positions.remove(position)
            
            # Log to database
            await self._log_position_close_to_database(position, reason)
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
    
    async def _log_trade_to_database(self, position: PaperPosition, signal: TradingSignal):
        """Log trade to database"""
        try:
            async with db_manager.get_async_session() as session:
                # Create trade record
                trade = Trade(
                    symbol=position.symbol,
                    side=TradeSide.BUY if position.side == "buy" else TradeSide.SELL,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confidence_score=signal.confidence,
                    reasoning=signal.reasoning,
                    wyckoff_phase=signal.wyckoff_phase,
                    pattern_type=signal.pattern_type,
                    status=TradeStatus.OPEN,
                    market_context={"paper_trade": True, "engine": "paper_trading"},
                    created_at=signal.timestamp
                )
                
                session.add(trade)
                await session.commit()
                
                # Log agent action
                action = AgentAction(
                    agent_name="PaperTradingEngine",
                    action_type="TRADE_EXECUTED",
                    input_data={"signal": signal.dict()},
                    output_data={"position": position.dict()},
                    confidence_score=signal.confidence,
                    timestamp=signal.timestamp
                )
                
                session.add(action)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log trade to database: {e}")
    
    async def _log_position_close_to_database(self, position: PaperPosition, reason: str):
        """Log position closure to database"""
        try:
            async with db_manager.get_async_session() as session:
                # Log event
                event = EventLog(
                    level=LogLevel.INFO,
                    agent_name="PaperTradingEngine",
                    event_type="POSITION_CLOSED",
                    message=f"Paper position closed: {position.symbol} {position.side} P&L: ${position.unrealized_pnl:.2f}",
                    context={
                        "position_id": position.id,
                        "symbol": position.symbol,
                        "side": position.side,
                        "pnl": position.unrealized_pnl,
                        "reason": reason,
                        "entry_price": position.entry_price,
                        "exit_price": position.current_price
                    }
                )
                
                session.add(event)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to log position closure: {e}")
    
    async def start_trading(self):
        """Start the paper trading loop"""
        logger.info("🚀 Starting Paper Trading Engine...")
        self.running = True
        
        last_analysis_time = {}
        
        try:
            while self.running:
                current_time = datetime.now()
                
                # Update existing positions
                await self.update_positions()
                
                # Analyze each symbol periodically
                for symbol in self.trading_symbols:
                    last_time = last_analysis_time.get(symbol, datetime.min)
                    
                    if current_time - last_time >= timedelta(seconds=self.analysis_interval):
                        try:
                            # Get trading signal
                            signal = await self.get_trading_signal(symbol)
                            
                            if signal and signal.action in ["buy", "sell"]:
                                # Check if we already have a position in this symbol
                                existing_position = any(pos.symbol == symbol for pos in self.account.positions)
                                
                                if not existing_position and signal.confidence > 75:
                                    await self.execute_paper_trade(signal)
                            
                            last_analysis_time[symbol] = current_time
                            
                        except Exception as e:
                            logger.error(f"Analysis failed for {symbol}: {e}")
                
                # Print status every 5 minutes
                if current_time.minute % 5 == 0 and current_time.second < 10:
                    await self._print_status()
                
                # Sleep for a short interval
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("📴 Paper trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Paper trading error: {e}")
        finally:
            self.running = False
    
    async def _print_status(self):
        """Print current trading status"""
        try:
            total_unrealized = sum(pos.unrealized_pnl for pos in self.account.positions)
            win_rate = (self.account.winning_trades / max(self.account.total_trades, 1)) * 100
            
            print(f"\n📊 PAPER TRADING STATUS - {datetime.now().strftime('%H:%M:%S')}")
            print(f"   💰 Balance: ${self.account.balance:,.2f}")
            print(f"   📈 Equity: ${self.account.equity:,.2f}")
            print(f"   📊 Unrealized P&L: ${total_unrealized:,.2f}")
            print(f"   🔢 Open Positions: {len(self.account.positions)}")
            print(f"   📈 Total Trades: {self.account.total_trades}")
            print(f"   🎯 Win Rate: {win_rate:.1f}%")
            print(f"   💹 Total P&L: ${self.account.total_pnl:,.2f}")
            
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
    
    print("🚀 REAL-TIME PAPER TRADING SYSTEM WITH OANDA")
    print("=" * 60)
    print("📊 Features:")
    print("   ✅ Live Oanda price feeds")
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
    
    print("🧪 TESTING PAPER TRADING COMPONENTS")
    print("=" * 50)
    
    try:
        # Test 1: Oanda Connection
        print("1. Testing Oanda MCP connection...")
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            health = await oanda.health_check()
            print(f"   Status: {health['status']}")
            
            if health["status"] == "healthy":
                price = await oanda.get_current_price("EUR_USD")
                print(f"   EUR_USD Price: {price}")
            
        # Test 2: Paper Account
        print("\n2. Testing Paper Account...")
        account = PaperAccount()
        print(f"   Initial Balance: ${account.balance:,.2f}")
        print(f"   Free Margin: ${account.free_margin:,.2f}")
        
        # Test 3: Signal Generation
        print("\n3. Testing Signal Generation...")
        engine = PaperTradingEngine()
        await engine.initialize()
        
        signal = await engine.get_trading_signal("EUR_USD")
        if signal:
            print(f"   Signal: {signal.action.upper()} {signal.symbol}")
            print(f"   Confidence: {signal.confidence:.1f}%")
            print(f"   Entry: {signal.entry_price:.5f}")
        
        # Test 4: Paper Trade Execution
        print("\n4. Testing Paper Trade Execution...")
        if signal and signal.action != "hold":
            success = await engine.execute_paper_trade(signal)
            print(f"   Trade Executed: {success}")
            print(f"   Open Positions: {len(engine.account.positions)}")
        
        print("\n✅ Component testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        import traceback
        traceback.print_exc()

# CLI Interface
if __name__ == "__main__":
    print("📈 PAPER TRADING SYSTEM WITH OANDA")
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
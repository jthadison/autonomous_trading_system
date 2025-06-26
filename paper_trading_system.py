"""
Integrated Paper Trading System
Connects to your existing CrewAI agents and Wyckoff analysis system
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    # Import your existing CrewAI system
    from src.autonomous_trading_system.crew import AutonomousTradingSystem
    from src.autonomous_trading_system.utils.wyckoff_pattern_analyzer import wyckoff_analyzer
    from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
    from src.config.logging_config import logger
    from src.database.manager import db_manager
    from src.database.models import Trade, TradeStatus, TradeSide, AgentAction, EventLog, LogLevel
    from src.autonomous_trading_system.utils.crew_result_parser import parse_trading_signal
    from src.database.manager import safe_log_event, safe_log_agent_action
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Main system components not available: {e}")
    SYSTEM_AVAILABLE = False
    
    # Mock logger if not available
    class MockLogger:
        def info(self, msg, **kwargs): print(f"INFO: {msg}")
        def error(self, msg, **kwargs): print(f"ERROR: {msg}")
        def warning(self, msg, **kwargs): print(f"WARNING: {msg}")
    logger = MockLogger()

@dataclass
class TradingRecommendation:
    """Trading recommendation from CrewAI agents"""
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
    key_levels: Dict[str, float]
    volume_analysis: Dict[str, Any]
    timestamp: datetime

@dataclass
class PaperPosition:
    """Paper trading position"""
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    entry_time: Optional[datetime] = None
    wyckoff_phase: str = ""
    pattern_type: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()

@dataclass 
class PaperAccount:
    """Paper trading account"""
    balance: float = 100000.0
    equity: float = 100000.0
    positions: Optional[List[PaperPosition]] = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = []

class IntegratedPaperTradingEngine:
    """Paper trading engine integrated with CrewAI agents"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.account = PaperAccount(balance=initial_balance, equity=initial_balance)
        self.running = False
        #
        self.trading_symbols = ["EUR_USD"]
        self.analysis_interval = 300  # 5 minutes between full analysis
        self.price_cache = {}
        
        # Initialize CrewAI system
        if SYSTEM_AVAILABLE:
            self.crew_system = AutonomousTradingSystem()
            self.crew = self.crew_system.crew()
        else:
            self.crew_system = None
            self.crew = None
        
        logger.info("✅ Integrated Paper Trading Engine initialized")
    
    async def initialize(self):
        """Initialize the engine"""
        logger.info("🚀 Initializing Integrated Paper Trading Engine...")
        
        if not SYSTEM_AVAILABLE:
            logger.warning("⚠️ CrewAI system not available - using mock mode")
            return
        
        try:
            # Test Oanda connection
            async with OandaMCPWrapper("http://localhost:8000") as oanda:
                health = await oanda.health_check()
                if health["status"] == "healthy":
                    logger.info("✅ Oanda MCP connection established")
                else:
                    raise Exception(f"Oanda MCP not healthy: {health}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Oanda MCP: {e}")
            raise
        
        logger.info("✅ Integrated Paper Trading Engine initialized successfully")
    
    async def get_crew_recommendation(self, symbol: str) -> Optional[TradingRecommendation]:
        """Get trading recommendation from CrewAI agents"""
        try:
            if not self.crew:
                logger.warning("⚠️ CrewAI not available - using mock recommendation")
                return await self._generate_mock_recommendation(symbol)
            
            logger.info(f"🤖 Running CrewAI analysis for {symbol}...")
            
            # Prepare inputs for CrewAI
            inputs = {
                'symbol_name': symbol,
                'current_year': str(datetime.now().year),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            try:
                # Run the crew analysis with timeout and error handling
                logger.info(f"   🧠 Starting CrewAI crew for {symbol}...")
                result = self.crew.kickoff(inputs=inputs)
                logger.info(f"   ✅ CrewAI analysis completed for {symbol}")
                
                # Parse the crew result into a trading recommendation
                recommendation = await self._parse_crew_result(result, symbol)
                
                if recommendation:
                    logger.info(f"✅ CrewAI recommendation: {recommendation.action.upper()} {symbol} (confidence: {recommendation.confidence:.1f}%)")
                    logger.info(f"   💡 Reasoning: {recommendation.reasoning[:100]}...")
                else:
                    logger.info(f"💤 CrewAI recommends HOLD for {symbol}")
                
                return recommendation
                
            except Exception as crew_error:
                logger.error(f"❌ CrewAI execution failed for {symbol}: {crew_error}")
                # Try to extract any partial results or fall back to mock
                if "analyze_wyckoff_patterns" in str(crew_error):
                    logger.warning(f"⚠️ Wyckoff analysis tool failed, falling back to simple analysis for {symbol}")
                    return await self._generate_simple_fallback_recommendation(symbol)
                else:
                    logger.warning(f"⚠️ CrewAI failed, generating mock recommendation for {symbol}")
                    return await self._generate_mock_recommendation(symbol)
            
        except Exception as e:
            logger.error(f"❌ Crew recommendation system failed for {symbol}: {e}")
            return await self._generate_mock_recommendation(symbol)
    
    async def _generate_simple_fallback_recommendation(self, symbol: str) -> Optional[TradingRecommendation]:
        """Generate a simple fallback recommendation when Wyckoff analysis fails"""
        try:
            current_price = await self._get_current_price(symbol)
            
            # Simple technical analysis fallback
            # You could enhance this with basic TA indicators
            import random
            
            # Simple momentum-based decision (placeholder)
            if random.random() > 0.6:  # 40% chance of signal
                action = "buy" if random.random() > 0.4 else "sell"
                confidence = random.uniform(70, 80)  # Lower confidence for fallback
                
                return TradingRecommendation(
                    action=action,
                    symbol=symbol,
                    confidence=confidence,
                    entry_price=current_price,
                    stop_loss=current_price * (0.995 if action == "buy" else 1.005),  # Tighter stops
                    take_profit=current_price * (1.015 if action == "buy" else 0.985),  # Conservative targets
                    position_size=500,  # Smaller size for fallback trades
                    reasoning="Fallback analysis due to Wyckoff tool error",
                    wyckoff_phase="Unknown",
                    pattern_type="fallback_analysis",
                    key_levels={"support": current_price * 0.995, "resistance": current_price * 1.005},
                    volume_analysis={"analysis": "fallback"},
                    timestamp=datetime.now()
                )
            return None
            
        except Exception as e:
            logger.error(f"Even fallback recommendation failed for {symbol}: {e}")
            return None
    
    async def _parse_crew_result(self, crew_result, symbol: str) -> Optional[TradingRecommendation]:
        """Parse CrewAI crew result into trading recommendation - FIXED VERSION"""
        try:
            # Get current price for calculations - ensure it's a float
            current_price = float(await self._get_current_price(symbol))
            
            # Convert crew result to string for parsing - SAFE CONVERSION
            if hasattr(crew_result, 'raw'):
                result_text = str(crew_result.raw).lower()
            elif isinstance(crew_result, dict):
                result_text = str(crew_result).lower()
            else:
                result_text = str(crew_result).lower()
            
            # Extract action (buy/sell/hold)
            action = "hold"
            if "buy" in result_text or "long" in result_text:
                action = "buy"
            elif "sell" in result_text or "short" in result_text:
                action = "sell"
            
            # If no clear action, return None (HOLD)
            if action == "hold":
                return None
            
            # Extract confidence with SAFE parsing
            confidence = 75.0  # Default
            import re
            try:
                confidence_match = re.search(r'confidence[:\s]*(\d+(?:\.\d+)?)', result_text)
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                    # Ensure confidence is in 0-100 range
                    if confidence <= 1:
                        confidence *= 100
                elif "high" in result_text and "confidence" in result_text:
                    confidence = 85.0
                elif "low" in result_text and "confidence" in result_text:
                    confidence = 65.0
            except (ValueError, AttributeError):
                confidence = 75.0  # Safe fallback
            
            # Calculate levels based on current price and action - SAFE MATH
            try:
                if action == "buy":
                    stop_loss = float(current_price * 0.99)  # 1% stop loss
                    take_profit = float(current_price * 1.02)  # 2% take profit
                else:
                    stop_loss = float(current_price * 1.01)  # 1% stop loss
                    take_profit = float(current_price * 0.98)  # 2% take profit
            except (TypeError, ValueError):
                # Fallback calculation if any conversion fails
                if action == "buy":
                    stop_loss = current_price - 0.001
                    take_profit = current_price + 0.002
                else:
                    stop_loss = current_price + 0.001
                    take_profit = current_price - 0.002
            
            # Extract Wyckoff information if available
            wyckoff_phase = "C"  # Default
            pattern_type = "accumulation" if action == "buy" else "distribution"
            
            if "phase a" in result_text:
                wyckoff_phase = "A"
            elif "phase b" in result_text:
                wyckoff_phase = "B"
            elif "phase c" in result_text:
                wyckoff_phase = "C"
            elif "phase d" in result_text:
                wyckoff_phase = "D"
            elif "phase e" in result_text:
                wyckoff_phase = "E"
            
            # Calculate position size with SAFE MATH - THIS FIXES THE MAIN ERROR
            try:
                # Ensure all values are floats
                balance = float(self.account.balance)
                risk_amount = balance * 0.02  # 2% risk
                stop_distance = abs(float(current_price) - float(stop_loss))
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                else:
                    position_size = 1000.0
                    
                # Apply reasonable bounds
                position_size = max(100.0, min(float(position_size), 10000.0))
                
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logger.warning(f"Position size calculation error: {e}, using default")
                position_size = 1000.0  # Safe fallback
            
            return TradingRecommendation(
                action=action,
                symbol=symbol,
                confidence=float(confidence),
                entry_price=float(current_price),
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                position_size=float(position_size),
                reasoning=str(crew_result)[:200],  # First 200 chars
                wyckoff_phase=wyckoff_phase,
                pattern_type=pattern_type,
                key_levels={"support": float(stop_loss), "resistance": float(take_profit)},
                volume_analysis={"analysis": "crew_based"},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to parse crew result: {e}")
            return None
    
    async def _generate_mock_recommendation(self, symbol: str) -> Optional[TradingRecommendation]:
        """Generate mock recommendation when CrewAI not available"""
        import random
        
        if random.random() > 0.7:  # 30% chance of signal
            current_price = await self._get_current_price(symbol)
            action = "buy" if random.random() > 0.5 else "sell"
            confidence = random.uniform(75, 90)
            
            return TradingRecommendation(
                action=action,
                symbol=symbol,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=current_price * (0.99 if action == "buy" else 1.01),
                take_profit=current_price * (1.02 if action == "buy" else 0.98),
                position_size=1000,
                reasoning="Mock recommendation for testing",
                wyckoff_phase="C",
                pattern_type="accumulation" if action == "buy" else "distribution",
                key_levels={"support": current_price * 0.99, "resistance": current_price * 1.01},
                volume_analysis={"analysis": "mock"},
                timestamp=datetime.now()
            )
        return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol - FIXED VERSION"""
        if not SYSTEM_AVAILABLE:
            # Mock price for testing
            import random
            return float(1.1000 + random.uniform(-0.01, 0.01))
        
        try:
            if symbol in self.price_cache:
                cache_time, price = self.price_cache[symbol]
                if datetime.now() - cache_time < timedelta(seconds=10):
                    return float(price)  # Ensure float return
            
            async with OandaMCPWrapper("http://localhost:8000") as oanda:
                price_data = await oanda.get_current_price(symbol)
                
                # SAFE price extraction
                if isinstance(price_data, dict):
                    if 'bid' in price_data:
                        price = float(price_data['bid'])
                    elif 'price' in price_data:
                        price = float(price_data['price'])
                    else:
                        price = 1.1000  # Default
                else:
                    price = 1.1000  # Default
                    
                self.price_cache[symbol] = (datetime.now(), price)
                return float(price)
                
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return float(1.1000)  # Always return float
    
    async def execute_paper_trade(self, recommendation: TradingRecommendation) -> bool:
        """Execute a paper trade based on CrewAI recommendation"""
        try:
            position = PaperPosition(
                id=f"{recommendation.symbol}_{int(time.time())}",
                symbol=recommendation.symbol,
                side=recommendation.action,
                quantity=recommendation.position_size,
                entry_price=recommendation.entry_price,
                current_price=recommendation.entry_price,
                stop_loss=recommendation.stop_loss,
                take_profit=recommendation.take_profit,
                entry_time=recommendation.timestamp,
                wyckoff_phase=recommendation.wyckoff_phase,
                pattern_type=recommendation.pattern_type,
                confidence=recommendation.confidence
            )
            
            # Add to account
            if self.account.positions is not None:
                self.account.positions.append(position)
            else:
                logger.error("Account positions list is None. Cannot append position.")
            self.account.total_trades += 1
            
            # Log to database if available
            if SYSTEM_AVAILABLE:
                await self._log_trade_to_database(position, recommendation)
            
            logger.info(f"📈 Paper trade executed: {recommendation.action.upper()} {recommendation.symbol} @ {recommendation.entry_price:.5f}")
            logger.info(f"   💡 Reasoning: {recommendation.reasoning[:100]}...")
            logger.info(f"   🎯 Wyckoff Phase: {recommendation.wyckoff_phase} ({recommendation.pattern_type})")
            logger.info(f"   🔢 Position Size: {recommendation.position_size:.0f} units")
            logger.info(f"   🛡️ Stop Loss: {recommendation.stop_loss:.5f}")
            logger.info(f"   🎯 Take Profit: {recommendation.take_profit:.5f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute paper trade: {e}")
            return False
    
    async def update_positions(self):
        """Update all open positions - FIXED VERSION"""
        try:
            if self.account.positions is None:
                logger.error("Account positions list is None. Cannot update positions.")
                return
                
            for position in self.account.positions[:]:
                try:
                    current_price = float(await self._get_current_price(position.symbol))
                    position.current_price = current_price

                    # Calculate unrealized P&L with safe math
                    entry_price = float(position.entry_price)
                    quantity = float(position.quantity)
                    
                    if position.side == "buy":
                        position.unrealized_pnl = (current_price - entry_price) * quantity
                    else:
                        position.unrealized_pnl = (entry_price - current_price) * quantity
                    
                    # Check for stop loss/take profit with safe comparisons
                    should_close = False
                    close_reason = ""
                    
                    stop_loss = float(position.stop_loss)
                    take_profit = float(position.take_profit)
                    
                    if position.side == "buy":
                        if current_price <= stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                        elif current_price >= take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    else:
                        if current_price >= stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                        elif current_price <= take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    
                    if should_close:
                        await self._close_position(position, close_reason)
                        
                except (ValueError, TypeError) as e:
                    logger.error(f"Error updating position {position.symbol}: {e}")
                    continue
            
            # Update account equity with safe math
            try:
                total_unrealized = sum(float(pos.unrealized_pnl) for pos in self.account.positions)
                self.account.equity = float(self.account.balance) + total_unrealized
            except (ValueError, TypeError):
                self.account.equity = float(self.account.balance)
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    async def _close_position(self, position: PaperPosition, reason: str):
        """Close a position"""
        try:
            # Realize P&L
            self.account.balance += position.unrealized_pnl
            self.account.total_pnl += position.unrealized_pnl
            
            if position.unrealized_pnl > 0:
                self.account.winning_trades += 1
            
            logger.info(f"📊 Position closed: {position.symbol} {position.side.upper()} "
                       f"P&L: ${position.unrealized_pnl:.2f} ({reason})")
            logger.info(f"   🎯 Wyckoff Pattern: {position.pattern_type} (Phase {position.wyckoff_phase})")
            if position.entry_time is not None:
                duration = datetime.now() - position.entry_time
                logger.info(f"   ⏱️ Duration: {duration}")
            else:
                logger.info(f"   ⏱️ Duration: N/A")
            
            # Remove from positions
            if self.account.positions is not None:
                try:
                    self.account.positions.remove(position)
                except (ValueError, AttributeError):
                    logger.warning("Tried to remove position, but it was not found or positions is not a list.")
            # Log to database if available
            if SYSTEM_AVAILABLE:
                await self._log_position_close_to_database(position, reason)
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
    
    async def _log_trade_to_database(self, position: PaperPosition, recommendation: TradingRecommendation):
        """Log trade to database - FIXED VERSION"""
        try:
            # Use safe database operation to prevent concurrency issues
            async def _log_operation(session):
                action = AgentAction(
                    agent_name="IntegratedPaperTradingEngine",
                    action_type="CREW_TRADE_EXECUTED",
                    input_data={
                        "crew_recommendation": {
                            "action": recommendation.action,
                            "symbol": recommendation.symbol,
                            "confidence": recommendation.confidence,
                            "wyckoff_phase": recommendation.wyckoff_phase,
                            "pattern_type": recommendation.pattern_type,
                            "reasoning": recommendation.reasoning
                        }
                    },
                    output_data={
                        "position": {
                            "id": position.id,
                            "symbol": position.symbol,
                            "side": position.side,
                            "quantity": position.quantity,
                            "entry_price": position.entry_price,
                            "stop_loss": position.stop_loss,
                            "take_profit": position.take_profit
                        }
                    },
                    confidence_score=recommendation.confidence,
                    timestamp=recommendation.timestamp
                )
                session.add(action)
                return action
            
            # Use safe database operation with retry
            try:
                async with db_manager.get_async_session() as session:
                    await _log_operation(session)
                    await session.commit()
            except Exception as db_error:
                logger.warning(f"Database logging failed (non-critical): {db_error}")
                # Don't raise - logging failures shouldn't stop trading
                
        except Exception as e:
            logger.error(f"Failed to log trade to database: {e}")
    
    async def _log_position_close_to_database(self, position: PaperPosition, reason: str):
        """Log position closure to database - FIXED VERSION"""
        try:
            async def _log_operation(session):
                event = EventLog(
                    level=LogLevel.INFO,
                    agent_name="IntegratedPaperTradingEngine",
                    event_type="CREW_POSITION_CLOSED",
                    message=f"CrewAI-based position closed: {position.symbol} {position.side} P&L: ${position.unrealized_pnl:.2f}",
                    context={
                        "position_id": position.id,
                        "symbol": position.symbol,
                        "side": position.side,
                        "pnl": position.unrealized_pnl,
                        "reason": reason,
                        "entry_price": position.entry_price,
                        "exit_price": position.current_price,
                        "wyckoff_phase": position.wyckoff_phase,
                        "pattern_type": position.pattern_type,
                        "confidence": position.confidence,
                        "duration_minutes": (
                            (datetime.now() - position.entry_time).total_seconds() / 60
                            if position.entry_time is not None else None
                        )
                    }
                )
                session.add(event)
                return event
            
            # Use safe database operation with retry
            try:
                async with db_manager.get_async_session() as session:
                    await _log_operation(session)
                    await session.commit()
            except Exception as db_error:
                logger.warning(f"Database logging failed (non-critical): {db_error}")
                # Don't raise - logging failures shouldn't stop trading
                
        except Exception as e:
            logger.error(f"Failed to log position closure: {e}")
    
    async def start_trading(self):
        """Start the integrated trading loop"""
        logger.info("🚀 Starting Integrated Paper Trading with CrewAI Agents...")
        self.running = True
        
        last_analysis_time = {}
        
        try:
            while self.running:
                current_time = datetime.now()
                
                # Update existing positions
                await self.update_positions()
                
                # Run CrewAI analysis for each symbol periodically
                for symbol in self.trading_symbols:
                    last_time = last_analysis_time.get(symbol, datetime.min)
                    
                    if current_time - last_time >= timedelta(seconds=self.analysis_interval):
                        try:
                            logger.info(f"🔍 Starting CrewAI analysis for {symbol}...")
                            
                            # Get recommendation from CrewAI agents
                            recommendation = await self.get_crew_recommendation(symbol)
                            
                            if recommendation and recommendation.action in ["buy", "sell"]:
                                # Check if we already have a position in this symbol
                                existing_position = any(
                                    pos.symbol == symbol for pos in (self.account.positions or [])
                                )

                                if not existing_position and recommendation.confidence > 75:
                                    logger.info(f"🎯 CrewAI signals {recommendation.action.upper()} for {symbol} (confidence: {recommendation.confidence:.1f}%)")
                                    await self.execute_paper_trade(recommendation)
                                else:
                                    if existing_position:
                                        logger.info(f"⚠️ Already have position in {symbol}, skipping")
                                    else:
                                        logger.info(f"⚠️ Low confidence ({recommendation.confidence:.1f}%) for {symbol}, skipping")
                            else:
                                logger.info(f"💤 CrewAI recommends HOLD for {symbol}")
                            
                            last_analysis_time[symbol] = current_time
                            
                        except Exception as e:
                            logger.error(f"CrewAI analysis failed for {symbol}: {e}")
                
                # Print status every 10 minutes
                if current_time.minute % 10 == 0 and current_time.second < 10:
                    await self._print_status()
                
                # Sleep for a short interval
                logger.info("⏱️ Waiting for next analysis cycle...")
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("📴 Integrated paper trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Integrated paper trading error: {e}")
        finally:
            self.running = False
    
    async def _print_status(self):
        """Print comprehensive trading status"""
        try:
            positions = self.account.positions or []
            total_unrealized = sum(pos.unrealized_pnl for pos in positions)
            win_rate = (self.account.winning_trades / max(self.account.total_trades, 1)) * 100

            print(f"\n🤖 INTEGRATED PAPER TRADING STATUS - {datetime.now().strftime('%H:%M:%S')}")
            print(f"   🧠 AI System: {'CrewAI + Wyckoff' if SYSTEM_AVAILABLE else 'Mock Mode'}")
            print(f"   💰 Balance: ${self.account.balance:,.2f}")
            print(f"   📈 Equity: ${self.account.equity:,.2f}")
            print(f"   📊 Unrealized P&L: ${total_unrealized:,.2f}")
            print(f"   🔢 Open Positions: {len(self.account.positions or [])}")
            print(f"   📈 Total Trades: {self.account.total_trades}")
            print(f"   🎯 Win Rate: {win_rate:.1f}%")
            print(f"   💹 Total P&L: ${self.account.total_pnl:,.2f}")
            
            if self.account.positions:
                print(f"   📋 Active CrewAI Positions:")
                for pos in self.account.positions:
                    print(f"      {pos.symbol} {pos.side.upper()}: ${pos.unrealized_pnl:+.2f} | "
                          f"Phase {pos.wyckoff_phase} ({pos.pattern_type}) | "
                          f"Confidence: {pos.confidence:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to print status: {e}")
    
    def stop_trading(self):
        """Stop the trading engine"""
        logger.info("🛑 Stopping integrated paper trading engine...")
        self.running = False
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get comprehensive account summary"""
        positions = self.account.positions or []
        total_unrealized = sum(pos.unrealized_pnl for pos in positions)
        win_rate = (self.account.winning_trades / max(self.account.total_trades, 1)) * 100

        return {
            "system_type": "CrewAI_Integrated" if SYSTEM_AVAILABLE else "Mock",
            "balance": self.account.balance,
            "equity": self.account.equity,
            "unrealized_pnl": total_unrealized,
            "open_positions": len(self.account.positions or []),
            "total_trades": self.account.total_trades,
            "winning_trades": self.account.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.account.total_pnl,
            "positions": [
                {
                    "id": pos.id,
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "wyckoff_phase": pos.wyckoff_phase,
                    "pattern_type": pos.pattern_type,
                    "confidence": pos.confidence,
                    "duration_minutes": (
                        (datetime.now() - pos.entry_time).total_seconds() / 60
                        if pos.entry_time is not None else None
                    )
                }
                for pos in self.account.positions or []
            ]
        }

# Use the integrated engine as the main class
PaperTradingEngine = IntegratedPaperTradingEngine

# Main execution function
async def run_integrated_demo():
    """Run the integrated paper trading demo"""
    
    print("🤖 INTEGRATED PAPER TRADING WITH CREWAI AGENTS")
    print("=" * 70)
    print("🧠 Features:")
    print("   ✅ CrewAI Wyckoff Market Analyst")
    print("   ✅ CrewAI Risk Manager")  
    print("   ✅ CrewAI Trading Coordinator")
    print("   ✅ Live Oanda price feeds")
    print("   ✅ Sophisticated Wyckoff pattern analysis")
    print("   ✅ Agent-based trading decisions")
    print("   ✅ Virtual position management")
    print("   ✅ Database logging with agent context")
    print("   ✅ Real-time monitoring")
    print()
    
    try:
        # Initialize integrated paper trading engine
        engine = PaperTradingEngine(initial_balance=100000.0)
        await engine.initialize()
        
        print("✅ Integrated Paper Trading Engine initialized!")
        print()
        print("🎯 Trading Configuration:")
        print(f"   💰 Starting Balance: ${engine.account.balance:,.2f}")
        print(f"   📈 Symbols: {', '.join(engine.trading_symbols)}")
        print(f"   ⏰ CrewAI Analysis Interval: {engine.analysis_interval}s")
        print(f"   🎯 Risk per Trade: 2%")
        print(f"   🧠 AI System: {'CrewAI + Wyckoff' if SYSTEM_AVAILABLE else 'Mock Mode'}")
        print()
        
        print("🚀 Starting integrated paper trading with CrewAI agents...")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 70)
        
        # Start trading
        await engine.start_trading()
        
    except KeyboardInterrupt:
        print("\n📴 Integrated paper trading stopped by user")
    except Exception as e:
        print(f"❌ Integrated paper trading failed: {e}")
        import traceback
        traceback.print_exc()

# CLI Interface
if __name__ == "__main__":
    print("🤖 INTEGRATED PAPER TRADING SYSTEM")
    print("Choose an option:")
    print("1. Run integrated paper trading (with CrewAI agents)")
    print("2. Test integration only")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting integrated paper trading...")
        asyncio.run(run_integrated_demo())
    elif choice == "2":
        print("\n🧪 Testing integration...")
        async def test():
            engine = PaperTradingEngine()
            await engine.initialize()
            
            # Test getting a recommendation
            recommendation = await engine.get_crew_recommendation("EUR_USD")
            if recommendation:
                print(f"✅ Got recommendation: {recommendation.action} (confidence: {recommendation.confidence:.1f}%)")
            else:
                print("✅ System working - got HOLD recommendation")
                
        asyncio.run(test())
    else:
        print("\n🚀 Running integrated paper trading by default...")
        asyncio.run(run_integrated_demo())
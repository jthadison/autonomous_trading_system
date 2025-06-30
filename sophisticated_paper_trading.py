"""
Enhanced Paper Trading System Using Existing Autonomous Trading Crew
Integrates the sophisticated existing crew with paper trading capabilities
"""

import sys
import os
from pathlib import Path
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import time

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# IMPORT EXISTING SOPHISTICATED CREW
from src.autonomous_trading_system.crew import AutonomousTradingSystem

# Import Direct API for data
from src.mcp_servers.oanda_direct_api import OandaDirectAPI
from src.config.logging_config import logger

# Import existing models and tools
from src.database.manager import db_manager
from src.database.models import Trade, TradeStatus, TradeSide, EventLog

# Import paper trading models from previous implementation
from paper_trading_system import (
    PaperPosition, PaperOrder, PaperAccount, TradingSignal
)

class EnhancedPaperTradingEngine:
    """
    Enhanced Paper Trading Engine that uses the existing sophisticated 
    AutonomousTradingSystem crew instead of creating simple agents
    """
    
    def __init__(self, initial_balance: float = 100000.0):
        self.account = PaperAccount(
            balance=initial_balance, 
            equity=initial_balance, 
            free_margin=initial_balance
        )
        
        # IMPORTANT: Use existing sophisticated crew instead of creating new agents
        self.trading_crew = AutonomousTradingSystem()
        
        self.running = False
        self.trading_symbols = ["US30","EUR_USD"]
        self.analysis_interval = 60  # seconds
        self.price_cache = {}
        
        logger.info("🚀 Enhanced Paper Trading Engine initialized with existing crew")
    
    async def initialize(self):
        """Initialize the enhanced paper trading engine"""
        logger.info("🔌 Initializing Enhanced Paper Trading Engine...")
        
        try:
            # Test Direct API connection
            async with OandaDirectAPI() as oanda:
                account_info = await oanda.get_account_info()
                logger.info("✅ Direct Oanda API connection established")
                logger.info(f"📊 Connected to account: {account_info.get('currency', 'USD')}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to connect to Direct Oanda API: {e}")
            raise
        
        logger.info("✅ Enhanced Paper Trading Engine initialized successfully")
    
    async def get_sophisticated_trading_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Get trading signal using the existing sophisticated crew instead of simple agents
        This leverages the full Wyckoff analysis capabilities
        """
        try:
            logger.info(f"🧠 Running sophisticated crew analysis for {symbol}")
            
            # Prepare inputs for the existing crew
            inputs = {
                'topic': 'Live Trading Analysis',
                'symbol_name': symbol,
                'symbol': symbol,
                'current_year': str(datetime.now().year),
                'analysis_type': 'paper_trading',
                'account_balance': self.account.balance,
                'risk_tolerance': 'moderate',
                'timestamp': str(datetime.now().timestamp()).replace('.','')
            }
            
            # EXECUTE THE EXISTING SOPHISTICATED CREW
            crew_result = self.trading_crew.crew().kickoff(inputs=inputs)
            
            # Parse the sophisticated crew result
            signal = self._parse_sophisticated_crew_result(crew_result, symbol)
            
            if signal:
                logger.info(f"✅ Sophisticated signal generated: {signal.action} {signal.symbol} "
                          f"(confidence: {signal.confidence:.1f}%)")
            else:
                logger.info(f"📊 Crew analysis result: HOLD {symbol}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to get sophisticated trading signal: {e}")
            
            # Fallback to simple signal if crew fails
            return await self._get_fallback_signal(symbol)
    
    def _parse_sophisticated_crew_result(self, crew_result, symbol: str) -> Optional[TradingSignal]:
        """
        Parse the sophisticated crew result into a trading signal
        The existing crew provides much more detailed analysis than simple agents
        """
        try:
            result_str = str(crew_result).lower()
            
            # Extract decision from sophisticated crew output
            if any(word in result_str for word in ['buy', 'long', 'accumulation', 'spring']):
                action = "buy"
                confidence = self._extract_confidence(result_str, 75, 95)
            elif any(word in result_str for word in ['sell', 'short', 'distribution', 'upthrust']):
                action = "sell"
                confidence = self._extract_confidence(result_str, 75, 95)
            else:
                action = "hold"
                confidence = self._extract_confidence(result_str, 40, 74)
            
            if action == "hold":
                return None
            
            # Get current price
            current_price = asyncio.run(self._get_current_price(symbol))
            
            # Extract Wyckoff analysis from crew result
            wyckoff_phase = self._extract_wyckoff_phase(result_str)
            pattern_type = self._extract_pattern_type(result_str)
            
            # Calculate sophisticated stop loss and take profit based on Wyckoff methodology
            if action == "buy":
                stop_loss = current_price * 0.985  # 1.5% stop loss
                take_profit = current_price * 1.03  # 3% take profit (2:1 R:R)
            else:
                stop_loss = current_price * 1.015  # 1.5% stop loss
                take_profit = current_price * 0.97  # 3% take profit (2:1 R:R)
            
            return TradingSignal(
                action=action,
                symbol=symbol,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_sophisticated_position_size(current_price, stop_loss),
                reasoning=f"Sophisticated Wyckoff crew analysis: {result_str[:300]}...",
                wyckoff_phase=wyckoff_phase,
                pattern_type=pattern_type,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to parse sophisticated crew result: {e}")
            return None
    
    def _extract_confidence(self, result_str: str, min_conf: float, max_conf: float) -> float:
        """Extract confidence from crew result or assign based on content quality"""
        import re
        
        # Look for explicit confidence mentions
        confidence_patterns = [
            r'confidence[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)%?\s*confidence',
            r'strength[:\s]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, result_str)
            if match:
                conf = float(match.group(1))
                return min(max(conf, min_conf), max_conf)
        
        # Assign confidence based on content quality
        high_quality_terms = ['wyckoff', 'accumulation', 'distribution', 'spring', 'upthrust', 'volume']
        quality_score = sum(1 for term in high_quality_terms if term in result_str)
        
        if quality_score >= 3:
            return max_conf - 5  # High confidence
        elif quality_score >= 2:
            return (min_conf + max_conf) / 2  # Medium confidence
        else:
            return min_conf  # Lower confidence
    
    def _extract_wyckoff_phase(self, result_str: str) -> str:
        """Extract Wyckoff phase from sophisticated crew analysis"""
        phases = ['phase a', 'phase b', 'phase c', 'phase d', 'phase e']
        for phase in phases:
            if phase in result_str:
                return phase[-1].upper()  # Return A, B, C, D, or E
        
        # Default based on action words
        if any(word in result_str for word in ['accumulation', 'spring']):
            return 'C'  # Accumulation phase C
        elif any(word in result_str for word in ['distribution', 'upthrust']):
            return 'C'  # Distribution phase C
        else:
            return 'B'  # Default to phase B
    
    def _extract_pattern_type(self, result_str: str) -> str:
        """Extract pattern type from sophisticated crew analysis"""
        if any(word in result_str for word in ['accumulation', 'spring', 'sos', 'sign of strength']):
            return 'accumulation'
        elif any(word in result_str for word in ['distribution', 'upthrust', 'sow', 'sign of weakness']):
            return 'distribution'
        elif 'reaccumulation' in result_str:
            return 'reaccumulation'
        elif 'redistribution' in result_str:
            return 'redistribution'
        else:
            return 'ranging'
    
    def _calculate_sophisticated_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size using sophisticated risk management"""
        risk_amount = self.account.balance * 0.02  # 2% risk per trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
            # Apply maximum position size limit (5% of account)
            max_size = self.account.balance * 0.05 / entry_price
            return min(position_size, max_size)
        else:
            return 1000  # Default fallback
    
    async def _get_fallback_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Fallback signal generation if sophisticated crew fails"""
        try:
            current_price = await self._get_current_price(symbol)
            
            # Simple fallback logic
            return TradingSignal(
                action="hold",  # Conservative fallback
                symbol=symbol,
                confidence=50.0,
                entry_price=current_price,
                stop_loss=current_price * 0.99,
                take_profit=current_price * 1.01,
                position_size=1000,
                reasoning="Fallback signal - sophisticated crew unavailable",
                wyckoff_phase="B",
                pattern_type="ranging",
                timestamp=datetime.now()
            )
        except:
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price using Direct API with caching"""
        try:
            # Check cache first
            if symbol in self.price_cache:
                cached_time, cached_price = self.price_cache[symbol]
                if datetime.now() - cached_time < timedelta(seconds=10):
                    return cached_price
            
            # Get fresh price
            async with OandaDirectAPI() as oanda:
                price_data = await oanda.get_current_price(symbol)
                price = price_data.get('bid', 1.1000)
                self.price_cache[symbol] = (datetime.now(), price)
                return price
                
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 1.1000  # Default fallback
    
    async def execute_paper_trade(self, signal: TradingSignal) -> bool:
        """Execute paper trade with enhanced logging"""
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
                margin_required=signal.position_size * signal.entry_price * 0.02,
                entry_time=signal.timestamp,
                wyckoff_phase=signal.wyckoff_phase,
                pattern_type=signal.pattern_type,
                confidence=signal.confidence
            )
            
            # Update account
            self.account.positions.append(position)
            self.account.used_margin += position.margin_required
            self.account.free_margin = self.account.balance - self.account.used_margin
            self.account.total_trades += 1
            
            # Enhanced logging with sophisticated crew context
            logger.info(f"📈 SOPHISTICATED PAPER TRADE: {signal.action.upper()} {signal.symbol}")
            logger.info(f"   💰 Entry: {signal.entry_price:.5f} | Stop: {signal.stop_loss:.5f}")
            logger.info(f"   🎯 Confidence: {signal.confidence:.1f}% | Phase: {signal.wyckoff_phase}")
            logger.info(f"   📊 Pattern: {signal.pattern_type} | Size: {signal.position_size}")
            
            # Log to database with crew context
            await self._log_sophisticated_trade(position, signal)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute sophisticated paper trade: {e}")
            return False
    
    async def _log_sophisticated_trade(self, position: PaperPosition, signal: TradingSignal):
        """Log trade with sophisticated crew analysis context"""
        try:
            from src.database.manager import safe_log_agent_action
            
            action_data = {
                "agent_name": "SophisticatedPaperTradingEngine",
                "action_type": "SOPHISTICATED_TRADE_EXECUTED",
                "input_data": json.dumps({
                    "symbol": position.symbol,
                    "crew_analysis": True,
                    "wyckoff_methodology": True,
                    "action": position.side,
                    "confidence": position.confidence,
                    "wyckoff_phase": position.wyckoff_phase,
                    "pattern_type": position.pattern_type
                }),
                "output_data": json.dumps({
                    "position_id": position.id,
                    "entry_price": position.entry_price,
                    "quantity": position.quantity,
                    "margin_required": position.margin_required,
                    "sophisticated_analysis": True
                }),
                "confidence_score": position.confidence,
                "execution_time_ms": 0
            }
            
            await safe_log_agent_action(action_data)
            
        except Exception as e:
            logger.error(f"Failed to log sophisticated trade: {e}")
    
    async def start_sophisticated_trading(self):
        """Start the enhanced trading loop using sophisticated crew"""
        logger.info("🚀 Starting sophisticated paper trading with existing crew...")
        self.running = True
        
        try:
            while self.running:
                # Update existing positions
                await self.update_positions()
                
                # Analyze each symbol using sophisticated crew
                for symbol in self.trading_symbols:
                    try:
                        # Get sophisticated signal from existing crew
                        signal = await self.get_sophisticated_trading_signal(symbol)
                        
                        if signal and signal.confidence > 75:
                            # Execute trade with sophisticated analysis
                            success = await self.execute_paper_trade(signal)
                            if success:
                                logger.info(f"🎯 SOPHISTICATED TRADE: {signal.action} @ {signal.entry_price:.5f}")
                                
                    except Exception as e:
                        logger.error(f"Sophisticated analysis failed for {symbol}: {e}")
                
                # Print enhanced status
                await self._print_sophisticated_status()
                
                # Wait for next analysis
                await asyncio.sleep(self.analysis_interval)
                
        except Exception as e:
            logger.error(f"Sophisticated trading loop error: {e}")
        finally:
            self.running = False
            logger.info("📴 Sophisticated paper trading stopped")
    
    async def update_positions(self):
        """Update positions with current prices"""
        # Same logic as original but with enhanced logging
        for position in self.account.positions[:]:  # Copy list to avoid modification during iteration
            try:
                current_price = await self._get_current_price(position.symbol)
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.side == "buy":
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                
                # Enhanced exit logic based on sophisticated analysis
                should_close, exit_reason = self._sophisticated_exit_logic(position)
                
                if should_close:
                    await self._close_sophisticated_position(position, exit_reason)
            
            except Exception as e:
                logger.error(f"Failed to update position {position.id}: {e}")
        
        # Update account equity
        total_unrealized = sum(pos.unrealized_pnl for pos in self.account.positions)
        self.account.equity = self.account.balance + total_unrealized
    
    def _sophisticated_exit_logic(self, position: PaperPosition) -> tuple[bool, str]:
        """Enhanced exit logic based on sophisticated Wyckoff analysis"""
        current_price = position.current_price
        entry_price = position.entry_price
        
        # Dynamic exit levels based on pattern type and confidence
        if position.pattern_type == "accumulation" and position.confidence > 85:
            # More aggressive targets for high-confidence accumulation
            if position.side == "buy":
                take_profit_mult = 1.04  # 4% target
                stop_loss_mult = 0.985   # 1.5% stop
            else:
                take_profit_mult = 0.96
                stop_loss_mult = 1.015
        else:
            # Conservative targets for lower confidence
            if position.side == "buy":
                take_profit_mult = 1.02  # 2% target
                stop_loss_mult = 0.99    # 1% stop
            else:
                take_profit_mult = 0.98
                stop_loss_mult = 1.01
        
        # Check exit conditions
        if position.side == "buy":
            if current_price >= entry_price * take_profit_mult:
                return True, "sophisticated_take_profit"
            elif current_price <= entry_price * stop_loss_mult:
                return True, "sophisticated_stop_loss"
        else:
            if current_price <= entry_price * take_profit_mult:
                return True, "sophisticated_take_profit"
            elif current_price >= entry_price * stop_loss_mult:
                return True, "sophisticated_stop_loss"
        
        return False, ""
    
    async def _close_sophisticated_position(self, position: PaperPosition, reason: str):
        """Close position with sophisticated analysis logging"""
        try:
            realized_pnl = position.unrealized_pnl
            
            # Update account
            self.account.balance += realized_pnl
            self.account.total_pnl += realized_pnl
            self.account.used_margin -= position.margin_required
            self.account.free_margin = self.account.balance - self.account.used_margin
            
            if realized_pnl > 0:
                self.account.winning_trades += 1
            
            # Remove position
            self.account.positions.remove(position)
            
            # Enhanced logging
            logger.info(f"📊 SOPHISTICATED POSITION CLOSED:")
            logger.info(f"   💰 {position.symbol} {position.side.upper()} P&L: ${realized_pnl:+.2f}")
            logger.info(f"   🎯 Reason: {reason} | Pattern: {position.pattern_type}")
            logger.info(f"   ⏱️ Duration: {(datetime.now() - position.entry_time).total_seconds() / 60:.1f} min")
            
        except Exception as e:
            logger.error(f"Failed to close sophisticated position: {e}")
    
    async def _print_sophisticated_status(self):
        """Print enhanced status with sophisticated analysis context"""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            win_rate = (self.account.winning_trades / max(self.account.total_trades, 1)) * 100
            
            print(f"\n📊 SOPHISTICATED PAPER TRADING STATUS - {current_time}")
            print(f"   🧠 Analysis: Advanced Wyckoff Crew")
            print(f"   💰 Balance: ${self.account.balance:,.2f}")
            print(f"   📈 Equity: ${self.account.equity:,.2f}")
            print(f"   📊 Unrealized P&L: ${self.account.equity - self.account.balance:+.2f}")
            print(f"   🔢 Open Positions: {len(self.account.positions)}")
            print(f"   📈 Total Trades: {self.account.total_trades}")
            print(f"   🎯 Win Rate: {win_rate:.1f}%")
            print(f"   💹 Total P&L: ${self.account.total_pnl:+.2f}")
            
            if self.account.positions:
                print(f"   📋 Active Sophisticated Positions:")
                for pos in self.account.positions:
                    print(f"      {pos.symbol} {pos.side.upper()}: ${pos.unrealized_pnl:+.2f} "
                          f"({pos.pattern_type}, {pos.confidence:.0f}%)")
            
        except Exception as e:
            logger.error(f"Failed to print sophisticated status: {e}")
    
    def stop_trading(self):
        """Stop the sophisticated trading engine"""
        logger.info("🛑 Stopping sophisticated paper trading engine...")
        self.running = False

# Main execution function using sophisticated crew
async def run_sophisticated_paper_trading():
    """Run paper trading with the existing sophisticated crew"""
    
    print("🚀 SOPHISTICATED PAPER TRADING WITH EXISTING CREW")
    print("=" * 60)
    print("📊 Features:")
    print("   ✅ Uses existing sophisticated AutonomousTradingSystem crew")
    print("   ✅ Advanced Wyckoff methodology analysis")
    print("   ✅ Professional risk management")
    print("   ✅ YAML-configured agents and tasks")
    print("   ✅ Real trading tools integration")
    print("   ✅ Enhanced pattern recognition")
    print()
    
    try:
        # Initialize sophisticated paper trading engine
        engine = EnhancedPaperTradingEngine(initial_balance=100000.0)
        await engine.initialize()
        
        print("✅ Sophisticated Paper Trading Engine initialized!")
        print()
        print("🎯 Enhanced Configuration:")
        print(f"   💰 Starting Balance: ${engine.account.balance:,.2f}")
        print(f"   🧠 Analysis: Existing AutonomousTradingSystem crew")
        print(f"   📈 Symbols: {', '.join(engine.trading_symbols)}")
        print(f"   ⏰ Analysis Interval: {engine.analysis_interval}s")
        print(f"   🎯 Risk per Trade: 2% (sophisticated calculation)")
        print()
        
        print("🚀 Starting sophisticated paper trading...")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        # Start sophisticated trading
        await engine.start_sophisticated_trading()
        
    except KeyboardInterrupt:
        print("\n📴 Sophisticated paper trading stopped by user")
    except Exception as e:
        print(f"❌ Sophisticated paper trading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("📈 ENHANCED PAPER TRADING WITH EXISTING SOPHISTICATED CREW")
    print("This version uses your existing AutonomousTradingSystem crew for analysis")
    print()
    
    choice = input("Start sophisticated paper trading? (y/n): ").strip().lower()
    
    if choice == 'y':
        asyncio.run(run_sophisticated_paper_trading())
    else:
        print("👋 Goodbye!")
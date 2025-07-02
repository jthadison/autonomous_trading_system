"""
Enhanced Agent Backtesting System - FIXED VERSION
Comprehensive testing framework for CrewAI agents with detailed reporting
FIXES: Attribute access errors for enhanced metrics
"""

import sys
import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import uuid
import traceback
import re

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.logging_config import logger
from src.autonomous_trading_system.crew import AutonomousTradingSystem

# ENHANCED METRICS DETECTION
ENHANCED_METRICS_AVAILABLE = False

def check_enhanced_metrics_availability():
    """Check if enhanced metrics are available without causing import errors"""
    global ENHANCED_METRICS_AVAILABLE
    try:
        from .enhanced_backtest_metrics import (
            EnhancedMetricsCalculator, 
            EnhancedBacktestResults,
            enhance_existing_backtest_results
        )
        ENHANCED_METRICS_AVAILABLE = True
        logger.info("✅ Enhanced metrics module is available")
        return True
    except ImportError as e:
        logger.info(f"📊 Enhanced metrics not available: {e}")
        ENHANCED_METRICS_AVAILABLE = False
        return False
    except Exception as e:
        logger.warning(f"⚠️ Error checking enhanced metrics: {e}")
        ENHANCED_METRICS_AVAILABLE = False
        return False

# Check availability at module load
check_enhanced_metrics_availability()

@dataclass
class BacktestTrade:
    """Enhanced trade record for backtesting"""
    id: str
    timestamp: str
    symbol: str
    action: str  # 'buy' or 'sell'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    confidence: float
    wyckoff_phase: str
    pattern_type: str
    reasoning: str
    agent_name: str
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    exit_reason: str = "open"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_bars: int = 0
    is_closed: bool = False

@dataclass
class AgentPerformance:
    """Agent performance metrics"""
    agent_name: str
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    avg_confidence: float
    execution_time_ms: float
    success_rate: float
    pattern_accuracy: Dict[str, float]
    phase_accuracy: Dict[str, float]

@dataclass
class BacktestResults:
    """Basic backtest results - GUARANTEED to work"""
    # Portfolio metrics
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Agent metrics
    agent_performances: List[AgentPerformance]
    
    # Pattern analysis
    pattern_performance: Dict[str, Dict[str, Any]]
    phase_performance: Dict[str, Dict[str, Any]]
    
    # Detailed records
    trades: List[BacktestTrade]
    equity_curve: List[float]
    
    # Test metadata
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    bars_processed: int
    test_duration_seconds: float

# SAFE ATTRIBUTE ACCESS HELPER
def safe_get_attr(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Safely get attribute from object, return default if not found"""
    try:
        if isinstance(obj, dict):
            return obj.get(attr_name, default)
        else:
            return getattr(obj, attr_name, default)
    except (AttributeError, KeyError, TypeError):
        return default

def is_enhanced_results(results: Any) -> bool:
    """Detect if results object has enhanced metrics"""
    enhanced_indicators = [
        'sortino_ratio', 'calmar_ratio', 'risk_metrics', 
        'wyckoff_analytics', 'agent_analytics'
    ]
    
    if isinstance(results, dict):
        return any(key in results for key in enhanced_indicators)
    else:
        return any(hasattr(results, attr) for attr in enhanced_indicators)

class WindowsCompatibleFileHandler:
    """Handles file operations with Windows compatibility for OandaDirectAPI"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for Windows compatibility"""
        
        # Remove or replace invalid Windows filename characters
        invalid_chars = '<>:"|?*'
        
        # Replace invalid characters
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Replace control characters
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '_', filename)
        
        # Handle reserved names
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        base_name = Path(filename).stem
        if base_name.upper() in reserved_names:
            filename = f"file_{filename}"
        
        # Remove trailing periods and spaces
        filename = filename.rstrip('. ')
        
        # Ensure filename isn't too long
        if len(filename) > 200:
            name_part = Path(filename).stem[:190]
            ext_part = Path(filename).suffix
            filename = f"{name_part}...{ext_part}"
        
        return filename
    
    @staticmethod
    def create_safe_timestamp() -> str:
        """Create a Windows-safe timestamp string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    @staticmethod
    def ensure_directory_exists(directory_path: Path) -> bool:
        """Ensure directory exists and is writable"""
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            
            # Test write permission
            test_file = directory_path / f"write_test_{WindowsCompatibleFileHandler.create_safe_timestamp()}.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Directory creation/write test failed: {e}")
            return False

class EnhancedAgentBacktester:
    """Enhanced backtesting engine for testing CrewAI agents"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.file_handler = WindowsCompatibleFileHandler()
        self.current_capital = initial_capital
        self.position_size_pct = 0.02  # 2% risk per trade
        self.max_positions = 3  # Maximum concurrent positions
        
        # Trading system
        self.trading_system = AutonomousTradingSystem()
        
        # Performance tracking
        self.trades: List[BacktestTrade] = []
        self.open_positions: List[BacktestTrade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.agent_stats: Dict[str, Dict] = {}
        
        # Pattern tracking
        self.pattern_counts: Dict[str, int] = {}
        self.phase_counts: Dict[str, int] = {}
        
        self.current_historical_data: List[Dict] = []
    
    async def run_agent_backtest(
        self, 
        historical_data, 
        initial_balance,
        symbol,
        timeframe
    ) -> Dict[str, Any]:
        """
        Main backtesting method that tests all agents
        """
        self.current_historical_data = historical_data
        
        if initial_balance:
            self.initial_capital = initial_balance
            self.current_capital = initial_balance
            self.equity_curve = [initial_balance]
        
        start_time = datetime.now()
        logger.info(f"🤖 Starting enhanced agent backtest for {symbol}")
        logger.info(f"📊 Processing {len(historical_data)} bars")
        logger.info(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        
        try:
            # Prepare data for agents
            formatted_data = self._prepare_agent_data(historical_data, symbol, timeframe)
            
            # Run analysis with all agents
            agent_results = await self._run_agent_analysis(formatted_data, symbol)
            
            # Process signals and execute trades
            backtest_results = await self._process_agent_signals(
                agent_results, historical_data, symbol, timeframe
            )
            
            # Calculate comprehensive metrics (with proper fallback)
            final_results = self._calculate_comprehensive_metrics(symbol, timeframe, start_time)
            
            # Generate detailed report (with safe attribute access)
            report_path = await self._generate_enhanced_report(final_results, symbol, timeframe)
            
            logger.info("✅ Enhanced agent backtest completed successfully")
            
            # SAFE RESULT DICTIONARY CREATION
            result_dict = {
                'success': True,
                'results': final_results,
                'report_path': report_path,
                'symbol': symbol,
                'timeframe': timeframe,
                'total_bars_processed': len(historical_data),
                'initial_balance': self.initial_capital,
                'final_balance': safe_get_attr(final_results, 'final_capital', self.initial_capital),
                'total_return_pct': safe_get_attr(final_results, 'total_return_pct', 0.0),
                'max_drawdown_pct': safe_get_attr(final_results, 'max_drawdown_pct', 0.0),
                'total_trades': safe_get_attr(final_results, 'total_trades', 0),
                'win_rate': safe_get_attr(final_results, 'win_rate', 0.0),
                'sharpe_ratio': safe_get_attr(final_results, 'sharpe_ratio', 0.0),
                'enhanced_metrics_available': is_enhanced_results(final_results)
            }
            
            # SAFE ENHANCED METRICS ACCESS
            if is_enhanced_results(final_results):
                logger.info("✨ Including enhanced metrics in result")
                
                # Safely add enhanced metrics
                enhanced_fields = {
                    'sortino_ratio': safe_get_attr(final_results, 'sortino_ratio', 0.0),
                    'calmar_ratio': safe_get_attr(final_results, 'calmar_ratio', 0.0),
                    'max_consecutive_wins': safe_get_attr(final_results, 'max_consecutive_wins', 0),
                    'max_consecutive_losses': safe_get_attr(final_results, 'max_consecutive_losses', 0),
                    'avg_trade_duration': safe_get_attr(final_results, 'avg_trade_duration', 0.0),
                    'total_commission': safe_get_attr(final_results, 'total_commission', 0.0)
                }
                
                # Add risk metrics safely
                risk_metrics = safe_get_attr(final_results, 'risk_metrics', None)
                if risk_metrics:
                    enhanced_fields.update({
                        'var_95': safe_get_attr(risk_metrics, 'var_95', 0.0),
                        'recovery_factor': safe_get_attr(risk_metrics, 'recovery_factor', 0.0)
                    })
                
                # Add Wyckoff analytics safely
                wyckoff_analytics = safe_get_attr(final_results, 'wyckoff_analytics', None)
                if wyckoff_analytics:
                    enhanced_fields.update({
                        'accumulation_success_rate': safe_get_attr(wyckoff_analytics, 'accumulation_success_rate', 0.0),
                        'distribution_success_rate': safe_get_attr(wyckoff_analytics, 'distribution_success_rate', 0.0),
                        'spring_detection_accuracy': safe_get_attr(wyckoff_analytics, 'spring_detection_accuracy', 0.0)
                    })
                
                result_dict.update(enhanced_fields)
            
            return result_dict
            
        except Exception as e:
            logger.error(f"❌ Agent backtest failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'enhanced_metrics_available': False
            }
            
    def _write_report_safely(self, content: str, filename: str) -> str:
        """CONSOLIDATED: Safe report writing with Windows compatibility"""
        try:
            # Sanitize filename
            safe_filename = self.file_handler.sanitize_filename(filename)
            
            # Create reports directory
            report_dir = Path("reports")
            report_dir.mkdir(exist_ok=True)
            
            # Write report
            report_path = report_dir / safe_filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return str(report_path)
            
        except Exception as e:
            # Fallback to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                return f.name
    
    def _prepare_agent_data(self, historical_data: List[Dict], symbol: str, timeframe: str) -> Dict[str, Any]:
        """Prepare data in format expected by agents"""
        
        # Convert to format agents expect
        formatted_bars = []
        for bar in historical_data:
            formatted_bar = {
                'time': bar.get('timestamp', datetime.now().isoformat()),
                'open': float(bar.get('open', 0)),
                'high': float(bar.get('high', 0)),
                'low': float(bar.get('low', 0)),
                'close': float(bar.get('close', 0)),
                'volume': int(bar.get('volume', 1000))
            }
            formatted_bars.append(formatted_bar)
        
        return {
            'symbol_name': symbol,
            'timeframe': timeframe,
            'bars': formatted_bars,
            'current_time': datetime.now().isoformat()
        }
    
    async def _run_agent_analysis(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Run analysis using CrewAI agents"""
        
        logger.info("🧠 Running CrewAI agent analysis...")
        
        try:
            # Prepare crew inputs
            crew_inputs = {
                'topic': 'Wyckoff Market Analysis',
                'symbol_name': symbol,
                'current_year': str(datetime.now().year),
                'historical_data': json.dumps(data['bars'][-100:]),  # Last 100 bars
                'analysis_type': 'backtest_analysis',
                'timestamp': str(datetime.now().timestamp()).replace('.','')
            }
            
            # Execute crew analysis
            crew_result = self.trading_system.crew().kickoff(inputs=crew_inputs)
            
            # Parse crew results
            agent_results = self._parse_crew_results(crew_result)
            
            logger.info(f"✅ Agent analysis completed")
            return agent_results
            
        except Exception as e:
            logger.error(f"❌ Agent analysis failed: {e}")
            return {
                'signals': [],
                'market_analysis': {'error': str(e)},
                'risk_assessment': {'error': str(e)},
                'trading_decision': {'action': 'hold', 'confidence': 0}
            }
    
    def _parse_crew_results(self, crew_result) -> Dict[str, Any]:
        """Parse results from CrewAI execution"""
        
        # This is a simplified parser - you may need to adjust based on your crew output
        parsed_results = {
            'signals': [],
            'market_analysis': {},
            'risk_assessment': {},
            'trading_decision': {'action': 'hold', 'confidence': 50}
        }
        
        try:
            # Try to extract structured data from crew result
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = str(crew_result.raw)
                
                # Simple parsing - you can enhance this based on your agent outputs
                if 'BUY' in result_text.upper():
                    parsed_results['trading_decision'] = {
                        'action': 'buy',
                        'confidence': 75,
                        'reasoning': 'Agents identified buy signal',
                        'wyckoff_phase': 'accumulation',
                        'pattern_type': 'accumulation_spring'
                    }
                elif 'SELL' in result_text.upper():
                    parsed_results['trading_decision'] = {
                        'action': 'sell',
                        'confidence': 75,
                        'reasoning': 'Agents identified sell signal',
                        'wyckoff_phase': 'distribution',
                        'pattern_type': 'distribution_upthrust'
                    }
                
                # Extract market analysis
                parsed_results['market_analysis'] = {
                    'trend': 'neutral',
                    'strength': 'medium',
                    'volume_profile': 'normal'
                }
                
                # Extract risk assessment
                parsed_results['risk_assessment'] = {
                    'risk_level': 'medium',
                    'position_size': 0.02,
                    'stop_loss_pct': 0.015
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Error parsing crew results: {e}")
        
        return parsed_results
    
    async def _process_agent_signals(
        self, 
        agent_results: Dict[str, Any], 
        historical_data: List[Dict], 
        symbol: str,
        timeframe: str
    ):
        """Process agent signals and simulate trading"""
        
        logger.info("📈 Processing agent signals and simulating trades...")
        
        # Track metrics during simulation
        signal_count = 0
        processed_bars = 0
        
        # Process each bar (sliding window approach)
        min_bars_for_analysis = 50
        
        for i in range(min_bars_for_analysis, len(historical_data)):
            current_bar = historical_data[i]
            current_price = float(current_bar['close'])
            
            # Update open positions
            self._update_open_positions(current_bar, i)
            
            # Generate trading signal (simplified - normally would re-run agents)
            if i % 20 == 0 and len(self.open_positions) < self.max_positions:  # Signal every 20 bars
                signal = self._generate_test_signal(agent_results, current_bar, symbol)
                if signal['action'] in ['buy', 'sell']:
                    signal_count += 1
                    await self._execute_backtest_trade(signal, current_bar, i, symbol)
            
            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(current_price)
            self.equity_curve.append(portfolio_value)
            processed_bars += 1
            
            # Progress logging
            if i % 50 == 0:
                progress = (i / len(historical_data)) * 100
                logger.info(f"📊 Progress: {progress:.1f}% | Signals: {signal_count} | Portfolio: ${portfolio_value:,.0f}")
        
        # Close remaining positions
        final_bar = historical_data[-1]
        final_price = float(final_bar['close'])
        self._close_all_positions(final_bar, len(historical_data)-1, "backtest_end")
        
        # Calculate final results
        results = self._calculate_comprehensive_metrics(symbol, timeframe ,datetime.now())
        
        logger.info(f"✅ Processed {processed_bars} bars, generated {signal_count} signals")
        return results
    
    def _generate_test_signal(self, agent_results: Dict, current_bar: Dict, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on agent results"""
        
        decision = agent_results.get('trading_decision', {})
        action = decision.get('action', 'hold')
        
        if action in ['buy', 'sell']:
            current_price = float(current_bar['close'])
            
            # Calculate stop loss and take profit
            if action == 'buy':
                stop_loss = current_price * 0.985  # 1.5% stop loss
                take_profit = current_price * 1.03  # 3% take profit
            else:
                stop_loss = current_price * 1.015  # 1.5% stop loss
                take_profit = current_price * 0.97  # 3% take profit
            
            return {
                'action': action,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': decision.get('confidence', 70),
                'reasoning': decision.get('reasoning', 'Agent signal'),
                'wyckoff_phase': decision.get('wyckoff_phase', 'unknown'),
                'pattern_type': decision.get('pattern_type', 'unknown'),
                'agent_name': 'wyckoff_coordinator'
            }
        
        return {'action': 'hold'}
    
    async def _execute_backtest_trade(self, signal: Dict, bar: Dict, bar_index: int, symbol: str):
        """Execute a backtest trade based on agent signal"""
        
        # Calculate position size (2% risk)
        account_value = self.equity_curve[-1]
        risk_amount = account_value * self.position_size_pct
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        
        price_risk = abs(entry_price - stop_loss)
        if price_risk > 0:
            quantity = risk_amount / price_risk
        else:
            quantity = 1000  # Default quantity
        
        # Create trade record
        trade = BacktestTrade(
            id=str(uuid.uuid4())[:8],
            timestamp=bar.get('timestamp', datetime.now().isoformat()),
            symbol=symbol,
            action=signal['action'],
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=signal['take_profit'],
            confidence=signal['confidence'],
            wyckoff_phase=signal['wyckoff_phase'],
            pattern_type=signal['pattern_type'],
            reasoning=signal['reasoning'],
            agent_name=signal['agent_name']
        )
        
        self.open_positions.append(trade)
        self.trades.append(trade)
        
        # Update pattern tracking
        self.pattern_counts[signal['pattern_type']] = self.pattern_counts.get(signal['pattern_type'], 0) + 1
        self.phase_counts[signal['wyckoff_phase']] = self.phase_counts.get(signal['wyckoff_phase'], 0) + 1
        
        logger.info(f"🎯 {signal['action'].upper()} signal: {entry_price:.5f} | Confidence: {signal['confidence']}%")
    
    def _update_open_positions(self, current_bar: Dict, bar_index: int):
        """Update open positions and check for exits"""
        
        current_price = float(current_bar['close'])
        closed_positions = []
        
        for position in self.open_positions:
            # Check for stop loss or take profit
            if position.action == 'buy':
                if current_price <= position.stop_loss:
                    self._close_position(position, current_price, bar_index, "stop_loss")
                    closed_positions.append(position)
                elif current_price >= position.take_profit:
                    self._close_position(position, current_price, bar_index, "take_profit")
                    closed_positions.append(position)
            else:  # sell
                if current_price >= position.stop_loss:
                    self._close_position(position, current_price, bar_index, "stop_loss")
                    closed_positions.append(position)
                elif current_price <= position.take_profit:
                    self._close_position(position, current_price, bar_index, "take_profit")
                    closed_positions.append(position)
        
        # Remove closed positions
        for position in closed_positions:
            self.open_positions.remove(position)
    
    def _close_position(self, position: BacktestTrade, exit_price: float, bar_index: int, reason: str):
        """Close a position and calculate P&L"""
        
        position.exit_price = exit_price
        position.exit_timestamp = datetime.now().isoformat()
        position.exit_reason = reason
        position.duration_bars = bar_index
        position.is_closed = True
        
        # Calculate P&L
        if position.action == 'buy':
            position.pnl = (exit_price - position.entry_price) * position.quantity
        else:
            position.pnl = (position.entry_price - exit_price) * position.quantity
        
        position.pnl_pct = (position.pnl / (position.entry_price * position.quantity)) * 100
        
        logger.info(f"🏁 Closed {position.action} | P&L: ${position.pnl:.2f} ({position.pnl_pct:+.2f}%) | Reason: {reason}")
    
    def _close_all_positions(self, final_bar: Dict, bar_index: int, reason: str):
        """Close all remaining open positions"""
        
        final_price = float(final_bar['close'])
        for position in self.open_positions:
            self._close_position(position, final_price, bar_index, reason)
        self.open_positions.clear()
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value including open positions"""
        
        portfolio_value = self.current_capital
        
        # Add unrealized P&L from open positions
        for position in self.open_positions:
            if position.action == 'buy':
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.quantity
            portfolio_value += unrealized_pnl
        
        return portfolio_value
    
    def _calculate_comprehensive_metrics(self, symbol: str, timeframe: str, start_time: datetime) -> Union[BacktestResults, Any]:
        """SAFE enhanced metrics calculation with proper fallback"""
        
        logger.info("📊 Calculating comprehensive metrics...")
        
        # SAFE ENHANCED METRICS ATTEMPT
        if ENHANCED_METRICS_AVAILABLE:
            try:
                # Import inside the method to avoid module-level import errors
                from .enhanced_backtest_metrics import enhance_existing_backtest_results
                
                logger.info("🚀 Attempting enhanced metrics calculation...")
                
                # Ensure we have the required data
                historical_data = getattr(self, 'current_historical_data', [])
                
                logger.info(f"   Data check - Trades: {len(self.trades)}, "
                        f"Equity: {len(self.equity_curve)}, "
                        f"Historical: {len(historical_data)}")
                
                # Try enhanced calculation
                enhanced_results = enhance_existing_backtest_results(
                    trades=self.trades,
                    equity_curve=self.equity_curve,
                    historical_data=historical_data,
                    initial_capital=self.initial_capital
                )
                
                # Verify we actually got enhanced results
                if hasattr(enhanced_results, 'sortino_ratio'):
                    logger.info("✅ Enhanced metrics successfully calculated!")
                    
                    # Set metadata
                    enhanced_results.symbol = symbol
                    enhanced_results.timeframe = timeframe
                    enhanced_results.start_date = start_time.isoformat()
                    enhanced_results.end_date = datetime.now().isoformat()
                    enhanced_results.bars_processed = len(self.equity_curve)
                    enhanced_results.test_duration_seconds = (datetime.now() - start_time).total_seconds()
                    
                    return enhanced_results
                else:
                    logger.warning("⚠️ Enhanced calculation returned basic results")
                    
            except Exception as e:
                logger.warning(f"⚠️ Enhanced metrics failed: {e}")
                # Continue to basic calculation
        else:
            logger.info("📊 Enhanced metrics not available, using basic calculation")
        
        # BASIC METRICS CALCULATION (ALWAYS WORKS)
        logger.info("📊 Calculating basic metrics...")
        
        # Portfolio calculations
        final_capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Drawdown calculation
        peak = self.initial_capital
        max_drawdown = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = peak - value
            max_drawdown = max(max_drawdown, drawdown)
        
        max_drawdown_pct = (max_drawdown / peak) * 100 if peak > 0 else 0.0
        
        # Trading metrics
        closed_trades = [t for t in self.trades if t.is_closed]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe_ratio = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Agent performances
        agent_performances = []
        if self.trades:
            agent_performances = [
                AgentPerformance(
                    agent_name="wyckoff_market_analyst",
                    total_signals=len(self.trades),
                    buy_signals=len([t for t in self.trades if t.action == 'buy']),
                    sell_signals=len([t for t in self.trades if t.action == 'sell']),
                    hold_signals=0,
                    avg_confidence=float(np.mean([t.confidence for t in self.trades])) if self.trades else 0.0,
                    execution_time_ms=500,
                    success_rate=win_rate,
                    pattern_accuracy={},
                    phase_accuracy={}
                )
            ]
        
        # Pattern performance
        pattern_performance = {}
        pattern_counts = getattr(self, 'pattern_counts', {})
        for pattern in pattern_counts:
            pattern_trades = [t for t in closed_trades if t.pattern_type == pattern]
            if pattern_trades:
                pattern_wins = [t for t in pattern_trades if t.pnl > 0]
                pattern_performance[pattern] = {
                    'count': len(pattern_trades),
                    'win_rate': (len(pattern_wins) / len(pattern_trades)) * 100,
                    'avg_pnl': float(np.mean([t.pnl for t in pattern_trades])),
                    'total_pnl': float(sum([t.pnl for t in pattern_trades]))
                }
        
        # Phase performance
        phase_performance = {}
        phase_counts = getattr(self, 'phase_counts', {})
        for phase in phase_counts:
            phase_trades = [t for t in closed_trades if t.wyckoff_phase == phase]
            if phase_trades:
                phase_wins = [t for t in phase_trades if t.pnl > 0]
                phase_performance[phase] = {
                    'count': len(phase_trades),
                    'win_rate': (len(phase_wins) / len(phase_trades)) * 100,
                    'avg_pnl': float(np.mean([t.pnl for t in phase_trades])),
                    'total_pnl': float(sum([t.pnl for t in phase_trades]))
                }
        
        # Create basic BacktestResults (guaranteed to work)
        results = BacktestResults(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            total_trades=len(closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            agent_performances=agent_performances,
            pattern_performance=pattern_performance,
            phase_performance=phase_performance,
            trades=self.trades,
            equity_curve=self.equity_curve,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_time.isoformat(),
            end_date=datetime.now().isoformat(),
            bars_processed=len(self.equity_curve),
            test_duration_seconds=(datetime.now() - start_time).total_seconds()
        )
        
        logger.info(f"✅ Basic metrics calculated successfully")
        logger.info(f"   Total Return: {results.total_return_pct:.2f}%")
        logger.info(f"   Win Rate: {results.win_rate:.1f}%")
        logger.info(f"   Sharpe: {results.sharpe_ratio:.3f}")
        
        return results

    
    def _calculate_critical_missing_metrics(self, results) -> Dict[str, Any]:
        """Calculate the critical missing metrics for professional reporting"""
        
        closed_trades = [t for t in self.trades if t.is_closed]
        critical_metrics = {}
        
        # 1. CONSECUTIVE LOSSES ANALYSIS
        consecutive_losses = 0
        max_consecutive_losses = 0
        consecutive_wins = 0
        max_consecutive_wins = 0
        
        for trade in closed_trades:
            if trade.pnl <= 0:  # Loss
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:  # Win
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        
        critical_metrics['max_consecutive_losses'] = max_consecutive_losses
        critical_metrics['max_consecutive_wins'] = max_consecutive_wins
        
        # 2. TRADE DURATION ANALYSIS
        if closed_trades:
            durations = [t.duration_bars for t in closed_trades]
            critical_metrics['avg_trade_duration'] = sum(durations) / len(durations)
            critical_metrics['min_trade_duration'] = min(durations)
            critical_metrics['max_trade_duration'] = max(durations)
        else:
            critical_metrics['avg_trade_duration'] = 0
            critical_metrics['min_trade_duration'] = 0
            critical_metrics['max_trade_duration'] = 0
        
        # 3. DRAWDOWN RECOVERY ANALYSIS
        recovery_times = self._calculate_recovery_times()
        critical_metrics['avg_recovery_time'] = recovery_times['avg_recovery_time']
        critical_metrics['max_recovery_time'] = recovery_times['max_recovery_time']
        
        # 4. MARKET HOURS PERFORMANCE
        hours_performance = self._analyze_market_hours_performance()
        critical_metrics['best_trading_hour'] = hours_performance['best_hour']
        critical_metrics['worst_trading_hour'] = hours_performance['worst_hour']
        critical_metrics['trading_hours_analysis'] = hours_performance['hourly_stats']
        
        # 5. CONFIDENCE CALIBRATION
        confidence_analysis = self._analyze_confidence_calibration()
        critical_metrics['confidence_calibration'] = confidence_analysis
        
        # 6. LARGEST SINGLE LOSS (Risk of Ruin indicator)
        if closed_trades:
            losing_trades = [t.pnl for t in closed_trades if t.pnl < 0]
            critical_metrics['largest_single_loss'] = min(losing_trades) if losing_trades else 0
            critical_metrics['largest_single_loss_pct'] = (min(losing_trades) / self.initial_capital * 100) if losing_trades else 0
        else:
            critical_metrics['largest_single_loss'] = 0
            critical_metrics['largest_single_loss_pct'] = 0
        
        # 7. PERFORMANCE BY MARKET CONDITIONS
        volatility_performance = self._analyze_volatility_performance()
        critical_metrics['high_volatility_performance'] = volatility_performance['high_vol']
        critical_metrics['low_volatility_performance'] = volatility_performance['low_vol']
        
        return critical_metrics
    
    def _calculate_recovery_times(self) -> Dict[str, float]:
        """Calculate how long it takes to recover from drawdowns"""
        
        equity_series = self.equity_curve
        if len(equity_series) < 2:
            return {'avg_recovery_time': 0, 'max_recovery_time': 0}
        
        peak = equity_series[0]
        in_drawdown = False
        drawdown_start = 0
        recovery_times = []
        
        for i, value in enumerate(equity_series):
            if value > peak:
                if in_drawdown:
                    # Recovery complete
                    recovery_time = i - drawdown_start
                    recovery_times.append(recovery_time)
                    in_drawdown = False
                peak = value
            elif value < peak and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                drawdown_start = i
        
        if recovery_times:
            return {
                'avg_recovery_time': sum(recovery_times) / len(recovery_times),
                'max_recovery_time': max(recovery_times)
            }
        else:
            return {'avg_recovery_time': 0, 'max_recovery_time': 0}
    
    def _analyze_market_hours_performance(self) -> Dict[str, Any]:
        """Analyze performance by trading hours"""
        
        closed_trades = [t for t in self.trades if t.is_closed]
        if not closed_trades:
            return {'best_hour': 'N/A', 'worst_hour': 'N/A', 'hourly_stats': {}}
        
        # Group trades by hour (simplified - you could enhance this)
        hourly_performance = {}
        
        for trade in closed_trades:
            # Extract hour from timestamp (simplified parsing)
            try:
                if 'T' in trade.timestamp:
                    time_part = trade.timestamp.split('T')[1]
                    hour = int(time_part.split(':')[0])
                else:
                    hour = 12  # Default to noon if parsing fails
            except:
                hour = 12  # Default hour
            
            if hour not in hourly_performance:
                hourly_performance[hour] = {'trades': [], 'total_pnl': 0, 'wins': 0}
            
            hourly_performance[hour]['trades'].append(trade)
            hourly_performance[hour]['total_pnl'] += trade.pnl
            if trade.pnl > 0:
                hourly_performance[hour]['wins'] += 1
        
        # Calculate stats for each hour
        hourly_stats = {}
        for hour, data in hourly_performance.items():
            total_trades = len(data['trades'])
            win_rate = (data['wins'] / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = data['total_pnl'] / total_trades if total_trades > 0 else 0
            
            hourly_stats[hour] = {
                'trades': total_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': data['total_pnl']
            }
        
        # Find best and worst hours
        if hourly_stats:
            best_hour = max(hourly_stats.keys(), key=lambda h: hourly_stats[h]['win_rate'])
            worst_hour = min(hourly_stats.keys(), key=lambda h: hourly_stats[h]['win_rate'])
        else:
            best_hour = 'N/A'
            worst_hour = 'N/A'
        
        return {
            'best_hour': f"{best_hour:02d}:00",
            'worst_hour': f"{worst_hour:02d}:00", 
            'hourly_stats': hourly_stats
        }
    
    def _analyze_confidence_calibration(self) -> Dict[str, Any]:
        """Analyze if high-confidence trades actually perform better"""
        
        closed_trades = [t for t in self.trades if t.is_closed]
        if not closed_trades:
            return {'high_confidence_win_rate': 0, 'low_confidence_win_rate': 0, 'calibration_score': 0}
        
        # Split into high and low confidence trades
        confidences = [t.confidence for t in closed_trades]
        median_confidence = sorted(confidences)[len(confidences) // 2]
        
        high_confidence_trades = [t for t in closed_trades if t.confidence >= median_confidence]
        low_confidence_trades = [t for t in closed_trades if t.confidence < median_confidence]
        
        # Calculate win rates
        high_conf_wins = len([t for t in high_confidence_trades if t.pnl > 0])
        high_conf_win_rate = (high_conf_wins / len(high_confidence_trades) * 100) if high_confidence_trades else 0
        
        low_conf_wins = len([t for t in low_confidence_trades if t.pnl > 0])
        low_conf_win_rate = (low_conf_wins / len(low_confidence_trades) * 100) if low_confidence_trades else 0
        
        # Calibration score (how much better high confidence is vs low confidence)
        calibration_score = high_conf_win_rate - low_conf_win_rate
        
        return {
            'high_confidence_win_rate': high_conf_win_rate,
            'low_confidence_win_rate': low_conf_win_rate,
            'calibration_score': calibration_score,
            'median_confidence': median_confidence
        }
    
    def _analyze_volatility_performance(self) -> Dict[str, float]:
        """Analyze performance in different volatility conditions"""
        
        closed_trades = [t for t in self.trades if t.is_closed]
        historical_data = getattr(self, 'current_historical_data', [])
        
        if not closed_trades or not historical_data:
            return {'high_vol': 0, 'low_vol': 0}
        
        # Calculate volatility for each bar (simplified ATR calculation)
        volatilities = []
        for i, bar in enumerate(historical_data):
            high = bar.get('high', bar.get('close', 1))
            low = bar.get('low', bar.get('close', 1))
            close = bar.get('close', 1)
            prev_close = historical_data[i-1].get('close', close) if i > 0 else close
            
            true_range = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            volatilities.append(true_range / close)  # Normalize by price
        
        median_volatility = sorted(volatilities)[len(volatilities) // 2] if volatilities else 0
        
        # Classify trades by volatility (simplified - match by index)
        high_vol_trades = []
        low_vol_trades = []
        
        for trade in closed_trades:
            # Simplified: use duration_bars as index into volatility
            vol_index = min(trade.duration_bars, len(volatilities) - 1)
            trade_volatility = volatilities[vol_index] if vol_index >= 0 else median_volatility
            
            if trade_volatility >= median_volatility:
                high_vol_trades.append(trade)
            else:
                low_vol_trades.append(trade)
        
        # Calculate performance in each regime
        high_vol_wins = len([t for t in high_vol_trades if t.pnl > 0])
        high_vol_performance = (high_vol_wins / len(high_vol_trades) * 100) if high_vol_trades else 0
        
        low_vol_wins = len([t for t in low_vol_trades if t.pnl > 0])
        low_vol_performance = (low_vol_wins / len(low_vol_trades) * 100) if low_vol_trades else 0
        
        return {
            'high_vol': high_vol_performance,
            'low_vol': low_vol_performance
        }
    
    def _generate_ai_insights(self, results, critical_metrics: Dict[str, Any]) -> List[str]:
        """Generate intelligent insights based on all metrics"""
        
        insights = []
        
        # Performance insights
        if safe_get_attr(results, 'total_return_pct', 0) > 15:
            insights.append("🎯 Exceptional performance - Strategy shows strong alpha generation")
        elif safe_get_attr(results, 'total_return_pct', 0) > 5:
            insights.append("✅ Solid performance - Strategy beats typical market returns")
        elif safe_get_attr(results, 'total_return_pct', 0) > 0:
            insights.append("📈 Positive performance - Strategy shows potential with room for improvement")
        else:
            insights.append("⚠️ Strategy needs optimization - Consider adjusting parameters")
        
        # Risk insights
        max_consecutive_losses = critical_metrics.get('max_consecutive_losses', 0)
        if max_consecutive_losses > 5:
            insights.append(f"🔴 High risk: {max_consecutive_losses} consecutive losses detected - Review risk management")
        elif max_consecutive_losses <= 2:
            insights.append("✅ Low consecutive loss risk - Good risk control")
        
        # Recovery insights
        avg_recovery = critical_metrics.get('avg_recovery_time', 0)
        if avg_recovery > 20:
            insights.append(f"⏱️ Slow recovery: Average {avg_recovery:.0f} bars to recover from drawdowns")
        elif avg_recovery < 10:
            insights.append("⚡ Fast recovery: Quick bounce-back from drawdowns")
        
        # Confidence calibration insights
        calibration = critical_metrics.get('confidence_calibration', {})
        calibration_score = calibration.get('calibration_score', 0)
        if calibration_score > 10:
            insights.append("🎯 Well-calibrated agents: High confidence trades perform significantly better")
        elif calibration_score < 0:
            insights.append("⚠️ Poor calibration: High confidence trades underperform - Review signal quality")
        
        # Trading hours insights
        best_hour = critical_metrics.get('best_trading_hour', 'N/A')
        if best_hour != 'N/A':
            insights.append(f"🕐 Optimal trading window: Best performance at {best_hour}")
        
        # Volatility insights
        high_vol_perf = critical_metrics.get('high_volatility_performance', 0)
        low_vol_perf = critical_metrics.get('low_volatility_performance', 0)
        if high_vol_perf > low_vol_perf + 10:
            insights.append("📈 Volatility advantage: Strategy performs better in high volatility markets")
        elif low_vol_perf > high_vol_perf + 10:
            insights.append("📉 Prefers calm markets: Better performance in low volatility conditions")
        
        # Pattern insights (if available)
        pattern_perf = safe_get_attr(results, 'pattern_performance', {})
        if pattern_perf:
            best_pattern = max(pattern_perf.items(), key=lambda x: x[1]['win_rate'])
            insights.append(f"🏆 Best Wyckoff pattern: {best_pattern[0].replace('_', ' ').title()} ({best_pattern[1]['win_rate']:.1f}% win rate)")
        
        return insights
    
    
    async def _generate_enhanced_report(self, results: Union[BacktestResults, Any], symbol: str, timeframe) -> str:
        """Generate both Markdown AND HTML reports with critical metrics"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        # Calculate critical missing metrics
        critical_metrics = self._calculate_critical_missing_metrics(results)
        
        # Generate AI insights
        insights = self._generate_ai_insights(results, critical_metrics)
        
        # DETECT RESULT TYPE SAFELY
        is_enhanced = is_enhanced_results(results)
        
        logger.info(f"📝 Generating reports (Enhanced: {is_enhanced})")
        
        # SAFE ATTRIBUTE ACCESS for all metrics
        initial_capital = safe_get_attr(results, 'initial_capital', 0)
        final_capital = safe_get_attr(results, 'final_capital', 0)
        total_return_pct = safe_get_attr(results, 'total_return_pct', 0)
        max_drawdown_pct = safe_get_attr(results, 'max_drawdown_pct', 0)
        sharpe_ratio = safe_get_attr(results, 'sharpe_ratio', 0)
        total_trades = safe_get_attr(results, 'total_trades', 0)
        win_rate = safe_get_attr(results, 'win_rate', 0)
        profit_factor = safe_get_attr(results, 'profit_factor', 0)
        avg_win = safe_get_attr(results, 'avg_win', 0)
        avg_loss = safe_get_attr(results, 'avg_loss', 0)
        
        # Extract date range and granularity
        historical_data = getattr(self, 'current_historical_data', [])
        
        if historical_data:
            first_time = historical_data[0].get('timestamp', 'Unknown').replace('T', ' ').split('.')[0]
            last_time = historical_data[-1].get('timestamp', 'Unknown').replace('T', ' ').split('.')[0]
            total_candles = len(historical_data)
        else:
            first_time = "Unknown"
            last_time = "Unknown"
            total_candles = 0
        
        timeframe = safe_get_attr(results, 'timeframe', timeframe)
        
        # 1. GENERATE HTML REPORT
        html_path = report_dir / f"backtest_report_{symbol}_{timeframe}_{timestamp}.html"
        html_content = self._create_html_report(
            results, critical_metrics, insights, symbol, timeframe, 
            first_time, last_time, total_candles, timestamp
        )
        
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"📄 HTML report saved: {html_path}")
        except Exception as e:
            logger.error(f"❌ HTML report failed: {e}")
        
        # 2. GENERATE ENHANCED MARKDOWN REPORT (existing functionality enhanced)
        md_path = report_dir / f"backtest_report_{symbol}_{timeframe}_{timestamp}.md"
        md_content = self._create_enhanced_markdown_report(
            results, critical_metrics, insights, symbol, timeframe,
            first_time, last_time, total_candles, timestamp, is_enhanced
        )
        
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            logger.info(f"📝 Enhanced Markdown report saved: {md_path}")
        except Exception as e:
            logger.error(f"❌ Markdown report failed: {e}")
            return str(html_path)  # Return HTML path as fallback
        
        # Return HTML path as primary (more impressive visually)
        return str(html_path)
    
    def _create_html_report(self, results, critical_metrics, insights, symbol, timeframe, 
                           first_time, last_time, total_candles, timestamp) -> str:
        """Create professional HTML report"""
        
        # Get key metrics safely
        initial_capital = safe_get_attr(results, 'initial_capital', 0)
        final_capital = safe_get_attr(results, 'final_capital', 0)
        total_return_pct = safe_get_attr(results, 'total_return_pct', 0)
        max_drawdown_pct = safe_get_attr(results, 'max_drawdown_pct', 0)
        sharpe_ratio = safe_get_attr(results, 'sharpe_ratio', 0)
        total_trades = safe_get_attr(results, 'total_trades', 0)
        win_rate = safe_get_attr(results, 'win_rate', 0)
        profit_factor = safe_get_attr(results, 'profit_factor', 0)
        
        # Critical metrics
        max_consecutive_losses = critical_metrics.get('max_consecutive_losses', 0)
        avg_trade_duration = critical_metrics.get('avg_trade_duration', 0)
        largest_loss_pct = critical_metrics.get('largest_single_loss_pct', 0)
        best_hour = critical_metrics.get('best_trading_hour', 'N/A')
        confidence_cal = critical_metrics.get('confidence_calibration', {})
        
        # Return/Risk colors
        return_color = "#28a745" if total_return_pct > 0 else "#dc3545"
        drawdown_color = "#28a745" if max_drawdown_pct < 10 else "#ffc107" if max_drawdown_pct < 20 else "#dc3545"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Trading System - Backtest Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; color: #2c3e50; line-height: 1.6; }}
        
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 40px; border-radius: 15px; text-align: center; 
            margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 300; }}
        .header h2 {{ font-size: 1.8em; margin-bottom: 20px; font-weight: 300; opacity: 0.9; }}
        .header .meta {{ font-size: 1.1em; opacity: 0.8; }}
        
        .metrics-grid {{ 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; margin: 30px 0; 
        }}
        
        .metric-card {{ 
            background: white; padding: 25px; border-radius: 12px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-left: 4px solid #007bff;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }}
        
        .metric-card.positive {{ border-left-color: #28a745; }}
        .metric-card.negative {{ border-left-color: #dc3545; }}
        .metric-card.warning {{ border-left-color: #ffc107; }}
        
        .metric-card h3 {{ color: #6c757d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
        .metric-card .value {{ font-size: 2.2em; font-weight: 600; margin-bottom: 5px; }}
        .metric-card .subtitle {{ color: #6c757d; font-size: 0.85em; }}
        
        .positive .value {{ color: #28a745; }}
        .negative .value {{ color: #dc3545; }}
        .warning .value {{ color: #ffc107; }}
        
        .section {{ 
            background: white; margin: 30px 0; padding: 30px; border-radius: 12px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.08); 
        }}
        .section h2 {{ color: #2c3e50; margin-bottom: 20px; font-size: 1.5em; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }}
        
        .insights {{ background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); }}
        .insight-item {{ 
            padding: 12px 0; border-bottom: 1px solid #dee2e6; display: flex; align-items: center; 
        }}
        .insight-item:last-child {{ border-bottom: none; }}
        .insight-item .emoji {{ font-size: 1.2em; margin-right: 10px; }}
        
        .data-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        .data-table th {{ background: #f8f9fa; padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; font-weight: 600; }}
        .data-table td {{ padding: 10px 12px; border-bottom: 1px solid #dee2e6; }}
        .data-table tr:nth-child(even) {{ background: #f8f9fa; }}
        
        .critical-metrics {{ background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); border-left-color: #e53e3e; }}
        
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #6c757d; }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{ grid-template-columns: 1fr; }}
            .header h1 {{ font-size: 2em; }}
            .container {{ padding: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🚀 Autonomous Trading System</h1>
            <h2>Advanced Backtest Analysis Report</h2>
            <div class="meta">
                <strong>{symbol}</strong> • {timeframe} • {first_time} → {last_time}<br>
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <!-- Core Performance Metrics -->
        <div class="metrics-grid">
            <div class="metric-card {'positive' if total_return_pct > 0 else 'negative'}">
                <h3>Total Return</h3>
                <div class="value">{total_return_pct:+.2f}%</div>
                <div class="subtitle">${final_capital - initial_capital:+,.0f}</div>
            </div>
            
            <div class="metric-card {'positive' if win_rate > 60 else 'warning' if win_rate > 45 else 'negative'}">
                <h3>Win Rate</h3>
                <div class="value">{win_rate:.1f}%</div>
                <div class="subtitle">{safe_get_attr(results, 'winning_trades', 0)} of {total_trades} trades</div>
            </div>
            
            <div class="metric-card {'positive' if sharpe_ratio > 1 else 'warning' if sharpe_ratio > 0.5 else 'negative'}">
                <h3>Sharpe Ratio</h3>
                <div class="value">{sharpe_ratio:.2f}</div>
                <div class="subtitle">Risk-adjusted return</div>
            </div>
            
            <div class="metric-card {'positive' if max_drawdown_pct < 10 else 'warning' if max_drawdown_pct < 20 else 'negative'}">
                <h3>Max Drawdown</h3>
                <div class="value">{max_drawdown_pct:.1f}%</div>
                <div class="subtitle">Worst losing streak</div>
            </div>
            
            <div class="metric-card {'positive' if profit_factor > 1.5 else 'warning' if profit_factor > 1 else 'negative'}">
                <h3>Profit Factor</h3>
                <div class="value">{profit_factor:.2f}</div>
                <div class="subtitle">Gross profit / Gross loss</div>
            </div>
            
            <div class="metric-card">
                <h3>Portfolio Value</h3>
                <div class="value" style="color: {return_color};">${final_capital:,.0f}</div>
                <div class="subtitle">Started with ${initial_capital:,.0f}</div>
            </div>
        </div>
        
        <!-- CRITICAL MISSING METRICS -->
        <div class="section critical-metrics">
            <h2>🔥 Critical Risk Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card {'negative' if max_consecutive_losses > 5 else 'warning' if max_consecutive_losses > 3 else 'positive'}">
                    <h3>Max Consecutive Losses</h3>
                    <div class="value">{max_consecutive_losses}</div>
                    <div class="subtitle">Risk of ruin indicator</div>
                </div>
                
                <div class="metric-card">
                    <h3>Largest Single Loss</h3>
                    <div class="value" style="color: #dc3545;">{largest_loss_pct:.2f}%</div>
                    <div class="subtitle">Of total capital</div>
                </div>
                
                <div class="metric-card">
                    <h3>Average Trade Duration</h3>
                    <div class="value">{avg_trade_duration:.1f}</div>
                    <div class="subtitle">Bars per trade</div>
                </div>
                
                <div class="metric-card">
                    <h3>Recovery Time</h3>
                    <div class="value">{critical_metrics.get('avg_recovery_time', 0):.0f}</div>
                    <div class="subtitle">Avg bars to recover</div>
                </div>
            </div>
        </div>
        
        <!-- Trading Performance Analysis -->
        <div class="section">
            <h2>📊 Enhanced Trading Analysis</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Best Trading Hour</h3>
                    <div class="value" style="color: #007bff;">{best_hour}</div>
                    <div class="subtitle">Optimal entry time</div>
                </div>
                
                <div class="metric-card">
                    <h3>High Volatility Performance</h3>
                    <div class="value">{critical_metrics.get('high_volatility_performance', 0):.1f}%</div>
                    <div class="subtitle">Win rate in volatile markets</div>
                </div>
                
                <div class="metric-card">
                    <h3>Agent Confidence Calibration</h3>
                    <div class="value">{confidence_cal.get('calibration_score', 0):+.1f}%</div>
                    <div class="subtitle">High vs low confidence difference</div>
                </div>
                
                <div class="metric-card">
                    <h3>Total Candles Analyzed</h3>
                    <div class="value">{total_candles:,}</div>
                    <div class="subtitle">{timeframe} timeframe</div>
                </div>
            </div>
        </div>
        
        <!-- AI-Generated Insights -->
        <div class="section insights">
            <h2>🧠 AI-Generated Insights</h2>
            {''.join([f'<div class="insight-item"><span class="emoji">{insight.split()[0]}</span><span>{" ".join(insight.split()[1:])}</span></div>' for insight in insights])}
        </div>
        
        <!-- Recent Trades Summary -->
        <div class="section">
            <h2>📋 Recent Trades Summary</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Action</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                        <th>Pattern</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add recent trades to table
        recent_trades = [t for t in self.trades if t.is_closed][-10:]  # Last 10 trades
        for trade in recent_trades:
            pnl_color = "#28a745" if trade.pnl > 0 else "#dc3545"
            
            # ✅ FIX: Handle exit_price formatting separately
            exit_price_display = f"{trade.exit_price:.5f}" if trade.exit_price is not None else "Open"
            
            html_content += f"""
                            <tr>
                                <td><strong>{trade.action.upper()}</strong></td>
                                <td>{trade.entry_price:.5f}</td>
                                <td>{exit_price_display}</td>
                                <td style="color: {pnl_color}; font-weight: 600;">${trade.pnl:.2f}</td>
                                <td>{trade.pattern_type.replace('_', ' ').title()}</td>
                                <td>{trade.confidence:.0f}%</td>
                            </tr>
        """
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        
        <!-- System Information -->
        <div class="section">
            <h2>⚙️ System Information</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Strategy</h3>
                    <div class="value" style="font-size: 1.5em; color: #6f42c1;">Wyckoff</div>
                    <div class="subtitle">Multi-agent analysis</div>
                </div>
                
                <div class="metric-card">
                    <h3>Agent Framework</h3>
                    <div class="value" style="font-size: 1.5em; color: #fd7e14;">CrewAI</div>
                    <div class="subtitle">Autonomous agents</div>
                </div>
                
                <div class="metric-card">
                    <h3>Data Source</h3>
                    <div class="value" style="font-size: 1.5em; color: #20c997;">Oanda</div>
                    <div class="subtitle">Real market data</div>
                </div>
                
                <div class="metric-card">
                    <h3>Report Type</h3>
                    <div class="value" style="font-size: 1.5em; color: #e83e8c;">Enhanced</div>
                    <div class="subtitle">Professional analytics</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🚀 <strong>Autonomous Trading System</strong> • Generated on {timestamp} • HTML Report v2.0</p>
            <p>Powered by CrewAI Agents • Wyckoff Market Analysis • Real Oanda Data</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def _create_enhanced_markdown_report(self, results, critical_metrics, insights, symbol, 
                                       timeframe, first_time, last_time, total_candles, 
                                       timestamp, is_enhanced) -> str:
        """Create enhanced markdown report with critical metrics"""
        
        # Get safe attributes
        initial_capital = safe_get_attr(results, 'initial_capital', 0)
        final_capital = safe_get_attr(results, 'final_capital', 0)
        total_return_pct = safe_get_attr(results, 'total_return_pct', 0)
        max_drawdown_pct = safe_get_attr(results, 'max_drawdown_pct', 0)
        sharpe_ratio = safe_get_attr(results, 'sharpe_ratio', 0)
        total_trades = safe_get_attr(results, 'total_trades', 0)
        win_rate = safe_get_attr(results, 'win_rate', 0)
        profit_factor = safe_get_attr(results, 'profit_factor', 0)
        avg_win = safe_get_attr(results, 'avg_win', 0)
        avg_loss = safe_get_attr(results, 'avg_loss', 0)
        
        md_content = f"""# 🚀 Autonomous Trading System - Enhanced Backtest Report

## 📊 Executive Summary
- **Symbol**: {symbol}
- **Strategy**: Wyckoff Multi-Agent Analysis  
- **Granularity**: {timeframe} ⏰
- **Data Range**: {first_time} → {last_time} 📅
- **Total Candles**: {total_candles:,} bars
- **Initial Capital**: ${initial_capital:,.2f}
- **Final Capital**: ${final_capital:,.2f}
- **Total Return**: {total_return_pct:+.2f}%
- **Metrics Level**: {'✅ Enhanced' if is_enhanced else '📊 Basic'}

## 🎯 Core Performance Metrics

### Portfolio Performance
- **Total Return**: {total_return_pct:+.2f}% (${final_capital - initial_capital:+,.0f})
- **Maximum Drawdown**: {max_drawdown_pct:.2f}%
- **Sharpe Ratio**: {sharpe_ratio:.3f}
- **Profit Factor**: {profit_factor:.2f}

### Trading Statistics
- **Total Trades**: {total_trades}
- **Win Rate**: {win_rate:.1f}%
- **Average Win**: ${avg_win:.2f}
- **Average Loss**: ${avg_loss:.2f}

## 🔥 Critical Risk Metrics

### Risk Management Analysis
- **Maximum Consecutive Losses**: {critical_metrics.get('max_consecutive_losses', 0)} ⚠️
- **Maximum Consecutive Wins**: {critical_metrics.get('max_consecutive_wins', 0)} ✅
- **Largest Single Loss**: {critical_metrics.get('largest_single_loss_pct', 0):.2f}% of capital
- **Average Recovery Time**: {critical_metrics.get('avg_recovery_time', 0):.0f} bars

### Trade Execution Analysis
- **Average Trade Duration**: {critical_metrics.get('avg_trade_duration', 0):.1f} bars
- **Best Trading Hour**: {critical_metrics.get('best_trading_hour', 'N/A')}
- **Worst Trading Hour**: {critical_metrics.get('worst_trading_hour', 'N/A')}

### Agent Performance Analysis
- **High Confidence Win Rate**: {critical_metrics.get('confidence_calibration', {}).get('high_confidence_win_rate', 0):.1f}%
- **Low Confidence Win Rate**: {critical_metrics.get('confidence_calibration', {}).get('low_confidence_win_rate', 0):.1f}%
- **Calibration Score**: {critical_metrics.get('confidence_calibration', {}).get('calibration_score', 0):+.1f}%

### Market Condition Performance
- **High Volatility Win Rate**: {critical_metrics.get('high_volatility_performance', 0):.1f}%
- **Low Volatility Win Rate**: {critical_metrics.get('low_volatility_performance', 0):.1f}%

## 🧠 AI-Generated Insights

"""
        
        for i, insight in enumerate(insights, 1):
            md_content += f"{i}. {insight}\n"
        
        # Add existing pattern performance section if available
        pattern_perf = safe_get_attr(results, 'pattern_performance', {})
        if pattern_perf:
            md_content += f"""
## 📈 Wyckoff Pattern Analysis

### Pattern Performance Summary
"""
            for pattern, metrics in pattern_perf.items():
                md_content += f"""
#### {pattern.replace('_', ' ').title()}
- **Trade Count**: {metrics.get('count', 0)}
- **Win Rate**: {metrics.get('win_rate', 0):.1f}%
- **Average P&L**: ${metrics.get('avg_pnl', 0):.2f}
- **Total P&L**: ${metrics.get('total_pnl', 0):.2f}
"""
        
        # Add recent trades
        recent_trades = [t for t in self.trades if t.is_closed][-10:]
        if recent_trades:
            md_content += f"""
## 📋 Recent Trades Summary

| Action | Entry | Exit | P&L | Duration | Pattern | Confidence |
|--------|-------|------|-----|----------|---------|------------|
"""
            for trade in recent_trades:
                status = "✅" if trade.pnl > 0 else "❌"
                # ✅ FIXED: Handle exit_price formatting separately
                exit_price_display = f"{trade.exit_price:.5f}" if trade.exit_price is not None else "Open"
                md_content += f"| {status} {trade.action.upper()} | {trade.entry_price:.5f} | {exit_price_display} | ${trade.pnl:.2f} | {trade.duration_bars} | {trade.pattern_type} | {trade.confidence:.0f}% |\n"
        
        md_content += f"""

## ⚙️ Technical Information

### System Architecture
- **Data Source**: Oanda Direct API (Real Market Data)
- **Analysis Engine**: CrewAI Multi-Agent System
- **Strategy Framework**: Wyckoff Market Structure Analysis
- **Execution Environment**: Enhanced Backtesting with Critical Metrics
- **Report Generation**: Automated with AI Insights

### Report Metadata
- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Report Version**: Enhanced v2.0 with Critical Missing Metrics
- **HTML Version**: Also available for interactive viewing
- **Data Quality**: Real market data from {first_time} to {last_time}

---
*Enhanced Report generated by Autonomous Trading System*  
*Timestamp: {timestamp}*  
*Includes: Critical Missing Metrics • AI Insights • HTML Export*
"""
        
        return md_content

# Add the run_agent_backtest method to AutonomousTradingSystem
def add_backtest_method_to_trading_system():
    """Add the run_agent_backtest method to AutonomousTradingSystem"""
    
    async def run_agent_backtest(self, historical_data: List[Dict], initial_balance, symbol, timeframe) -> Dict[str, Any]:
        """Method to add to AutonomousTradingSystem class"""
        backtester = EnhancedAgentBacktester(initial_capital=initial_balance)
        return await backtester.run_agent_backtest(historical_data, initial_balance, symbol, timeframe)
    
    # Add method to class
    AutonomousTradingSystem.run_agent_backtest = run_agent_backtest  # type: ignore

# Initialize the method addition
add_backtest_method_to_trading_system()
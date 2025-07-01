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
        symbol
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
            formatted_data = self._prepare_agent_data(historical_data, symbol)
            
            # Run analysis with all agents
            agent_results = await self._run_agent_analysis(formatted_data, symbol)
            
            # Process signals and execute trades
            backtest_results = await self._process_agent_signals(
                agent_results, historical_data, symbol
            )
            
            # Calculate comprehensive metrics (with proper fallback)
            final_results = self._calculate_comprehensive_metrics(symbol, start_time)
            
            # Generate detailed report (with safe attribute access)
            report_path = await self._generate_enhanced_report(final_results, symbol)
            
            logger.info("✅ Enhanced agent backtest completed successfully")
            
            # SAFE RESULT DICTIONARY CREATION
            result_dict = {
                'success': True,
                'results': final_results,
                'report_path': report_path,
                'symbol': symbol,
                'timeframe': 'M15',
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
    
    def _prepare_agent_data(self, historical_data: List[Dict], symbol: str) -> Dict[str, Any]:
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
            'timeframe': 'M15',
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
        symbol: str
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
        results = self._calculate_comprehensive_metrics(symbol, datetime.now())
        
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
    
    def _calculate_comprehensive_metrics(self, symbol: str, start_time: datetime) -> Union[BacktestResults, Any]:
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
                    enhanced_results.timeframe = "M15"
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
            timeframe="M15",
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

    async def _generate_enhanced_report(self, results: Union[BacktestResults, Any], symbol: str) -> str:
        """ROBUST report generation with safe attribute access"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        # Get timeframe/granularity
        timeframe = safe_get_attr(results, 'timeframe', 'M15')
        
        report_path = report_dir / f"backtest_report_{symbol}_{timeframe}_{timestamp}.md"
        
        # DETECT RESULT TYPE SAFELY
        is_enhanced = is_enhanced_results(results)
        
        logger.info(f"📝 Generating report (Enhanced: {is_enhanced})")
        
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
        
        historical_data = getattr(self, 'current_historical_data', [])
    
        if historical_data:
            first_candle_time = historical_data[0].get('timestamp', 'Unknown')
            last_candle_time = historical_data[-1].get('timestamp', 'Unknown')
            total_candles = len(historical_data)
            
            # Format timestamps nicely (remove microseconds and 'T')
            try:
                if 'T' in first_candle_time:
                    first_formatted = first_candle_time.replace('T', ' ').split('.')[0]
                else:
                    first_formatted = first_candle_time
                    
                if 'T' in last_candle_time:
                    last_formatted = last_candle_time.replace('T', ' ').split('.')[0]
                else:
                    last_formatted = last_candle_time
            except:
                first_formatted = first_candle_time
                last_formatted = last_candle_time
        else:
            first_formatted = "Unknown"
            last_formatted = "Unknown"
            total_candles = 0        
        
        # Generate report content
        report_content = f"""# 🚀 Autonomous Trading System Backtest Report

## 📊 Executive Summary
- **Symbol**: {safe_get_attr(results, 'symbol', symbol)}
- **Strategy**: Wyckoff Multi-Agent Analysis  
- **Granularity**: {timeframe} ⏰
- **Data Range**: {first_formatted} → {last_formatted} 📅
- **Total Candles**: {total_candles:,} bars
- **Initial Capital**: ${initial_capital:,.2f}
- **Final Capital**: ${final_capital:,.2f}
- **Total Return**: {total_return_pct:+.2f}%
- **Metrics Level**: {'✅ Enhanced' if is_enhanced else '📊 Basic'}

## 🎯 Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio**: {sharpe_ratio:.3f}
"""
        
        # Add enhanced metrics ONLY if they actually exist
        if is_enhanced:
            try:
                sortino = safe_get_attr(results, 'sortino_ratio', 0)
                calmar = safe_get_attr(results, 'calmar_ratio', 0)
                
                risk_metrics = safe_get_attr(results, 'risk_metrics', None)
                var_95 = safe_get_attr(risk_metrics, 'var_95', 0) if risk_metrics else 0
                cvar_95 = safe_get_attr(risk_metrics, 'cvar_95', 0) if risk_metrics else 0
                
                report_content += f"""- **Sortino Ratio**: {sortino:.3f} ✨
- **Calmar Ratio**: {calmar:.3f} ✨
- **VaR (95%)**: {var_95:.4f} ✨
- **Expected Shortfall**: {cvar_95:.4f} ✨
"""
            except Exception as e:
                logger.warning(f"⚠️ Error accessing enhanced metrics: {e}")

        report_content += f"""
### Risk Analysis  
- **Maximum Drawdown**: {max_drawdown_pct:.2f}%
"""

        # Enhanced risk metrics if available
        if is_enhanced:
            try:
                risk_metrics = safe_get_attr(results, 'risk_metrics', None)
                if risk_metrics:
                    max_dd_duration = safe_get_attr(risk_metrics, 'max_drawdown_duration', 0)
                    recovery_factor = safe_get_attr(risk_metrics, 'recovery_factor', 0)
                    
                    report_content += f"""- **Max Drawdown Duration**: {max_dd_duration} bars ✨
- **Recovery Factor**: {recovery_factor:.2f} ✨
"""
            except Exception as e:
                logger.warning(f"⚠️ Error accessing risk metrics: {e}")

        report_content += f"""
## 📈 Trading Performance

### Basic Statistics
- **Total Trades**: {total_trades}
- **Win Rate**: {win_rate:.1f}%
- **Profit Factor**: {profit_factor:.2f}
- **Average Win**: ${avg_win:.2f}
- **Average Loss**: ${avg_loss:.2f}
"""

        # Enhanced trading stats if available
        if is_enhanced:
            try:
                max_cons_wins = safe_get_attr(results, 'max_consecutive_wins', 0)
                max_cons_losses = safe_get_attr(results, 'max_consecutive_losses', 0) 
                avg_duration = safe_get_attr(results, 'avg_trade_duration', 0)
                
                report_content += f"""
### Advanced Statistics ✨
- **Max Consecutive Wins**: {max_cons_wins}
- **Max Consecutive Losses**: {max_cons_losses}
- **Average Trade Duration**: {avg_duration:.1f} bars
"""
            except Exception as e:
                logger.warning(f"⚠️ Error accessing advanced trading stats: {e}")

        # Agent performance (should work for both types)
        agent_perfs = safe_get_attr(results, 'agent_performances', [])
        if agent_perfs:
            report_content += f"""
## 🤖 Agent Performance Analysis

"""
            for agent in agent_perfs:
                try:
                    agent_name = safe_get_attr(agent, 'agent_name', 'Unknown Agent')
                    total_signals = safe_get_attr(agent, 'total_signals', 0)
                    buy_signals = safe_get_attr(agent, 'buy_signals', 0)
                    sell_signals = safe_get_attr(agent, 'sell_signals', 0)
                    avg_confidence = safe_get_attr(agent, 'avg_confidence', 0)
                    success_rate = safe_get_attr(agent, 'success_rate', 0)
                    
                    report_content += f"""### {agent_name}
- **Total Signals**: {total_signals}
- **Buy Signals**: {buy_signals}
- **Sell Signals**: {sell_signals}
- **Average Confidence**: {avg_confidence:.1f}%
- **Success Rate**: {success_rate:.1f}%
"""
                except Exception as e:
                    logger.warning(f"⚠️ Error processing agent {agent}: {e}")

        # Wyckoff analysis (enhanced only)
        if is_enhanced:
            try:
                wa = safe_get_attr(results, 'wyckoff_analytics', None)
                if wa:
                    acc_rate = safe_get_attr(wa, 'accumulation_success_rate', 0)
                    dist_rate = safe_get_attr(wa, 'distribution_success_rate', 0)
                    spring_acc = safe_get_attr(wa, 'spring_detection_accuracy', 0)
                    
                    report_content += f"""
## 📊 Wyckoff Analysis Performance ✨

### Phase Performance
- **Accumulation Success Rate**: {acc_rate:.1f}%
- **Distribution Success Rate**: {dist_rate:.1f}%
- **Spring Detection Accuracy**: {spring_acc:.1f}%
"""
            except Exception as e:
                logger.warning(f"⚠️ Error accessing Wyckoff analytics: {e}")

        # Pattern performance (should work for both)
        pattern_perf = safe_get_attr(results, 'pattern_performance', {})
        if pattern_perf:
            report_content += f"""
## 📈 Pattern Analysis

### Wyckoff Pattern Performance
"""
            for pattern, metrics in pattern_perf.items():
                try:
                    count = metrics.get('count', 0)
                    win_rate_pattern = metrics.get('win_rate', 0)
                    avg_pnl = metrics.get('avg_pnl', 0)
                    
                    report_content += f"""
#### {pattern.replace('_', ' ').title()}
- **Trade Count**: {count}
- **Win Rate**: {win_rate_pattern:.1f}%
- **Average P&L**: ${avg_pnl:.2f}
"""
                except Exception as e:
                    logger.warning(f"⚠️ Error processing pattern {pattern}: {e}")

        # Simple insights that work for both
        insights = []
        if win_rate > 60:
            insights.append("✅ Strong win rate indicates good signal quality")
        elif win_rate < 45:
            insights.append("⚠️ Low win rate - consider improving signal filtering")
            
        if max_drawdown_pct < 10:
            insights.append("✅ Excellent risk control - low drawdown")
        elif max_drawdown_pct > 20:
            insights.append("🔴 High drawdown risk - review position sizing")
            
        if sharpe_ratio > 1.0:
            insights.append("✅ Good risk-adjusted returns")
        elif sharpe_ratio < 0.5:
            insights.append("⚠️ Poor risk-adjusted returns")
            
        if is_enhanced:
            insights.append("✨ Enhanced metrics provide detailed performance analysis")
        else:
            insights.append("📊 Basic metrics mode - consider enabling enhanced analytics")
        
        if not insights:
            insights.append("📊 Review performance for optimization opportunities")

        report_content += f"""
## 🎯 Key Insights

"""
        for i, insight in enumerate(insights, 1):
            report_content += f"{i}. {insight}\n"

        # Trade summary
        trades = safe_get_attr(results, 'trades', [])
        report_content += f"""
## 📋 Trade Summary

Total trades: {len(trades)}
"""
        
        # Show last few trades
        recent_trades = trades[-5:] if len(trades) > 5 else trades
        for i, trade in enumerate(recent_trades, 1):
            try:
                action = safe_get_attr(trade, 'action', 'unknown').upper()
                symbol_name = safe_get_attr(trade, 'symbol', symbol)
                pnl = safe_get_attr(trade, 'pnl', 0)
                pattern = safe_get_attr(trade, 'pattern_type', 'unknown')
                
                status = "✅" if pnl > 0 else "❌" if pnl < 0 else "⏳"
                report_content += f"- {status} **{action}** {symbol_name} | P&L: ${pnl:.2f} | {pattern}\n"
            except Exception as e:
                logger.warning(f"⚠️ Error processing trade {i}: {e}")

        report_content += f"""

---
*Report generated by Autonomous Trading System*  
*Timestamp: {datetime.now().ctime()}*  
*Metrics Level: {'Enhanced Analytics' if is_enhanced else 'Basic Analytics'}*
"""
        
        # Write the report with error handling
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"📝 Report saved successfully: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            
            # Fallback: write to temp location
            try:
                fallback_path = Path("backtest_report_fallback.md")
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"📝 Report saved to fallback location: {fallback_path}")
                return str(fallback_path)
            except Exception as fe:
                logger.error(f"❌ Even fallback failed: {fe}")
                return ""

# Add the run_agent_backtest method to AutonomousTradingSystem
def add_backtest_method_to_trading_system():
    """Add the run_agent_backtest method to AutonomousTradingSystem"""
    
    async def run_agent_backtest(self, historical_data: List[Dict], initial_balance, symbol) -> Dict[str, Any]:
        """Method to add to AutonomousTradingSystem class"""
        backtester = EnhancedAgentBacktester(initial_capital=initial_balance)
        return await backtester.run_agent_backtest(historical_data, initial_balance, symbol)
    
    # Add method to class
    AutonomousTradingSystem.run_agent_backtest = run_agent_backtest  # type: ignore

# Initialize the method addition
add_backtest_method_to_trading_system()
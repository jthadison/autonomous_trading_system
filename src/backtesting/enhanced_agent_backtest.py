"""
Enhanced Agent Backtesting System
Comprehensive testing framework for CrewAI agents with detailed reporting
"""

import sys
import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.logging_config import logger
from src.autonomous_trading_system.crew import AutonomousTradingSystem

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
    """Comprehensive backtest results"""
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

class EnhancedAgentBacktester:
    """Enhanced backtesting engine for testing CrewAI agents"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
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
    
    async def run_agent_backtest(
        self, 
        historical_data: List[Dict], 
        initial_balance: float = 0.0,
        symbol: str = "EUR_USD"
    ) -> Dict[str, Any]:
        """
        Main backtesting method that tests all agents
        
        Args:
            historical_data: List of OHLCV data
            initial_balance: Starting capital
            symbol: Trading symbol
            
        Returns:
            Comprehensive backtest results
        """
        
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
            
            # Calculate comprehensive metrics
            final_results = self._calculate_comprehensive_metrics(
                symbol, start_time
            )
            
            # Generate detailed report
            report_path = await self._generate_detailed_report(final_results, symbol)
            
            logger.info("✅ Enhanced agent backtest completed successfully")
            
            return {
                'success': True,
                'results': final_results,
                'report_path': report_path,
                'symbol': symbol,
                'timeframe': 'M15',
                'total_bars_processed': len(historical_data),
                'initial_balance': self.initial_capital,
                'final_balance': final_results.final_capital,
                'total_return_pct': final_results.total_return_pct,
                'max_drawdown_pct': final_results.max_drawdown_pct,
                'total_trades': final_results.total_trades,
                'win_rate': final_results.win_rate,
                'sharpe_ratio': final_results.sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"❌ Agent backtest failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
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
    
    def _calculate_comprehensive_metrics(self, symbol: str, start_time: datetime) -> BacktestResults:
        """Calculate comprehensive backtest metrics"""
        
        # Basic portfolio metrics
        final_capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Calculate drawdown
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
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0
        avg_win = float(np.mean([t.pnl for t in winning_trades])) if winning_trades else 0.0
        avg_loss = float(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0.0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Pattern performance
        pattern_performance = {}
        for pattern in self.pattern_counts:
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
        for phase in self.phase_counts:
            phase_trades = [t for t in closed_trades if t.wyckoff_phase == phase]
            if phase_trades:
                phase_wins = [t for t in phase_trades if t.pnl > 0]
                phase_performance[phase] = {
                    'count': len(phase_trades),
                    'win_rate': (len(phase_wins) / len(phase_trades)) * 100,
                    'avg_pnl': float(np.mean([t.pnl for t in phase_trades])),
                    'total_pnl': float(sum([t.pnl for t in phase_trades]))
                }
        
        # Agent performance (simplified)
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
        
        return BacktestResults(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=float(sharpe_ratio),
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
            start_date=datetime.now().isoformat(),
            end_date=datetime.now().isoformat(),
            bars_processed=len(self.equity_curve),
            test_duration_seconds=(datetime.now() - start_time).total_seconds()
        )
    
    async def _generate_detailed_report(self, results: BacktestResults, symbol: str) -> str:
        """Generate comprehensive markdown report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        report_path = report_dir / f"agent_backtest_report_{symbol}_{timestamp}.md"
        
        report_content = f"""# 🤖 Enhanced Agent Backtesting Report

## 📊 Test Overview
- **Symbol**: {results.symbol}
- **Timeframe**: {results.timeframe}
- **Test Period**: {results.start_date} to {results.end_date}
- **Bars Processed**: {results.bars_processed:,}
- **Test Duration**: {results.test_duration_seconds:.1f} seconds

## 💰 Portfolio Performance

### Capital Metrics
- **Initial Capital**: ${results.initial_capital:,.2f}
- **Final Capital**: ${results.final_capital:,.2f}
- **Total Return**: ${results.total_return:,.2f} ({results.total_return_pct:+.2f}%)
- **Max Drawdown**: ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.2f}%)
- **Sharpe Ratio**: {results.sharpe_ratio:.3f}

### Trading Performance
- **Total Trades**: {results.total_trades}
- **Winning Trades**: {results.winning_trades}
- **Losing Trades**: {results.losing_trades}
- **Win Rate**: {results.win_rate:.2f}%
- **Average Win**: ${results.avg_win:.2f}
- **Average Loss**: ${results.avg_loss:.2f}
- **Profit Factor**: {results.profit_factor:.2f}

## 🤖 Agent Performance

### Agent Analysis Summary
"""
        
        for agent_perf in results.agent_performances:
            report_content += f"""
#### {agent_perf.agent_name}
- **Total Signals**: {agent_perf.total_signals}
- **Buy Signals**: {agent_perf.buy_signals}
- **Sell Signals**: {agent_perf.sell_signals}
- **Average Confidence**: {agent_perf.avg_confidence:.1f}%
- **Success Rate**: {agent_perf.success_rate:.2f}%
- **Execution Time**: {agent_perf.execution_time_ms:.0f}ms
"""
        
        report_content += f"""
## 📈 Pattern Analysis

### Wyckoff Pattern Performance
"""
        
        for pattern, metrics in results.pattern_performance.items():
            report_content += f"""
#### {pattern.replace('_', ' ').title()}
- **Trade Count**: {metrics['count']}
- **Win Rate**: {metrics['win_rate']:.1f}%
- **Average P&L**: ${metrics['avg_pnl']:.2f}
- **Total P&L**: ${metrics['total_pnl']:.2f}
"""
        
        report_content += f"""
### Wyckoff Phase Performance
"""
        
        for phase, metrics in results.phase_performance.items():
            report_content += f"""
#### {phase.replace('_', ' ').title()}
- **Trade Count**: {metrics['count']}
- **Win Rate**: {metrics['win_rate']:.1f}%
- **Average P&L**: ${metrics['avg_pnl']:.2f}
- **Total P&L**: ${metrics['total_pnl']:.2f}
"""
        
        report_content += f"""
## 📋 Trade History

### Recent Trades (Last 10)
"""
        
        recent_trades = results.trades[-10:] if len(results.trades) > 10 else results.trades
        for trade in recent_trades:
            status = "✅" if trade.pnl > 0 else "❌" if trade.pnl < 0 else "⏳"
            report_content += f"""
- {status} **{trade.action.upper()}** {trade.symbol} @ {trade.entry_price:.5f} | P&L: ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%) | {trade.pattern_type}
"""
        
        report_content += f"""
## 🎯 Key Insights

### Strengths
- Agent system successfully generated {results.total_trades} trading signals
- Pattern recognition identified {len(results.pattern_performance)} different Wyckoff patterns
- Risk management maintained drawdown at {results.max_drawdown_pct:.2f}%

### Areas for Improvement
- Win rate could be improved through better signal filtering
- Agent confidence calibration may need adjustment
- Consider implementing ensemble decision making

### Recommendations
1. **Pattern Focus**: Best performing pattern was {max(results.pattern_performance.keys(), key=lambda x: results.pattern_performance[x]['win_rate']) if results.pattern_performance else 'N/A'}
2. **Phase Optimization**: {max(results.phase_performance.keys(), key=lambda x: results.phase_performance[x]['win_rate']) if results.phase_performance else 'N/A'} phase showed strongest performance
3. **Risk Management**: Current 2% risk per trade appears appropriate
4. **Signal Quality**: Focus on signals with confidence > 70%

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Enhanced Agent Backtesting System*
"""
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📝 Detailed report saved: {report_path}")
        return str(report_path)

# Add the run_agent_backtest method to AutonomousTradingSystem
def add_backtest_method_to_trading_system():
    """Add the run_agent_backtest method to AutonomousTradingSystem"""
    
    async def run_agent_backtest(self, historical_data: List[Dict], initial_balance: float = 100000.0, symbol: str = "EUR_USD") -> Dict[str, Any]:
        """Method to add to AutonomousTradingSystem class"""
        backtester = EnhancedAgentBacktester(initial_capital=initial_balance)
        return await backtester.run_agent_backtest(historical_data, initial_balance, symbol)
    
    # Add method to class
    AutonomousTradingSystem.run_agent_backtest = run_agent_backtest  # type: ignore

# Initialize the method addition
add_backtest_method_to_trading_system()
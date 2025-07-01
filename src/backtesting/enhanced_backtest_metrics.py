"""
Enhanced Backtest Metrics System - Phase 1
Comprehensive metrics enhancement for your autonomous trading system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
from collections import defaultdict

# =====================================================
# ENHANCED DATA STRUCTURES
# =====================================================

@dataclass
class EnhancedBacktestTrade:
    """Enhanced trade record with advanced metrics"""
    # Existing fields (maintaining compatibility)
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
    
    # NEW ENHANCED FIELDS
    # Execution metrics
    entry_slippage: float = 0.0  # Actual vs expected entry price
    exit_slippage: float = 0.0   # Actual vs expected exit price
    commission_paid: float = 0.0
    
    # Risk metrics  
    max_adverse_excursion: float = 0.0  # Worst unrealized loss
    max_favorable_excursion: float = 0.0  # Best unrealized profit
    max_risk_pct: float = 0.0  # Maximum risk taken as % of capital
    
    # Timing metrics
    entry_hour: int = 0  # Hour of day (0-23)
    entry_day_of_week: int = 0  # Day of week (0-6)
    bars_to_profit: int = 0  # Bars until first profit
    bars_to_max_profit: int = 0  # Bars to maximum profit
    
    # Market context
    market_volatility: float = 0.0  # ATR at entry
    trend_strength: float = 0.0  # ADX or similar
    volume_ratio: float = 0.0  # Volume vs average
    
    # Wyckoff specific
    phase_confidence: float = 0.0  # Confidence in phase identification
    cause_measurement: float = 0.0  # Accumulation/distribution size
    effort_vs_result_score: float = 0.0  # Volume/price relationship quality
    composite_operator_detected: bool = False


@dataclass 
class DrawdownPeriod:
    """Individual drawdown period analysis"""
    start_date: datetime
    end_date: datetime
    duration_bars: int
    peak_value: float
    trough_value: float
    drawdown_amount: float
    drawdown_pct: float
    recovery_bars: int
    recovered: bool = False


@dataclass
class PerformanceByPeriod:
    """Performance broken down by time periods"""
    hourly_performance: Dict[int, Dict] = field(default_factory=dict)  # 0-23
    daily_performance: Dict[int, Dict] = field(default_factory=dict)   # 0-6 (Mon-Sun)
    monthly_performance: Dict[int, Dict] = field(default_factory=dict) # 1-12
    
    
@dataclass
class RiskMetrics:
    """Comprehensive risk analysis"""
    # Value at Risk
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    
    # Drawdown analysis
    max_drawdown_duration: int = 0  # Longest drawdown in bars
    avg_drawdown_duration: float = 0.0
    drawdown_periods: List[DrawdownPeriod] = field(default_factory=list)
    recovery_factor: float = 0.0  # Return / Max Drawdown
    
    # Return distribution
    return_skewness: float = 0.0
    return_kurtosis: float = 0.0
    positive_periods_pct: float = 0.0
    
    # Tail risk
    tail_ratio: float = 0.0  # 95th percentile return / 5th percentile return
    largest_loss_vs_avg: float = 0.0


@dataclass
class AgentAnalytics:
    """Enhanced agent performance analytics"""
    agent_name: str
    
    # Decision quality
    signal_accuracy: float = 0.0  # Correct directional calls
    confidence_calibration: float = 0.0  # How well confidence predicts success
    timing_accuracy: float = 0.0  # Entry timing quality
    
    # Collaboration metrics
    agreement_with_other_agents: float = 0.0
    decision_consistency: float = 0.0  # Similar decisions in similar conditions
    
    # Efficiency metrics
    avg_decision_time: float = 0.0  # Time to make decisions
    signal_frequency: float = 0.0  # Signals per bar
    
    # Pattern-specific performance
    pattern_success_rates: Dict[str, float] = field(default_factory=dict)
    phase_identification_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Learning indicators
    performance_trend: float = 0.0  # Improving/declining over time
    adaptability_score: float = 0.0  # Performance in different market conditions


@dataclass
class WyckoffAnalytics:
    """Wyckoff-specific performance metrics"""
    # Phase performance
    accumulation_success_rate: float = 0.0
    distribution_success_rate: float = 0.0
    markup_success_rate: float = 0.0
    markdown_success_rate: float = 0.0
    
    # Pattern performance
    spring_detection_accuracy: float = 0.0
    upthrust_detection_accuracy: float = 0.0
    trading_range_breakout_success: float = 0.0
    
    # Cause and effect analysis
    avg_cause_measurement_accuracy: float = 0.0
    effort_vs_result_correlation: float = 0.0
    volume_confirmation_rate: float = 0.0
    
    # Composite operator analysis
    institutional_behavior_detection: float = 0.0
    absorption_identification: float = 0.0
    
    # Trading range analysis
    support_resistance_accuracy: float = 0.0
    range_duration_estimation: float = 0.0


@dataclass
class EnhancedBacktestResults:
    """Comprehensive backtest results with advanced metrics"""
    
    # ===== EXISTING FIELDS (maintaining compatibility) =====
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    agent_performances: List = field(default_factory=list)
    pattern_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    phase_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trades: List = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    symbol: str = ""
    timeframe: str = ""
    start_date: str = ""
    end_date: str = ""
    bars_processed: int = 0
    test_duration_seconds: float = 0.0
    
    # ===== NEW ENHANCED METRICS =====
    
    # Advanced performance metrics
    sortino_ratio: float = 0.0  # Risk-adjusted return using downside deviation
    calmar_ratio: float = 0.0   # Annual return / Max Drawdown
    omega_ratio: float = 0.0    # Probability-weighted gains vs losses
    
    # Risk metrics
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    
    # Time-based analysis
    performance_by_period: PerformanceByPeriod = field(default_factory=PerformanceByPeriod)
    
    # Enhanced agent analytics
    agent_analytics: List[AgentAnalytics] = field(default_factory=list)
    
    # Wyckoff-specific metrics
    wyckoff_analytics: WyckoffAnalytics = field(default_factory=WyckoffAnalytics)
    
    # Trade sequence analysis
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Execution analysis
    avg_trade_duration: float = 0.0  # Average bars per trade
    avg_slippage: float = 0.0
    total_commission: float = 0.0
    
    # Market condition performance
    trending_market_performance: float = 0.0
    ranging_market_performance: float = 0.0
    high_volatility_performance: float = 0.0
    low_volatility_performance: float = 0.0
    
    # Benchmark comparisons
    vs_buy_hold_return: float = 0.0
    vs_random_strategy: float = 0.0
    market_correlation: float = 0.0


# =====================================================
# ENHANCED METRICS CALCULATOR
# =====================================================

class EnhancedMetricsCalculator:
    """Calculator for all enhanced backtest metrics"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_enhanced_metrics(
        self, 
        trades: List[EnhancedBacktestTrade],
        equity_curve: List[float],
        historical_data: List[Dict],
        initial_capital: float
    ) -> EnhancedBacktestResults:
        """Calculate all enhanced metrics from trade and equity data"""
        
        # Start with basic results structure
        results = EnhancedBacktestResults(
            initial_capital=initial_capital,
            final_capital=equity_curve[-1] if equity_curve else initial_capital,
            equity_curve=equity_curve,
            trades=trades,
            total_trades=len([t for t in trades if t.is_closed]),
            total_return=0.0,
            total_return_pct=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0
        )
        
        # Calculate basic metrics
        self._calculate_basic_metrics(results)
        # Calculate advanced risk metrics
        self._calculate_risk_metrics(results)
        
        # Calculate time-based performance
        self._calculate_time_based_performance(results)
        
        # Calculate Wyckoff-specific metrics
        self._calculate_wyckoff_metrics(results)
        
        # Calculate agent analytics
        self._calculate_agent_analytics(results)
        
        # Calculate execution metrics
        self._calculate_execution_metrics(results)
        
        # Calculate market condition performance
        self._calculate_market_condition_performance(results, historical_data)
        
        return results
    
    def _calculate_basic_metrics(self, results: EnhancedBacktestResults):
        """Calculate enhanced basic performance metrics"""
        closed_trades = [t for t in results.trades if t.is_closed]
        
        if not closed_trades:
            return
            
        # Basic trade statistics
        results.winning_trades = len([t for t in closed_trades if t.pnl > 0])
        results.losing_trades = len([t for t in closed_trades if t.pnl <= 0])
        results.win_rate = (results.winning_trades / len(closed_trades)) * 100
        
        # P&L metrics
        winning_pnl = [t.pnl for t in closed_trades if t.pnl > 0]
        losing_pnl = [t.pnl for t in closed_trades if t.pnl <= 0]
        
        results.avg_win = float(np.mean(winning_pnl) if winning_pnl else 0)
        results.avg_loss = float(np.mean(losing_pnl) if losing_pnl else 0)
        
        gross_profit = sum(winning_pnl) if winning_pnl else 0
        gross_loss = abs(sum(losing_pnl)) if losing_pnl else 0
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        results.total_return = results.final_capital - results.initial_capital
        results.total_return_pct = (results.total_return / results.initial_capital) * 100
        
        # Enhanced ratios
        returns = self._calculate_returns(results.equity_curve)
        if len(returns) > 1:
            results.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            results.sortino_ratio = self._calculate_sortino_ratio(returns)
            results.calmar_ratio = self._calculate_calmar_ratio(results)
            results.omega_ratio = self._calculate_omega_ratio(returns)
    
    def _calculate_risk_metrics(self, results: EnhancedBacktestResults):
        """Calculate comprehensive risk metrics"""
        returns = self._calculate_returns(results.equity_curve)
        
        if len(returns) < 2:
            return
            
        # VaR calculations
        results.risk_metrics.var_95 = float(np.percentile(returns, 5))
        results.risk_metrics.var_99 = float(np.percentile(returns, 1))
        results.risk_metrics.cvar_95 = float(np.mean(returns[returns <= results.risk_metrics.var_95]))
        
        # Drawdown analysis
        self._calculate_drawdown_analysis(results)
        
        # Return distribution
        results.risk_metrics.return_skewness = float(pd.Series(returns).skew())
        results.risk_metrics.return_kurtosis = float(pd.Series(returns).kurtosis())
        results.risk_metrics.positive_periods_pct = (np.sum(np.array(returns) > 0) / len(returns)) * 100
        
        # Tail risk
        if len(returns) >= 20:  # Need sufficient data
            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)
            results.risk_metrics.tail_ratio = float(p95 / abs(p5) if p5 != 0 else 0)
        
        # Recovery factor
        max_dd = results.max_drawdown
        if max_dd > 0:
            results.risk_metrics.recovery_factor = results.total_return / max_dd
    
    def _calculate_drawdown_analysis(self, results: EnhancedBacktestResults):
        """Detailed drawdown analysis"""
        equity_series = np.array(results.equity_curve)
        
        # Find all drawdown periods
        peak = equity_series[0]
        in_drawdown = False
        drawdown_start = 0
        drawdowns = []
        
        for i, value in enumerate(equity_series):
            if value > peak:
                if in_drawdown:
                    # End of drawdown period
                    dd_period = DrawdownPeriod(
                        start_date=datetime.now(),  # Would use actual dates
                        end_date=datetime.now(),
                        duration_bars=i - drawdown_start,
                        peak_value=peak,
                        trough_value=min(equity_series[drawdown_start:i]),
                        drawdown_amount=peak - min(equity_series[drawdown_start:i]),
                        drawdown_pct=((peak - min(equity_series[drawdown_start:i])) / peak) * 100,
                        recovery_bars=i - drawdown_start,
                        recovered=True
                    )
                    drawdowns.append(dd_period)
                    in_drawdown = False
                peak = value
            elif value < peak and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
        
        results.risk_metrics.drawdown_periods = drawdowns
        
        if drawdowns:
            durations = [dd.duration_bars for dd in drawdowns]
            results.risk_metrics.max_drawdown_duration = max(durations)
            results.risk_metrics.avg_drawdown_duration = float(np.mean(durations))
    
    def _calculate_time_based_performance(self, results: EnhancedBacktestResults):
        """Calculate performance by time periods"""
        closed_trades = [t for t in results.trades if t.is_closed]
        
        # Initialize performance dictionaries
        hourly_stats = defaultdict(list)
        daily_stats = defaultdict(list)
        monthly_stats = defaultdict(list)
        
        for trade in closed_trades:
            # Extract time components (would use actual datetime parsing)
            hour = trade.entry_hour
            day = trade.entry_day_of_week
            month = datetime.now().month  # Would extract from actual timestamp
            
            hourly_stats[hour].append(trade.pnl)
            daily_stats[day].append(trade.pnl)
            monthly_stats[month].append(trade.pnl)
        
        # Calculate statistics for each period
        for hour, pnls in hourly_stats.items():
            results.performance_by_period.hourly_performance[hour] = {
                'total_trades': len(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': (sum(1 for p in pnls if p > 0) / len(pnls)) * 100,
                'total_pnl': sum(pnls)
            }
        
        # Similar calculations for daily and monthly
        for day, pnls in daily_stats.items():
            results.performance_by_period.daily_performance[day] = {
                'total_trades': len(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': (sum(1 for p in pnls if p > 0) / len(pnls)) * 100,
                'total_pnl': sum(pnls)
            }
    
    def _calculate_wyckoff_metrics(self, results: EnhancedBacktestResults):
        """Calculate Wyckoff-specific performance metrics"""
        closed_trades = [t for t in results.trades if t.is_closed]
        
        if not closed_trades:
            return
            
        # Phase performance
        phase_trades = defaultdict(list)
        pattern_trades = defaultdict(list)
        
        for trade in closed_trades:
            phase_trades[trade.wyckoff_phase].append(trade)
            pattern_trades[trade.pattern_type].append(trade)
        
        # Calculate phase success rates
        accumulation_trades = [t for t in closed_trades if 'accumulation' in t.wyckoff_phase.lower()]
        if accumulation_trades:
            wins = sum(1 for t in accumulation_trades if t.pnl > 0)
            results.wyckoff_analytics.accumulation_success_rate = (wins / len(accumulation_trades)) * 100
        
        distribution_trades = [t for t in closed_trades if 'distribution' in t.wyckoff_phase.lower()]
        if distribution_trades:
            wins = sum(1 for t in distribution_trades if t.pnl > 0)
            results.wyckoff_analytics.distribution_success_rate = (wins / len(distribution_trades)) * 100
        
        # Pattern accuracy
        spring_trades = [t for t in closed_trades if 'spring' in t.pattern_type.lower()]
        if spring_trades:
            wins = sum(1 for t in spring_trades if t.pnl > 0)
            results.wyckoff_analytics.spring_detection_accuracy = (wins / len(spring_trades)) * 100
        
        # Effort vs Result analysis
        effort_results = [t.effort_vs_result_score for t in closed_trades if hasattr(t, 'effort_vs_result_score')]
        if effort_results:
            results.wyckoff_analytics.effort_vs_result_correlation = np.corrcoef(
                effort_results, 
                [t.pnl for t in closed_trades if hasattr(t, 'effort_vs_result_score')]
            )[0, 1] if len(effort_results) > 1 else 0
    
    def _calculate_agent_analytics(self, results: EnhancedBacktestResults):
        """Calculate agent-specific performance analytics"""
        closed_trades = [t for t in results.trades if t.is_closed]
        
        # Group trades by agent
        agent_trades = defaultdict(list)
        for trade in closed_trades:
            agent_trades[trade.agent_name].append(trade)
        
        for agent_name, trades in agent_trades.items():
            analytics = AgentAnalytics(agent_name=agent_name)
            
            # Signal accuracy
            winning_trades = [t for t in trades if t.pnl > 0]
            analytics.signal_accuracy = (len(winning_trades) / len(trades)) * 100 if trades else 0
            
            # Confidence calibration
            if trades:
                confidences = [t.confidence for t in trades]
                outcomes = [1 if t.pnl > 0 else 0 for t in trades]
                analytics.confidence_calibration = np.corrcoef(confidences, outcomes)[0, 1] if len(trades) > 1 else 0
            
            # Pattern-specific performance
            pattern_performance = defaultdict(list)
            for trade in trades:
                pattern_performance[trade.pattern_type].append(trade.pnl > 0)
            
            for pattern, outcomes in pattern_performance.items():
                analytics.pattern_success_rates[pattern] = (sum(outcomes) / len(outcomes)) * 100
            
            results.agent_analytics.append(analytics)
    
    def _calculate_execution_metrics(self, results: EnhancedBacktestResults):
        """Calculate trade execution quality metrics"""
        closed_trades = [t for t in results.trades if t.is_closed]
        
        if not closed_trades:
            return
            
        # Average trade duration
        durations = [t.duration_bars for t in closed_trades]
        results.avg_trade_duration = float(np.mean(durations))
        
        # Slippage analysis
        entry_slippages = [getattr(t, 'entry_slippage', 0) for t in closed_trades]
        exit_slippages = [getattr(t, 'exit_slippage', 0) for t in closed_trades]
        results.avg_slippage = float(np.mean(entry_slippages + exit_slippages))
        
        # Commission analysis
        commissions = [getattr(t, 'commission_paid', 0) for t in closed_trades]
        results.total_commission = sum(commissions)
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in closed_trades:
            if trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        results.max_consecutive_wins = max_consecutive_wins
        results.max_consecutive_losses = max_consecutive_losses
    
    def _calculate_market_condition_performance(self, results: EnhancedBacktestResults, historical_data: List[Dict]):
        """Calculate performance under different market conditions"""
        # This would analyze market conditions during each trade
        # For now, placeholder implementation
        closed_trades = [t for t in results.trades if t.is_closed]
        
        if not closed_trades:
            return
            
        # Placeholder - would implement actual market condition detection
        results.trending_market_performance = float(np.mean([t.pnl for t in closed_trades[:len(closed_trades)//2]]))
        results.ranging_market_performance = float(np.mean([t.pnl for t in closed_trades[len(closed_trades)//2:]]))
    
    def _calculate_returns(self, equity_curve: List[float]) -> np.ndarray:
        """Calculate period returns from equity curve"""
        if len(equity_curve) < 2:
            return np.array([])
        
        equity_series = np.array(equity_curve)
        returns = np.diff(equity_series) / equity_series[:-1]
        return returns
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        return float(np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (uses only downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        return float(np.mean(excess_returns) / downside_deviation) if downside_deviation > 0 else 0.0
    
    def _calculate_calmar_ratio(self, results: EnhancedBacktestResults) -> float:
        """Calculate Calmar ratio (Annual return / Max Drawdown)"""
        if results.max_drawdown == 0:
            return float('inf')
        
        annual_return = results.total_return_pct  # Simplified - would annualize properly
        return annual_return / results.max_drawdown_pct
    
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns <= 0]
        
        if len(losses) == 0:
            return float('inf')
        if len(gains) == 0:
            return 0.0
        
        return float(np.sum(gains) / abs(np.sum(losses)))


# =====================================================
# USAGE EXAMPLE
# =====================================================

def enhance_existing_backtest_results(
    trades: List,  # Your existing BacktestTrade objects
    equity_curve: List[float],
    historical_data: List[Dict],
    initial_capital: float
) -> EnhancedBacktestResults:
    """
    Convert your existing backtest results to enhanced format
    
    Usage:
        enhanced_results = enhance_existing_backtest_results(
            trades=your_existing_trades,
            equity_curve=your_equity_curve,
            historical_data=your_historical_data,
            initial_capital=50000
        )
    """
    
    calculator = EnhancedMetricsCalculator()
    
    # Convert existing trades to enhanced format (if needed)
    enhanced_trades = []
    for trade in trades:
        # Create enhanced trade from existing trade
        enhanced_trade = EnhancedBacktestTrade(
            id=getattr(trade, 'id', ''),
            timestamp=getattr(trade, 'timestamp', ''),
            symbol=getattr(trade, 'symbol', ''),
            action=getattr(trade, 'action', ''),
            entry_price=getattr(trade, 'entry_price', 0.0),
            quantity=getattr(trade, 'quantity', 0.0),
            stop_loss=getattr(trade, 'stop_loss', 0.0),
            take_profit=getattr(trade, 'take_profit', 0.0),
            confidence=getattr(trade, 'confidence', 0.0),
            wyckoff_phase=getattr(trade, 'wyckoff_phase', ''),
            pattern_type=getattr(trade, 'pattern_type', ''),
            reasoning=getattr(trade, 'reasoning', ''),
            agent_name=getattr(trade, 'agent_name', ''),
            exit_price=getattr(trade, 'exit_price', None),
            exit_timestamp=getattr(trade, 'exit_timestamp', None),
            exit_reason=getattr(trade, 'exit_reason', 'open'),
            pnl=getattr(trade, 'pnl', 0.0),
            pnl_pct=getattr(trade, 'pnl_pct', 0.0),
            duration_bars=getattr(trade, 'duration_bars', 0),
            is_closed=getattr(trade, 'is_closed', False)
        )
        enhanced_trades.append(enhanced_trade)
    
    # Calculate all enhanced metrics
    enhanced_results = calculator.calculate_enhanced_metrics(
        trades=enhanced_trades,
        equity_curve=equity_curve,
        historical_data=historical_data,
        initial_capital=initial_capital
    )
    
    return enhanced_results
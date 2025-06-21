"""
Backtesting Engine for Autonomous Trading System
Tests trading strategies on historical data
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow running this script directly
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

#from autonomous_trading_system.utils.wyckoff_data_processor import IntegratedWyckoffBacktester
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from src.config.logging_config import logger
#from src.autonomous_trading_system.utils.wyckoff_data_processor import IntegratedWyckoffBacktester

import asyncio
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

import logging
from src.config.logging_config import logger

warnings.filterwarnings('ignore')

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.database.manager import db_manager
from src.database.models import Trade, TradeStatus, TradeSide
from src.autonomous_trading_system.utils.wyckoff_pattern_analyzer import wyckoff_analyzer

def fix_json_serialization_error():
    """Comprehensive fix for all JSON serialization errors"""
    
    def make_json_serializable(obj):
        """Convert any object to JSON-serializable format"""
        
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):  # This fixes WyckoffPattern enum
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif hasattr(obj, '__dict__'):
            return make_json_serializable(obj.__dict__)
        else:
            try:
                return str(obj)
            except:
                return None
    
    # Patch JSON dumps globally
    original_dumps = json.dumps
    def safe_dumps(obj, **kwargs):
        try:
            cleaned_obj = make_json_serializable(obj)
            return original_dumps(cleaned_obj, **kwargs)
        except Exception as e:
            return original_dumps(str(obj), **kwargs)
    
    json.dumps = safe_dumps
    print("✅ Comprehensive JSON serialization fix applied")
    return make_json_serializable

# Apply the fix
make_json_serializable = fix_json_serialization_error()

# Patch database manager
def patch_database_manager():
    try:
        from src.database.manager import db_manager
        
        if not hasattr(db_manager, '_original_log_pattern_detection'):
            print()
            #db_manager._original_log_pattern_detection = db_manager.log_pattern_detection
        
        def patched_log_pattern_detection(self, detection_data):
            try:
                # Clean all data
                cleaned_data = make_json_serializable(detection_data)
                print('*'*40)
                print(cleaned_data)
                # Ensure specific field types
                if 'confidence_score' in cleaned_data:
                    cleaned_data['confidence_score'] = float(cleaned_data['confidence_score'])
                
                if 'invalidation_level' in cleaned_data and cleaned_data['invalidation_level'] is not None:
                    cleaned_data['invalidation_level'] = float(cleaned_data['invalidation_level'])
                
                # Handle JSON fields
                json_fields = ['key_levels', 'volume_analysis', 'market_context']
                for field in json_fields:
                    if field in cleaned_data and cleaned_data[field] is not None:
                        if isinstance(cleaned_data[field], (dict, list)):
                            cleaned_data[field] = json.dumps(make_json_serializable(cleaned_data[field]))
                
                return self._original_log_pattern_detection(cleaned_data)
                
            except Exception as e:
                # Fallback to minimal safe data
                safe_data = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': str(detection_data.get('symbol', 'UNKNOWN')),
                    'timeframe': str(detection_data.get('timeframe', 'UNKNOWN')),
                    'pattern_type': str(detection_data.get('pattern_type', 'unknown')),
                    'confidence_score': float(detection_data.get('confidence_score', 0)),
                    'key_levels': '{}',
                    'volume_analysis': '{}',
                    'market_context': '{}',
                    'invalidation_level': None,
                    'trade_id': None
                }
                return self._original_log_pattern_detection(safe_data)
        
        db_manager.log_pattern_detection = patched_log_pattern_detection.__get__(db_manager, type(db_manager))
        print("✅ Database manager patched")
        
    except Exception as e:
        print(f"⚠️ Could not patch database manager: {e}")

# Apply database patch
patch_database_manager()

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class BacktestTrade:
    """Individual trade in backtest"""
    id: int
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: TradeDirection = TradeDirection.LONG
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    wyckoff_pattern: str = "unknown"
    pattern_confidence: float = 0.0
    phase: str = "unknown"
    entry_reason: str = ""
    exit_reason: str = ""
    bars_in_trade: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    status: str = "open"
    volume_confirmation: bool = False
    market_context: str = "unknown"

@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    parameters: Dict[str, float]
    total_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    score: float  # Composite optimization score

@dataclass
class BacktestResults:
    """Complete backtest results"""
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Performance Metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk Metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    
    # Time-based Metrics
    avg_trade_duration: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Wyckoff-specific Metrics
    accumulation_trades: int = 0
    distribution_trades: int = 0
    spring_trades: int = 0
    upthrust_trades: int = 0
    pattern_success_rates: Dict[str, float] = field(default_factory=dict)
    phase_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Advanced Metrics
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    monthly_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    # Trade List
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # Enhanced OANDA-specific metrics
    avg_spread: Optional[float] = None
    max_spread: Optional[float] = None
    min_spread: Optional[float] = None
    spread_volatility: Optional[float] = None
    data_completeness: Optional[float] = None
    volume_availability: Optional[float] = None
    

class AutomatedWyckoffBacktester:
    """Professional automated backtesting engine for Wyckoff strategies"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_per_trade: float = 5.0,
                 risk_per_trade: float = 0.02,
                 max_position_size: float = 0.1):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.trade_counter = 0
        
        # Wyckoff-specific settings (can be optimized)
        self.min_pattern_confidence = 60  
        self.min_risk_reward = 2.0       
        self.volume_threshold = 1.5      
        self.structure_confirmation_bars = 3
        self.max_trades_per_day = 2
        
        # Optimization parameters
        self.optimization_params = {
            'min_pattern_confidence': [50, 60, 70, 80],
            'min_risk_reward': [1.5, 2.0, 2.5, 3.0],
            'volume_threshold': [1.2, 1.5, 2.0, 2.5],
            'risk_per_trade': [0.01, 0.015, 0.02, 0.025]
        }
        
    async def run_automated_backtest_suite(self,
                                         symbols: List[str] = ["EUR_USD"],
                                         timeframes: List[str] = ["M15", "H1"],
                                         optimization: bool = True,
                                         walk_forward: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive automated backtest suite
        
        Args:
            symbols: List of currency pairs to test
            timeframes: List of timeframes to test
            optimization: Whether to run parameter optimization
            walk_forward: Whether to run walk-forward analysis
            
        Returns:
            Complete suite results with optimization and validation
        """
        
        logger.info("🚀 Starting automated Wyckoff backtest suite", 
                   symbols=symbols, timeframes=timeframes)
        
        suite_results = {
            'timestamp': datetime.now(),
            'symbols_tested': symbols,
            'timeframes_tested': timeframes,
            'optimization_results': {},
            'backtest_results': {},
            'walk_forward_results': {},
            'best_parameters': {},
            'summary_statistics': {}
        }
        
        try:
            for symbol in symbols:
                logger.info(f"📊 Testing symbol: {symbol}")
                suite_results['backtest_results'][symbol] = {}
                
                for timeframe in timeframes:
                    logger.info(f"⏰ Testing timeframe: {timeframe}")
                    
                    # Get historical data
                    historical_data = await self._get_historical_data(symbol, timeframe)
                    
                    if not historical_data or len(historical_data) < 500:
                        logger.warning(f"Insufficient data for {symbol} {timeframe}")
                        continue
                    
                    # Parameter optimization
                    if optimization:
                        opt_result = await self._optimize_parameters(historical_data)
                        suite_results['optimization_results'][f"{symbol}_{timeframe}"] = opt_result
                        
                        # Use optimized parameters
                        self._apply_optimized_parameters(opt_result.parameters)
                    
                    # Main backtest
                    backtest_result = await self.run_backtest(historical_data)
                    suite_results['backtest_results'][symbol][timeframe] = backtest_result
                    
                    # Walk-forward analysis
                    if walk_forward:
                        wf_result = await self._run_walk_forward_analysis(historical_data)
                        suite_results['walk_forward_results'][f"{symbol}_{timeframe}"] = wf_result
                    
                    # Reset for next test
                    self._reset_backtest_state()
            
            # Generate comprehensive report
            suite_results['summary_statistics'] = self._calculate_suite_statistics(suite_results)
            
            # Save results
            await self._save_suite_results(suite_results)
            
            # Generate visual reports
            await self._generate_visual_reports(suite_results)
            
            logger.info("✅ Automated backtest suite completed successfully")
            return suite_results
            
        except Exception as e:
            logger.error("❌ Automated backtest suite failed", error=str(e))
            raise
    
    async def _optimize_parameters(self, historical_data: List[Dict[str, Any]]) -> OptimizationResult:
        """
        Optimize strategy parameters using grid search
        """
        logger.info("🔧 Starting parameter optimization...")
        
        best_score = -float('inf')
        best_params = {}
        best_result = None
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        total_combinations = len(param_combinations)
        
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        for i, params in enumerate(param_combinations):
            try:
                # Apply parameters
                self._apply_parameters(params)
                
                # Run backtest with current parameters
                result = await self.run_backtest(historical_data.copy())
                
                # Calculate optimization score
                score = self._calculate_optimization_score(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                
                # Progress update
                if (i + 1) % 10 == 0:
                    progress = ((i + 1) / total_combinations) * 100
                    logger.info(f"Optimization progress: {progress:.1f}% ({i+1}/{total_combinations})")
                
                # Reset for next iteration
                self._reset_backtest_state()
                
            except Exception as e:
                logger.warning(f"Parameter combination failed: {params}", error=str(e))
                continue
        
        optimization_result = OptimizationResult(
            parameters=best_params,
            total_return=best_result.total_pnl_pct if best_result else 0,
            win_rate=best_result.win_rate if best_result else 0,
            profit_factor=best_result.profit_factor if best_result else 0,
            max_drawdown=best_result.max_drawdown_pct if best_result else 0,
            sharpe_ratio=best_result.sharpe_ratio if best_result else 0,
            total_trades=best_result.total_trades if best_result else 0,
            score=best_score
        )
        
        logger.info("✅ Parameter optimization completed", 
                   best_score=best_score, best_params=best_params)
        
        return optimization_result
    
    def _generate_parameter_combinations(self) -> List[Dict[str, float]]:
        """Generate all combinations of optimization parameters"""
        from itertools import product
        
        param_names = list(self.optimization_params.keys())
        param_values = [self.optimization_params[name] for name in param_names]
        
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _calculate_optimization_score(self, result: BacktestResults) -> float:
        """
        Calculate composite optimization score
        Balances return, risk, and trade frequency
        """
        if result.total_trades == 0:
            return -float('inf')
        
        # Components (normalized to 0-100 scale)
        return_score = min(result.total_pnl_pct, 100)  # Cap at 100%
        
        win_rate_score = result.win_rate  # Already 0-100
        
        # Profit factor score (optimal around 2.0)
        pf_score = min(result.profit_factor * 25, 100) if result.profit_factor > 0 else 0
        
        # Drawdown penalty (lower is better)
        dd_penalty = max(0, 100 - abs(result.max_drawdown_pct * 2))
        
        # Trade frequency score (prefer 20-100 trades)
        trade_score = min(result.total_trades * 2, 100) if result.total_trades >= 10 else result.total_trades * 5
        
        # Sharpe ratio score
        sharpe_score = min(max(result.sharpe_ratio * 25, 0), 100)
        
        # Weighted composite score
        score = (
            return_score * 0.25 +
            win_rate_score * 0.15 +
            pf_score * 0.20 +
            dd_penalty * 0.20 +
            trade_score * 0.10 +
            sharpe_score * 0.10
        )
        
        return score
    
    async def _run_walk_forward_analysis(self, 
                                       historical_data: List[Dict[str, Any]],
                                       walk_periods: int = 6) -> Dict[str, Any]:
        """
        Run walk-forward analysis to validate strategy robustness
        """
        logger.info("🚶 Starting walk-forward analysis...")
        
        total_bars = len(historical_data)
        optimization_period = total_bars // (walk_periods * 2)  # 50% for optimization
        test_period = optimization_period  # 50% for testing
        
        wf_results = {
            'periods': [],
            'optimization_results': [],
            'test_results': [],
            'parameter_stability': {},
            'performance_consistency': {}
        }
        
        for period in range(walk_periods):
            try:
                # Define data windows
                opt_start = period * test_period
                opt_end = opt_start + optimization_period
                test_start = opt_end
                test_end = test_start + test_period
                
                if test_end > total_bars:
                    break
                
                opt_data = historical_data[opt_start:opt_end]
                test_data = historical_data[test_start:test_end]
                
                logger.info(f"Walk-forward period {period + 1}: "
                           f"Opt {len(opt_data)} bars, Test {len(test_data)} bars")
                
                # Optimize on training data
                opt_result = await self._optimize_parameters(opt_data)
                self._apply_optimized_parameters(opt_result.parameters)
                
                # Test on out-of-sample data
                test_result = await self.run_backtest(test_data)
                
                wf_results['periods'].append({
                    'period': period + 1,
                    'opt_start': opt_start,
                    'opt_end': opt_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
                
                wf_results['optimization_results'].append(opt_result)
                wf_results['test_results'].append(test_result)
                
                # Reset for next period
                self._reset_backtest_state()
                
            except Exception as e:
                logger.warning(f"Walk-forward period {period + 1} failed", error=str(e))
                continue
        
        # Analyze parameter stability and performance consistency
        wf_results['parameter_stability'] = self._analyze_parameter_stability(
            wf_results['optimization_results']
        )
        wf_results['performance_consistency'] = self._analyze_performance_consistency(
            wf_results['test_results']
        )
        
        logger.info("✅ Walk-forward analysis completed")
        return wf_results
    
    def _analyze_parameter_stability(self, opt_results: List[OptimizationResult]) -> Dict[str, float]:
        """Analyze how stable optimized parameters are across periods"""
        if not opt_results:
            return {}
        
        stability_metrics = {}
        param_names = list(opt_results[0].parameters.keys())
        
        for param_name in param_names:
            values = [result.parameters[param_name] for result in opt_results]
            stability_metrics[param_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stability_metrics
    
    def _analyze_performance_consistency(self, test_results: List[BacktestResults]) -> Dict[str, float]:
        """Analyze consistency of out-of-sample performance"""
        if not test_results:
            return {}
        
        returns = [result.total_pnl_pct for result in test_results]
        win_rates = [result.win_rate for result in test_results]
        sharpe_ratios = [result.sharpe_ratio for result in test_results]
        
        return {
            'avg_return': float(np.mean(returns)),
            'return_volatility': float(np.std(returns)),
            'positive_periods': float(sum(1 for r in returns if r > 0) / len(returns) * 100),
            'avg_win_rate': float(np.mean(win_rates)),
            'win_rate_stability': float(np.std(win_rates)),
            'avg_sharpe': float(np.mean(sharpe_ratios)),
            'sharpe_stability': float(np.std(sharpe_ratios))
        }
    
    async def run_backtest(self, 
                          price_data: List[Dict[str, Any]], 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> BacktestResults:
        """
        Run comprehensive Wyckoff strategy backtest
        """
        
        logger.info("🧪 Starting Wyckoff strategy backtest", 
                   data_points=len(price_data),
                   initial_capital=self.initial_capital)
        
        try:
            # Prepare data
            df = self._prepare_backtest_data(price_data, start_date, end_date)
            
            if len(df) < 100:
                raise ValueError("Insufficient data for backtesting (minimum 100 bars required)")
            
            # Run the backtest simulation
            await self._run_simulation(df)
            
            # Calculate results
            results = self._calculate_results(df)
            
            return results
            
        except Exception as e:
            logger.error("❌ Backtest failed", error=str(e))
            raise
    
    def _prepare_backtest_data(self, 
                              price_data: List[Dict[str, Any]], 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Prepare and validate backtest data"""
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df = self._ensure_dataframe(df)
            print(f"Backtest Dataframe: {df}")
            # Ensure we have the required columns
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            elif 'timestamp' not in df.columns:
                raise ValueError("Data must contain 'time' or 'timestamp' column")
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp - ensure we keep DataFrame structure
            df = df.sort_values('timestamp')
            df = self._ensure_dataframe(df)
            df = df.reset_index(drop=True)
            
            # Filter by date range if specified
            if start_date:
                mask = df['timestamp'] >= start_date
                df = df.loc[mask]
                df = self._ensure_dataframe(df)
            
            if end_date:
                mask = df['timestamp'] <= end_date
                df = df.loc[mask]
                df = self._ensure_dataframe(df)
            
            # Reset index after filtering
            df = df.reset_index(drop=True)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            df = self._ensure_dataframe(df)
            
            # Add required columns for backtesting
            df['equity'] = float(self.initial_capital)
            df['trades_open'] = 0
            df['daily_pnl'] = 0.0
            df['drawdown'] = 0.0
            
            # Final validation
            if len(df) == 0:
                raise ValueError("No data remaining after filtering")
            
            return df
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise ValueError(f"Failed to prepare backtest data: {str(e)}")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for enhanced analysis"""
        
        # Ensure input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("Input to _add_technical_indicators is not a DataFrame, converting...")
            df = pd.DataFrame(df)
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns for technical indicators: {missing_columns}")
            # Add missing columns with default values
            for col in missing_columns:
                if col == 'volume':
                    df[col] = 1000  # Default volume
                else:
                    df[col] = 1.0  # Default price
        
        try:
            # Ensure all required columns are numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill any NaN values in the base columns first
            df[required_columns] = df[required_columns].ffill().bfill()
            
            # Volume moving average and ratio
            if len(df) >= 20:
                df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
            else:
                df['volume_ma'] = df['volume'].mean()
                df['volume_ratio'] = 1.0
            
            # Price moving averages
            if len(df) >= 20:
                df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            else:
                df['sma_20'] = df['close'].mean()
                
            if len(df) >= 50:
                df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
            else:
                df['sma_50'] = df['close'].mean()
            
            # ATR for volatility
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_close'] = abs(df['low'] - df['close'].shift(1))
            
            # Ensure we have a DataFrame for max operation
            temp_df = df[['high_low', 'high_close', 'low_close']].copy()
            df['true_range'] = temp_df.max(axis=1)
            
            if len(df) >= 14:
                df['atr'] = df['true_range'].rolling(window=14, min_periods=1).mean()
            else:
                df['atr'] = df['true_range'].mean()
            
            # Support and resistance levels
            if len(df) >= 20:
                df['resistance'] = df['high'].rolling(window=20, min_periods=1).max()
                df['support'] = df['low'].rolling(window=20, min_periods=1).min()
            else:
                df['resistance'] = df['high'].max()
                df['support'] = df['low'].min()
            
            # Fill any remaining NaN values
            df = df.ffill().bfill()
            
            # Final check to ensure we return a DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.error("Technical indicators function did not return a DataFrame")
                df = pd.DataFrame(df)
            
        except Exception as e:
            logger.warning(f"Error adding technical indicators: {str(e)}")
            # If there's an error, add default values
            df['volume_ma'] = df.get('volume', pd.Series([1000] * len(df)))
            df['volume_ratio'] = 1.0
            df['sma_20'] = df.get('close', pd.Series([1.0] * len(df)))
            df['sma_50'] = df.get('close', pd.Series([1.0] * len(df)))
            df['atr'] = 0.01
            df['resistance'] = df.get('high', pd.Series([1.0] * len(df)))
            df['support'] = df.get('low', pd.Series([1.0] * len(df)))
        
        return df
    
    async def _run_simulation(self, df: pd.DataFrame):
        """Run the main backtest simulation with enhanced logic"""
        
        logger.info("🔄 Running enhanced backtest simulation...")
        
        analysis_window = 200  # bars for pattern analysis
        trades_today = 0
        last_trade_date = None
        
        for i in range(analysis_window, len(df)):
            current_bar = df.iloc[i]
            current_time = self._convert_to_datetime(current_bar['timestamp'])
            current_price = float(current_bar['close'])
            
            # Reset daily trade counter
            if last_trade_date is None or current_time.date() != last_trade_date:
                trades_today = 0
                last_trade_date = current_time.date()
            
            # Update existing trades
            self._update_open_trades(current_bar)
            
            # Check if we can open new trades
            if (trades_today >= self.max_trades_per_day or 
                not self._can_open_new_trade()):
                continue
            
            # Get data window for analysis - convert to list of dictionaries manually
            analysis_slice = df.iloc[i-analysis_window:i]
            analysis_data = []
            
            for idx in range(len(analysis_slice)):
                row = analysis_slice.iloc[idx]
                row_dict = {
                    'time': str(row.get('timestamp', row.get('time', ''))),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'close': float(row.get('close', 0)),
                    'volume': int(row.get('volume', 0))
                }
                analysis_data.append(row_dict)
            
            # Run Wyckoff analysis
            try:
                wyckoff_analysis = await wyckoff_analyzer.analyze_market_data(analysis_data, "M15")
                
                if isinstance(wyckoff_analysis, dict) and "error" not in wyckoff_analysis:
                    # Enhanced signal evaluation
                    signal = self._evaluate_enhanced_trading_signal(
                        wyckoff_analysis, current_bar, df.iloc[i-10:i]
                    )
                    
                    if signal:
                        await self._execute_trade(signal, current_bar, wyckoff_analysis)
                        trades_today += 1
                        
            except Exception as e:
                logger.warning(f"Analysis failed at bar {i}", error=str(e))
                continue
            
            # Update equity curve and drawdown
            current_equity = self._calculate_current_equity(current_price)
            df.at[i, 'equity'] = current_equity
            self.equity_curve.append(current_equity)
            
            # Calculate drawdown
            peak = max(self.equity_curve)
            drawdown = (peak - current_equity) / peak * 100
            df.at[i, 'drawdown'] = drawdown
            
            # Progress logging
            if i % 500 == 0:
                progress = (i / len(df)) * 100
                logger.info(f"📈 Backtest progress: {progress:.1f}% "
                           f"(Bar {i}/{len(df)}, Equity: ${current_equity:,.0f})")
    
    def _evaluate_enhanced_trading_signal(self, 
                                        wyckoff_analysis: Dict[str, Any], 
                                        current_bar: pd.Series,
                                        recent_bars: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Enhanced signal evaluation with additional filters"""
        
        try:
            # Get basic signal
            signal = self._evaluate_trading_signal(wyckoff_analysis, current_bar)
            if not signal:
                return None
            
            # Additional volume confirmation
            volume_confirmed = self._check_volume_confirmation(current_bar, recent_bars)
            if not volume_confirmed:
                return None
            
            # Market context filter
            market_context = self._determine_market_context(recent_bars)
            
            # Trend alignment filter
            if not self._check_trend_alignment(signal, recent_bars):
                return None
            
            # Time-based filters (avoid low-volume periods)
            if not self._check_time_filters(current_bar['timestamp']):
                return None
            
            # Enhance signal with additional data
            signal['volume_confirmation'] = volume_confirmed
            signal['market_context'] = market_context
            signal['trend_aligned'] = True
            
            return signal
            
        except Exception as e:
            logger.warning("Enhanced signal evaluation failed", error=str(e))
            return None
    
    def _check_volume_confirmation(self, current_bar: pd.Series, recent_bars: pd.DataFrame) -> bool:
        """Check for volume confirmation of the signal"""
        
        if 'volume_ratio' not in current_bar:
            return True  # Skip if no volume data
        
        # Current bar should have above-average volume
        return current_bar['volume_ratio'] >= self.volume_threshold
    
    def _determine_market_context(self, recent_bars: pd.DataFrame) -> str:
        """Determine current market context (trending/ranging)"""
        
        if len(recent_bars) < 20:
            return "unknown"
        
        # Calculate ADX-like trend strength
        high_low_pct = (recent_bars['high'].max() - recent_bars['low'].min()) / recent_bars['close'].mean()
        
        if high_low_pct > 0.03:  # 3% range suggests trending
            return "trending"
        else:
            return "ranging"
    
    def _check_trend_alignment(self, signal: Dict[str, Any], recent_bars: pd.DataFrame) -> bool:
        """Check if signal aligns with medium-term trend"""
        
        if len(recent_bars) < 20:
            return True
        
        # Simple trend check using moving averages
        if 'sma_20' in recent_bars.columns and 'sma_50' in recent_bars.columns:
            last_bar = recent_bars.iloc[-1]
            trend_up = last_bar['sma_20'] > last_bar['sma_50']
            
            if signal['direction'] == TradeDirection.LONG and trend_up:
                return True
            elif signal['direction'] == TradeDirection.SHORT and not trend_up:
                return True
            else:
                return False  # Counter-trend trades filtered out
        
        return True
    
    def _check_time_filters(self, timestamp) -> bool:
        """Apply time-based filters to avoid low-quality periods"""
        
        # Convert to datetime if needed
        timestamp = self._convert_to_datetime(timestamp)
        
        # Avoid weekend gaps (if present in data)
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Focus on active trading hours (London/NY overlap)
        hour = timestamp.hour
        if hour < 8 or hour > 17:  # Outside 8 AM - 5 PM UTC
            return False
        
        return True
    
    # Continue with the rest of the original methods...
    def _evaluate_trading_signal(self, 
                                wyckoff_analysis: Dict[str, Any], 
                                current_bar: pd.Series) -> Optional[Dict[str, Any]]:
        """Evaluate if current Wyckoff analysis generates a trading signal"""
        
        try:
            structure_analysis = wyckoff_analysis.get("structure_analysis", {})
            recommendations = wyckoff_analysis.get("trading_recommendations", {})
            spring_upthrust = wyckoff_analysis.get("spring_upthrust", {})
            confidence = wyckoff_analysis.get("confidence_score", 0)
            
            # Check minimum confidence threshold
            if confidence < self.min_pattern_confidence:
                return None
            
            # Check for actionable recommendations
            action = recommendations.get("action", "wait")
            if action in ["wait", "monitor_closely"]:
                return None
            
            # Extract signal details
            signal = {
                "action": action,
                "confidence": float(confidence),
                "current_price": float(current_bar['close']),
                "timestamp": self._convert_to_datetime(current_bar['timestamp']),
                "wyckoff_pattern": str(structure_analysis.get("current_structure", "unknown")),
                "phase": str(structure_analysis.get("phase", "unknown")),
                "entry_reason": "; ".join(recommendations.get("reasoning", [])),
                "key_levels": structure_analysis.get("key_levels", {}),
                "targets": recommendations.get("targets", []),
                "stop_loss": None
            }
            
            # Determine trade direction and levels
            if action == "prepare_long":
                signal["direction"] = TradeDirection.LONG
                signal["stop_loss"] = self._calculate_long_stop(structure_analysis, spring_upthrust, float(current_bar['close']))
                
            elif action == "prepare_short":
                signal["direction"] = TradeDirection.SHORT
                signal["stop_loss"] = self._calculate_short_stop(structure_analysis, spring_upthrust, float(current_bar['close']))
            
            else:
                return None
            
            # Validate risk/reward ratio
            if not self._validate_risk_reward(signal):
                return None
            
            return signal
            
        except Exception as e:
            logger.warning("Signal evaluation failed", error=str(e))
            return None
    
    def _calculate_long_stop(self, 
                           structure_analysis: Dict[str, Any], 
                           spring_upthrust: Dict[str, Any], 
                           current_price: float) -> float:
        """Calculate stop loss for long positions based on Wyckoff logic"""
        
        # Priority 1: Below spring low if spring detected
        springs = spring_upthrust.get("springs_detected", [])
        if springs:
            latest_spring = max(springs, key=lambda x: x['timestamp'])
            spring_low = latest_spring.get("price", current_price)
            return spring_low * 0.999  
        
        # Priority 2: Below structure support
        key_levels = structure_analysis.get("key_levels", {})
        support = key_levels.get("support")
        if support and support < current_price:
            return support * 0.999
        
        # Fallback: Percentage-based stop
        return current_price * (1 - self.risk_per_trade * 2)
    
    def _calculate_short_stop(self, 
                            structure_analysis: Dict[str, Any], 
                            spring_upthrust: Dict[str, Any], 
                            current_price: float) -> float:
        """Calculate stop loss for short positions based on Wyckoff logic"""
        
        # Priority 1: Above upthrust high if upthrust detected
        upthrusts = spring_upthrust.get("upthrusts_detected", [])
        if upthrusts:
            latest_upthrust = max(upthrusts, key=lambda x: x['timestamp'])
            upthrust_high = latest_upthrust.get("price", current_price)
            return upthrust_high * 1.001  
        
        # Priority 2: Above structure resistance
        key_levels = structure_analysis.get("key_levels", {})
        resistance = key_levels.get("resistance")
        if resistance and resistance > current_price:
            return resistance * 1.001
        
        # Fallback: Percentage-based stop
        return current_price * (1 + self.risk_per_trade * 2)
    
    def _validate_risk_reward(self, signal: Dict[str, Any]) -> bool:
        """Validate risk/reward ratio meets minimum requirements"""
        
        current_price = signal["current_price"]
        stop_loss = signal["stop_loss"]
        targets = signal.get("targets", [])
        
        if not stop_loss or not targets:
            return False
        
        # Calculate risk
        if signal["direction"] == TradeDirection.LONG:
            risk = current_price - stop_loss
            reward = targets[0] - current_price if targets else 0
        else:
            risk = stop_loss - current_price
            reward = current_price - targets[0] if targets else 0
        
        if risk <= 0 or reward <= 0:
            return False
        
        risk_reward_ratio = reward / risk
        return risk_reward_ratio >= self.min_risk_reward
    
    def _can_open_new_trade(self) -> bool:
        """Check if we can open a new trade based on risk management rules"""
        
        # Check open trades count
        open_trades = [trade for trade in self.trades if trade.status == "open"]
        max_concurrent_trades = 1  # Conservative approach
        
        return len(open_trades) < max_concurrent_trades
    
    async def _execute_trade(self, 
                           signal: Dict[str, Any], 
                           current_bar: pd.Series, 
                           wyckoff_analysis: Dict[str, Any]):
        """Execute a new trade based on signal"""
        
        try:
            # Calculate position size
            position_size = self._calculate_position_size(
                float(signal["current_price"]), 
                float(signal["stop_loss"])
            )
            
            # Create new trade
            self.trade_counter += 1
            trade = BacktestTrade(
                id=self.trade_counter,
                entry_time=self._convert_to_datetime(current_bar['timestamp']),
                direction=signal["direction"],
                entry_price=float(signal["current_price"]),
                stop_loss=float(signal["stop_loss"]),
                take_profit=float(signal["targets"][0]) if signal["targets"] else 0,
                position_size=float(position_size),
                commission=float(self.commission_per_trade),
                wyckoff_pattern=str(signal["wyckoff_pattern"]),
                pattern_confidence=float(signal["confidence"]),
                phase=str(signal["phase"]),
                entry_reason=str(signal["entry_reason"]),
                volume_confirmation=bool(signal.get("volume_confirmation", False)),
                market_context=str(signal.get("market_context", "unknown")),
                status="open"
            )
            
            self.trades.append(trade)
            
            logger.info("📈 Trade executed", 
                       trade_id=trade.id,
                       direction=trade.direction.value,
                       entry_price=trade.entry_price,
                       pattern=trade.wyckoff_pattern,
                       confidence=trade.pattern_confidence)
            
        except Exception as e:
            logger.error("Trade execution failed", error=str(e))
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management rules"""
        
        # Risk amount in dollars
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Distance to stop loss
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        # Position size calculation
        position_size = risk_amount / stop_distance
        
        # Apply maximum position size limit
        max_size = self.current_capital * self.max_position_size / entry_price
        position_size = min(position_size, max_size)
        
        return round(position_size, 0)
    
    def _ensure_dataframe(self, data, columns=None) -> pd.DataFrame:
        """Ensure data is a proper DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pd.Series):
            return pd.DataFrame(data).T
        elif isinstance(data, np.ndarray):
            if columns:
                return pd.DataFrame(data, columns=columns)
            else:
                return pd.DataFrame(data)
        elif isinstance(data, (list, dict)):
            return pd.DataFrame(data)
        else:
            logger.warning(f"Unexpected data type: {type(data)}, converting to DataFrame")
            return pd.DataFrame(data)

    def _convert_to_datetime(self, timestamp_value) -> datetime:
        """Convert various timestamp formats to datetime object"""
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        elif isinstance(timestamp_value, str):
            # Handle ISO format strings
            return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
        elif hasattr(timestamp_value, 'to_pydatetime'):
            # Handle pandas Timestamp
            return timestamp_value.to_pydatetime()
        else:
            # Fallback to pandas conversion
            ts = pd.to_datetime(timestamp_value)
            if hasattr(ts, 'to_pydatetime'):
                return ts.to_pydatetime()
            else:
                return ts
    
    def _update_open_trades(self, current_bar: pd.Series):
        """Update all open trades with current market data"""
        
        current_price = float(current_bar['close'])
        current_time = self._convert_to_datetime(current_bar['timestamp'])
        
        for trade in self.trades:
            if trade.status != "open":
                continue
            
            # Update trade metrics
            trade.bars_in_trade += 1
            
            # Calculate current P&L
            if trade.direction == TradeDirection.LONG:
                unrealized_pnl = (current_price - trade.entry_price) * trade.position_size
                
                # Update max favorable/adverse
                trade.max_favorable = max(trade.max_favorable, current_price - trade.entry_price)
                trade.max_adverse = min(trade.max_adverse, current_price - trade.entry_price)
                
                # Check stop loss
                if current_price <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, current_time, "Stop Loss Hit")
                
                # Check take profit
                elif trade.take_profit and current_price >= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, current_time, "Take Profit Hit")
            
            else:  # SHORT
                unrealized_pnl = (trade.entry_price - current_price) * trade.position_size
                
                # Update max favorable/adverse
                trade.max_favorable = max(trade.max_favorable, trade.entry_price - current_price)
                trade.max_adverse = min(trade.max_adverse, trade.entry_price - current_price)
                
                # Check stop loss
                if current_price >= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, current_time, "Stop Loss Hit")
                
                # Check take profit
                elif trade.take_profit and current_price <= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, current_time, "Take Profit Hit")
    
    def _close_trade(self, trade: BacktestTrade, exit_price: float, exit_time: datetime, reason: str):
        """Close an open trade"""
        
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.status = "closed"
        
        # Calculate final P&L
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size - trade.commission
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size - trade.commission
        
        trade.pnl_pct = (trade.pnl / self.current_capital) * 100
        
        # Update capital
        self.current_capital += trade.pnl
        
        logger.info("💰 Trade closed", 
                   trade_id=trade.id,
                   pnl=trade.pnl,
                   reason=reason,
                   duration_bars=trade.bars_in_trade)
    
    def _calculate_current_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized P&L"""
        
        equity = self.current_capital
        
        # Add unrealized P&L from open trades
        for trade in self.trades:
            if trade.status == "open":
                if trade.direction == TradeDirection.LONG:
                    unrealized_pnl = (current_price - trade.entry_price) * trade.position_size
                else:
                    unrealized_pnl = (trade.entry_price - current_price) * trade.position_size
                
                equity += unrealized_pnl
        
        return equity
    
    def _calculate_results(self, df: pd.DataFrame) -> BacktestResults:
        """Calculate comprehensive backtest results with enhanced metrics"""
        
        # Close any remaining open trades at final price
        if self.trades:
            final_bar = df.iloc[-1]
            final_time = self._convert_to_datetime(final_bar['timestamp'])
            final_price = float(final_bar['close'])
            
            for trade in self.trades:
                if trade.status == "open":
                    self._close_trade(trade, final_price, final_time, "Backtest End")
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        
        if total_trades == 0:
            return BacktestResults()
        
        # Performance calculations
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_pnl_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        winning_pnls = [t.pnl for t in self.trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
        avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
        largest_win = max(winning_pnls) if winning_pnls else 0
        largest_loss = min(losing_pnls) if losing_pnls else 0
        
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100
        
        # Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calmar ratio
        annual_return = total_pnl_pct
        calmar_ratio = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive(self.trades, True)
        consecutive_losses = self._calculate_max_consecutive(self.trades, False)
        
        # Time-based metrics
        avg_trade_duration = sum(trade.bars_in_trade for trade in self.trades) / total_trades
        
        # Wyckoff-specific metrics
        pattern_trades = {}
        phase_distribution = {}
        
        for trade in self.trades:
            # Pattern analysis
            pattern = trade.wyckoff_pattern
            if pattern not in pattern_trades:
                pattern_trades[pattern] = {"total": 0, "wins": 0}
            pattern_trades[pattern]["total"] += 1
            if trade.pnl > 0:
                pattern_trades[pattern]["wins"] += 1
            
            # Phase distribution
            phase = trade.phase
            phase_distribution[phase] = phase_distribution.get(phase, 0) + 1
        
        pattern_success_rates = {
            pattern: (data["wins"] / data["total"]) * 100 
            for pattern, data in pattern_trades.items()
        }
        
        # Count specific trade types
        accumulation_trades = len([t for t in self.trades if "accumulation" in t.wyckoff_pattern.lower()])
        distribution_trades = len([t for t in self.trades if "distribution" in t.wyckoff_pattern.lower()])
        spring_trades = len([t for t in self.trades if "spring" in t.entry_reason.lower()])
        upthrust_trades = len([t for t in self.trades if "upthrust" in t.entry_reason.lower()])
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(df)
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=float(sharpe_ratio),
            calmar_ratio=float(calmar_ratio),
            var_95=float(var_95),
            avg_trade_duration=avg_trade_duration,
            start_date=df.iloc[0]['timestamp'] if len(df) > 0 else None,
            end_date=df.iloc[-1]['timestamp'] if len(df) > 0 else None,
            accumulation_trades=accumulation_trades,
            distribution_trades=distribution_trades,
            spring_trades=spring_trades,
            upthrust_trades=upthrust_trades,
            pattern_success_rates=pattern_success_rates,
            phase_distribution=phase_distribution,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            monthly_returns=monthly_returns,
            equity_curve=self.equity_curve.copy(),
            trades=self.trades.copy()
        )
    
    def _calculate_max_consecutive(self, trades: List[BacktestTrade], wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if (trade.pnl > 0) == wins:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_monthly_returns(self, df: pd.DataFrame) -> List[float]:
        """Calculate monthly returns from equity curve"""
        if 'equity' not in df.columns or len(df) == 0:
            return []
        
        try:
            df['month'] = df['timestamp'].dt.to_period('M')
            monthly_equity = df.groupby('month')['equity'].last()
            monthly_returns = monthly_equity.pct_change().dropna() * 100
            return monthly_returns.tolist()
        except:
            return []
    
    async def _get_historical_data(self, symbol: str, timeframe: str, 
                                 bars: int = 2000) -> List[Dict[str, Any]]:
        """Get historical data for backtesting"""
        try:
            from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
            
            async with OandaMCPWrapper("http://localhost:8000") as oanda:
                historical_data = await oanda.get_historical_data(symbol, timeframe, bars)
            
            if "error" in historical_data:
                logger.error(f"Failed to get historical data: {historical_data['error']}")
                return []
            
            return historical_data['data']
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return []
    
    def _apply_parameters(self, params: Dict[str, float]):
        """Apply parameter set to the backtest engine"""
        for param_name, value in params.items():
            if hasattr(self, param_name):
                setattr(self, param_name, value)
    
    def _apply_optimized_parameters(self, params: Dict[str, float]):
        """Apply optimized parameters"""
        self._apply_parameters(params)
        logger.info("Applied optimized parameters", parameters=params)
    
    def _reset_backtest_state(self):
        """Reset backtest state for new run"""
        self.current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.trade_counter = 0
    
    def _calculate_suite_statistics(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall statistics across all backtest runs"""
        
        all_results = []
        for symbol_results in suite_results['backtest_results'].values():
            for result in symbol_results.values():
                if isinstance(result, BacktestResults):
                    all_results.append(result)
        
        if not all_results:
            return {}
        
        return {
            'total_backtests': len(all_results),
            'avg_return': np.mean([r.total_pnl_pct for r in all_results]),
            'avg_win_rate': np.mean([r.win_rate for r in all_results]),
            'avg_sharpe': np.mean([r.sharpe_ratio for r in all_results]),
            'avg_max_drawdown': np.mean([r.max_drawdown_pct for r in all_results]),
            'profitable_backtests': sum(1 for r in all_results if r.total_pnl_pct > 0),
            'profitability_rate': sum(1 for r in all_results if r.total_pnl_pct > 0) / len(all_results) * 100
        }
    
    async def _save_suite_results(self, suite_results: Dict[str, Any]):
        """Save comprehensive suite results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path("backtest_results")
            results_dir.mkdir(exist_ok=True)
            
            # Save JSON summary
            json_file = results_dir / f"wyckoff_backtest_suite_{timestamp}.json"
            
            # Convert results to JSON-serializable format
            serializable_results = self._make_json_serializable(suite_results)
            
            with open(json_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"✅ Suite results saved to {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to save suite results: {str(e)}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (BacktestResults, OptimizationResult, BacktestTrade)):
            return self._make_json_serializable(obj.__dict__)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, TradeDirection):
            return obj.value
        else:
            return obj
    
    async def _generate_visual_reports(self, suite_results: Dict[str, Any]):
        """Generate visual reports and charts"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            charts_dir = Path("backtest_charts")
            charts_dir.mkdir(exist_ok=True)
            
            # Set up plotting style - use a more compatible style
            try:
                plt.style.use('seaborn-v0_8')
            except OSError:
                try:
                    plt.style.use('seaborn')
                except OSError:
                    plt.style.use('default')
            
            # Set color palette
            try:
                sns.set_palette("husl")
            except Exception:
                pass  # Continue without seaborn palette if it fails
            
            # Generate equity curves
            self._plot_equity_curves(suite_results, charts_dir, timestamp)
            
            # Generate performance comparison
            self._plot_performance_comparison(suite_results, charts_dir, timestamp)
            
            # Generate drawdown analysis
            self._plot_drawdown_analysis(suite_results, charts_dir, timestamp)
            
            # Generate optimization results
            if suite_results.get('optimization_results'):
                self._plot_optimization_results(suite_results, charts_dir, timestamp)
            
            logger.info(f"✅ Visual reports generated in {charts_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate visual reports: {str(e)}")
    
    def _plot_equity_curves(self, suite_results: Dict[str, Any], charts_dir: Path, timestamp: str):
        """Plot equity curves for all backtests"""
        
        plt.figure(figsize=(15, 10))
        
        # Define colors manually for compatibility
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        plot_count = 0
        for symbol, symbol_results in suite_results['backtest_results'].items():
            for timeframe, result in symbol_results.items():
                if isinstance(result, BacktestResults) and result.equity_curve:
                    plt.subplot(2, 2, (plot_count % 4) + 1)
                    color = colors[plot_count % len(colors)]
                    plt.plot(result.equity_curve, label=f"{symbol} {timeframe}", color=color)
                    plt.title(f"Equity Curve - {symbol} {timeframe}")
                    plt.xlabel("Bars")
                    plt.ylabel("Equity ($)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plot_count += 1
                    
                    # Limit to 4 plots to avoid overcrowding
                    if plot_count >= 4:
                        break
            if plot_count >= 4:
                break
        
        plt.tight_layout()
        plt.savefig(charts_dir / f"equity_curves_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, suite_results: Dict[str, Any], charts_dir: Path, timestamp: str):
        """Plot performance comparison across different configurations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Collect data
        returns = []
        win_rates = []
        sharpe_ratios = []
        max_drawdowns = []
        profit_factors = []
        labels = []
        
        for symbol, symbol_results in suite_results['backtest_results'].items():
            for timeframe, result in symbol_results.items():
                if isinstance(result, BacktestResults):
                    returns.append(result.total_pnl_pct)
                    win_rates.append(result.win_rate)
                    sharpe_ratios.append(result.sharpe_ratio)
                    max_drawdowns.append(abs(result.max_drawdown_pct))
                    profit_factors.append(min(result.profit_factor, 5))  # Cap for visualization
                    labels.append(f"{symbol}\n{timeframe}")
        
        # Create a simple color scheme that works with all matplotlib versions
        if len(returns) > 0:
            # Use a simple color cycle
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            colors = colors * (len(returns) // len(colors) + 1)  # Repeat colors if needed
            colors = colors[:len(returns)]  # Trim to exact length needed
        else:
            colors = ['blue']
        
        # Plot comparisons
        metrics = [
            (returns, "Total Return (%)", 0),
            (win_rates, "Win Rate (%)", 1),
            (sharpe_ratios, "Sharpe Ratio", 2),
            (max_drawdowns, "Max Drawdown (%)", 3),
            (profit_factors, "Profit Factor", 4)
        ]
        
        for data, title, idx in metrics:
            if idx < len(axes) and len(data) > 0:
                axes[idx].bar(range(len(data)), data, color=colors[:len(data)])
                axes[idx].set_title(title)
                axes[idx].set_xticks(range(len(labels)))
                axes[idx].set_xticklabels(labels, rotation=45, ha='right')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplot
        if len(axes) > 5:
            axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(charts_dir / f"performance_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown_analysis(self, suite_results: Dict[str, Any], charts_dir: Path, timestamp: str):
        """Plot drawdown analysis"""
        
        plt.figure(figsize=(15, 8))
        
        # Define colors manually for compatibility
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        color_idx = 0
        
        for i, (symbol, symbol_results) in enumerate(suite_results['backtest_results'].items()):
            for j, (timeframe, result) in enumerate(symbol_results.items()):
                if isinstance(result, BacktestResults) and result.equity_curve:
                    # Calculate drawdown
                    equity = np.array(result.equity_curve)
                    peak = np.maximum.accumulate(equity)
                    drawdown = (equity - peak) / peak * 100
                    
                    plt.subplot(len(suite_results['backtest_results']), 1, i + 1)
                    color = colors[color_idx % len(colors)]
                    plt.plot(drawdown, label=f"{timeframe}", color=color)
                    plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color=color)
                    plt.title(f"Drawdown Analysis - {symbol}")
                    plt.xlabel("Bars")
                    plt.ylabel("Drawdown (%)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    color_idx += 1
        
        plt.tight_layout()
        plt.savefig(charts_dir / f"drawdown_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_optimization_results(self, suite_results: Dict[str, Any], charts_dir: Path, timestamp: str):
        """Plot optimization results"""
        
        opt_results = suite_results.get('optimization_results', {})
        if not opt_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Define colors manually for compatibility
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']
        
        for idx, (config, result) in enumerate(opt_results.items()):
            if idx >= 4:  # Limit to 4 plots
                break
            
            if isinstance(result, OptimizationResult):
                # Create parameter sensitivity analysis
                params = result.parameters
                param_names = list(params.keys())
                param_values = list(params.values())
                
                color = colors[idx % len(colors)]
                axes[idx].bar(param_names, param_values, color=color)
                axes[idx].set_title(f"Optimized Parameters - {config}")
                axes[idx].set_ylabel("Parameter Value")
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(opt_results), 4):
            if idx < len(axes):
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(charts_dir / f"optimization_results_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

class EnhancedWyckoffDataProcessor:
    """
    Enhanced data processor that integrates with your existing AutomatedWyckoffBacktester
    Fixes the OANDA nested candles structure and prepares data for your system
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    def fix_oanda_candles_structure(self, raw_dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Fix the nested candles structure from your OANDA data and convert to format
        expected by your AutomatedWyckoffBacktester._prepare_backtest_data() method
        
        This directly addresses the error: "Data must contain 'time' or 'timestamp' column"
        """
        try:
            self.logger.info(f"🔧 Fixing OANDA data structure for {len(raw_dataframe)} rows")
            
            processed_data = []
            
            for idx, row in raw_dataframe.iterrows():
                try:
                    # Extract the nested candle data
                    candle_data = row['candles']
                    
                    # Handle different formats of nested data
                    if isinstance(candle_data, str):
                        # Parse JSON-like string
                        candle_data = self._parse_candle_string(candle_data)
                    elif isinstance(candle_data, dict):
                        # Already a dictionary
                        pass
                    else:
                        self.logger.warning(f"Unexpected candle data format at row {idx}: {type(candle_data)}")
                        continue
                    
                    # Extract OHLCV data in the format your backtest expects
                    processed_row = self._extract_ohlcv_data(candle_data, row)
                    if processed_row:
                        processed_data.append(processed_row)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process row {idx}: {str(e)}")
                    continue
            
            self.logger.info(f"✅ Successfully processed {len(processed_data)} candles")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Data structure fix failed: {str(e)}")
            raise ValueError(f"Failed to fix OANDA data structure: {str(e)}")
    
    def _parse_candle_string(self, candle_string: str) -> Dict[str, Any]:
        """Parse candle data from string format"""
        try:
            # Replace single quotes with double quotes for JSON parsing
            json_string = candle_string.replace("'", '"')
            return json.loads(json_string)
        except json.JSONDecodeError:
            # Try eval as fallback (be careful with this in production)
            try:
                return eval(candle_string)
            except:
                raise ValueError(f"Could not parse candle string: {candle_string[:100]}...")
    
    def _extract_ohlcv_data(self, candle_data: Dict[str, Any], original_row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Extract OHLCV data in the exact format expected by your backtesting engine
        """
        try:
            # Extract timestamp (multiple possible locations)
            timestamp = (
                candle_data.get('time') or 
                candle_data.get('timestamp') or 
                original_row.get('time') or
                original_row.get('timestamp')
            )
            
            if not timestamp:
                return None
            
            # Extract mid prices (OANDA format)
            mid_data = candle_data.get('mid', {})
            if not mid_data:
                # Try alternative structures
                mid_data = candle_data
            
            # Extract bid/ask if available (for spread analysis)
            bid_data = candle_data.get('bid', {})
            ask_data = candle_data.get('ask', {})
            
            # Create the data structure expected by your _prepare_backtest_data method
            processed_row = {
                'time': timestamp,  # This fixes the "must contain 'time'" error
                'timestamp': timestamp,  # Alternative timestamp field
                'open': float(mid_data.get('o', mid_data.get('open', 0))),
                'high': float(mid_data.get('h', mid_data.get('high', 0))),
                'low': float(mid_data.get('l', mid_data.get('low', 0))),
                'close': float(mid_data.get('c', mid_data.get('close', 0))),
                'volume': int(candle_data.get('volume', 1000)),  # Default volume if missing
                'complete': candle_data.get('complete', True),
                
                # Additional OANDA-specific data
                'instrument': original_row.get('instrument', 'EUR_USD'),
                'granularity': original_row.get('granularity', 'M15'),
                
                # Spread data if available
                'bid_open': float(bid_data.get('o', 0)) if bid_data else None,
                'bid_high': float(bid_data.get('h', 0)) if bid_data else None,
                'bid_low': float(bid_data.get('l', 0)) if bid_data else None,
                'bid_close': float(bid_data.get('c', 0)) if bid_data else None,
                'ask_open': float(ask_data.get('o', 0)) if ask_data else None,
                'ask_high': float(ask_data.get('h', 0)) if ask_data else None,
                'ask_low': float(ask_data.get('l', 0)) if ask_data else None,
                'ask_close': float(ask_data.get('c', 0)) if ask_data else None,
            }
            
            # Validate the data
            if self._validate_candle_data(processed_row):
                return processed_row
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to extract OHLCV data: {str(e)}")
            return None
    
    def _validate_candle_data(self, candle: Dict[str, Any]) -> bool:
        """Validate that candle data is reasonable"""
        try:
            # Check required fields
            required_fields = ['open', 'high', 'low', 'close']
            for field in required_fields:
                if field not in candle or candle[field] <= 0:
                    return False
            
            # Check OHLC relationships
            if not (candle['low'] <= candle['open'] <= candle['high'] and
                    candle['low'] <= candle['close'] <= candle['high']):
                return False
            
            # Check for reasonable price ranges (EUR/USD example)
            if candle['close'] < 0.5 or candle['close'] > 2.0:
                return False
            
            return True
            
        except:
            return False



class IntegratedWyckoffBacktester(AutomatedWyckoffBacktester):
    """
    Enhanced version of your AutomatedWyckoffBacktester that integrates the data fix
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.data_processor = EnhancedWyckoffDataProcessor(logger=self.logger)
    
    def _prepare_backtest_data(self, 
                              price_data: List[Dict[str, Any]], 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Enhanced version of your _prepare_backtest_data method that handles the nested structure
        """
        
        try:
            # Check if we have the problematic nested structure
            if self._has_nested_candles_structure(price_data):
                self.logger.info("🔧 Detected nested candles structure, applying fix...")
                
                # Convert to DataFrame first if it's not already
                if isinstance(price_data, list):
                    df = pd.DataFrame(price_data)
                else:
                    df = price_data
                
                # Apply the fix
                fixed_data = self.data_processor.fix_oanda_candles_structure(df)
                print(f"fixed_data*****: {fixed_data}")
                
                # Continue with your original logic using the fixed data
                return super()._prepare_backtest_data(fixed_data, start_date, end_date)
            
            else:
                # Data is already in correct format, use original method
                return super()._prepare_backtest_data(price_data, start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"Enhanced data preparation failed: {str(e)}")
            raise ValueError(f"Failed to prepare backtest data: {str(e)}")
    
    def _has_nested_candles_structure(self, data) -> bool:
        """
        Detect if data has the problematic nested candles structure
        """
        try:
            if isinstance(data, list) and len(data) > 0:
                # Convert to DataFrame to check structure
                df = pd.DataFrame(data)
                return 'candles' in df.columns and 'time' not in df.columns and 'timestamp' not in df.columns
            elif isinstance(data, pd.DataFrame):
                return 'candles' in data.columns and 'time' not in data.columns and 'timestamp' not in data.columns
            return False
        except:
            return False
    
    async def run_backtest_with_enhanced_data_handling(self, 
                                                     raw_oanda_data,
                                                     start_date: Optional[datetime] = None,
                                                     end_date: Optional[datetime] = None) -> BacktestResults:
        """
        New method that specifically handles your OANDA data format
        """
        
        self.logger.info("🧪 Starting enhanced Wyckoff backtest with OANDA data fix")
        
        try:
            # Step 1: Fix the data structure
            if isinstance(raw_oanda_data, pd.DataFrame):
                processed_data = self.data_processor.fix_oanda_candles_structure(raw_oanda_data)
            else:
                # Assume it's already a list
                processed_data = raw_oanda_data
            
            # Step 2: Run your existing backtest logic
            results = await self.run_backtest(processed_data, start_date, end_date)
            
            # Step 3: Add enhanced reporting for OANDA-specific metrics
            self._add_oanda_specific_metrics(results, processed_data)
            
            return results
            
        except Exception as e:
            self.logger.error("❌ Enhanced backtest failed", error=str(e))
            raise
    
    def _add_oanda_specific_metrics(self, results: BacktestResults, processed_data: List[Dict[str, Any]]):
        """Add OANDA-specific metrics to results"""
        
        try:
            # Calculate spread metrics if available
            spreads = []
            for candle in processed_data:
                if (candle.get('ask_close') and candle.get('bid_close') and 
                    candle['ask_close'] > 0 and candle['bid_close'] > 0):
                    spread = candle['ask_close'] - candle['bid_close']
                    spreads.append(spread)
            
            if spreads:
                # Add spread analysis to results
                results.avg_spread = float(np.mean(spreads))
                results.max_spread = float(np.max(spreads))
                results.min_spread = float(np.min(spreads))
                results.spread_volatility = float(np.std(spreads))
            
            # Add data quality metrics
            results.data_completeness = len([c for c in processed_data if c.get('complete', True)]) / len(processed_data) * 100
            results.volume_availability = len([c for c in processed_data if c.get('volume', 0) > 0]) / len(processed_data) * 100
            
        except Exception as e:
            self.logger.warning(f"Failed to add OANDA-specific metrics: {str(e)}")


# Global automated backtest engine
#automated_backtest_engine = AutomatedWyckoffBacktester()
automated_backtest_engine = IntegratedWyckoffBacktester()

async def run_automated_wyckoff_backtest():
    """Run the complete automated Wyckoff backtest suite"""
    
    print("🤖 AUTOMATED WYCKOFF STRATEGY BACKTESTING SUITE")
    print("=" * 60)
    print("🚀 Initializing comprehensive automated testing...")
    
    try:
        # Configure test parameters
        symbols = ["EUR_USD", "GBP_USD", "USD_JPY"]  # Multiple currency pairs
        timeframes = ["M15", "H1"]  # Multiple timeframes
        
        print(f"📊 Testing Configuration:")
        print(f"   💱 Symbols: {', '.join(symbols)}")
        print(f"   ⏰ Timeframes: {', '.join(timeframes)}")
        print(f"   🔧 Parameter Optimization: Enabled")
        print(f"   🚶 Walk-Forward Analysis: Enabled")
        print(f"   📈 Visual Reports: Enabled")
        
        # Initialize automated backtest engine
        engine = IntegratedWyckoffBacktester(
            initial_capital=100000,
            commission_per_trade=5.0,
            risk_per_trade=0.02,
            max_position_size=0.1
        )
        
        print("\n⚙️ Engine Configuration:")
        print(f"   💰 Initial Capital: ${engine.initial_capital:,.2f}")
        print(f"   📊 Risk per Trade: {engine.risk_per_trade*100}%")
        print(f"   💸 Commission: ${engine.commission_per_trade}")
        print(f"   🎯 Min Pattern Confidence: {engine.min_pattern_confidence}%")
        print(f"   ⚖️ Min Risk/Reward: {engine.min_risk_reward}:1")
        
        print("\n🚀 Starting automated backtest suite...")
        print("This includes:")
        print("   1. Multi-symbol testing")
        print("   2. Multi-timeframe analysis") 
        print("   3. Parameter optimization")
        print("   4. Walk-forward validation")
        print("   5. Comprehensive reporting")
        
        # Run the automated suite
        suite_results = await engine.run_automated_backtest_suite(
            symbols=symbols,
            timeframes=timeframes,
            optimization=True,
            walk_forward=True
        )
        
        # Display summary results
        print("\n" + "=" * 60)
        print("📋 AUTOMATED BACKTEST SUITE RESULTS")
        print("=" * 60)
        
        summary = suite_results.get('summary_statistics', {})
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Backtests: {summary.get('total_backtests', 0)}")
        print(f"   Average Return: {summary.get('avg_return', 0):.2f}%")
        print(f"   Average Win Rate: {summary.get('avg_win_rate', 0):.1f}%")
        print(f"   Average Sharpe Ratio: {summary.get('avg_sharpe', 0):.2f}")
        print(f"   Average Max Drawdown: {summary.get('avg_max_drawdown', 0):.2f}%")
        print(f"   Profitable Backtests: {summary.get('profitable_backtests', 0)}/{summary.get('total_backtests', 0)}")
        print(f"   Profitability Rate: {summary.get('profitability_rate', 0):.1f}%")
        
        # Detailed results by symbol and timeframe
        print(f"\n📈 DETAILED RESULTS BY CONFIGURATION:")
        for symbol, symbol_results in suite_results['backtest_results'].items():
            print(f"\n{symbol}:")
            for timeframe, result in symbol_results.items():
                if isinstance(result, BacktestResults):
                    print(f"   {timeframe}:")
                    print(f"      Return: {result.total_pnl_pct:.2f}%")
                    print(f"      Win Rate: {result.win_rate:.1f}%")
                    print(f"      Trades: {result.total_trades}")
                    print(f"      Sharpe: {result.sharpe_ratio:.2f}")
                    print(f"      Max DD: {result.max_drawdown_pct:.2f}%")
        
        # Optimization results
        if suite_results.get('optimization_results'):
            print(f"\n🔧 OPTIMIZATION RESULTS:")
            for config, opt_result in suite_results['optimization_results'].items():
                if isinstance(opt_result, OptimizationResult):
                    print(f"\n{config}:")
                    print(f"   Optimization Score: {opt_result.score:.1f}")
                    print(f"   Best Parameters:")
                    for param, value in opt_result.parameters.items():
                        print(f"      {param}: {value}")
        
        # Walk-forward analysis summary
        if suite_results.get('walk_forward_results'):
            print(f"\n🚶 WALK-FORWARD VALIDATION:")
            for config, wf_result in suite_results['walk_forward_results'].items():
                if 'performance_consistency' in wf_result:
                    consistency = wf_result['performance_consistency']
                    print(f"\n{config}:")
                    print(f"   Avg Out-of-Sample Return: {consistency.get('avg_return', 0):.2f}%")
                    print(f"   Return Volatility: {consistency.get('return_volatility', 0):.2f}%")
                    print(f"   Positive Periods: {consistency.get('positive_periods', 0):.1f}%")
                    print(f"   Avg Sharpe: {consistency.get('avg_sharpe', 0):.2f}")
        
        print(f"\n📊 Reports and charts saved to:")
        print(f"   JSON Results: backtest_results/")
        print(f"   Visual Charts: backtest_charts/")
        
        print(f"\n✅ Automated backtest suite completed successfully!")
        print(f"   Total runtime: {datetime.now() - suite_results['timestamp']}")
        
        return suite_results
        
    except Exception as e:
        print(f"❌ Automated backtest suite failed: {str(e)}")
        logger.error(f"Automated backtest suite failed: {str(e)}")
        return None


async def run_quick_wyckoff_validation():
    """Run a quick validation backtest for development/testing"""
    
    print("⚡ QUICK WYCKOFF VALIDATION")
    print("=" * 40)
    
    try:
        # Initialize with faster settings
        engine = IntegratedWyckoffBacktester(
            initial_capital=50000,
            commission_per_trade=3.0,
            risk_per_trade=0.015,
            max_position_size=0.05
        )
        
        # Reduce optimization parameters for speed
        engine.optimization_params = {
            'min_pattern_confidence': [60, 70],
            'min_risk_reward': [2.0, 2.5],
            'volume_threshold': [1.5, 2.0]
        }
        
        print("🚀 Running quick validation on EUR_USD M15...")
        
        # Get limited data for quick test
        historical_data = await engine._get_historical_data("EUR_USD", "M15", 500)
        processor = EnhancedWyckoffDataProcessor()
        fixed_data = processor.fix_oanda_candles_structure(pd.DataFrame(historical_data))
        #print(f"-----historical data-----: {fixed_data}")
        if not fixed_data:
            print("❌ No data available for quick test")
            return
        
        # Run single backtest
        result = await engine.run_backtest(fixed_data)
        
        print(f"\n📊 QUICK VALIDATION RESULTS:")
        print(f"   Return: {result.total_pnl_pct:.2f}%")
        print(f"   Win Rate: {result.win_rate:.1f}%")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        print(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        if result.total_trades > 0:
            print(f"\n🎯 WYCKOFF PATTERN ANALYSIS:")
            for pattern, success_rate in result.pattern_success_rates.items():
                trades_count = sum(1 for t in result.trades if t.wyckoff_pattern == pattern)
                print(f"   {pattern}: {success_rate:.1f}% ({trades_count} trades)")
        
        print(f"\n✅ Quick validation completed!")
        return result
        
    except Exception as e:
        print(f"❌ Quick validation failed: {str(e)}")
        return None


class BacktestScheduler:
    """Scheduler for automated periodic backtesting"""
    
    def __init__(self, engine: IntegratedWyckoffBacktester):
        self.engine = engine
        self.running = False
        
    async def start_scheduled_backtests(self, 
                                      interval_hours: int = 24,
                                      symbols: Optional[List[str]] = None,
                                      timeframes: Optional[List[str]] = None):
        """Start scheduled automated backtests"""
        
        symbols = symbols or ["EUR_USD", "GBP_USD"]
        timeframes = timeframes or ["M15", "H1"]
        
        logger.info("📅 Starting scheduled backtests", 
                   interval_hours=interval_hours,
                   symbols=symbols,
                   timeframes=timeframes)
        
        self.running = True
        
        while self.running:
            try:
                print(f"\n🕐 Running scheduled backtest at {datetime.now()}")
                
                # Run automated suite
                results = await self.engine.run_automated_backtest_suite(
                    symbols=symbols,
                    timeframes=timeframes,
                    optimization=True,
                    walk_forward=False  # Skip walk-forward for scheduled runs
                )
                
                # Log results
                summary = results.get('summary_statistics', {})
                logger.info("📊 Scheduled backtest completed",
                           avg_return=summary.get('avg_return', 0),
                           profitability_rate=summary.get('profitability_rate', 0))
                
                # Wait for next interval
                await asyncio.sleep(interval_hours * 3600)  # Convert hours to seconds
                
            except Exception as e:
                logger.error(f"Scheduled backtest failed: {str(e)}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    def stop_scheduled_backtests(self):
        """Stop scheduled backtests"""
        self.running = False
        logger.info("📅 Scheduled backtests stopped")


# Utility functions for external integration
async def run_parameter_sensitivity_analysis(symbol: str = "EUR_USD", 
                                           timeframe: str = "M15") -> Dict[str, Any]:
    """Run sensitivity analysis on key parameters"""
    
    print("🔬 PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 50)
    
    engine = IntegratedWyckoffBacktester()
    historical_data = await engine._get_historical_data(symbol, timeframe, 1000)
    
    if not historical_data:
        return {"error": "No data available"}
    
    # Parameters to test
    sensitivity_params = {
        'min_pattern_confidence': [40, 50, 60, 70, 80, 90],
        'min_risk_reward': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        'volume_threshold': [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        'risk_per_trade': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    }
    
    sensitivity_results = {}
    
    for param_name, values in sensitivity_params.items():
        print(f"\n🧪 Testing {param_name} sensitivity...")
        param_results = []
        
        for value in values:
            try:
                # Reset and configure
                engine._reset_backtest_state()
                setattr(engine, param_name, value)
                
                # Run backtest
                result = await engine.run_backtest(historical_data.copy())
                
                param_results.append({
                    'value': value,
                    'return': result.total_pnl_pct,
                    'win_rate': result.win_rate,
                    'trades': result.total_trades,
                    'sharpe': result.sharpe_ratio,
                    'max_dd': result.max_drawdown_pct
                })
                
                print(f"   {param_name}={value}: Return={result.total_pnl_pct:.1f}%, Trades={result.total_trades}")
                
            except Exception as e:
                logger.warning(f"Sensitivity test failed for {param_name}={value}", error=str(e))
                continue
        
        sensitivity_results[param_name] = param_results
    
    print("\n✅ Parameter sensitivity analysis completed!")
    return sensitivity_results


async def validate_wyckoff_system_robustness() -> Dict[str, Any]:
    """Comprehensive system robustness validation"""
    
    print("🛡️ WYCKOFF SYSTEM ROBUSTNESS VALIDATION")
    print("=" * 50)
    
    validation_results = {
        'monte_carlo_results': {},
        'stress_test_results': {},
        'regime_analysis': {},
        'overfitting_checks': {}
    }
    
    try:
        engine = IntegratedWyckoffBacktester()
        
        # 1. Monte Carlo simulation
        print("\n🎲 Running Monte Carlo simulation...")
        mc_results = await _run_monte_carlo_validation(engine)
        validation_results['monte_carlo_results'] = mc_results
        
        # 2. Stress testing
        print("\n⚡ Running stress tests...")
        stress_results = await _run_stress_tests(engine)
        validation_results['stress_test_results'] = stress_results
        
        # 3. Market regime analysis
        print("\n📊 Analyzing market regime performance...")
        regime_results = await _analyze_market_regimes(engine)
        validation_results['regime_analysis'] = regime_results
        
        # 4. Overfitting checks
        print("\n🔍 Checking for overfitting...")
        overfitting_results = await _check_overfitting(engine)
        validation_results['overfitting_checks'] = overfitting_results
        
        print("\n✅ System robustness validation completed!")
        return validation_results
        
    except Exception as e:
        logger.error(f"Robustness validation failed: {str(e)}")
        return {"error": str(e)}


async def _run_monte_carlo_validation(engine: IntegratedWyckoffBacktester, 
                                    iterations: int = 100) -> Dict[str, Any]:
    """Run Monte Carlo simulation for strategy validation"""
    
    # Get base data
    historical_data = await engine._get_historical_data("EUR_USD", "M15", 1000)
    if not historical_data:
        return {"error": "No data available"}
    
    mc_results = []
    
    for i in range(iterations):
        try:
            # Shuffle data while maintaining basic structure
            shuffled_data = historical_data.copy()
            # Convert to numpy array for shuffling, then back to list
            shuffled_array = np.array(shuffled_data, dtype=object)
            np.random.shuffle(shuffled_array)
            shuffled_data = shuffled_array.tolist()
            
            # Run backtest on shuffled data
            engine._reset_backtest_state()
            result = await engine.run_backtest(shuffled_data)
            
            mc_results.append({
                'iteration': i + 1,
                'return': result.total_pnl_pct,
                'trades': result.total_trades,
                'win_rate': result.win_rate
            })
            
        except Exception as e:
            continue
    
    if mc_results:
        returns = [r['return'] for r in mc_results]
        return {
            'iterations': len(mc_results),
            'avg_return': np.mean(returns),
            'return_std': np.std(returns),
            'positive_results': sum(1 for r in returns if r > 0) / len(returns) * 100,
            'results': mc_results
        }
    else:
        return {"error": "Monte Carlo simulation failed"}


async def _run_stress_tests(engine: IntegratedWyckoffBacktester) -> Dict[str, Any]:
    """Run stress tests with extreme market conditions"""
    
    stress_scenarios = {
        'high_volatility': {'volume_multiplier': 3.0, 'price_volatility': 2.0},
        'low_volatility': {'volume_multiplier': 0.3, 'price_volatility': 0.5},
        'trending_market': {'trend_strength': 2.0},
        'choppy_market': {'noise_level': 3.0}
    }
    
    stress_results = {}
    base_data = await engine._get_historical_data("EUR_USD", "M15", 500)
    
    for scenario_name, params in stress_scenarios.items():
        try:
            # Generate stressed data
            stressed_data = _generate_stressed_data(base_data, params)
            
            # Run backtest
            engine._reset_backtest_state()
            result = await engine.run_backtest(stressed_data)
            
            stress_results[scenario_name] = {
                'return': result.total_pnl_pct,
                'max_drawdown': result.max_drawdown_pct,
                'trades': result.total_trades,
                'win_rate': result.win_rate
            }
            
        except Exception as e:
            stress_results[scenario_name] = {"error": str(e)}
    
    return stress_results


def _generate_stressed_data(base_data: List[Dict[str, Any]], 
                          stress_params: Dict[str, float]) -> List[Dict[str, Any]]:
    """Generate stressed market data for testing"""
    
    stressed_data = []
    
    for bar in base_data:
        new_bar = bar.copy()
        
        # Apply stress parameters
        if 'volume_multiplier' in stress_params:
            new_bar['volume'] = int(bar['volume'] * stress_params['volume_multiplier'])
        
        if 'price_volatility' in stress_params:
            vol_factor = stress_params['price_volatility']
            price_change = (bar['high'] - bar['low']) * (vol_factor - 1)
            new_bar['high'] += price_change / 2
            new_bar['low'] -= price_change / 2
        
        stressed_data.append(new_bar)
    
    return stressed_data


async def _analyze_market_regimes(engine: IntegratedWyckoffBacktester) -> Dict[str, Any]:
    """Analyze performance across different market regimes"""
    
    # This would ideally use actual regime classification
    # For now, use a simplified approach based on volatility
    
    historical_data = await engine._get_historical_data("EUR_USD", "M15", 2000)
    if not historical_data:
        return {"error": "No data available"}
    
    # Split data into volatility regimes
    df = pd.DataFrame(historical_data)
    df['volatility'] = (df['high'] - df['low']) / df['close']
    vol_threshold = df['volatility'].median()
    
    # Filter and ensure we maintain DataFrame structure
    low_vol_mask = df['volatility'] <= vol_threshold
    low_vol_df = df[low_vol_mask].copy()
    
    high_vol_mask = df['volatility'] > vol_threshold
    high_vol_df = df[high_vol_mask].copy()
    
    # Convert to list of dictionaries manually
    low_vol_data = []
    for idx in range(len(low_vol_df)):
        row = low_vol_df.iloc[idx]
        row_dict = {
            'time': str(row.get('time', '')),
            'open': float(row.get('open', 0)),
            'high': float(row.get('high', 0)),
            'low': float(row.get('low', 0)),
            'close': float(row.get('close', 0)),
            'volume': int(row.get('volume', 0))
        }
        low_vol_data.append(row_dict)
    
    high_vol_data = []
    for idx in range(len(high_vol_df)):
        row = high_vol_df.iloc[idx]
        row_dict = {
            'time': str(row.get('time', '')),
            'open': float(row.get('open', 0)),
            'high': float(row.get('high', 0)),
            'low': float(row.get('low', 0)),
            'close': float(row.get('close', 0)),
            'volume': int(row.get('volume', 0))
        }
        high_vol_data.append(row_dict)
    
    regime_results = {}
    
    for regime_name, data in [('low_volatility', low_vol_data), ('high_volatility', high_vol_data)]:
        if len(data) > 100:  # Minimum data requirement
            try:
                engine._reset_backtest_state()
                result = await engine.run_backtest(data)
                
                regime_results[regime_name] = {
                    'return': result.total_pnl_pct,
                    'win_rate': result.win_rate,
                    'trades': result.total_trades,
                    'sharpe': result.sharpe_ratio,
                    'data_points': len(data)
                }
            except Exception as e:
                regime_results[regime_name] = {"error": str(e)}
    
    return regime_results


async def _check_overfitting(engine: IntegratedWyckoffBacktester) -> Dict[str, Any]:
    """Check for overfitting using out-of-sample testing"""
    
    historical_data = await engine._get_historical_data("EUR_USD", "M15", 1500)
    if not historical_data or len(historical_data) < 1000:
        return {"error": "Insufficient data for overfitting check"}
    
    # Split data: 70% training, 30% testing
    split_point = int(len(historical_data) * 0.7)
    training_data = historical_data[:split_point]
    testing_data = historical_data[split_point:]
    
    try:
        # Optimize on training data
        opt_result = await engine._optimize_parameters(training_data)
        engine._apply_optimized_parameters(opt_result.parameters)
        
        # Test on out-of-sample data
        engine._reset_backtest_state()
        in_sample_result = await engine.run_backtest(training_data)
        
        engine._reset_backtest_state()
        out_sample_result = await engine.run_backtest(testing_data)
        
        # Calculate overfitting metrics
        return_degradation = (in_sample_result.total_pnl_pct - out_sample_result.total_pnl_pct) / abs(in_sample_result.total_pnl_pct) * 100
        
        return {
            'in_sample_return': in_sample_result.total_pnl_pct,
            'out_sample_return': out_sample_result.total_pnl_pct,
            'return_degradation_pct': return_degradation,
            'in_sample_sharpe': in_sample_result.sharpe_ratio,
            'out_sample_sharpe': out_sample_result.sharpe_ratio,
            'overfitting_detected': return_degradation > 50  # Threshold for overfitting
        }
        
    except Exception as e:
        return {"error": str(e)}


# Main execution functions
if __name__ == "__main__":
    #from autonomous_trading_system.utils.wyckoff_data_processor import IntegratedWyckoffBacktester
    print("🤖 Automated Wyckoff Backtesting Engine")
    print("Choose an option:")
    print("1. Run full automated suite")
    print("2. Run quick validation") 
    print("3. Run parameter sensitivity analysis")
    print("4. Run robustness validation")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        asyncio.run(run_automated_wyckoff_backtest())
    elif choice == "2":
        asyncio.run(run_quick_wyckoff_validation())
    elif choice == "3":
        asyncio.run(run_parameter_sensitivity_analysis())
    elif choice == "4":
        asyncio.run(validate_wyckoff_system_robustness())
    else:
        print("Invalid choice. Exiting.")
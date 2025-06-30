"""
Backtesting Simulation Tools for Orchestrator Agent
Tools for realistic market simulation and agent coordination
"""

import sys
from pathlib import Path
import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from crewai.tools import tool
from config.logging_config import logger

# Backtesting simulation models
class BacktestMarketContext(BaseModel):
    """Realistic market context for agents"""
    current_price: float
    bid: float 
    ask: float
    spread: float
    timestamp: datetime
    volatility: float
    market_hours: str
    volume: float
    historical_bars: List[Dict]
    account_balance: float
    equity: float
    margin_available: float

class BacktestExecutionResult(BaseModel):
    """Trade execution simulation result"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    requested_price: float
    executed_price: float
    slippage: float
    commission: float
    execution_time_ms: int
    status: str
    partial_fill: bool = False

@dataclass
class BacktestPortfolio:
    """Portfolio state during backtesting"""
    initial_balance: float
    current_balance: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: List[Dict]
    closed_trades: List[Dict]
    margin_used: float
    free_margin: float

# Simulation Tools
@tool
def simulate_historical_market_context(
    historical_bars: str,  # JSON string of bars
    current_bar_index: int,
    account_info: str  # JSON string of account state
) -> str:
    """
    Convert historical price data into realistic market context that matches live trading conditions.
    
    Args:
        historical_bars: JSON string containing historical OHLC data
        current_bar_index: Index of current bar being processed
        account_info: JSON string containing current account state
    
    Returns:
        JSON string with realistic market context for trading agents
    """
    try:
        bars = json.loads(historical_bars)
        account = json.loads(account_info)
        
        if current_bar_index >= len(bars):
            return json.dumps({"error": "Bar index out of range"})
        
        current_bar = bars[current_bar_index]
        current_price = current_bar['close']
        
        # Calculate realistic spread based on volatility and market hours
        volatility = calculate_recent_volatility(bars[max(0, current_bar_index-20):current_bar_index+1])
        spread = calculate_realistic_spread(current_price, volatility, current_bar['timestamp'])
        
        # Create market context that mimics live conditions
        market_context = BacktestMarketContext(
            current_price=current_price,
            bid=current_price - spread/2,
            ask=current_price + spread/2,
            spread=spread,
            timestamp=datetime.fromisoformat(current_bar['timestamp']),
            volatility=volatility,
            market_hours=determine_market_session(current_bar['timestamp']),
            volume=current_bar.get('volume', 1000),
            historical_bars=bars[max(0, current_bar_index-100):current_bar_index+1],
            account_balance=account['balance'],
            equity=account['equity'],
            margin_available=account['margin_available']
        )
        
        return market_context.model_dump_json()
        
    except Exception as e:
        logger.error(f"Error simulating market context: {e}")
        return json.dumps({"error": str(e)})

@tool 
def simulate_trade_execution(
    trade_decision: str,  # JSON string with trade details
    market_context: str   # JSON string with market conditions
) -> str:
    """
    Simulate realistic trade execution with slippage, commissions, and market impact.
    
    Args:
        trade_decision: JSON string containing trade parameters
        market_context: JSON string with current market conditions
    
    Returns:
        JSON string with execution results
    """
    try:
        decision = json.loads(trade_decision)
        context = json.loads(market_context)
        
        # Extract trade parameters
        side = decision.get('side', 'buy')
        quantity = float(decision.get('quantity', 0))
        order_type = decision.get('order_type', 'market')
        requested_price = float(decision.get('price', context['current_price']))
        
        # Simulate realistic execution
        execution_price = simulate_execution_price(
            requested_price, 
            side, 
            quantity, 
            context,
            order_type
        )
        
        # Calculate slippage and commission
        slippage = abs(execution_price - requested_price)
        commission = calculate_commission(quantity, execution_price)
        
        # Simulate execution time (in milliseconds)
        execution_time = simulate_execution_latency(order_type, quantity)
        
        result = BacktestExecutionResult(
            order_id=f"BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000,9999)}",
            symbol=decision.get('symbol', 'EUR_USD'),
            side=side,
            quantity=quantity,
            requested_price=requested_price,
            executed_price=execution_price,
            slippage=slippage,
            commission=commission,
            execution_time_ms=execution_time,
            status='filled',
            partial_fill=quantity > 100000  # Large orders might get partial fills
        )
        
        return result.model_dump_json()
        
    except Exception as e:
        logger.error(f"Error simulating trade execution: {e}")
        return json.dumps({"error": str(e)})

@tool
def update_backtest_portfolio(
    portfolio_state: str,    # JSON string with current portfolio
    execution_result: str,   # JSON string with trade execution
    current_prices: str      # JSON string with current market prices
) -> str:
    """
    Update backtest portfolio state after trade execution or price changes.
    
    Args:
        portfolio_state: JSON string containing current portfolio state
        execution_result: JSON string with trade execution details
        current_prices: JSON string with current market prices
    
    Returns:
        JSON string with updated portfolio state
    """
    try:
        portfolio_data = json.loads(portfolio_state)
        execution = json.loads(execution_result) if execution_result != "null" else None
        prices = json.loads(current_prices)
        
        # Convert to portfolio object for easier manipulation
        portfolio = BacktestPortfolio(**portfolio_data)
        
        # Process new trade execution if provided
        if execution and execution.get('status') == 'filled':
            process_trade_execution(portfolio, execution)
        
        # Update unrealized P&L for open positions
        update_unrealized_pnl(portfolio, prices)
        
        # Update equity calculation
        portfolio.equity = portfolio.current_balance + portfolio.unrealized_pnl
        
        # Calculate free margin
        portfolio.free_margin = portfolio.equity - portfolio.margin_used
        
        # Convert back to JSON
        return json.dumps({
            'initial_balance': portfolio.initial_balance,
            'current_balance': portfolio.current_balance,
            'equity': portfolio.equity,
            'unrealized_pnl': portfolio.unrealized_pnl,
            'realized_pnl': portfolio.realized_pnl,
            'open_positions': portfolio.open_positions,
            'closed_trades': portfolio.closed_trades,
            'margin_used': portfolio.margin_used,
            'free_margin': portfolio.free_margin
        })
        
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")
        return json.dumps({"error": str(e)})

@tool
def calculate_backtest_performance_metrics(
    portfolio_history: str,  # JSON string with portfolio history
    trade_history: str       # JSON string with completed trades
) -> str:
    """
    Calculate comprehensive performance metrics for backtest results.
    
    Args:
        portfolio_history: JSON string with portfolio value over time
        trade_history: JSON string with all completed trades
    
    Returns:
        JSON string with performance metrics
    """
    try:
        portfolio_data = json.loads(portfolio_history)
        trades_data = json.loads(trade_history)
        
        if not trades_data:
            return json.dumps({"error": "No trades to analyze"})
        
        # Calculate basic metrics
        total_trades = len(trades_data)
        winning_trades = sum(1 for trade in trades_data if trade.get('pnl', 0) > 0)
        losing_trades = total_trades - winning_trades
        
        # Calculate P&L metrics
        total_pnl = sum(trade.get('pnl', 0) for trade in trades_data)
        avg_win = np.mean([trade['pnl'] for trade in trades_data if trade.get('pnl', 0) > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([trade['pnl'] for trade in trades_data if trade.get('pnl', 0) < 0]) if losing_trades > 0 else 0
        
        # Calculate advanced metrics
        equity_curve = [point['equity'] for point in portfolio_data]
        max_drawdown = calculate_max_drawdown(equity_curve)
        sharpe_ratio = calculate_sharpe_ratio(equity_curve)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return_pct': (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100 if equity_curve else 0
        }
        
        return json.dumps(metrics)
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return json.dumps({"error": str(e)})

# Helper functions
def calculate_recent_volatility(bars: List[Dict]) -> float:
    """Calculate recent price volatility"""
    if len(bars) < 2:
        return 0.001  # Default low volatility
    
    prices = [bar['close'] for bar in bars]
    returns = np.diff(np.log(prices))
    return float(np.std(returns) * np.sqrt(24))  # Annualized volatility

def calculate_realistic_spread(price: float, volatility: float, timestamp: str) -> float:
    """Calculate realistic bid-ask spread based on market conditions"""
    base_spread = price * 0.00002  # 0.2 pips base spread
    volatility_adjustment = volatility * 0.1
    
    # Adjust for market hours (wider spreads during off-hours)
    hour = datetime.fromisoformat(timestamp).hour
    if hour < 7 or hour > 17:  # Outside main trading hours
        base_spread *= 1.5
    
    return base_spread + volatility_adjustment

def determine_market_session(timestamp: str) -> str:
    """Determine market session based on time"""
    hour = datetime.fromisoformat(timestamp).hour
    if 7 <= hour < 17:
        return "active"
    else:
        return "quiet"

def simulate_execution_price(requested_price: float, side: str, quantity: float, 
                           context: Dict, order_type: str) -> float:
    """Simulate realistic execution price with slippage"""
    base_price = requested_price
    
    if order_type == 'market':
        # Market orders get current bid/ask
        base_price = context['ask'] if side == 'buy' else context['bid']
    
    # Add slippage based on quantity and volatility
    slippage_factor = min(quantity / 100000 * 0.1, 0.5)  # Max 0.5% slippage
    volatility_slippage = context['volatility'] * 0.01
    
    total_slippage = (slippage_factor + volatility_slippage) * base_price
    
    if side == 'buy':
        return base_price + total_slippage
    else:
        return base_price - total_slippage

def calculate_commission(quantity: float, price: float) -> float:
    """Calculate trading commission"""
    notional = quantity * price
    return max(notional * 0.000025, 2.50)  # 2.5 bps min $2.50

def simulate_execution_latency(order_type: str, quantity: float) -> int:
    """Simulate execution latency in milliseconds"""
    base_latency = 50 if order_type == 'market' else 100
    size_penalty = min(quantity / 10000 * 10, 100)
    return int(base_latency + size_penalty + np.random.normal(0, 20))

def process_trade_execution(portfolio: BacktestPortfolio, execution: Dict):
    """Process trade execution and update portfolio"""
    side = execution['side']
    quantity = execution['quantity']
    price = execution['executed_price']
    commission = execution['commission']
    
    # Update cash balance
    if side == 'buy':
        portfolio.current_balance -= (quantity * price + commission)
    else:
        portfolio.current_balance += (quantity * price - commission)
    
    # Add to open positions
    position = {
        'symbol': execution['symbol'],
        'side': side,
        'quantity': quantity,
        'entry_price': price,
        'timestamp': datetime.now().isoformat(),
        'commission': commission
    }
    portfolio.open_positions.append(position)

def update_unrealized_pnl(portfolio: BacktestPortfolio, current_prices: Dict):
    """Update unrealized P&L for all open positions"""
    total_unrealized = 0
    
    for position in portfolio.open_positions:
        symbol = position['symbol']
        if symbol in current_prices:
            current_price = current_prices[symbol]
            entry_price = position['entry_price']
            quantity = position['quantity']
            side = position['side']
            
            if side == 'buy':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            position['unrealized_pnl'] = pnl
            total_unrealized += pnl
    
    portfolio.unrealized_pnl = total_unrealized

def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown percentage"""
    if not equity_curve:
        return 0
    
    peak = equity_curve[0]
    max_dd = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    
    return max_dd * 100

def calculate_sharpe_ratio(equity_curve: List[float]) -> float:
    """Calculate Sharpe ratio"""
    if len(equity_curve) < 2:
        return 0
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if np.std(returns) == 0:
        return 0
    
    return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
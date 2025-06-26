"""
Fixed Database Models - Resolves all timezone, field, and query issues
src/database/models.py
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Text, Numeric, DateTime, Boolean, 
    ForeignKey, JSON, Index, func
)
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# CRITICAL FIX: Use timezone-naive datetime for database compatibility
def utc_now():
    """Generate timezone-naive UTC datetime for database compatibility"""
    return datetime.utcnow()  # Keep using utcnow for now, will be timezone-naive

# Enums
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class TradeStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class TradeSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    
class MarketRegime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"

class WyckoffPattern(Enum):
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    REACCUMULATION = "REACCUMULATION"
    REDISTRIBUTION = "REDISTRIBUTION"
    UNKNOWN = "UNKNOWN"

# Main Models
class Trade(Base):
    """Enhanced Trade model with all required fields"""
    __tablename__ = 'trades'
    
    trade_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    # Basic trade info
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(ENUM(TradeSide), nullable=False)
    quantity = Column(Numeric(15, 5), nullable=False)
    entry_price = Column(Numeric(15, 5))
    exit_price = Column(Numeric(15, 5))
    
    # P&L tracking
    realized_pnl = Column(Numeric(15, 2), default=0)
    unrealized_pnl = Column(Numeric(15, 2), default=0)
    commission = Column(Numeric(10, 2), default=0)
    
    # Risk management
    stop_loss = Column(Numeric(15, 5))
    take_profit = Column(Numeric(15, 5))
    trailing_stop = Column(Numeric(15, 5))
    risk_amount = Column(Numeric(15, 2))
    
    # Wyckoff-specific fields
    confidence_score = Column(Numeric(5, 2))
    wyckoff_phase = Column(String(20))
    pattern_type = Column(String(50))
    
    # FIXED: Add missing reasoning field
    reasoning = Column(Text)  # This field was missing!
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    
    # Context and decision tracking
    agent_decisions = Column(JSON)
    market_context = Column(JSON)
    status = Column(ENUM(TradeStatus), nullable=False)
    broker_trade_id = Column(String(50), index=True)
    session_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    
    # Relationships
    orders = relationship("Order", back_populates="trade")
    pattern_detections = relationship("PatternDetection", back_populates="trade")
    risk_calculations = relationship("RiskCalculation", back_populates="trade")


class EventLog(Base):
    """Event logging for debugging and audit trails"""
    __tablename__ = 'events_log'
    
    event_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    level = Column(ENUM(LogLevel), nullable=False, index=True)
    agent_name = Column(String(100), index=True)
    event_type = Column(String(50), index=True)
    message = Column(Text, nullable=False)
    context = Column(JSON)
    stack_trace = Column(Text)
    session_id = Column(UUID(as_uuid=True), index=True)
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)


class AgentAction(Base):
    """Agent decision tracking for transparency and optimization"""
    __tablename__ = 'agent_actions'
    
    action_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    agent_name = Column(String(100), nullable=False, index=True)
    action_type = Column(String(50), index=True)
    input_data = Column(JSON)
    output_data = Column(JSON)
    confidence_score = Column(Numeric(5, 2))
    execution_time_ms = Column(Integer)
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    session_id = Column(UUID(as_uuid=True), index=True)
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)


class PerformanceMetric(Base):
    """Real-time performance metrics for monitoring"""
    __tablename__ = 'performance_metrics'
    
    metric_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    timeframe = Column(String(10), nullable=False, index=True)
    
    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    # Financial metrics
    total_pnl = Column(Numeric(15, 2), default=0)
    unrealized_pnl = Column(Numeric(15, 2), default=0)
    account_balance = Column(Numeric(15, 2))
    equity = Column(Numeric(15, 2))
    margin_used = Column(Numeric(15, 2))
    free_margin = Column(Numeric(15, 2))
    
    # Risk metrics
    sharpe_ratio = Column(Numeric(8, 4))
    max_drawdown = Column(Numeric(8, 4))
    win_rate = Column(Numeric(5, 2))
    profit_factor = Column(Numeric(8, 4))
    avg_win = Column(Numeric(10, 2))
    avg_loss = Column(Numeric(10, 2))
    largest_win = Column(Numeric(10, 2))
    largest_loss = Column(Numeric(10, 2))
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)


class PatternDetection(Base):
    """Wyckoff pattern detection results with full context"""
    __tablename__ = 'pattern_detections'
    
    detection_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    pattern_type = Column(ENUM(WyckoffPattern), nullable=False, index=True)
    confidence_score = Column(Numeric(5, 2), nullable=False)
    
    # Structure timing - FIXED: Use timezone-naive datetime
    structure_start_time = Column(DateTime)
    structure_end_time = Column(DateTime)
    
    # Pattern geometry and levels
    key_levels = Column(JSON)
    volume_analysis = Column(JSON)
    market_context = Column(JSON)
    pattern_geometry = Column(JSON)
    
    # Trading levels
    invalidation_level = Column(Numeric(15, 5))
    target_levels = Column(JSON)
    
    # Relationships
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    trade = relationship("Trade", back_populates="pattern_detections")
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)


class Position(Base):
    """Real-time position tracking"""
    __tablename__ = 'positions'
    
    position_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(ENUM(TradeSide), nullable=False)
    quantity = Column(Numeric(15, 5), nullable=False)
    entry_price = Column(Numeric(15, 5))
    current_price = Column(Numeric(15, 5))
    unrealized_pnl = Column(Numeric(15, 2))
    margin_required = Column(Numeric(15, 2))
    
    # Position management
    stop_loss = Column(Numeric(15, 5))
    take_profit = Column(Numeric(15, 5))
    trailing_stop = Column(Numeric(15, 5))
    max_pnl = Column(Numeric(15, 2))
    max_adverse = Column(Numeric(15, 2))
    
    status = Column(String(20), default="open")
    broker_position_id = Column(String(50), index=True)
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    session_id = Column(UUID(as_uuid=True), index=True)
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class Order(Base):
    """Order tracking with full execution details"""
    __tablename__ = 'orders'
    
    order_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    symbol = Column(String(10), nullable=False, index=True)
    order_type = Column(ENUM(OrderType), nullable=False)
    side = Column(ENUM(TradeSide), nullable=False)
    quantity = Column(Numeric(15, 5), nullable=False)
    price = Column(Numeric(15, 5))
    stop_price = Column(Numeric(15, 5))
    filled_quantity = Column(Numeric(15, 5), default=0)
    avg_fill_price = Column(Numeric(15, 5))
    commission = Column(Numeric(10, 2))
    
    status = Column(ENUM(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    broker_order_id = Column(String(50), index=True)
    error_message = Column(Text)
    
    # Relationships
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    trade = relationship("Trade", back_populates="orders")
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)
    filled_at = Column(DateTime)


class RiskCalculation(Base):
    """Risk calculations audit trail"""
    __tablename__ = 'risk_calculations'
    
    calc_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    symbol = Column(String(10), nullable=False, index=True)
    
    # Account and risk data
    account_balance = Column(Numeric(15, 2))
    risk_per_trade_pct = Column(Numeric(5, 2))
    stop_distance_pips = Column(Numeric(8, 2))
    position_size = Column(Numeric(15, 5))
    risk_amount = Column(Numeric(15, 2))
    risk_reward_ratio = Column(Numeric(8, 2))
    kelly_criterion = Column(Numeric(8, 4))
    atr_stop_distance = Column(Numeric(8, 2))
    
    calculation_method = Column(String(50))
    reasoning = Column(Text)
    
    # Relationships
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    trade = relationship("Trade", back_populates="risk_calculations")
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)


class ExecutionMetric(Base):
    """Execution quality metrics"""
    __tablename__ = 'execution_metrics'
    
    metric_id = Column(Integer, primary_key=True)
    
    # FIXED: Use timezone-naive datetime
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)
    
    symbol = Column(String(10), nullable=False, index=True)
    order_id = Column(Integer, ForeignKey('orders.order_id'))
    
    # Execution quality
    expected_price = Column(Numeric(15, 5))
    actual_price = Column(Numeric(15, 5))
    slippage_pips = Column(Numeric(8, 2))
    slippage_cost = Column(Numeric(10, 2))
    spread_at_execution = Column(Numeric(8, 2))
    market_impact = Column(Numeric(8, 2))
    execution_latency_ms = Column(Integer)
    fill_rate = Column(Numeric(5, 2))
    
    market_conditions = Column(JSON)
    
    # FIXED: Use timezone-naive datetime
    created_at = Column(DateTime, default=utc_now)


# Indexes for performance
Index('idx_trades_symbol_timestamp', Trade.symbol, Trade.timestamp)
Index('idx_events_agent_type', EventLog.agent_name, EventLog.event_type)
Index('idx_actions_agent_timestamp', AgentAction.agent_name, AgentAction.timestamp)
Index('idx_patterns_symbol_type', PatternDetection.symbol, PatternDetection.pattern_type)
Index('idx_positions_symbol_status', Position.symbol, Position.status)
"""
Database Models for Autonomous Trading System - FIXED VERSION
SQLAlchemy models with datetime deprecation warnings resolved
"""

from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import Column, Integer, String, Numeric, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ENUM
import uuid

Base = declarative_base()

# Helper function for timezone-aware datetime
def utc_now():
    """Get current UTC time in timezone-aware format"""
    return datetime.now(timezone.utc)

# Enums for type safety
class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MarketRegime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"

class WyckoffPattern(str, Enum):
    SPRING = "spring"
    UPTHRUST = "upthrust"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    REACCUMULATION = "reaccumulation"
    REDISTRIBUTION = "redistribution"


class Trade(Base):
    """Complete trade records with full metadata and Wyckoff context"""
    __tablename__ = 'trades'
    
    trade_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now)  # FIXED
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(ENUM(TradeSide), nullable=False)
    quantity = Column(Numeric(15, 5), nullable=False)
    entry_price = Column(Numeric(15, 5))
    exit_price = Column(Numeric(15, 5))
    stop_loss = Column(Numeric(15, 5))
    take_profit = Column(Numeric(15, 5))
    pnl = Column(Numeric(15, 2))
    commission = Column(Numeric(10, 2))
    duration_minutes = Column(Integer)
    
    # Wyckoff-specific fields
    wyckoff_pattern = Column(ENUM(WyckoffPattern))
    pattern_confidence = Column(Numeric(5, 2))  # 0-100%
    market_regime = Column(ENUM(MarketRegime))
    timeframe_analysis = Column(String(10))  # 1H, 15m, 5m
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    
    # Context and decision tracking
    agent_decisions = Column(JSON)  # Full agent decision tree
    market_context = Column(JSON)   # Market conditions at entry
    status = Column(ENUM(TradeStatus), nullable=False)
    broker_trade_id = Column(String(50), index=True)
    session_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    
    created_at = Column(DateTime, default=utc_now)  # FIXED
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)  # FIXED
    
    # Relationships
    orders = relationship("Order", back_populates="trade")
    pattern_detections = relationship("PatternDetection", back_populates="trade")
    risk_calculations = relationship("RiskCalculation", back_populates="trade")


class EventLog(Base):
    """Event logging for debugging and audit trails"""
    __tablename__ = 'events_log'
    
    event_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
    level = Column(ENUM(LogLevel), nullable=False, index=True)
    agent_name = Column(String(100), index=True)
    event_type = Column(String(50), index=True)  # SIGNAL_GENERATED, TRADE_EXECUTED, etc.
    message = Column(Text, nullable=False)
    context = Column(JSON)  # Additional context data
    stack_trace = Column(Text)  # For errors
    session_id = Column(UUID(as_uuid=True), index=True)
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    
    created_at = Column(DateTime, default=utc_now)  # FIXED


class AgentAction(Base):
    """Agent decision tracking for transparency and optimization"""
    __tablename__ = 'agent_actions'
    
    action_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
    agent_name = Column(String(100), nullable=False, index=True)
    action_type = Column(String(50), index=True)  # ANALYZE_PATTERN, CALCULATE_RISK, etc.
    input_data = Column(JSON)  # What data the agent received
    output_data = Column(JSON)  # What the agent decided/calculated
    confidence_score = Column(Numeric(5, 2))  # Agent confidence in decision
    execution_time_ms = Column(Integer)  # How long the action took
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    session_id = Column(UUID(as_uuid=True), index=True)
    
    created_at = Column(DateTime, default=utc_now)  # FIXED


class PerformanceMetric(Base):
    """Real-time performance metrics for monitoring"""
    __tablename__ = 'performance_metrics'
    
    metric_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
    timeframe = Column(String(10), nullable=False, index=True)  # 1m, 5m, 15m, 1h, 1d, 1w
    
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
    
    created_at = Column(DateTime, default=utc_now)  # FIXED


class PatternDetection(Base):
    """Wyckoff pattern detection results with full context"""
    __tablename__ = 'pattern_detections'
    
    detection_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    pattern_type = Column(ENUM(WyckoffPattern), nullable=False, index=True)
    confidence_score = Column(Numeric(5, 2), nullable=False)  # 0-100%
    
    # Pattern timing
    structure_start_time = Column(DateTime)
    structure_end_time = Column(DateTime)
    
    # Pattern data
    key_levels = Column(JSON)  # Support/resistance levels identified
    volume_analysis = Column(JSON)  # Volume profile data
    market_context = Column(JSON)  # Volatility, trend, regime
    pattern_geometry = Column(JSON)  # Coordinates for visualization
    invalidation_level = Column(Numeric(15, 5))  # Where pattern becomes invalid
    target_levels = Column(JSON)  # Projected targets
    
    # Relationships
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    trade = relationship("Trade", back_populates="pattern_detections")
    
    created_at = Column(DateTime, default=utc_now)  # FIXED


class MarketContext(Base):
    """Market regime and context tracking"""
    __tablename__ = 'market_context'
    
    context_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    
    # Market regime
    regime = Column(ENUM(MarketRegime), index=True)
    trend_direction = Column(String(10))  # Bullish, Bearish, Neutral
    volatility_percentile = Column(Numeric(5, 2))  # 0-100%
    atr_value = Column(Numeric(15, 5))
    
    # Volume and structure
    volume_profile = Column(JSON)  # Current session profile
    key_levels = Column(JSON)  # Important S/R levels
    sentiment_score = Column(Numeric(5, 2))  # -100 to +100
    economic_events = Column(JSON)  # Upcoming news events
    
    created_at = Column(DateTime, default=utc_now)  # FIXED


class Position(Base):
    """Position tracking with full lifecycle"""
    __tablename__ = 'positions'
    
    position_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
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
    max_pnl = Column(Numeric(15, 2))  # High water mark
    max_adverse = Column(Numeric(15, 2))  # Maximum adverse excursion
    
    status = Column(String(20), default="open")
    broker_position_id = Column(String(50), index=True)
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    session_id = Column(UUID(as_uuid=True), index=True)
    
    created_at = Column(DateTime, default=utc_now)  # FIXED
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)  # FIXED


class Order(Base):
    """Order tracking with full execution details"""
    __tablename__ = 'orders'
    
    order_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
    symbol = Column(String(10), nullable=False, index=True)
    order_type = Column(ENUM(OrderType), nullable=False)
    side = Column(ENUM(TradeSide), nullable=False)
    quantity = Column(Numeric(15, 5), nullable=False)
    price = Column(Numeric(15, 5))  # For limit orders
    stop_price = Column(Numeric(15, 5))  # For stop orders
    filled_quantity = Column(Numeric(15, 5), default=0)
    avg_fill_price = Column(Numeric(15, 5))
    commission = Column(Numeric(10, 2))
    
    status = Column(ENUM(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    broker_order_id = Column(String(50), index=True)
    error_message = Column(Text)
    
    # Relationships
    trade_id = Column(Integer, ForeignKey('trades.trade_id'))
    trade = relationship("Trade", back_populates="orders")
    
    created_at = Column(DateTime, default=utc_now)  # FIXED
    filled_at = Column(DateTime)


class RiskCalculation(Base):
    """Risk calculations audit trail"""
    __tablename__ = 'risk_calculations'
    
    calc_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
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
    
    created_at = Column(DateTime, default=utc_now)  # FIXED


class ExecutionMetric(Base):
    """Execution quality metrics"""
    __tablename__ = 'execution_metrics'
    
    metric_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now, index=True)  # FIXED
    symbol = Column(String(10), nullable=False, index=True)
    order_id = Column(Integer, ForeignKey('orders.order_id'))
    
    # Execution quality
    expected_price = Column(Numeric(15, 5))
    actual_price = Column(Numeric(15, 5))
    slippage_pips = Column(Numeric(8, 2))
    slippage_cost = Column(Numeric(10, 2))  # Dollar cost of slippage
    spread_at_execution = Column(Numeric(8, 2))
    market_impact = Column(Numeric(8, 2))  # Price impact of order
    execution_latency_ms = Column(Integer)
    fill_rate = Column(Numeric(5, 2))  # % of order filled
    
    market_conditions = Column(JSON)  # Volatility, volume at execution
    created_at = Column(DateTime, default=utc_now)  # FIXED
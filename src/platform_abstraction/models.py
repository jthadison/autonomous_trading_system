"""
Universal Trading Data Models for Platform Abstraction
File: src/platform_abstraction/models.py

These models provide standardized data structures that work across all trading platforms.
Every platform adapter must convert their native data formats to these universal models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime, timezone
from enum import Enum
import uuid


# ================================
# ENUMS FOR TYPE SAFETY
# ================================

class TradeSide(Enum):
    """Standardized trade sides"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Standardized order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Standardized order statuses"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class PositionSide(Enum):
    """Standardized position sides"""
    LONG = "long"
    SHORT = "short"

class Platform(Enum):
    """Supported trading platforms"""
    OANDA = "oanda"
    METATRADER4 = "metatrader4"
    METATRADER5 = "metatrader5"
    TRADELOCKER = "tradelocker"
    DXTRADE = "dxtrade"
    CTRADER = "ctrader"
    MATCH_TRADER = "match_trader"


# ================================
# CORE DATA MODELS
# ================================

@dataclass
class UniversalMarketPrice:
    """Standardized market price data across all platforms"""
    instrument: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    platform_source: Platform
    
    # Optional additional data
    volume: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_24h_pct: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2
    
    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage"""
        return (self.spread / self.mid_price) * 100 if self.mid_price > 0 else 0


@dataclass
class UniversalAccountInfo:
    """Standardized account information across all platforms"""
    account_id: str
    currency: str
    balance: float
    equity: float
    free_margin: float
    used_margin: float
    margin_rate: float
    platform_source: Platform
    timestamp: datetime
    
    # Optional fields
    unrealized_pl: Optional[float] = None
    realized_pl: Optional[float] = None
    margin_call_level: Optional[float] = None
    stop_out_level: Optional[float] = None
    leverage: Optional[float] = None
    
    @property
    def margin_level(self) -> float:
        """Calculate margin level percentage"""
        return (self.equity / self.used_margin * 100) if self.used_margin > 0 else float('inf')


@dataclass
class UniversalPosition:
    """Standardized position data across all platforms"""
    position_id: str
    instrument: str
    side: PositionSide
    units: float
    average_price: float
    current_price: float
    unrealized_pl: float
    timestamp: datetime
    platform_source: Platform
    
    # Optional fields
    swap: Optional[float] = None
    commission: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    margin_used: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position"""
        return abs(self.units) * self.current_price
    
    @property
    def unrealized_pl_pct(self) -> float:
        """Calculate unrealized P&L as percentage"""
        if self.side == PositionSide.LONG:
            return ((self.current_price - self.average_price) / self.average_price) * 100
        else:
            return ((self.average_price - self.current_price) / self.average_price) * 100


@dataclass
class UniversalOrder:
    """Standardized order data across all platforms"""
    order_id: str
    instrument: str
    side: TradeSide
    order_type: OrderType
    status: OrderStatus
    units: float
    timestamp: datetime
    platform_source: Platform
    
    # Price fields (depend on order type)
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    
    # Execution details
    filled_units: Optional[float] = None
    remaining_units: Optional[float] = None
    average_fill_price: Optional[float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Timing
    expiry_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate how much of the order has been filled"""
        if self.filled_units is None:
            return 0.0
        return (self.filled_units / self.units) * 100 if self.units > 0 else 0.0


@dataclass
class TradeParams:
    """Parameters for executing a trade - input to trading functions"""
    instrument: str
    side: TradeSide
    units: float
    order_type: OrderType = OrderType.MARKET
    
    # Price parameters
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Order management
    expiry_time: Optional[datetime] = None
    max_slippage: Optional[float] = None
    
    # Metadata
    reason: str = "Platform abstraction trade"
    platform_preference: Optional[Platform] = None
    
    def __post_init__(self):
        """Validate trade parameters"""
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders must specify a price")
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop orders must specify a stop price")


@dataclass
class UniversalTradeResult:
    """Standardized trade execution result across all platforms"""
    success: bool
    trade_reference: str
    platform_source: Platform
    timestamp: datetime
    
    # Order details
    order_id: Optional[str] = None
    transaction_id: Optional[str] = None
    
    # Execution details
    instrument: Optional[str] = None
    side: Optional[TradeSide] = None
    units: Optional[float] = None
    execution_price: Optional[float] = None
    
    # Status information
    status: Optional[OrderStatus] = None
    
    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success_result(
        cls,
        trade_reference: str,
        platform_source: Platform,
        order_id: str,
        instrument: str,
        side: TradeSide,
        units: float,
        execution_price: float,
        **kwargs
    ) -> 'UniversalTradeResult':
        """Create a successful trade result"""
        return cls(
            success=True,
            trade_reference=trade_reference,
            platform_source=platform_source,
            timestamp=datetime.now(timezone.utc),
            order_id=order_id,
            instrument=instrument,
            side=side,
            units=units,
            execution_price=execution_price,
            status=OrderStatus.FILLED,
            **kwargs
        )
    
    @classmethod
    def error_result(
        cls,
        trade_reference: str,
        platform_source: Platform,
        error: str,
        error_type: Optional[str] = None,
        **kwargs
    ) -> 'UniversalTradeResult':
        """Create an error trade result"""
        return cls(
            success=False,
            trade_reference=trade_reference,
            platform_source=platform_source,
            timestamp=datetime.now(timezone.utc),
            error=error,
            error_type=error_type,
            **kwargs
        )


@dataclass
class UniversalPortfolioStatus:
    """Comprehensive portfolio status across all platforms"""
    account_info: UniversalAccountInfo
    positions: List[UniversalPosition]
    orders: List[UniversalOrder]
    timestamp: datetime
    platform_sources: List[Platform]
    
    @property
    def total_unrealized_pl(self) -> float:
        """Calculate total unrealized P&L across all positions"""
        return sum(pos.unrealized_pl for pos in self.positions)
    
    @property
    def total_margin_used(self) -> float:
        """Calculate total margin used across all positions"""
        return sum(pos.margin_used or 0 for pos in self.positions)
    
    @property
    def active_orders_count(self) -> int:
        """Count active orders"""
        return len([order for order in self.orders if order.is_active])
    
    @property
    def positions_by_instrument(self) -> Dict[str, List[UniversalPosition]]:
        """Group positions by instrument"""
        grouped = {}
        for position in self.positions:
            if position.instrument not in grouped:
                grouped[position.instrument] = []
            grouped[position.instrument].append(position)
        return grouped


@dataclass
class HistoricalData:
    """Standardized historical market data"""
    instrument: str
    timeframe: str
    data: List[Dict[str, Any]]  # OHLCV data
    platform_source: Platform
    timestamp: datetime
    
    @property
    def candle_count(self) -> int:
        """Number of candles in the dataset"""
        return len(self.data)


# ================================
# UTILITY FUNCTIONS
# ================================

def generate_trade_reference(platform: Platform) -> str:
    """Generate a unique trade reference"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{platform.value.upper()}_{timestamp}_{unique_id}"


def convert_side_to_enum(side_str: str) -> TradeSide:
    """Convert string side to TradeSide enum"""
    side_str = side_str.lower()
    if side_str in ["buy", "long"]:
        return TradeSide.BUY
    elif side_str in ["sell", "short"]:
        return TradeSide.SELL
    else:
        raise ValueError(f"Invalid trade side: {side_str}")


def convert_position_side_to_enum(side_str: str) -> PositionSide:
    """Convert string position side to PositionSide enum"""
    side_str = side_str.lower()
    if side_str in ["long", "buy"]:
        return PositionSide.LONG
    elif side_str in ["short", "sell"]:
        return PositionSide.SHORT
    else:
        raise ValueError(f"Invalid position side: {side_str}")


# ================================
# VALIDATION FUNCTIONS
# ================================

def validate_trade_params(params: TradeParams) -> bool:
    """Validate trade parameters"""
    try:
        # Basic validation
        if not params.instrument:
            raise ValueError("Instrument is required")
        if params.units <= 0:
            raise ValueError("Units must be positive")
        
        # Order type specific validation
        if params.order_type == OrderType.LIMIT and params.price is None:
            raise ValueError("Limit orders require a price")
        
        if params.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and params.stop_price is None:
            raise ValueError("Stop orders require a stop price")
        
        # Risk management validation
        if params.stop_loss is not None and params.stop_loss <= 0:
            raise ValueError("Stop loss must be positive")
        
        if params.take_profit is not None and params.take_profit <= 0:
            raise ValueError("Take profit must be positive")
        
        return True
        
    except ValueError as e:
        raise ValueError(f"Trade parameter validation failed: {str(e)}")


# ================================
# PLATFORM CONVERSION HELPERS
# ================================

class PlatformConverter:
    """Helper class for converting platform-specific data to universal models"""
    
    @staticmethod
    def to_platform_enum(platform_str: str) -> Platform:
        """Convert platform string to Platform enum"""
        platform_str = platform_str.lower()
        for platform in Platform:
            if platform.value == platform_str:
                return platform
        raise ValueError(f"Unsupported platform: {platform_str}")
    
    @staticmethod
    def from_platform_enum(platform: Platform) -> str:
        """Convert Platform enum to string"""
        return platform.value
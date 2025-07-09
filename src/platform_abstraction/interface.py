"""
Universal Trading Interface for Platform Abstraction
File: src/platform_abstraction/interface.py

Abstract base class that defines the contract all trading platform adapters must implement.
This ensures consistent behavior across all platforms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .models import (
    UniversalMarketPrice,
    UniversalAccountInfo,
    UniversalPosition,
    UniversalOrder,
    UniversalTradeResult,
    UniversalPortfolioStatus,
    TradeParams,
    HistoricalData,
    Platform,
    TradeSide,
    OrderType,
    OrderStatus
)


class UniversalTradingInterface(ABC):
    """
    Abstract base class defining the universal trading interface.
    All platform adapters must implement these methods.
    """
    
    def __init__(self, platform: Platform, config: Dict[str, Any]):
        self.platform = platform
        self.config = config
        self._connected = False
        
    @property
    def is_connected(self) -> bool:
        """Check if platform is connected"""
        return self._connected
    
    @property
    def platform_name(self) -> str:
        """Get platform name"""
        return self.platform.value
    
    # ================================
    # CONNECTION MANAGEMENT
    # ================================
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the trading platform
        Returns: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the trading platform
        Returns: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check platform health and connectivity
        Returns: Health status information
        """
        pass
    
    # ================================
    # MARKET DATA OPERATIONS
    # ================================
    
    @abstractmethod
    async def get_live_price(self, instrument: str) -> UniversalMarketPrice:
        """
        Get current live price for an instrument
        Args:
            instrument: Trading instrument (e.g., "EUR_USD", "US30_USD")
        Returns:
            UniversalMarketPrice with current bid/ask
        """
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        instrument: str,
        timeframe: str,
        count: int = 200,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> HistoricalData:
        """
        Get historical market data
        Args:
            instrument: Trading instrument
            timeframe: Timeframe (M1, M5, M15, H1, H4, D1)
            count: Number of candles to retrieve
            start_time: Start time for historical data
            end_time: End time for historical data
        Returns:
            HistoricalData with OHLCV data
        """
        pass
    
    # ================================
    # ACCOUNT OPERATIONS
    # ================================
    
    @abstractmethod
    async def get_account_info(self) -> UniversalAccountInfo:
        """
        Get account information
        Returns:
            UniversalAccountInfo with balance, equity, margin info
        """
        pass
    
    @abstractmethod
    async def get_portfolio_status(self) -> UniversalPortfolioStatus:
        """
        Get comprehensive portfolio status
        Returns:
            UniversalPortfolioStatus with account, positions, and orders
        """
        pass
    
    # ================================
    # POSITION MANAGEMENT
    # ================================
    
    @abstractmethod
    async def get_open_positions(self) -> List[UniversalPosition]:
        """
        Get all open positions
        Returns:
            List of UniversalPosition objects
        """
        pass
    
    @abstractmethod
    async def get_position(self, instrument: str) -> Optional[UniversalPosition]:
        """
        Get position for specific instrument
        Args:
            instrument: Trading instrument
        Returns:
            UniversalPosition if exists, None otherwise
        """
        pass
    
    @abstractmethod
    async def close_position(
        self,
        instrument: str,
        units: Optional[float] = None,
        reason: str = "Manual close"
    ) -> UniversalTradeResult:
        """
        Close position (full or partial)
        Args:
            instrument: Trading instrument
            units: Units to close (None for full close)
            reason: Reason for closing
        Returns:
            UniversalTradeResult with execution details
        """
        pass
    
    @abstractmethod
    async def cancel_pending_order(
        self,
        order_id: str,
        reason: str = "Order cancellation"
    ) -> UniversalTradeResult:
        """Cancel a pending order by ID"""
        pass
    
    # ================================
    # ORDER MANAGEMENT
    # ================================
    
    @abstractmethod
    async def get_pending_orders(self) -> List[UniversalOrder]:
        """
        Get all pending orders
        Returns:
            List of UniversalOrder objects
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[UniversalOrder]:
        """
        Get specific order by ID
        Args:
            order_id: Order identifier
        Returns:
            UniversalOrder if exists, None otherwise
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> UniversalTradeResult:
        """
        Cancel pending order
        Args:
            order_id: Order identifier
        Returns:
            UniversalTradeResult with cancellation details
        """
        pass
    
    # ================================
    # TRADE EXECUTION
    # ================================
    
    @abstractmethod
    async def execute_market_trade(self, params: TradeParams) -> UniversalTradeResult:
        """
        Execute market order
        Args:
            params: TradeParams with trade details
        Returns:
            UniversalTradeResult with execution details
        """
        pass
    
    @abstractmethod
    async def execute_limit_trade(self, params: TradeParams) -> UniversalTradeResult:
        """
        Execute limit order
        Args:
            params: TradeParams with trade details (must include price)
        Returns:
            UniversalTradeResult with order placement details
        """
        pass
    
    @abstractmethod
    async def execute_stop_trade(self, params: TradeParams) -> UniversalTradeResult:
        """
        Execute stop order
        Args:
            params: TradeParams with trade details (must include stop_price)
        Returns:
            UniversalTradeResult with order placement details
        """
        pass
    
    # ================================
    # PLATFORM-SPECIFIC OPERATIONS
    # ================================
    
    @abstractmethod
    async def get_platform_info(self) -> Dict[str, Any]:
        """
        Get platform-specific information
        Returns:
            Dictionary with platform capabilities and metadata
        """
        pass
    
    @abstractmethod
    async def get_available_instruments(self) -> List[str]:
        """
        Get list of available trading instruments
        Returns:
            List of instrument names
        """
        pass
    
    # ================================
    # VALIDATION AND UTILITIES
    # ================================
    
    async def validate_trade_params(self, params: TradeParams) -> bool:
        """
        Validate trade parameters for this platform
        Args:
            params: TradeParams to validate
        Returns:
            True if valid, raises exception if invalid
        """
        # Basic validation that applies to all platforms
        if not params.instrument:
            raise ValueError("Instrument is required")
        if params.units <= 0:
            raise ValueError("Units must be positive")
            
        # Order type specific validation
        if params.order_type == OrderType.LIMIT and params.price is None:
            raise ValueError("Limit orders require a price")
        if params.order_type == OrderType.STOP and params.stop_price is None:
            raise ValueError("Stop orders require a stop price")
            
        # Platform-specific validation (can be overridden)
        return await self._platform_specific_validation(params)
    
    async def _platform_specific_validation(self, params: TradeParams) -> bool:
        """
        Platform-specific validation logic
        Override in platform adapters for custom validation
        """
        return True
    
    async def calculate_position_size(
        self,
        instrument: str,
        risk_amount: float,
        stop_loss_distance: float
    ) -> float:
        """
        Calculate position size based on risk parameters
        Args:
            instrument: Trading instrument
            risk_amount: Amount to risk (in account currency)
            stop_loss_distance: Distance to stop loss in price units
        Returns:
            Calculated position size in units
        """
        # Basic position sizing calculation
        # Platform adapters can override for more sophisticated calculations
        if stop_loss_distance <= 0:
            raise ValueError("Stop loss distance must be positive")
            
        # Get current price for pip value calculation
        current_price = await self.get_live_price(instrument)
        
        # Basic calculation - can be enhanced per platform
        position_size = risk_amount / stop_loss_distance
        
        return abs(position_size)
    
    # ================================
    # CONTEXT MANAGERS
    # ================================
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
        return False


class TradingPlatformError(Exception):
    """Base exception for trading platform errors"""
    
    def __init__(self, message: str, platform: Platform, error_code: Optional[str] = None):
        self.platform = platform
        self.error_code = error_code
        super().__init__(f"[{platform.value}] {message}")


class ConnectionError(TradingPlatformError):
    """Platform connection errors"""
    pass


class ExecutionError(TradingPlatformError):
    """Trade execution errors"""
    pass


class ValidationError(TradingPlatformError):
    """Parameter validation errors"""
    pass


class InsufficientFundsError(TradingPlatformError):
    """Insufficient funds errors"""
    pass


class MarketClosedError(TradingPlatformError):
    """Market closed errors"""
    pass


# ================================
# PLATFORM ADAPTER REGISTRY
# ================================

class PlatformRegistry:
    """Registry for tracking available platform adapters"""
    
    _adapters: Dict[Platform, type] = {}
    
    @classmethod
    def register(cls, platform: Platform, adapter_class: type):
        """Register a platform adapter"""
        if not issubclass(adapter_class, UniversalTradingInterface):
            raise ValueError(f"Adapter must inherit from UniversalTradingInterface")
        cls._adapters[platform] = adapter_class
    
    @classmethod
    def get_adapter_class(cls, platform: Platform) -> type:
        """Get adapter class for platform"""
        if platform not in cls._adapters:
            raise ValueError(f"No adapter registered for platform: {platform}")
        return cls._adapters[platform]
    
    @classmethod
    def get_available_platforms(cls) -> List[Platform]:
        """Get list of available platforms"""
        return list(cls._adapters.keys())
    
    @classmethod
    def create_adapter(cls, platform: Platform, config: Dict[str, Any]) -> UniversalTradingInterface:
        """Create adapter instance for platform"""
        adapter_class = cls.get_adapter_class(platform)
        return adapter_class(platform, config)


# ================================
# HELPER FUNCTIONS
# ================================

def platform_supports_feature(platform: Platform, feature: str) -> bool:
    """Check if platform supports a specific feature"""
    # This can be expanded with platform capability matrices
    feature_matrix = {
        Platform.OANDA: {
            "limit_orders", "stop_orders", "trailing_stops", 
            "partial_closes", "guaranteed_stops"
        },
        Platform.METATRADER5: {
            "limit_orders", "stop_orders", "pending_orders",
            "expert_advisors", "custom_indicators"
        },
        Platform.TRADELOCKER: {
            "limit_orders", "stop_orders", "bracket_orders",
            "market_depth", "level2_data"
        }
        # Add more platforms as needed
    }
    
    return feature in feature_matrix.get(platform, set())


async def test_platform_connection(adapter: UniversalTradingInterface) -> Dict[str, Any]:
    """Test platform connection and basic functionality"""
    test_results = {
        "platform": adapter.platform_name,
        "connection": False,
        "account_info": False,
        "market_data": False,
        "health_check": False,
        "errors": []
    }
    
    try:
        # Test connection
        await adapter.connect()
        test_results["connection"] = True
        
        # Test account info
        account_info = await adapter.get_account_info()
        test_results["account_info"] = True
        
        # Test market data (try EUR_USD as it's commonly available)
        try:
            price_data = await adapter.get_live_price("EUR_USD")
            test_results["market_data"] = True
        except Exception as e:
            test_results["errors"].append(f"Market data test failed: {str(e)}")
        
        # Test health check
        health = await adapter.health_check()
        test_results["health_check"] = True
        
    except Exception as e:
        test_results["errors"].append(f"Connection test failed: {str(e)}")
    
    finally:
        try:
            await adapter.disconnect()
        except Exception as e:
            test_results["errors"].append(f"Disconnect failed: {str(e)}")
    
    return test_results
"""
Oanda Platform Adapter for Universal Trading Interface
File: src/platform_abstraction/adapters/oanda_adapter.py

Wraps the existing OandaDirectAPI functionality with the universal trading interface.
This allows your existing Oanda integration to work through the platform abstraction layer.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import asyncio

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import your existing Oanda integration
from src.mcp_servers.oanda_direct_api import OandaDirectAPI
from src.config.logging_config import logger

# Import the universal interface and models
from ..interface import UniversalTradingInterface, TradingPlatformError, ConnectionError, ExecutionError
from ..models import (
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
    PositionSide,
    OrderType,
    OrderStatus,
    convert_side_to_enum,
    convert_position_side_to_enum,
    generate_trade_reference
)


class OandaAdapter(UniversalTradingInterface):
    """
    Oanda platform adapter that wraps your existing OandaDirectAPI
    """
    
    def __init__(self, platform: Platform, config: Dict[str, Any]):
        super().__init__(platform, config)
        self._oanda_api = None
        self._connection_verified = False
        
    # ================================
    # CONNECTION MANAGEMENT
    # ================================
    
    async def connect(self) -> bool:
        """Connect to Oanda platform"""
        try:
            # Test connection by creating API instance and checking account
            async with OandaDirectAPI() as oanda:
                account_info = await oanda.get_account_info()
                if account_info.get("success"):
                    self._connected = True
                    self._connection_verified = True
                    logger.info(f"✅ Oanda adapter connected successfully")
                    return True
                else:
                    raise ConnectionError(
                        f"Failed to connect to Oanda: {account_info.get('error')}",
                        Platform.OANDA
                    )
        except Exception as e:
            logger.error(f"❌ Oanda connection failed: {str(e)}")
            raise ConnectionError(f"Oanda connection failed: {str(e)}", Platform.OANDA)
    
    async def disconnect(self) -> bool:
        """Disconnect from Oanda platform"""
        self._connected = False
        self._connection_verified = False
        logger.info("🔌 Oanda adapter disconnected")
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Oanda platform health"""
        try:
            async with OandaDirectAPI() as oanda:
                # Test basic operations
                account_info = await oanda.get_account_info()
                
                # Try to get a price quote
                try:
                    price_data = await oanda.get_current_price("EUR_USD")
                    market_data_ok = price_data.get("success", False)
                except Exception:
                    market_data_ok = False
                
                return {
                    "platform": "oanda",
                    "connected": account_info.get("success", False),
                    "account_accessible": account_info.get("success", False),
                    "market_data_accessible": market_data_ok,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": None,  # Could add latency measurement
                    "status": "healthy" if account_info.get("success") else "degraded"
                }
        except Exception as e:
            return {
                "platform": "oanda",
                "connected": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    # ================================
    # MARKET DATA OPERATIONS
    # ================================
    
    async def get_live_price(self, instrument: str) -> UniversalMarketPrice:
        """Get current live price from Oanda"""
        try:
            async with OandaDirectAPI() as oanda:
                price_data = await oanda.get_current_price(instrument)
                
                if not price_data.get("success"):
                    raise TradingPlatformError(
                        f"Failed to get price for {instrument}: {price_data.get('error')}",
                        Platform.OANDA
                    )
                
                bid = float(price_data.get("bid", 0))
                ask = float(price_data.get("ask", 0))
                
                return UniversalMarketPrice(
                    instrument=instrument,
                    bid=bid,
                    ask=ask,
                    spread=ask - bid,
                    timestamp=datetime.now(timezone.utc),
                    platform_source=Platform.OANDA
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to get live price for {instrument}: {str(e)}")
            raise TradingPlatformError(f"Live price retrieval failed: {str(e)}", Platform.OANDA)
    
    async def get_historical_data(
        self,
        instrument: str,
        timeframe: str,
        count: int = 200,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> HistoricalData:
        """Get historical data from Oanda"""
        try:
            async with OandaDirectAPI() as oanda:
                # Use your existing historical data method
                # This might need adjustment based on your actual OandaDirectAPI implementation
                historical_data = await oanda.get_historical_data(
                    instrument=instrument,
                    granularity=timeframe,
                    count=count
                )
                
                if not historical_data.get("success"):
                    raise TradingPlatformError(
                        f"Failed to get historical data: {historical_data.get('error')}",
                        Platform.OANDA
                    )
                
                return HistoricalData(
                    instrument=instrument,
                    timeframe=timeframe,
                    data=historical_data.get("data", []),
                    platform_source=Platform.OANDA,
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to get historical data for {instrument}: {str(e)}")
            raise TradingPlatformError(f"Historical data retrieval failed: {str(e)}", Platform.OANDA)
    
    # ================================
    # ACCOUNT OPERATIONS
    # ================================
    
    async def get_account_info(self) -> UniversalAccountInfo:
        """Get account information from Oanda"""
        try:
            async with OandaDirectAPI() as oanda:
                account_data = await oanda.get_account_info()
                
                if not account_data.get("success"):
                    raise TradingPlatformError(
                        f"Failed to get account info: {account_data.get('error')}",
                        Platform.OANDA
                    )
                
                return UniversalAccountInfo(
                    account_id=account_data.get("account_id", "unknown"),
                    currency=account_data.get("currency", "USD"),
                    balance=float(account_data.get("balance", 0)),
                    equity=float(account_data.get("nav", account_data.get("balance", 0))),
                    free_margin=float(account_data.get("margin_available", 0)),
                    used_margin=float(account_data.get("margin_used", 0)),
                    margin_rate=float(account_data.get("margin_rate", 0.02)),
                    platform_source=Platform.OANDA,
                    timestamp=datetime.now(timezone.utc),
                    unrealized_pl=float(account_data.get("unrealized_pl", 0)),
                    leverage=float(account_data.get("margin_rate", 0.02)) * 100 if account_data.get("margin_rate") else None
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to get account info: {str(e)}")
            raise TradingPlatformError(f"Account info retrieval failed: {str(e)}", Platform.OANDA)
    
    async def get_portfolio_status(self) -> UniversalPortfolioStatus:
        """Get comprehensive portfolio status"""
        try:
            # Get all components
            account_info = await self.get_account_info()
            positions = await self.get_open_positions()
            orders = await self.get_pending_orders()
            
            return UniversalPortfolioStatus(
                account_info=account_info,
                positions=positions,
                orders=orders,
                timestamp=datetime.now(timezone.utc),
                platform_sources=[Platform.OANDA]
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio status: {str(e)}")
            raise TradingPlatformError(f"Portfolio status retrieval failed: {str(e)}", Platform.OANDA)
    
    # ================================
    # POSITION MANAGEMENT
    # ================================
    
    async def get_open_positions(self) -> List[UniversalPosition]:
        """Get all open positions from Oanda"""
        try:
            async with OandaDirectAPI() as oanda:
                positions_data = await oanda.get_positions()
                
                if not positions_data.get("success"):
                    raise TradingPlatformError(
                        f"Failed to get positions: {positions_data.get('error')}",
                        Platform.OANDA
                    )
                
                universal_positions = []
                positions = positions_data.get("positions", [])
                
                for position_data in positions:
                    # Convert Oanda position data to universal format
                    units = float(position_data.get("units", 0))
                    side = PositionSide.LONG if units > 0 else PositionSide.SHORT
                    
                    universal_position = UniversalPosition(
                        position_id=position_data.get("id", f"oanda_{position_data.get('instrument')}"),
                        instrument=position_data.get("instrument", ""),
                        side=side,
                        units=abs(units),
                        average_price=float(position_data.get("average_price", 0)),
                        current_price=float(position_data.get("current_price", 0)),
                        unrealized_pl=float(position_data.get("unrealized_pl", 0)),
                        timestamp=datetime.now(timezone.utc),
                        platform_source=Platform.OANDA,
                        margin_used=float(position_data.get("margin_used", 0))
                    )
                    universal_positions.append(universal_position)
                
                return universal_positions
                
        except Exception as e:
            logger.error(f"❌ Failed to get open positions: {str(e)}")
            raise TradingPlatformError(f"Position retrieval failed: {str(e)}", Platform.OANDA)
    
    async def get_position(self, instrument: str) -> Optional[UniversalPosition]:
        """Get position for specific instrument"""
        positions = await self.get_open_positions()
        for position in positions:
            if position.instrument == instrument:
                return position
        return None
    
    async def close_position(
        self,
        instrument: str,
        units: Optional[float] = None,
        reason: str = "Manual close"
    ) -> UniversalTradeResult:
        """Close position using existing close_position functionality"""
        try:
            async with OandaDirectAPI() as oanda:
                close_result = await oanda.close_position(
                    instrument=instrument,
                    units=str(units) if units else None
                )
                
                if not close_result.get("success"):
                    return UniversalTradeResult.error_result(
                        trade_reference=generate_trade_reference(Platform.OANDA),
                        platform_source=Platform.OANDA,
                        error=f"Position close failed: {close_result.get('error')}",
                        error_type="close_position_error"
                    )
                
                # Calculate total P&L
                total_pl = 0
                if close_result.get("long_close"):
                    total_pl += close_result["long_close"].get("pl", 0)
                if close_result.get("short_close"):
                    total_pl += close_result["short_close"].get("pl", 0)
                
                return UniversalTradeResult.success_result(
                    trade_reference=generate_trade_reference(Platform.OANDA),
                    platform_source=Platform.OANDA,
                    order_id=f"close_{instrument}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    instrument=instrument,
                    side=TradeSide.SELL,  # Closing is always opposite side
                    units=units or 0,
                    execution_price=0,  # Price not available in close result
                    metadata={
                        "total_pl": total_pl,
                        "reason": reason,
                        "close_details": close_result
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to close position {instrument}: {str(e)}")
            return UniversalTradeResult.error_result(
                trade_reference=generate_trade_reference(Platform.OANDA),
                platform_source=Platform.OANDA,
                error=f"Close position failed: {str(e)}",
                error_type="exception"
            )
    
    # ================================
    # ORDER MANAGEMENT
    # ================================
    
    async def get_pending_orders(self) -> List[UniversalOrder]:
        """Get all pending orders from Oanda"""
        try:
            async with OandaDirectAPI() as oanda:
                orders_data = await oanda.get_orders()
                
                if not orders_data.get("success"):
                    raise TradingPlatformError(
                        f"Failed to get orders: {orders_data.get('error')}",
                        Platform.OANDA
                    )
                
                universal_orders = []
                orders = orders_data.get("orders", [])
                
                for order_data in orders:
                    # Convert Oanda order data to universal format
                    order_type_str = order_data.get("type", "market").lower()
                    order_type = OrderType.LIMIT if "limit" in order_type_str else OrderType.MARKET
                    
                    side_str = order_data.get("side", "buy")
                    trade_side = convert_side_to_enum(side_str)
                    
                    universal_order = UniversalOrder(
                        order_id=order_data.get("id", "unknown"),
                        instrument=order_data.get("instrument", ""),
                        side=trade_side,
                        order_type=order_type,
                        status=OrderStatus.PENDING,  # Assume pending since it's in pending orders
                        units=float(order_data.get("units", 0)),
                        timestamp=datetime.now(timezone.utc),
                        platform_source=Platform.OANDA,
                        price=float(order_data.get("price", 0)) if order_data.get("price") else None,
                        stop_loss=float(order_data.get("stop_loss", 0)) if order_data.get("stop_loss") else None,
                        take_profit=float(order_data.get("take_profit", 0)) if order_data.get("take_profit") else None
                    )
                    universal_orders.append(universal_order)
                
                return universal_orders
                
        except Exception as e:
            logger.error(f"❌ Failed to get pending orders: {str(e)}")
            raise TradingPlatformError(f"Order retrieval failed: {str(e)}", Platform.OANDA)
    
    async def get_order(self, order_id: str) -> Optional[UniversalOrder]:
        """Get specific order by ID"""
        orders = await self.get_pending_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
        return None
    
    async def cancel_order(self, order_id: str) -> UniversalTradeResult:
        """Cancel pending order"""
        try:
            async with OandaDirectAPI() as oanda:
                cancel_result = await oanda.cancel_order(order_id)
                
                if not cancel_result.get("success"):
                    return UniversalTradeResult.error_result(
                        trade_reference=generate_trade_reference(Platform.OANDA),
                        platform_source=Platform.OANDA,
                        error=f"Order cancellation failed: {cancel_result.get('error')}",
                        error_type="cancel_order_error"
                    )
                
                return UniversalTradeResult.success_result(
                    trade_reference=generate_trade_reference(Platform.OANDA),
                    platform_source=Platform.OANDA,
                    order_id=order_id,
                    instrument="unknown",  # Not available in cancel result
                    side=TradeSide.BUY,  # Placeholder
                    units=0,
                    execution_price=0,
                    metadata={"action": "cancelled", "cancel_details": cancel_result}
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to cancel order {order_id}: {str(e)}")
            return UniversalTradeResult.error_result(
                trade_reference=generate_trade_reference(Platform.OANDA),
                platform_source=Platform.OANDA,
                error=f"Cancel order failed: {str(e)}",
                error_type="exception"
            )
    
    # ================================
    # TRADE EXECUTION
    # ================================
    
    async def execute_market_trade(self, params: TradeParams) -> UniversalTradeResult:
        """Execute market order using existing execute_market_trade_async"""
        try:
            # Import your existing function
            from src.autonomous_trading_system.tools.trading_execution_tools_sync import execute_market_trade_async
            
            # Convert TradeSide enum to string
            side_str = "buy" if params.side == TradeSide.BUY else "sell"
            
            # Call your existing function
            result = await execute_market_trade_async(
                instrument=params.instrument,
                side=side_str,
                units=params.units,
                stop_loss=params.stop_loss,
                take_profit=params.take_profit,
                reason=params.reason
            )
            
            # Convert result to universal format
            if result.get("success"):
                return UniversalTradeResult.success_result(
                    trade_reference=result.get("trade_reference", generate_trade_reference(Platform.OANDA)),
                    platform_source=Platform.OANDA,
                    order_id=result.get("order_id", "unknown"),
                    instrument=params.instrument,
                    side=params.side,
                    units=params.units,
                    execution_price=result.get("execution_price", 0),
                    metadata={"oanda_result": result}
                )
            else:
                return UniversalTradeResult.error_result(
                    trade_reference=result.get("trade_reference", generate_trade_reference(Platform.OANDA)),
                    platform_source=Platform.OANDA,
                    error=result.get("error", "Unknown error"),
                    error_type=result.get("error_type", "execution_error")
                )
                
        except Exception as e:
            logger.error(f"❌ Market trade execution failed: {str(e)}")
            return UniversalTradeResult.error_result(
                trade_reference=generate_trade_reference(Platform.OANDA),
                platform_source=Platform.OANDA,
                error=f"Market trade execution failed: {str(e)}",
                error_type="exception"
            )
    
    async def execute_limit_trade(self, params: TradeParams) -> UniversalTradeResult:
        """Execute limit order using existing execute_limit_trade_async"""
        try:
            # Import your existing function
            from src.autonomous_trading_system.tools.trading_execution_tools_sync import execute_limit_trade_async
            
            # Convert TradeSide enum to string
            side_str = "buy" if params.side == TradeSide.BUY else "sell"
            
            # Call your existing function
            result = await execute_limit_trade_async(
                instrument=params.instrument,
                side=side_str,
                units=params.units,
                price=params.price if params.price is not None else 0.0,
                stop_loss=params.stop_loss,
                take_profit=params.take_profit,
                reason=params.reason
            )
            
            # Convert result to universal format
            if result.get("success"):
                return UniversalTradeResult.success_result(
                    trade_reference=result.get("trade_reference", generate_trade_reference(Platform.OANDA)),
                    platform_source=Platform.OANDA,
                    order_id=result.get("order_id", "unknown"),
                    instrument=params.instrument,
                    side=params.side,
                    units=params.units,
                    execution_price=result.get("limit_price", params.price),
                    metadata={"oanda_result": result, "order_status": "pending"}
                )
            else:
                return UniversalTradeResult.error_result(
                    trade_reference=result.get("trade_reference", generate_trade_reference(Platform.OANDA)),
                    platform_source=Platform.OANDA,
                    error=result.get("error", "Unknown error"),
                    error_type=result.get("error_type", "execution_error")
                )
                
        except Exception as e:
            logger.error(f"❌ Limit trade execution failed: {str(e)}")
            return UniversalTradeResult.error_result(
                trade_reference=generate_trade_reference(Platform.OANDA),
                platform_source=Platform.OANDA,
                error=f"Limit trade execution failed: {str(e)}",
                error_type="exception"
            )
    
    async def execute_stop_trade(self, params: TradeParams) -> UniversalTradeResult:
        """Execute stop order - implement based on Oanda's stop order capabilities"""
        # Oanda might not have dedicated stop orders, so this could create a limit order
        # at the stop price or use a different approach
        return UniversalTradeResult.error_result(
            trade_reference=generate_trade_reference(Platform.OANDA),
            platform_source=Platform.OANDA,
            error="Stop orders not yet implemented for Oanda",
            error_type="not_implemented"
        )
    
    # ================================
    # PLATFORM-SPECIFIC OPERATIONS
    # ================================
    
    async def get_platform_info(self) -> Dict[str, Any]:
        """Get Oanda platform information"""
        return {
            "platform": "oanda",
            "name": "Oanda v20 API",
            "version": "v20",
            "supported_features": [
                "market_orders", "limit_orders", "stop_loss", "take_profit",
                "position_closing", "streaming_prices", "historical_data"
            ],
            "supported_instruments": ["forex", "indices", "commodities", "bonds"],
            "max_leverage": 50,  # Varies by region
            "min_trade_size": 1,
            "api_rate_limits": {
                "requests_per_second": 20,
                "streaming_connections": 20
            }
        }
    
    async def get_available_instruments(self) -> List[str]:
        """Get available trading instruments from Oanda"""
        try:
            async with OandaDirectAPI() as oanda:
                # This method might need to be implemented in your OandaDirectAPI
                # For now, return common forex pairs
                return [
                    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
                    "EUR_GBP", "EUR_JPY", "GBP_JPY", "US30_USD", "UK100_GBP", "DE30_EUR"
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get available instruments: {str(e)}")
            # Return default list if API call fails
            return ["EUR_USD", "GBP_USD", "USD_JPY", "US30_USD"]


# Register the Oanda adapter
from ..interface import PlatformRegistry
PlatformRegistry.register(Platform.OANDA, OandaAdapter)
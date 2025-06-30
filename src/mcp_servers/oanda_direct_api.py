"""
Direct Oanda API Wrapper
Replaces MCP server with direct oandapyV20 integration
"""

import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union, cast
from datetime import datetime, timezone
import json
import pandas as pd
from decimal import Decimal
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    import oandapyV20
    from oandapyV20 import API
    from oandapyV20.endpoints import accounts, instruments, orders, positions, trades, pricing
    from oandapyV20.exceptions import V20Error
except ImportError:
    print("❌ oandapyV20 not installed. Run: pip install oandapyV20")
    sys.exit(1)

from src.config.logging_config import logger
from src.database.manager import db_manager
from src.database.models import EventLog, LogLevel


class OandaDirectAPI:
    """Direct Oanda API wrapper using oandapyV20"""

    def __init__(self):
        # Load configuration from environment
        self.access_token = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.environment = os.getenv('OANDA_ENVIRONMENT', 'practice')  # practice or live
        
        if not self.access_token:
            raise ValueError("OANDA_API_KEY environment variable is required")
        if not self.account_id:
            raise ValueError("OANDA_ACCOUNT_ID environment variable is required")
        
        # Set the correct API endpoint
        if self.environment.lower() == 'live':
            self.api_url = "https://api-fxtrade.oanda.com"
        else:
            self.api_url = "https://api-fxpractice.oanda.com"
        
        # Initialize API client
        self.client = API(access_token=self.access_token, environment=self.environment)
        
        logger.info(
            "Oanda Direct API initialized",
            environment=self.environment,
            account_id=self.account_id[:8] + "..." if self.account_id else None
        )

    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass

    def _handle_oanda_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle Oanda API errors consistently"""
        if isinstance(error, V20Error):
            error_msg = f"Oanda API error in {operation}: {error.msg}"
            error_code = getattr(error, 'code', 'UNKNOWN')
        else:
            error_msg = f"Error in {operation}: {str(error)}"
            error_code = 'GENERAL_ERROR'
        
        logger.error(error_msg, operation=operation, error_code=error_code)
        
        return {
            "success": False,
            "error": error_msg,
            "error_code": error_code,
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check if Oanda API is accessible and account is valid"""
        try:
            # Test API connectivity by getting account info
            request = accounts.AccountDetails(self.account_id)
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary
            if isinstance(response, dict) and 'account' in response:
                return {
                    "status": "healthy",
                    "server": "oanda-direct-api",
                    "environment": self.environment,
                    "account_id": self.account_id[:8] + "..." if self.account_id else None,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Invalid response from Oanda API",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            return self._handle_oanda_error(e, "health_check")

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            request = accounts.AccountDetails(self.account_id)
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary with account data
            if isinstance(response, dict) and 'account' in response:
                account = response['account']
                
                # Extract and format key account information
                account_info = {
                    "success": True,
                    "account_id": account.get('id'),
                    "balance": float(account.get('balance', 0)),
                    "currency": account.get('currency'),
                    "margin_used": float(account.get('marginUsed', 0)),
                    "margin_available": float(account.get('marginAvailable', 0)),
                    "margin_rate": float(account.get('marginRate', 0)),
                    "nav": float(account.get('NAV', 0)),
                    "unrealized_pl": float(account.get('unrealizedPL', 0)),
                    "pl": float(account.get('pl', 0)),
                    "open_trade_count": int(account.get('openTradeCount', 0)),
                    "open_position_count": int(account.get('openPositionCount', 0)),
                    "pending_order_count": int(account.get('pendingOrderCount', 0)),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info(
                    "Account info retrieved",
                    balance=account_info["balance"],
                    currency=account_info["currency"],
                    margin_available=account_info["margin_available"]
                )
                
                return account_info
            else:
                raise Exception("Invalid response format from Oanda API")
                
        except Exception as e:
            return self._handle_oanda_error(e, "get_account_info")

    async def get_current_price(self, instrument: str) -> Dict[str, Any]:
        """Get current price for an instrument"""
        try:
            # Get current pricing
            request = pricing.PricingInfo(
                accountID=self.account_id,
                params={"instruments": instrument}
            )
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary with prices
            if isinstance(response, dict) and 'prices' in response and len(response['prices']) > 0:
                price_data = response['prices'][0]
                
                # Extract pricing information
                bids = price_data.get('bids', [{}])
                asks = price_data.get('asks', [{}])
                
                bid_price = float(bids[0].get('price', 0)) if bids else 0
                ask_price = float(asks[0].get('price', 0)) if asks else 0
                spread = round(ask_price - bid_price, 5) if bid_price and ask_price else 0
                
                pricing_info = {
                    "success": True,
                    "instrument": instrument,
                    "bid": bid_price,
                    "ask": ask_price,
                    "spread": spread,
                    "price": (bid_price + ask_price) / 2 if bid_price and ask_price else 0,
                    "timestamp": price_data.get('time'),
                    "tradeable": price_data.get('tradeable', False),
                    "query_time": datetime.now(timezone.utc).isoformat()
                }
                
                logger.debug(
                    f"{instrument} price retrieved",
                    bid=bid_price,
                    ask=ask_price,
                    spread=spread
                )
                
                return pricing_info
            else:
                raise Exception(f"No price data available for {instrument}")
                
        except Exception as e:
            return self._handle_oanda_error(e, f"get_current_price({instrument})")

    async def get_historical_data(
        self, 
        instrument: str, 
        granularity: str = "M1", 
        count: int = 100,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get historical price data"""
        try:
            # Prepare parameters with proper typing
            params: Dict[str, Any] = {
                "granularity": granularity,
                "count": count
            }
            
            if from_time:
                params["from"] = from_time
            if to_time:
                params["to"] = to_time
            
            # Get historical data
            request = instruments.InstrumentsCandles(
                instrument=instrument,
                params=params
            )
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary with candles
            if isinstance(response, dict) and 'candles' in response:
                candles = response['candles']
                
                # Process candle data
                processed_candles = []
                for candle in candles:
                    if candle.get('complete', False):  # Only include complete candles
                        mid = candle.get('mid', {})
                        processed_candle = {
                            "time": candle.get('time'),
                            "volume": int(candle.get('volume', 0)),
                            "mid": {
                                "o": float(mid.get('o', 0)),
                                "h": float(mid.get('h', 0)),
                                "l": float(mid.get('l', 0)),
                                "c": float(mid.get('c', 0))
                            }
                        }
                        processed_candles.append(processed_candle)
                
                historical_data = {
                    "success": True,
                    "instrument": instrument,
                    "granularity": granularity,
                    "data": {
                        "candles": processed_candles
                    },
                    "count": len(processed_candles),
                    "query_time": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info(
                    f"Historical data retrieved for {instrument}",
                    granularity=granularity,
                    candle_count=len(processed_candles)
                )
                
                return historical_data
            else:
                raise Exception(f"No historical data available for {instrument}")
                
        except Exception as e:
            return self._handle_oanda_error(e, f"get_historical_data({instrument})")

    async def get_positions(self) -> Dict[str, Any]:
        """Get all open positions"""
        try:
            request = positions.OpenPositions(self.account_id)
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary
            if isinstance(response, dict) and 'positions' in response:
                positions_data = response['positions']
                
                # Process position data
                processed_positions = []
                for position in positions_data:
                    processed_position = {
                        "instrument": position.get('instrument'),
                        "pl": float(position.get('pl', 0)),
                        "unrealized_pl": float(position.get('unrealizedPL', 0)),
                        "margin_used": float(position.get('marginUsed', 0)),
                        "commission": float(position.get('commission', 0)),
                        
                        # Long position details
                        "long": {
                            "units": float(position.get('long', {}).get('units', 0)),
                            "pl": float(position.get('long', {}).get('pl', 0)),
                            "unrealized_pl": float(position.get('long', {}).get('unrealizedPL', 0)),
                            "avg_price": float(position.get('long', {}).get('averagePrice', 0)) if position.get('long', {}).get('averagePrice') else None
                        },
                        
                        # Short position details
                        "short": {
                            "units": float(position.get('short', {}).get('units', 0)),
                            "pl": float(position.get('short', {}).get('pl', 0)),
                            "unrealized_pl": float(position.get('short', {}).get('unrealizedPL', 0)),
                            "avg_price": float(position.get('short', {}).get('averagePrice', 0)) if position.get('short', {}).get('averagePrice') else None
                        }
                    }
                    processed_positions.append(processed_position)
                
                return {
                    "success": True,
                    "positions": processed_positions,
                    "position_count": len(processed_positions),
                    "query_time": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "success": True,
                    "positions": [],
                    "position_count": 0,
                    "query_time": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            return self._handle_oanda_error(e, "get_positions")

    async def get_orders(self) -> Dict[str, Any]:
        """Get all pending orders"""
        try:
            request = orders.OrdersPending(self.account_id)
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary
            if isinstance(response, dict) and 'orders' in response:
                orders_data = response['orders']
                
                # Process order data
                processed_orders = []
                for order in orders_data:
                    processed_order = {
                        "id": order.get('id'),
                        "type": order.get('type'),
                        "instrument": order.get('instrument'),
                        "units": float(order.get('units', 0)),
                        "price": float(order.get('price', 0)) if order.get('price') else None,
                        "stop_loss_price": float(order.get('stopLossOnFill', {}).get('price', 0)) if order.get('stopLossOnFill') else None,
                        "take_profit_price": float(order.get('takeProfitOnFill', {}).get('price', 0)) if order.get('takeProfitOnFill') else None,
                        "time_in_force": order.get('timeInForce'),
                        "created_time": order.get('createTime'),
                        "state": order.get('state')
                    }
                    processed_orders.append(processed_order)
                
                return {
                    "success": True,
                    "orders": processed_orders,
                    "order_count": len(processed_orders),
                    "query_time": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "success": True,
                    "orders": [],
                    "order_count": 0,
                    "query_time": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            return self._handle_oanda_error(e, "get_orders")

    async def create_market_order(
        self,
        instrument: str,
        units: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a market order"""
        try:
            # Prepare order data with proper typing
            order_data: Dict[str, Any] = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(int(units)),
                    "timeInForce": "FOK",  # Fill or Kill
                    "positionFill": "DEFAULT"
                }
            }
            
            # Add stop loss if specified
            if stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
            
            # Add take profit if specified
            if take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit),
                    "timeInForce": "GTC"
                }
            
            # Submit order
            request = orders.OrderCreate(self.account_id, data=order_data)
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary
            if isinstance(response, dict):
                transaction = response.get('orderFillTransaction', response.get('orderCreateTransaction', {}))
                
                result = {
                    "success": True,
                    "order_id": transaction.get('id'),
                    "transaction_id": transaction.get('id'),
                    "instrument": instrument,
                    "units": units,
                    "type": "MARKET",
                    "price": float(transaction.get('price', 0)) if transaction.get('price') else None,
                    "time": transaction.get('time'),
                    "pl": float(transaction.get('pl', 0)) if transaction.get('pl') else 0,
                    "commission": float(transaction.get('commission', 0)) if transaction.get('commission') else 0,
                    "financing": float(transaction.get('financing', 0)) if transaction.get('financing') else 0,
                    "reason": transaction.get('reason'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info(
                    "Market order created",
                    instrument=instrument,
                    units=units,
                    order_id=result["order_id"],
                    price=result["price"]
                )
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="ORDER_CREATED",
                    message=f"Market order created for {instrument}",
                    context=result
                )
                
                return result
            else:
                raise Exception("No response from order creation")
                
        except Exception as e:
            return self._handle_oanda_error(e, f"create_market_order({instrument}, {units})")

    async def create_limit_order(
        self,
        instrument: str,
        units: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a limit order"""
        try:
            # Prepare order data with proper typing
            order_data: Dict[str, Any] = {
                "order": {
                    "type": "LIMIT",
                    "instrument": instrument,
                    "units": str(int(units)),
                    "price": str(price),
                    "timeInForce": "GTC",  # Good Till Cancelled
                    "positionFill": "DEFAULT"
                }
            }
            
            # Add stop loss if specified
            if stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
            
            # Add take profit if specified
            if take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit),
                    "timeInForce": "GTC"
                }
            
            # Submit order
            request = orders.OrderCreate(self.account_id, data=order_data)
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary
            if isinstance(response, dict):
                transaction = response.get('orderCreateTransaction', {})
                
                result = {
                    "success": True,
                    "order_id": transaction.get('id'),
                    "transaction_id": transaction.get('id'),
                    "instrument": instrument,
                    "units": units,
                    "type": "LIMIT",
                    "price": price,
                    "time": transaction.get('time'),
                    "reason": transaction.get('reason'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info(
                    "Limit order created",
                    instrument=instrument,
                    units=units,
                    order_id=result["order_id"],
                    price=price
                )
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="ORDER_CREATED",
                    message=f"Limit order created for {instrument} at {price}",
                    context=result
                )
                
                return result
            else:
                raise Exception("No response from order creation")
                
        except Exception as e:
            return self._handle_oanda_error(e, f"create_limit_order({instrument}, {units}, {price})")

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        try:
            request = orders.OrderCancel(self.account_id, order_id)
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary
            if isinstance(response, dict):
                transaction = response.get('orderCancelTransaction', {})
                
                result = {
                    "success": True,
                    "order_id": order_id,
                    "transaction_id": transaction.get('id'),
                    "reason": transaction.get('reason'),
                    "time": transaction.get('time'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info(
                    "Order cancelled",
                    order_id=order_id,
                    reason=result["reason"]
                )
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="ORDER_CANCELLED",
                    message=f"Order {order_id} cancelled",
                    context=result
                )
                
                return result
            else:
                raise Exception("No response from order cancellation")
                
        except Exception as e:
            return self._handle_oanda_error(e, f"cancel_order({order_id})")

    async def close_position(self, instrument: str, units: Optional[str] = None) -> Dict[str, Any]:
        """Close a position (partially or completely)"""
        try:
            # Prepare close data with proper typing
            close_data: Dict[str, str] = {}
            if units:
                if float(units) > 0:
                    close_data["longUnits"] = str(units)
                else:
                    close_data["shortUnits"] = str(abs(float(units)))
            else:
                close_data["longUnits"] = "ALL"
                close_data["shortUnits"] = "ALL"
            
            # Submit position close
            request = positions.PositionClose(
                self.account_id,
                instrument,
                data=close_data
            )
            response = self.client.request(request)
            
            # Type guard: ensure response is a dictionary
            if isinstance(response, dict):
                long_transaction = response.get('longOrderFillTransaction', {})
                short_transaction = response.get('shortOrderFillTransaction', {})
                
                result = {
                    "success": True,
                    "instrument": instrument,
                    "long_close": {
                        "units": float(long_transaction.get('units', 0)),
                        "price": float(long_transaction.get('price', 0)) if long_transaction.get('price') else None,
                        "pl": float(long_transaction.get('pl', 0)),
                        "commission": float(long_transaction.get('commission', 0)) if long_transaction.get('commission') else 0
                    } if long_transaction else None,
                    "short_close": {
                        "units": float(short_transaction.get('units', 0)),
                        "price": float(short_transaction.get('price', 0)) if short_transaction.get('price') else None,
                        "pl": float(short_transaction.get('pl', 0)),
                        "commission": float(short_transaction.get('commission', 0)) if short_transaction.get('commission') else 0
                    } if short_transaction else None,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info(
                    "Position closed",
                    instrument=instrument,
                    units=units,
                    long_pl=result["long_close"]["pl"] if result["long_close"] else 0,
                    short_pl=result["short_close"]["pl"] if result["short_close"] else 0
                )
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="POSITION_CLOSED",
                    message=f"Position closed for {instrument}",
                    context=result
                )
                
                return result
            else:
                raise Exception("No response from position close")
                
        except Exception as e:
            return self._handle_oanda_error(e, f"close_position({instrument})")

    async def _log_event(
        self, 
        level: LogLevel, 
        event_type: str, 
        message: str, 
        context: Optional[Dict] = None
    ):
        """Log events to database"""
        try:
            async with db_manager.get_async_session() as session:
                event = EventLog(
                    level=level,
                    agent_name="OandaDirectAPI",
                    event_type=event_type,
                    message=message,
                    context=context or {}
                )
                session.add(event)
                await session.commit()
        except Exception as e:
            logger.error("Failed to log event to database", error=str(e))


# Convenience functions for backward compatibility
async def test_oanda_direct_integration():
    """Test the direct Oanda API integration"""
    
    logger.info("🧪 Testing Direct Oanda API Integration...")
    
    async with OandaDirectAPI() as oanda:
        try:
            # Test health check
            health = await oanda.health_check()
            logger.info("Health check", status=health.get("status"))
            
            if not health.get("status") == "healthy":
                logger.error("❌ Direct Oanda API not available")
                return False
            
            # Test account info
            account = await oanda.get_account_info()
            if account.get("success"):
                logger.info("✅ Account info retrieved", balance=account.get("balance"))
            else:
                logger.error("❌ Failed to get account info", error=account.get("error"))
                return False
            
            # Test price data
            price = await oanda.get_current_price("EUR_USD")
            if price.get("success"):
                logger.info("✅ EUR_USD price retrieved", price=price.get("price"))
            else:
                logger.error("❌ Failed to get price", error=price.get("error"))
                return False
            
            # Test historical data
            historical = await oanda.get_historical_data("EUR_USD", "M1", 10)
            if historical.get("success"):
                logger.info("✅ Historical data retrieved", 
                           bars=historical.get("count"))
            else:
                logger.error("❌ Failed to get historical data", error=historical.get("error"))
                return False
            
            logger.info("🎉 Direct Oanda API integration successful!")
            return True
            
        except Exception as e:
            logger.error("❌ Integration test failed", error=str(e))
            return False


if __name__ == "__main__":
    import asyncio
    
    async def main():
        success = await test_oanda_direct_integration()
        if success:
            print("✅ Direct Oanda API ready!")
        else:
            print("❌ Setup needed - check environment variables")
    
    asyncio.run(main())
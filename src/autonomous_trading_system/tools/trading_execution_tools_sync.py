"""
Enhanced Trading Execution Tools for CrewAI Autonomous Trading System
Updated for OandaDirectAPI (replacing MCP wrapper)
"""

import asyncio
import sys
import concurrent.futures
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
import json
import uuid

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from crewai.tools import tool
from src.config.logging_config import logger
# UPDATED: Import OandaDirectAPI instead of OandaMCPWrapper
from src.mcp_servers.oanda_direct_api import OandaDirectAPI
from src.database.manager import db_manager
from src.database.models import EventLog, Trade, TradeStatus, TradeSide, Order, OrderType, OrderStatus, AgentAction, LogLevel


class TradeExecutionError(Exception):
    """Custom exception for trade execution errors"""
    pass


class RiskValidationError(Exception):
    """Custom exception for risk validation errors"""
    pass


class ThreadSafeAsyncRunner:
    """Thread-safe async runner for CrewAI tools - solves event loop issues"""
    
    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        
    def run_async(self, coro_func, *args, **kwargs):
        """Run async function in thread pool to avoid event loop conflicts"""
        try:
            def run_in_thread():
                # Create new event loop in thread to avoid "no current event loop" error
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro_func(*args, **kwargs))
                finally:
                    loop.close()
            
            with self._lock:  # Thread safety
                future = self._executor.submit(run_in_thread)
                return future.result(timeout=45)  # 45 second timeout for complex operations
                
        except concurrent.futures.TimeoutError:
            logger.error("Async operation timed out after 45 seconds")
            return {"success": False, "error": "Request timed out after 45 seconds"}
        except Exception as e:
            logger.error("Thread-safe async runner failed", error=str(e))
            return {"success": False, "error": f"Execution failed: {str(e)}"}

# Global runner instance
async_runner = ThreadSafeAsyncRunner()


def _generate_trade_reference() -> str:
    """Generate a unique trade reference for tracking"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"TRADE_{timestamp}_{unique_id}"


# ================================
# ASYNC CORE IMPLEMENTATIONS
# ================================

async def execute_market_trade_async(
    instrument: str,
    side: str,  # "buy" or "sell"
    units: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    reason: str = "Wyckoff signal",
    max_slippage: float = 0.001  # 0.1% max slippage
) -> Dict[str, Any]:
    """Execute market order with comprehensive risk management and logging"""
    
    try:
        # Generate unique trade reference for tracking
        trade_reference = _generate_trade_reference()
        
        # UPDATED: Use OandaDirectAPI instead of OandaMCPWrapper
        async with OandaDirectAPI() as oanda:
            account_info = await oanda.get_account_info()
            
            if not account_info.get("success"):
                raise TradeExecutionError(f"Failed to get account info: {account_info.get('error')}")
            
            account_balance = float(account_info.get("balance", 0))
            
            # Validate trade parameters
            validation = await _validate_trade_parameters(
                instrument, side, units, account_balance
            )
            
            # Get current price for slippage validation
            current_price_data = await oanda.get_current_price(instrument)
            if not current_price_data.get("success"):
                raise TradeExecutionError(f"Failed to get current price: {current_price_data.get('error')}")
            
            current_price = float(current_price_data.get("bid" if side == "sell" else "ask", 0))
            
            # Convert side to units (positive for buy, negative for sell)
            signed_units = abs(units) if side.lower() == "buy" else -abs(units)
            
            logger.info(
                "Executing market order",
                instrument=instrument,
                side=side,
                units=signed_units,
                current_price=current_price
            )
            
            # UPDATED: Use create_market_order instead of place_market_order
            order_result = await oanda.create_market_order(
                instrument=instrument,
                units=signed_units,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if not order_result.get("success"):
                raise TradeExecutionError(f"Order execution failed: {order_result.get('error')}")
            
            # Extract execution details
            execution_price = float(order_result.get("price", current_price))
            order_id = order_result.get("order_id", "unknown")
            
            # Validate slippage
            if side.lower() == "buy":
                slippage = (execution_price - current_price) / current_price
            else:
                slippage = (current_price - execution_price) / current_price
            
            if abs(slippage) > max_slippage:
                logger.warning(
                    "High slippage detected",
                    expected_price=current_price,
                    execution_price=execution_price,
                    slippage_pct=slippage * 100
                )
            
            # Log trade to database (non-critical - trade succeeds even if logging fails)
            trade_logged = await _log_trade_to_database(
                instrument=instrument,
                side=side,
                units=abs(units),
                entry_price=execution_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id,
                agent_decision={
                    "trade_reference": trade_reference,
                    "reason": reason,
                    "validation": validation,
                    "slippage": slippage
                }
            )
            
            result = {
                "success": True,
                "trade_reference": trade_reference,
                "trade_logged": trade_logged,
                "order_id": order_id,
                "transaction_id": order_result.get("transaction_id"),
                "instrument": instrument,
                "side": side,
                "units": abs(units),
                "execution_price": execution_price,
                "expected_price": current_price,
                "slippage": slippage,
                "slippage_pct": round(slippage * 100, 4),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "account_balance": account_balance,
                "position_value": validation["position_value"],
                "risk_ratio": validation["risk_ratio"],
                "execution_time": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "pl": order_result.get("pl", 0),
                "commission": order_result.get("commission", 0),
                "financing": order_result.get("financing", 0)
            }
            
            logger.info("Market order executed successfully", trade_reference=trade_reference)
            return result
            
    except Exception as e:
        trade_reference = _generate_trade_reference()
        error_result = {
            "success": False,
            "trade_reference": trade_reference,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "side": side,
            "units": units,
            "execution_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Market order execution failed", **error_result)
        return error_result


async def execute_limit_trade_async(
    instrument: str,
    side: str,  # "buy" or "sell"
    units: float,
    price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    expiry_time: Optional[str] = None,  # ISO format or "GTC"
    reason: str = "Wyckoff limit order"
) -> Dict[str, Any]:
    """Execute limit order with risk management and expiry control"""
    
    try:
        trade_reference = _generate_trade_reference()
        
        # UPDATED: Use OandaDirectAPI instead of OandaMCPWrapper
        async with OandaDirectAPI() as oanda:
            # Get account info for validation
            account_info = await oanda.get_account_info()
            if not account_info.get("success"):
                raise TradeExecutionError(f"Failed to get account info: {account_info.get('error')}")
            
            account_balance = float(account_info.get("balance", 0))
            
            # Validate trade parameters
            validation = await _validate_trade_parameters(
                instrument, side, units, account_balance
            )
            
            # Validate limit price against current market
            current_price_data = await oanda.get_current_price(instrument)
            if not current_price_data.get("success"):
                raise TradeExecutionError(f"Failed to get current price: {current_price_data.get('error')}")
            
            current_price = float(current_price_data.get("bid" if side == "sell" else "ask", 0))
            
            # Validate limit price logic
            if side.lower() == "buy" and price >= current_price:
                logger.warning(
                    "Buy limit price above market - will execute immediately",
                    limit_price=price,
                    current_price=current_price
                )
            elif side.lower() == "sell" and price <= current_price:
                logger.warning(
                    "Sell limit price below market - will execute immediately",
                    limit_price=price,
                    current_price=current_price
                )
            
            # Convert side to units (positive for buy, negative for sell)
            signed_units = abs(units) if side.lower() == "buy" else -abs(units)
            
            logger.info(
                "Placing limit order",
                instrument=instrument,
                side=side,
                units=signed_units,
                price=price,
                current_price=current_price
            )
            
            # UPDATED: Use create_limit_order instead of place_limit_order
            order_result = await oanda.create_limit_order(
                instrument=instrument,
                units=signed_units,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if not order_result.get("success"):
                raise TradeExecutionError(f"Limit order failed: {order_result.get('error')}")
            
            order_id = order_result.get("order_id", "unknown")
            
            result = {
                "success": True,
                "trade_reference": trade_reference,
                "order_id": order_id,
                "transaction_id": order_result.get("transaction_id"),
                "order_type": "limit",
                "status": "pending",
                "instrument": instrument,
                "side": side,
                "units": abs(units),
                "limit_price": price,
                "current_price": current_price,
                "price_distance": abs(price - current_price),
                "price_distance_pct": round(abs(price - current_price) / current_price * 100, 4),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "expiry_time": expiry_time or "GTC",
                "account_balance": account_balance,
                "position_value": validation["position_value"],
                "risk_ratio": validation["risk_ratio"],
                "placement_time": datetime.now(timezone.utc).isoformat(),
                "reason": reason
            }
            
            logger.info("Limit order placed successfully", **result)
            return result
            
    except Exception as e:
        trade_reference = _generate_trade_reference()
        error_result = {
            "success": False,
            "trade_reference": trade_reference,
            "error": str(e),
            "error_type": type(e).__name__,
            "order_type": "limit",
            "instrument": instrument,
            "side": side,
            "units": units,
            "price": price,
            "placement_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Limit order placement failed", **error_result)
        return error_result


async def get_portfolio_status_async() -> Dict[str, Any]:
    """Get comprehensive portfolio status"""
    
    try:
        portfolio_summary = {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "LIVE"
        }
        
        async with OandaDirectAPI() as oanda:
            # Get account information
            account_info = await oanda.get_account_info()
            
            if account_info.get("success"):
                balance = float(account_info.get("balance", 0))
                margin_used = float(account_info.get("margin_used", 0))
                margin_available = float(account_info.get("margin_available", 0))
                nav = float(account_info.get("nav", 0))
                unrealized_pl = float(account_info.get("unrealized_pl", 0))
                
                portfolio_summary.update({
                    "account_balance": balance,
                    "nav": nav,
                    "margin_used": margin_used,
                    "margin_available": margin_available,
                    "unrealized_pl": unrealized_pl,
                    "open_trades": account_info.get("open_trade_count", 0),
                    "open_positions": account_info.get("open_position_count", 0),
                    "pending_orders": account_info.get("pending_order_count", 0)
                })
            
            # Get detailed position information
            positions_data = await oanda.get_positions()
            if positions_data.get("success"):
                portfolio_summary.update({
                    "positions": positions_data.get("positions", []),
                    "total_positions": positions_data.get("position_count", 0)
                })
            
            # Get pending orders
            orders_data = await oanda.get_orders()
            if orders_data.get("success"):
                portfolio_summary.update({
                    "pending_orders_detail": orders_data.get("orders", []),
                    "total_pending_orders": orders_data.get("order_count", 0)
                })
            
            # Calculate exposure
            total_exposure = 0.0
            for position in portfolio_summary.get("positions", []):
                # Add long and short exposure
                long_units = abs(float(position.get("long", {}).get("units", 0)))
                short_units = abs(float(position.get("short", {}).get("units", 0)))
                total_exposure += long_units + short_units
            
            portfolio_summary.update({
                "total_exposure": total_exposure,
                "exposure_pct": round((total_exposure / balance * 100), 2) if balance > 0 else 0
            })
            
            logger.info(
                "Portfolio status retrieved",
                balance=balance,
                unrealized_pl=unrealized_pl,
                total_positions=portfolio_summary.get("total_positions"),
                total_exposure=total_exposure
            )
            
            return portfolio_summary
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Failed to get portfolio status", **error_result)
        return error_result


async def get_open_positions_async() -> Dict[str, Any]:
    """Get all open positions with current P&L"""
    
    try:
        async with OandaDirectAPI() as oanda:
            # Get positions data
            positions_data = await oanda.get_positions()
            
            if not positions_data.get("success"):
                raise Exception(f"Failed to get positions: {positions_data.get('error')}")
            
            # Process positions data
            positions = positions_data.get("positions", [])
            
            # Calculate portfolio metrics
            total_unrealized_pnl = 0.0
            total_exposure = 0.0
            position_details = []
            
            for position in positions:
                long_info = position.get("long", {})
                short_info = position.get("short", {})
                
                long_units = float(long_info.get("units", 0))
                short_units = float(short_info.get("units", 0))
                
                net_units = long_units + short_units  # short_units is negative
                unrealized_pnl = float(position.get("unrealized_pl", 0))
                
                # Calculate exposure (absolute value of all units)
                exposure = abs(long_units) + abs(short_units)
                
                total_unrealized_pnl += unrealized_pnl
                total_exposure += exposure
                
                # Add processed position details
                position_details.append({
                    "instrument": position.get("instrument"),
                    "net_units": net_units,
                    "side": "LONG" if net_units > 0 else "SHORT" if net_units < 0 else "FLAT",
                    "unrealized_pl": unrealized_pnl,
                    "total_pl": float(position.get("pl", 0)),
                    "exposure": exposure,
                    "margin_used": float(position.get("margin_used", 0)),
                    "commission": float(position.get("commission", 0)),
                    "long": {
                        "units": long_units,
                        "avg_price": long_info.get("avg_price"),
                        "pl": float(long_info.get("pl", 0)),
                        "unrealized_pl": float(long_info.get("unrealized_pl", 0))
                    },
                    "short": {
                        "units": short_units,
                        "avg_price": short_info.get("avg_price"),
                        "pl": float(short_info.get("pl", 0)),
                        "unrealized_pl": float(short_info.get("unrealized_pl", 0))
                    }
                })
            
            result = {
                "success": True,
                "positions": position_details,
                "position_count": len(position_details),
                "total_unrealized_pnl": round(total_unrealized_pnl, 2),
                "total_exposure": round(total_exposure, 2),
                "query_time": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(
                "Positions retrieved successfully",
                position_count=len(position_details),
                total_pnl=total_unrealized_pnl,
                total_exposure=total_exposure
            )
            
            return result
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "query_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Failed to get positions", **error_result)
        return error_result


async def get_pending_orders_async() -> Dict[str, Any]:
    """Get all pending orders"""
    
    try:
        async with OandaDirectAPI() as oanda:
            # Get orders data
            orders_data = await oanda.get_orders()
            
            if not orders_data.get("success"):
                raise Exception(f"Failed to get orders: {orders_data.get('error')}")
            
            # Process orders data
            orders = orders_data.get("orders", [])
            
            processed_orders = []
            for order in orders:
                processed_order = {
                    "id": order.get("id"),
                    "type": order.get("type"),
                    "instrument": order.get("instrument"),
                    "units": float(order.get("units", 0)),
                    "price": order.get("price"),
                    "stop_loss_price": order.get("stop_loss_price"),
                    "take_profit_price": order.get("take_profit_price"),
                    "time_in_force": order.get("time_in_force"),
                    "created_time": order.get("created_time"),
                    "state": order.get("state"),
                    "side": "BUY" if float(order.get("units", 0)) > 0 else "SELL"
                }
                processed_orders.append(processed_order)
            
            result = {
                "success": True,
                "orders": processed_orders,
                "order_count": len(processed_orders),
                "query_time": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(
                "Orders retrieved successfully",
                order_count=len(processed_orders)
            )
            
            return result
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "query_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Failed to get orders", **error_result)
        return error_result


async def close_position_async(
    instrument: str,
    units: Optional[float] = None,
    reason: str = "Manual close"
) -> Dict[str, Any]:
    """Close an existing position (full or partial)"""
    
    try:
        close_reference = f"CLOSE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        async with OandaDirectAPI() as oanda:
            logger.info(
                "Closing position",
                instrument=instrument,
                units=units,
                reason=reason
            )
            
            # Close position using Oanda Direct API
            close_result = await oanda.close_position(
                instrument=instrument,
                units=str(units) if units else None
            )
            
            if not close_result.get("success"):
                raise Exception(f"Position close failed: {close_result.get('error')}")
            
            result = {
                "success": True,
                "close_reference": close_reference,
                "instrument": instrument,
                "units_closed": units,
                "long_close": close_result.get("long_close"),
                "short_close": close_result.get("short_close"),
                "total_pl": 0,
                "close_time": datetime.now(timezone.utc).isoformat(),
                "reason": reason
            }
            
            # Calculate total P&L
            if close_result.get("long_close"):
                result["total_pl"] += close_result["long_close"].get("pl", 0)
            if close_result.get("short_close"):
                result["total_pl"] += close_result["short_close"].get("pl", 0)
            
            logger.info(
                "Position closed successfully",
                instrument=instrument,
                close_reference=close_reference,
                total_pl=result["total_pl"]
            )
            
            return result
            
    except Exception as e:
        close_reference = f"CLOSE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        error_result = {
            "success": False,
            "close_reference": close_reference,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "units": units,
            "close_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Position closure failed", **error_result)
        return error_result


async def cancel_pending_order_async(
    order_id: str,
    reason: str = "Order cancellation"
) -> Dict[str, Any]:
    """Cancel a pending order by ID"""
    
    try:
        cancel_reference = f"CANCEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        async with OandaDirectAPI() as oanda:
            logger.info(
                "Cancelling order",
                order_id=order_id,
                reason=reason
            )
            
            # Cancel order using Oanda Direct API
            cancel_result = await oanda.cancel_order(order_id)
            
            if not cancel_result.get("success"):
                raise Exception(f"Order cancellation failed: {cancel_result.get('error')}")
            
            result = {
                "success": True,
                "cancel_reference": cancel_reference,
                "order_id": order_id,
                "transaction_id": cancel_result.get("transaction_id"),
                "cancel_reason": cancel_result.get("reason"),
                "cancel_time": cancel_result.get("time"),
                "request_reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(
                "Order cancelled successfully",
                order_id=order_id,
                cancel_reference=cancel_reference
            )
            
            return result
            
    except Exception as e:
        cancel_reference = f"CANCEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        error_result = {
            "success": False,
            "cancel_reference": cancel_reference,
            "error": str(e),
            "error_type": type(e).__name__,
            "order_id": order_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Order cancellation failed", **error_result)
        return error_result


# ================================
# VALIDATION AND HELPER FUNCTIONS
# ================================

async def _validate_trade_parameters(
    instrument: str,
    side: str,
    units: float,
    account_balance: float,
    max_risk_per_trade: float = 0.02,  # 2% max risk
    max_position_size: float = 0.10    # 10% max position size
) -> Dict[str, Any]:
    """Validate trade parameters against risk management rules"""
    
    # Basic parameter validation
    if not instrument or not isinstance(instrument, str):
        raise RiskValidationError("Invalid instrument specified")
    
    if side not in ["buy", "sell"]:
        raise RiskValidationError("Side must be 'buy' or 'sell'")
    
    if units <= 0:
        raise RiskValidationError("Units must be positive")
    
    if account_balance <= 0:
        raise RiskValidationError("Invalid account balance")
    
    # Risk validation
    position_value = abs(units)  # Simplified - in real implementation, multiply by price
    max_allowed_position = account_balance * max_position_size
    
    if position_value > max_allowed_position:
        raise RiskValidationError(
            f"Position size {position_value} exceeds maximum allowed {max_allowed_position}"
        )
    
    return {
        "validated": True,
        "position_value": position_value,
        "max_allowed": max_allowed_position,
        "risk_ratio": position_value / account_balance
    }


async def _log_trade_to_database(
    instrument: str,
    side: str,
    units: float,
    entry_price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    order_id: Optional[str] = None,
    agent_decision: Optional[Dict[str, Any]] = None
) -> bool:
    """Log trade execution to database"""
    try:
        async with db_manager.get_async_session() as session:
            trade = Trade(
                instrument=instrument,
                side=TradeSide.BUY if side.lower() == "buy" else TradeSide.SELL,
                units=abs(units),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=TradeStatus.OPEN,
                broker_trade_id=order_id,
                agent_decisions=agent_decision or {},
                market_context={
                    "execution_time": datetime.now(timezone.utc).isoformat(),
                    "execution_type": "agent_decision"
                }
            )
            
            session.add(trade)
            await session.commit()
            
            logger.info(
                "Trade logged to database successfully",
                instrument=instrument,
                side=side,
                units=units,
                order_id=order_id
            )
            
            return True
            
    except Exception as e:
        logger.error(
            "Failed to log trade to database",
            error=str(e),
            instrument=instrument,
            side=side,
            units=units
        )
        return False


# ================================
# SYNC WRAPPERS FOR CREWAI TOOLS
# ================================

@tool
def get_portfolio_status() -> Dict[str, Any]:
    """Get comprehensive portfolio status (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(get_portfolio_status_async)
        if isinstance(result, dict) and result.get("success"):
            logger.info("✅ Portfolio status retrieved")
        return result
    except Exception as e:
        error_msg = f"Portfolio status framework error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()}


@tool
def get_open_positions() -> Dict[str, Any]:
    """Get all open positions with current P&L (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(get_open_positions_async)
        if isinstance(result, dict) and result.get("success"):
            logger.info("✅ Open positions retrieved")
        return result
    except Exception as e:
        error_msg = f"Position query framework error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


@tool
def get_pending_orders() -> Dict[str, Any]:
    """Get all pending orders (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(get_pending_orders_async)
        if isinstance(result, dict) and result.get("success"):
            logger.info("✅ Pending orders retrieved")
        return result
    except Exception as e:
        error_msg = f"Orders query framework error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


@tool
def execute_market_trade(
    instrument: str,
    side: str,
    units: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    reason: str = "Wyckoff signal",
    max_slippage: float = 0.001
) -> Dict[str, Any]:
    """Execute market order with comprehensive risk management and logging (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(
            execute_market_trade_async, 
            instrument, side, units, stop_loss, take_profit, reason, max_slippage
        )
        
        if result.get("success"):
            logger.info(f"✅ Market trade executed", 
                       instrument=instrument, 
                       side=side, 
                       units=units,
                       reference=result.get("trade_reference"))
        else:
            logger.error(f"❌ Market trade failed", result=result)
        
        return result
    except Exception as e:
        error_msg = f"Execution framework error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "trade_reference": _generate_trade_reference(),
            "error": error_msg,
            "instrument": instrument,
            "side": side,
            "units": units
        }


@tool
def execute_limit_trade(
    instrument: str,
    side: str,
    units: float,
    price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    expiry_time: Optional[str] = "GTC",
    reason: str = "Wyckoff limit order"
) -> Dict[str, Any]:
    """Execute limit order with risk management and expiry control (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(
            execute_limit_trade_async,
            instrument, side, units, price, stop_loss, take_profit, expiry_time, reason
        )
        
        if result.get("success"):
            logger.info(f"✅ Limit order placed", 
                       instrument=instrument, 
                       side=side, 
                       units=units,
                       price=price,
                       reference=result.get("trade_reference"))
        else:
            logger.error(f"❌ Limit order failed", result=result)
        
        return result
    except Exception as e:
        error_msg = f"Limit order framework error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "trade_reference": _generate_trade_reference(),
            "error": error_msg,
            "instrument": instrument,
            "side": side,
            "units": units,
            "price": price
        }


@tool
def close_position(
    instrument: str,
    units: Optional[float] = None,
    reason: str = "Manual close"
) -> Dict[str, Any]:
    """Close an existing position (full or partial) (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(close_position_async, instrument, units, reason)
        
        if result.get("success"):
            logger.info(f"✅ Position closed", 
                       instrument=instrument, 
                       units=units,
                       reference=result.get("close_reference"))
        else:
            logger.error(f"❌ Position closure failed", result=result)
        
        return result
    except Exception as e:
        error_msg = f"Position closure framework error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "close_reference": f"CLOSE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
            "error": error_msg,
            "instrument": instrument,
            "units": units
        }


@tool
def cancel_pending_order(
    order_id: str,
    reason: str = "Order cancellation"
) -> Dict[str, Any]:
    """Cancel a pending order by ID (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(cancel_pending_order_async, order_id, reason)
        
        if result.get("success"):
            logger.info(f"✅ Order cancelled", 
                       order_id=order_id,
                       reference=result.get("cancel_reference"))
        else:
            logger.error(f"❌ Order cancellation failed", result=result)
        
        return result
    except Exception as e:
        error_msg = f"Order cancellation framework error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "cancel_reference": f"CANCEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
            "error": error_msg,
            "order_id": order_id
        }


# Additional convenience functions for backward compatibility
@tool
def get_account_info() -> Dict[str, Any]:
    """Get account information (SYNC VERSION FOR CREWAI)"""
    try:
        async def _get_account():
            async with OandaDirectAPI() as oanda:
                return await oanda.get_account_info()
        
        result = async_runner.run_async(_get_account)
        if isinstance(result, dict) and result.get("success"):
            logger.info("✅ Account info retrieved")
        return result
    except Exception as e:
        error_msg = f"Account info framework error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


@tool
def calculate_position_size(
    account_balance: float,
    risk_per_trade: float = 0.02,
    stop_distance_pips: float = 20,
    pip_value: float = 1.0
) -> Dict[str, Any]:
    """Calculate optimal position size based on risk management rules"""
    try:
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / (stop_distance_pips * pip_value)
        
        result = {
            "success": True,
            "position_size": round(position_size, 0),
            "risk_amount": round(risk_amount, 2),
            "risk_per_trade_pct": risk_per_trade * 100,
            "stop_distance_pips": stop_distance_pips,
            "pip_value": pip_value,
            "account_balance": account_balance,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(
            "Position size calculated",
            position_size=result["position_size"],
            risk_amount=result["risk_amount"]
        )
        
        return result
    except Exception as e:
        error_msg = f"Position size calculation error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
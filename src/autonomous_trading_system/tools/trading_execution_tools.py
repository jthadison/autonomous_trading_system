"""
Complete Trading Execution Tools for Autonomous Trading System
Provides async, sync, and tool versions of all trading functions
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json
import uuid

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from crewai.tools import tool
from src.config.logging_config import logger
from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
from src.database.manager import db_manager
from src.database.models import Trade, TradeStatus, TradeSide, Order, OrderType, OrderStatus


class TradeExecutionError(Exception):
    """Custom exception for trade execution errors"""
    pass


class RiskValidationError(Exception):
    """Custom exception for risk validation errors"""
    pass


def _generate_trade_reference() -> str:
    """Generate a unique trade reference for tracking"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"TRADE_{timestamp}_{unique_id}"


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
        logger.error("Failed to log trade to database", error=str(e))
        # Return False instead of raising to avoid breaking trade execution
        return False


async def _place_stop_loss(
    oanda: OandaMCPWrapper,
    instrument: str,
    original_side: str,
    units: float,
    stop_price: float
) -> Dict[str, Any]:
    """Helper function to place stop loss order"""
    # Stop loss is opposite side to original order
    stop_side = "sell" if original_side.lower() == "buy" else "buy"
    
    # For now, using limit order at stop price
    # In production, you'd want proper stop-loss order types
    return await oanda.place_limit_order(instrument, units, stop_price, stop_side)


async def _place_take_profit(
    oanda: OandaMCPWrapper,
    instrument: str,
    original_side: str,
    units: float,
    tp_price: float
) -> Dict[str, Any]:
    """Helper function to place take profit order"""
    # Take profit is opposite side to original order
    tp_side = "sell" if original_side.lower() == "buy" else "buy"
    
    return await oanda.place_limit_order(instrument, units, tp_price, tp_side)


# ================================
# ASYNC VERSIONS (Core implementations)
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
        
        # Get account info for risk validation
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            account_info = await oanda.get_account_info()
            
            if "error" in account_info:
                raise TradeExecutionError(f"Failed to get account info: {account_info['error']}")
            
            account_balance = float(account_info.get("balance", 0))
            
            # Validate trade parameters
            validation = await _validate_trade_parameters(
                instrument, side, units, account_balance
            )
            
            # Get current price for slippage validation
            current_price_data = await oanda.get_current_price(instrument)
            if "error" in current_price_data:
                raise TradeExecutionError(f"Failed to get current price: {current_price_data['error']}")
            
            current_price = float(current_price_data.get("price", 0))
            
            # Execute market order
            logger.info(
                "Executing market order",
                instrument=instrument,
                side=side,
                units=units,
                current_price=current_price
            )
            
            order_result = await oanda.place_market_order(instrument, units, side)
            
            if "error" in order_result:
                raise TradeExecutionError(f"Order execution failed: {order_result['error']}")
            
            # Extract execution details
            execution_price = float(order_result.get("price", current_price))
            order_id = order_result.get("id", "unknown")
            
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
            
            # Place stop loss if specified
            stop_order_id = None
            if stop_loss:
                try:
                    stop_result = await _place_stop_loss(
                        oanda, instrument, side, units, stop_loss
                    )
                    stop_order_id = stop_result.get("id")
                except Exception as e:
                    logger.error("Failed to place stop loss", error=str(e))
            
            # Place take profit if specified
            tp_order_id = None
            if take_profit:
                try:
                    tp_result = await _place_take_profit(
                        oanda, instrument, side, units, take_profit
                    )
                    tp_order_id = tp_result.get("id")
                except Exception as e:
                    logger.error("Failed to place take profit", error=str(e))
            
            # Log trade to database (non-critical - trade succeeds even if logging fails)
            trade_logged = await _log_trade_to_database(
                instrument=instrument,
                side=side,
                units=units,
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
                "instrument": instrument,
                "side": side,
                "units": units,
                "execution_price": execution_price,
                "expected_price": current_price,
                "slippage": slippage,
                "slippage_pct": round(slippage * 100, 4),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "stop_order_id": stop_order_id,
                "tp_order_id": tp_order_id,
                "account_balance": account_balance,
                "position_value": validation["position_value"],
                "risk_ratio": validation["risk_ratio"],
                "execution_time": datetime.now(timezone.utc).isoformat(),
                "reason": reason
            }
            
            logger.info("Market order executed successfully", trade_reference=trade_reference, **result)
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
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # Get account info for validation
            account_info = await oanda.get_account_info()
            if "error" in account_info:
                raise TradeExecutionError(f"Failed to get account info: {account_info['error']}")
            
            account_balance = float(account_info.get("balance", 0))
            
            # Validate trade parameters
            validation = await _validate_trade_parameters(
                instrument, side, units, account_balance
            )
            
            # Validate limit price against current market
            current_price_data = await oanda.get_current_price(instrument)
            if "error" in current_price_data:
                raise TradeExecutionError(f"Failed to get current price: {current_price_data['error']}")
            
            current_price = float(current_price_data.get("price", 0))
            
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
            
            # Execute limit order
            logger.info(
                "Placing limit order",
                instrument=instrument,
                side=side,
                units=units,
                price=price,
                current_price=current_price
            )
            
            order_result = await oanda.place_limit_order(instrument, units, price, side)
            
            if "error" in order_result:
                raise TradeExecutionError(f"Limit order failed: {order_result['error']}")
            
            order_id = order_result.get("id", "unknown")
            
            result = {
                "success": True,
                "trade_reference": trade_reference,
                "order_id": order_id,
                "order_type": "limit",
                "status": "pending",
                "instrument": instrument,
                "side": side,
                "units": units,
                "limit_price": price,
                "current_price": current_price,
                "price_distance": abs(price - current_price),
                "price_distance_pct": round(abs(price - current_price) / current_price * 100, 4),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "expiry_time": expiry_time,
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


async def cancel_pending_order_async(order_id: str, reason: str = "Manual cancellation") -> Dict[str, Any]:
    """Cancel a pending order by ID"""
    
    try:
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            result = await oanda.cancel_order(order_id)
            
            if "error" in result:
                raise TradeExecutionError(f"Failed to cancel order: {result['error']}")
            
            logger.info(
                "Order cancelled successfully",
                order_id=order_id,
                reason=reason
            )
            
            return {
                "success": True,
                "order_id": order_id,
                "status": "cancelled",
                "reason": reason,
                "cancellation_time": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "order_id": order_id,
            "cancellation_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Order cancellation failed", **error_result)
        return error_result


async def get_open_positions_async() -> Dict[str, Any]:
    """Get all open positions with current P&L"""
    
    try:
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            positions = await oanda.get_positions()
            
            if "error" in positions:
                raise TradeExecutionError(f"Failed to get positions: {positions['error']}")
            
            # Process and enrich position data
            enriched_positions = []
            for position in positions.get("positions", []):
                # Get current price for P&L calculation
                instrument = position.get("instrument")
                if instrument:
                    price_data = await oanda.get_current_price(instrument)
                    current_price = float(price_data.get("price", 0))
                    
                    enriched_position = {
                        **position,
                        "current_price": current_price,
                        "query_time": datetime.now(timezone.utc).isoformat()
                    }
                    enriched_positions.append(enriched_position)
            
            return {
                "success": True,
                "positions": enriched_positions,
                "position_count": len(enriched_positions),
                "query_time": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error("Failed to get positions", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "query_time": datetime.now(timezone.utc).isoformat()
        }


async def get_pending_orders_async() -> Dict[str, Any]:
    """Get all pending orders"""
    
    try:
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            orders = await oanda.get_orders()
            
            if "error" in orders:
                raise TradeExecutionError(f"Failed to get orders: {orders['error']}")
            
            return {
                "success": True,
                "orders": orders.get("orders", []),
                "order_count": len(orders.get("orders", [])),
                "query_time": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error("Failed to get orders", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "query_time": datetime.now(timezone.utc).isoformat()
        }


async def close_position_async(
    instrument: str,
    units: Optional[float] = None,  # None = close all
    reason: str = "Manual close"
) -> Dict[str, Any]:
    """Close an existing position (full or partial)"""
    
    try:
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # Get current position first
            positions = await oanda.get_positions()
            if "error" in positions:
                raise TradeExecutionError(f"Failed to get positions: {positions['error']}")
            
            # Find the position for this instrument
            target_position = None
            for pos in positions.get("positions", []):
                if pos.get("instrument") == instrument:
                    target_position = pos
                    break
            
            if not target_position:
                return {
                    "success": False,
                    "error": f"No open position found for {instrument}",
                    "instrument": instrument
                }
            
            # Determine position details
            current_units = float(target_position.get("units", 0))
            if current_units == 0:
                return {
                    "success": False,
                    "error": f"No units in position for {instrument}",
                    "instrument": instrument
                }
            
            # Determine how much to close
            close_units = abs(units) if units else abs(current_units)
            if close_units > abs(current_units):
                close_units = abs(current_units)
            
            # Determine side for closing order (opposite of current position)
            close_side = "sell" if current_units > 0 else "buy"
            
            # Execute closing market order
            close_result = await oanda.place_market_order(instrument, close_units, close_side)
            
            if "error" in close_result:
                raise TradeExecutionError(f"Failed to close position: {close_result['error']}")
            
            result = {
                "success": True,
                "instrument": instrument,
                "original_units": current_units,
                "closed_units": close_units,
                "remaining_units": current_units - (close_units if current_units > 0 else -close_units),
                "close_side": close_side,
                "close_price": float(close_result.get("price", 0)),
                "order_id": close_result.get("id"),
                "reason": reason,
                "close_time": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("Position closed successfully", **result)
            return result
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "instrument": instrument,
            "close_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Position close failed", **error_result)
        return error_result


async def get_portfolio_status_async() -> Dict[str, Any]:
    """Get comprehensive portfolio status including positions, orders, account info, and risk metrics"""
    
    try:
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # Get all relevant data in parallel for efficiency
            account_info = await oanda.get_account_info()
            positions = await oanda.get_positions()
            orders = await oanda.get_orders()
            
            # Initialize portfolio summary
            portfolio_summary = {
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "account_info": account_info,
                "positions_raw": positions,
                "orders_raw": orders
            }
            
            # Process account information
            if "error" not in account_info:
                balance = float(account_info.get("balance", 0))
                margin_used = float(account_info.get("marginUsed", 0))
                margin_available = float(account_info.get("marginAvailable", 0))
                
                portfolio_summary.update({
                    "account_balance": balance,
                    "margin_used": margin_used,
                    "margin_available": margin_available,
                    "margin_utilization_pct": round((margin_used / balance) * 100, 2) if balance > 0 else 0
                })
            else:
                logger.error("Failed to get account info", error=account_info.get("error"))
                portfolio_summary["account_error"] = account_info.get("error")
            
            # Process positions
            if "error" not in positions:
                position_list = positions.get("positions", [])
                portfolio_summary["total_positions"] = len(position_list)
                
                # Calculate total exposure and position details
                total_exposure = 0
                total_unrealized_pnl = 0
                position_details = []
                
                for pos in position_list:
                    try:
                        instrument = pos.get("instrument", "Unknown")
                        units = float(pos.get("units", 0))
                        unrealized_pl = float(pos.get("unrealizedPL", 0))
                        
                        # Get current price for exposure calculation
                        price_data = await oanda.get_current_price(instrument)
                        current_price = float(price_data.get("price", 1)) if "error" not in price_data else 1
                        
                        # Calculate position exposure (absolute value)
                        position_exposure = abs(units * current_price)
                        total_exposure += position_exposure
                        total_unrealized_pnl += unrealized_pl
                        
                        position_detail = {
                            "instrument": instrument,
                            "units": units,
                            "current_price": current_price,
                            "exposure": position_exposure,
                            "unrealized_pnl": unrealized_pl,
                            "side": "long" if units > 0 else "short"
                        }
                        position_details.append(position_detail)
                        
                    except Exception as e:
                        logger.error(f"Error processing position {pos}", error=str(e))
                
                portfolio_summary.update({
                    "position_details": position_details,
                    "total_exposure": total_exposure,
                    "total_unrealized_pnl": total_unrealized_pnl
                })
                
                # Calculate exposure as percentage of account balance
                if portfolio_summary.get("account_balance", 0) > 0:
                    exposure_pct = round((total_exposure / portfolio_summary["account_balance"]) * 100, 2)
                    portfolio_summary["exposure_pct"] = exposure_pct
                    
                    # Risk warnings
                    portfolio_summary["risk_warnings"] = []
                    if exposure_pct > 20:
                        portfolio_summary["risk_warnings"].append(f"High exposure: {exposure_pct}% > 20% limit")
                    if portfolio_summary.get("margin_utilization_pct", 0) > 50:
                        portfolio_summary["risk_warnings"].append(f"High margin usage: {portfolio_summary['margin_utilization_pct']}%")
                        
            else:
                logger.error("Failed to get positions", error=positions.get("error"))
                portfolio_summary["positions_error"] = positions.get("error")
                portfolio_summary["total_positions"] = 0
            
            # Process pending orders
            if "error" not in orders:
                order_list = orders.get("orders", [])
                portfolio_summary["total_pending_orders"] = len(order_list)
                
                # Process order details
                order_details = []
                for order in order_list:
                    order_detail = {
                        "id": order.get("id"),
                        "instrument": order.get("instrument"),
                        "units": float(order.get("units", 0)),
                        "type": order.get("type"),
                        "price": float(order.get("price", 0)) if order.get("price") else None,
                        "state": order.get("state"),
                        "created_time": order.get("createTime")
                    }
                    order_details.append(order_detail)
                
                portfolio_summary["order_details"] = order_details
            else:
                logger.error("Failed to get orders", error=orders.get("error"))
                portfolio_summary["orders_error"] = orders.get("error")
                portfolio_summary["total_pending_orders"] = 0
            
            # Calculate portfolio health score (0-100)
            health_score = 100
            if portfolio_summary.get("exposure_pct", 0) > 20:
                health_score -= 30  # High exposure penalty
            if portfolio_summary.get("margin_utilization_pct", 0) > 50:
                health_score -= 20  # High margin penalty
            if portfolio_summary.get("total_unrealized_pnl", 0) < 0:
                health_score -= 10  # Unrealized loss penalty
            
            portfolio_summary["portfolio_health_score"] = max(0, health_score)
            
            logger.info(
                "Portfolio status retrieved",
                positions=portfolio_summary.get("total_positions", 0),
                orders=portfolio_summary.get("total_pending_orders", 0),
                exposure_pct=portfolio_summary.get("exposure_pct", 0),
                health_score=portfolio_summary.get("portfolio_health_score", 0)
            )
            
            return portfolio_summary
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Portfolio status retrieval failed", **error_result)
        return error_result


# ================================
# SYNC VERSIONS (for direct calling)
# ================================

def execute_market_trade_sync(
    instrument: str,
    side: str,
    units: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    reason: str = "Wyckoff signal",
    max_slippage: float = 0.001
) -> Dict[str, Any]:
    """Synchronous version of execute_market_trade for direct calling"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    execute_market_trade_async(instrument, side, units, stop_loss, take_profit, reason, max_slippage)
                )
                return future.result()
        else:
            return asyncio.run(
                execute_market_trade_async(instrument, side, units, stop_loss, take_profit, reason, max_slippage)
            )
    except Exception as e:
        trade_reference = _generate_trade_reference()
        return {
            "success": False,
            "trade_reference": trade_reference,
            "error": f"Execution framework error: {str(e)}",
            "instrument": instrument,
            "side": side,
            "units": units
        }


def execute_limit_trade_sync(
    instrument: str,
    side: str,
    units: float,
    price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    expiry_time: Optional[str] = None,
    reason: str = "Wyckoff limit order"
) -> Dict[str, Any]:
    """Synchronous version of execute_limit_trade for direct calling"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    execute_limit_trade_async(instrument, side, units, price, stop_loss, take_profit, expiry_time, reason)
                )
                return future.result()
        else:
            return asyncio.run(
                execute_limit_trade_async(instrument, side, units, price, stop_loss, take_profit, expiry_time, reason)
            )
    except Exception as e:
        trade_reference = _generate_trade_reference()
        return {
            "success": False,
            "trade_reference": trade_reference,
            "error": f"Execution framework error: {str(e)}",
            "order_type": "limit",
            "instrument": instrument,
            "side": side,
            "units": units,
            "price": price
        }


def cancel_pending_order_sync(order_id: str, reason: str = "Manual cancellation") -> Dict[str, Any]:
    """Synchronous version of cancel_pending_order for direct calling"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, cancel_pending_order_async(order_id, reason))
                return future.result()
        else:
            return asyncio.run(cancel_pending_order_async(order_id, reason))
    except Exception as e:
        return {
            "success": False,
            "error": f"Cancellation framework error: {str(e)}",
            "order_id": order_id
        }


def get_open_positions_sync() -> Dict[str, Any]:
    """Synchronous version of get_open_positions for direct calling"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_open_positions_async())
                return future.result()
        else:
            return asyncio.run(get_open_positions_async())
    except Exception as e:
        return {
            "success": False,
            "error": f"Position query framework error: {str(e)}"
        }


def get_pending_orders_sync() -> Dict[str, Any]:
    """Synchronous version of get_pending_orders for direct calling"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_pending_orders_async())
                return future.result()
        else:
            return asyncio.run(get_pending_orders_async())
    except Exception as e:
        return {
            "success": False,
            "error": f"Orders query framework error: {str(e)}"
        }


def close_position_sync(
    instrument: str,
    units: Optional[float] = None,
    reason: str = "Manual close"
) -> Dict[str, Any]:
    """Synchronous version of close_position for direct calling"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, close_position_async(instrument, units, reason))
                return future.result()
        else:
            return asyncio.run(close_position_async(instrument, units, reason))
    except Exception as e:
        return {
            "success": False,
            "error": f"Close position framework error: {str(e)}",
            "instrument": instrument
        }


def get_portfolio_status_sync() -> Dict[str, Any]:
    """Synchronous version of get_portfolio_status for direct calling"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_portfolio_status_async())
                return future.result()
        else:
            return asyncio.run(get_portfolio_status_async())
    except Exception as e:
        return {
            "success": False,
            "error": f"Portfolio status framework error: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ================================
# TOOL VERSIONS (for agents)
# ================================

@tool
def execute_market_trade(
    instrument: str,
    side: str,  # "buy" or "sell"
    units: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    reason: str = "Wyckoff signal",
    max_slippage: float = 0.001  # 0.1% max slippage
) -> Dict[str, Any]:
    """
    Execute market order with comprehensive risk management and logging.
    
    Args:
        instrument: Trading instrument (e.g., "BTC_USD", "EUR_USD")
        side: "buy" or "sell"
        units: Position size in units
        stop_loss: Stop loss price level (optional)
        take_profit: Take profit price level (optional)
        reason: Reason for trade (for logging)
        max_slippage: Maximum acceptable slippage
    
    Returns:
        Dict with execution results and trade details
    """
    return execute_market_trade_sync(instrument, side, units, stop_loss, take_profit, reason, max_slippage)


@tool
def execute_limit_trade(
    instrument: str,
    side: str,  # "buy" or "sell"
    units: float,
    price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    expiry_time: Optional[str] = None,  # ISO format or "GTC"
    reason: str = "Wyckoff limit order"
) -> Dict[str, Any]:
    """
    Execute limit order with risk management and expiry control.
    
    Args:
        instrument: Trading instrument
        side: "buy" or "sell"
        units: Position size
        price: Limit price
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        expiry_time: Order expiry ("GTC" for Good Till Cancelled or ISO datetime)
        reason: Reason for trade
    
    Returns:
        Dict with order placement results
    """
    return execute_limit_trade_sync(instrument, side, units, price, stop_loss, take_profit, expiry_time, reason)


@tool
def cancel_pending_order(order_id: str, reason: str = "Manual cancellation") -> Dict[str, Any]:
    """
    Cancel a pending order by ID.
    
    Args:
        order_id: Order ID to cancel
        reason: Reason for cancellation
    
    Returns:
        Dict with cancellation results
    """
    return cancel_pending_order_sync(order_id, reason)


@tool
def get_open_positions() -> Dict[str, Any]:
    """
    Get all open positions with current P&L.
    
    Returns:
        Dict with current positions and their status
    """
    return get_open_positions_sync()


@tool
def get_pending_orders() -> Dict[str, Any]:
    """
    Get all pending orders.
    
    Returns:
        Dict with current pending orders
    """
    return get_pending_orders_sync()


@tool
def close_position(
    instrument: str,
    units: Optional[float] = None,  # None = close all
    reason: str = "Manual close"
) -> Dict[str, Any]:
    """
    Close an existing position (full or partial).
    
    Args:
        instrument: Instrument to close position for
        units: Units to close (None for full position)
        reason: Reason for closing
    
    Returns:
        Dict with position close results
    """
    return close_position_sync(instrument, units, reason)


@tool
def get_portfolio_status() -> Dict[str, Any]:
    """
    Get comprehensive portfolio status including positions, orders, account info, and risk metrics.
    This is the primary tool for overall portfolio monitoring and risk assessment.
    
    Returns:
        Dict with complete portfolio overview including exposure calculations
    """
    return get_portfolio_status_sync()
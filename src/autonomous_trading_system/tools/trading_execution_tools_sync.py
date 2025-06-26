"""
Enhanced Trading Execution Tools for CrewAI Autonomous Trading System
Combines comprehensive business logic with sync/async compatibility fixes
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
from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
from src.database.manager import db_manager
from src.database.models import Trade, TradeStatus, TradeSide, Order, OrderType, OrderStatus, AgentAction, LogLevel


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
        
        # Get account info for risk validation
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            account_info = await oanda.get_account_info()
            
            if "error" in str(account_info):
                raise TradeExecutionError(f"Failed to get account info: {account_info}")
            
            account_balance = float(account_info.get("balance", 0))
            
            # Validate trade parameters
            validation = await _validate_trade_parameters(
                instrument, side, units, account_balance
            )
            
            # Get current price for slippage validation
            current_price_data = await oanda.get_current_price(instrument)
            if "error" in str(current_price_data):
                raise TradeExecutionError(f"Failed to get current price: {current_price_data}")
            
            current_price = float(current_price_data.get("bid" if side == "sell" else "ask", 0))
            
            # Execute market order (simulated for now)
            logger.info(
                "Executing market order",
                instrument=instrument,
                side=side,
                units=units,
                current_price=current_price
            )
            
            # TODO: Replace with actual order execution when MCP server supports it
            order_result = {
                "success": True,
                "id": f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
                "price": current_price,
                "status": "FILLED"
            }
            
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
                "account_balance": account_balance,
                "position_value": validation["position_value"],
                "risk_ratio": validation["risk_ratio"],
                "execution_time": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "status": "SIMULATED"  # Will be "FILLED" when live
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
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # Get account info for validation
            account_info = await oanda.get_account_info()
            if "error" in str(account_info):
                raise TradeExecutionError(f"Failed to get account info: {account_info}")
            
            account_balance = float(account_info.get("balance", 0))
            
            # Validate trade parameters
            validation = await _validate_trade_parameters(
                instrument, side, units, account_balance
            )
            
            # Validate limit price against current market
            current_price_data = await oanda.get_current_price(instrument)
            if "error" in str(current_price_data):
                raise TradeExecutionError(f"Failed to get current price: {current_price_data}")
            
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
            
            # Execute limit order (simulated for now)
            logger.info(
                "Placing limit order",
                instrument=instrument,
                side=side,
                units=units,
                price=price,
                current_price=current_price
            )
            
            # TODO: Replace with actual limit order when MCP server supports it
            order_result = {
                "success": True,
                "id": f"LIMIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
                "status": "PENDING"
            }
            
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
    """Get comprehensive portfolio status including positions, orders, account info, and risk metrics"""
    
    try:
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # Get account information
            account_info = await oanda.get_account_info()
            
            # Initialize portfolio summary
            portfolio_summary = {
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "account_info": account_info
            }
            
            # Process account information
            if "error" not in str(account_info):
                balance = float(account_info.get("balance", 0))
                margin_used = float(account_info.get("margin_used", 0))
                margin_available = float(account_info.get("margin_available", 0))
                
                portfolio_summary.update({
                    "account_balance": balance,
                    "margin_used": margin_used,
                    "margin_available": margin_available,
                    "margin_utilization_pct": round((margin_used / balance) * 100, 2) if balance > 0 else 0
                })
            else:
                logger.error("Failed to get account info", error=str(account_info))
                portfolio_summary["account_error"] = str(account_info)
            
            # TODO: Add positions and orders when MCP server supports them
            # For now, simulate empty portfolio
            portfolio_summary.update({
                "total_positions": 0,
                "position_details": [],
                "total_exposure": 0.0,
                "total_unrealized_pnl": 0.0,
                "exposure_pct": 0.0,
                "total_pending_orders": 0,
                "order_details": [],
                "risk_warnings": [],
                "portfolio_health_score": 100
            })
            
            logger.info(
                "Portfolio status retrieved",
                positions=0,
                orders=0,
                balance=portfolio_summary.get("account_balance", 0)
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


async def get_open_positions_async() -> Dict[str, Any]:
    """Get all open positions with current P&L"""
    
    try:
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # TODO: Implement when positions endpoint is available
            # For now, return empty positions
            return {
                "success": True,
                "positions": [],
                "position_count": 0,
                "total_unrealized_pnl": 0.0,
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
            # TODO: Implement when orders endpoint is available
            # For now, return empty orders
            return {
                "success": True,
                "orders": [],
                "order_count": 0,
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
        close_reference = f"CLOSE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # TODO: Implement actual position closure when available
            # For now, simulate the closure
            
            closure_result = {
                "success": True,
                "close_reference": close_reference,
                "instrument": instrument,
                "units_closed": units or "ALL",
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "SIMULATED"
            }
            
            logger.info("Position closure simulated", **closure_result)
            return closure_result
            
    except Exception as e:
        error_result = {
            "success": False,
            "close_reference": f"CLOSE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
            "error": str(e),
            "instrument": instrument,
            "units": units,
            "close_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Position close failed", **error_result)
        return error_result


async def cancel_pending_order_async(order_id: str, reason: str = "Manual cancellation") -> Dict[str, Any]:
    """Cancel a pending order by ID"""
    
    try:
        cancel_reference = f"CANCEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            # TODO: Implement actual order cancellation when available
            # For now, simulate the cancellation
            
            cancellation_result = {
                "success": True,
                "cancel_reference": cancel_reference,
                "order_id": order_id,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "CANCELLED"
            }
            
            logger.info("Order cancellation simulated", **cancellation_result)
            return cancellation_result
            
    except Exception as e:
        error_result = {
            "success": False,
            "cancel_reference": f"CANCEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
            "error": str(e),
            "order_id": order_id,
            "cancellation_time": datetime.now(timezone.utc).isoformat()
        }
        logger.error("Order cancellation failed", **error_result)
        return error_result


# ================================
# MARKET DATA AND ANALYSIS TOOLS
# ================================

async def get_live_price_async(instrument: str) -> Dict[str, Any]:
    """Get live price for a forex instrument"""
    async def _get_price():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            return await oanda.get_current_price(instrument)
    
    try:
        result = await _get_price()
        logger.info(f"✅ Live price retrieved for {instrument}")
        return result
    except Exception as e:
        error_msg = f"Failed to get live price for {instrument}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


async def get_historical_data_async(instrument: str, timeframe: str = "M15", count: int = 200) -> Dict[str, Any]:
    """Get historical price data for Wyckoff analysis"""
    async def _get_historical():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            return await oanda.get_historical_data(instrument, timeframe, count)
    
    try:
        result = await _get_historical()
        logger.info(f"✅ Historical data retrieved for {instrument}")
        return result
    except Exception as e:
        error_msg = f"Failed to get historical data for {instrument}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


async def get_account_info_async() -> Dict[str, Any]:
    """Get current account information"""
    async def _get_account():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            return await oanda.get_account_info()
    
    try:
        result = await _get_account()
        logger.info("✅ Account info retrieved")
        return result
    except Exception as e:
        error_msg = f"Failed to get account info: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


# ================================
# SYNC WRAPPER TOOLS (CREWAI COMPATIBLE)
# ================================

@tool
def get_live_price(instrument: str) -> Dict[str, Any]:
    """Get live price for a forex instrument (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(get_live_price_async, instrument)
        if isinstance(result, dict) and "error" not in result:
            logger.info(f"✅ Live price retrieved for {instrument}")
        return result
    except Exception as e:
        error_msg = f"Failed to get live price for {instrument}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool  
def get_historical_data(instrument: str, timeframe: str = "M15", count: int = 200) -> Dict[str, Any]:
    """Get historical price data for Wyckoff analysis (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(get_historical_data_async, instrument, timeframe, count)
        if isinstance(result, dict) and "error" not in result:
            logger.info(f"✅ Historical data retrieved for {instrument}")
        return result
    except Exception as e:
        error_msg = f"Failed to get historical data for {instrument}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def get_account_info() -> Dict[str, Any]:
    """Get current account information (SYNC VERSION FOR CREWAI)"""
    try:
        result = async_runner.run_async(get_account_info_async)
        if isinstance(result, dict) and "error" not in result:
            logger.info("✅ Account info retrieved")
        return result
    except Exception as e:
        error_msg = f"Failed to get account info: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def get_portfolio_status() -> Dict[str, Any]:
    """Get comprehensive portfolio status including positions, orders, account info, and risk metrics (SYNC VERSION FOR CREWAI)"""
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
                       reference=result.get("order_reference"))
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


@tool
def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    stop_distance_pips: float,
    pip_value: float
) -> Dict[str, Any]:
    """Calculate position size based on Wyckoff levels and risk management (SYNC VERSION FOR CREWAI)"""
    try:
        # Convert risk_per_trade to percentage if it's in dollar amount
        if risk_per_trade > 1.0:
            risk_per_trade_pct = (risk_per_trade / account_balance) * 100
            risk_amount = risk_per_trade
        else:
            risk_per_trade_pct = risk_per_trade * 100
            risk_amount = account_balance * risk_per_trade
        
        # Calculate position size
        position_size = risk_amount / (stop_distance_pips * pip_value)
        
        # Apply maximum position size limit (10% of account)
        max_position_size = account_balance * 0.1
        
        result = {
            "account_balance": account_balance,
            "risk_per_trade_pct": risk_per_trade_pct,
            "risk_amount": risk_amount,
            "stop_distance_pips": stop_distance_pips,
            "position_size": position_size,
            "pip_value": pip_value,
            "max_position_size": max_position_size
        }
        
        logger.info("✅ Position size calculated", 
                   risk_pct=risk_per_trade_pct, 
                   position_size=position_size)
        
        return result
        
    except Exception as e:
        error_msg = f"Position size calculation failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


# ================================
# WYCKOFF ANALYSIS INTEGRATION (COMPLETELY REFACTORED)
# ================================

@tool
def analyze_wyckoff_patterns(instrument: str, timeframe: str = "M15") -> Dict[str, Any]:
    """Perform comprehensive Wyckoff pattern analysis (SYNC VERSION FOR CREWAI)"""
    
    try:
        logger.info(f"🧠 Starting Wyckoff analysis for {instrument} on {timeframe}")
        
        # Step 1: Get historical data using the sync wrapper (no async calls here)
        historical_result = get_historical_data(instrument, timeframe, 200)
        
        if "error" in historical_result:
            error_msg = f"Failed to get historical data: {historical_result['error']}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Step 2: Extract and validate price data structure
        data_section = historical_result.get('data', {})
        if not data_section:
            return {"error": "No data section in historical result"}
        
        # Handle the nested data structure from your MCP server
        candles = data_section.get('candles', [])
        if not candles:
            return {"error": "No candles data available for analysis"}
        
        logger.info(f"📊 Processing {len(candles)} candles for Wyckoff analysis")
        
        # Step 3: Validate candle data structure
        if not isinstance(candles, list) or len(candles) < 20:
            return {"error": f"Insufficient data for Wyckoff analysis: {len(candles)} candles (minimum 20 required)"}
        
        # Step 4: Prepare data for Wyckoff analyzer
        processed_candles = []
        for i, candle in enumerate(candles):
            try:
                # Extract OHLC data from candle structure
                mid_data = candle.get('mid', {})
                processed_candle = {
                    'time': candle.get('time', ''),
                    'open': float(mid_data.get('o', 0)),
                    'high': float(mid_data.get('h', 0)),
                    'low': float(mid_data.get('l', 0)),
                    'close': float(mid_data.get('c', 0)),
                    'volume': int(candle.get('volume', 0)),
                    'complete': candle.get('complete', True)
                }
                processed_candles.append(processed_candle)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid candle at index {i}: {e}")
                continue
        
        if len(processed_candles) < 20:
            return {"error": f"Too few valid candles after processing: {len(processed_candles)}"}
        
        # Step 5: Import Wyckoff analyzer with error handling
        try:
            from src.autonomous_trading_system.utils.wyckoff_pattern_analyzer import wyckoff_analyzer
        except ImportError as e:
            error_msg = f"Failed to import Wyckoff analyzer: {str(e)}"
            logger.error(error_msg)
            # Return manual fallback analysis instead of failing completely
            return _manual_wyckoff_analysis_fallback(processed_candles, instrument, timeframe)
        
        # Step 6: Define async analysis function for the thread-safe runner
        async def _run_wyckoff_analysis():
            try:
                # Call the wyckoff analyzer with processed data
                return await wyckoff_analyzer.analyze_market_data(processed_candles, timeframe)
            except Exception as e:
                logger.error(f"Wyckoff analyzer internal error: {str(e)}")
                return {"error": f"Wyckoff analyzer failed: {str(e)}"}
        
        # Step 7: Run analysis using ThreadSafeAsyncRunner
        logger.info(f"🔍 Running Wyckoff analysis using ThreadSafeAsyncRunner...")
        analysis_result = async_runner.run_async(_run_wyckoff_analysis)
        
        # Step 8: Validate and return results
        if not isinstance(analysis_result, dict):
            error_msg = f"Invalid analysis result type: {type(analysis_result)}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        if "error" in analysis_result:
            logger.warning(f"Main Wyckoff analysis failed, trying manual fallback: {analysis_result.get('error')}")
            # Try manual fallback instead of failing
            return _manual_wyckoff_analysis_fallback(processed_candles, instrument, timeframe)
        
        # Step 9: Add metadata to successful results
        analysis_result.update({
            "instrument": instrument,
            "timeframe": timeframe,
            "candles_analyzed": len(processed_candles),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_success": True
        })
        
        logger.info(f"✅ Wyckoff analysis completed successfully for {instrument}")
        return analysis_result
        
    except Exception as e:
        error_msg = f"Critical error in Wyckoff analysis for {instrument}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Try manual fallback as last resort
        try:
            historical_result = get_historical_data(instrument, timeframe, 100)
            if "error" not in historical_result:
                candles = historical_result.get('data', {}).get('candles', [])
                if candles:
                    processed_candles = []
                    for candle in candles:
                        mid_data = candle.get('mid', {})
                        processed_candles.append({
                            'time': candle.get('time', ''),
                            'open': float(mid_data.get('o', 0)),
                            'high': float(mid_data.get('h', 0)),
                            'low': float(mid_data.get('l', 0)),
                            'close': float(mid_data.get('c', 0)),
                            'volume': int(candle.get('volume', 0))
                        })
                    
                    fallback_result = _manual_wyckoff_analysis_fallback(processed_candles, instrument, timeframe)
                    logger.info(f"✅ Manual fallback analysis completed for {instrument}")
                    return fallback_result
        except Exception as fallback_error:
            logger.error(f"Even fallback analysis failed: {str(fallback_error)}")
        
        return {
            "error": error_msg,
            "instrument": instrument,
            "timeframe": timeframe,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_success": False
        }


def _manual_wyckoff_analysis_fallback(candles: List[Dict], instrument: str, timeframe: str) -> Dict[str, Any]:
    """Fallback manual Wyckoff analysis when the main analyzer fails"""
    
    try:
        logger.info(f"🔄 Running fallback Wyckoff analysis for {instrument}")
        
        if len(candles) < 20:
            return {"error": "Insufficient data for manual analysis"}
        
        # Extract price arrays
        highs = [candle['high'] for candle in candles if candle.get('high', 0) > 0]
        lows = [candle['low'] for candle in candles if candle.get('low', 0) > 0]
        closes = [candle['close'] for candle in candles if candle.get('close', 0) > 0]
        volumes = [candle.get('volume', 1) for candle in candles]  # Default to 1 if no volume
        
        if len(closes) < 20:
            return {"error": "Insufficient valid price data for analysis"}
        
        # Calculate basic metrics
        current_price = closes[-1]
        price_range = max(highs) - min(lows)
        avg_volume = sum(volumes) / len(volumes) if volumes else 1
        recent_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else avg_volume
        
        # Simple trend analysis
        short_ma = sum(closes[-5:]) / 5 if len(closes) >= 5 else current_price
        long_ma = sum(closes[-20:]) / 20 if len(closes) >= 20 else current_price
        trend_direction = "bullish" if short_ma > long_ma else "bearish"
        
        # Basic accumulation/distribution detection
        volume_trend = "increasing" if recent_volume > avg_volume else "decreasing"
        
        # Simple phase identification
        if trend_direction == "bullish" and volume_trend == "increasing":
            wyckoff_phase = "phase_c_or_d"
            phase_confidence = 65
        elif trend_direction == "bearish" and volume_trend == "increasing":
            wyckoff_phase = "phase_a_or_b"
            phase_confidence = 60
        else:
            wyckoff_phase = "phase_b"
            phase_confidence = 45
        
        # Generate trading recommendations
        if phase_confidence > 50:
            if trend_direction == "bullish":
                action = "buy"
                entry_level = current_price * 0.999  # Slight pullback entry
                stop_loss = min(lows[-10:]) * 0.995 if len(lows) >= 10 else current_price * 0.98
                take_profit = current_price * 1.02   # 2% target
            else:
                action = "sell"
                entry_level = current_price * 1.001
                stop_loss = max(highs[-10:]) * 1.005 if len(highs) >= 10 else current_price * 1.02
                take_profit = current_price * 0.98
        else:
            action = "wait"
            entry_level = current_price
            stop_loss = None
            take_profit = None
        
        return {
            "structure_analysis": {
                "structure_type": "accumulation" if trend_direction == "bullish" else "distribution",
                "phase": wyckoff_phase,
                "confidence": phase_confidence,
                "key_levels": {
                    "current_price": current_price,
                    "support": min(lows[-10:]) if len(lows) >= 10 else current_price * 0.98,
                    "resistance": max(highs[-10:]) if len(highs) >= 10 else current_price * 1.02
                }
            },
            "market_regime": {
                "trend_direction": trend_direction,
                "volume_trend": volume_trend,
                "volatility": price_range / current_price if current_price > 0 else 0.01
            },
            "trading_recommendations": {
                "action": action,
                "confidence": phase_confidence,
                "entry_level": entry_level,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reasoning": [
                    f"Trend is {trend_direction}",
                    f"Volume is {volume_trend}",
                    f"Phase identified as {wyckoff_phase}",
                    f"Manual fallback analysis used"
                ]
            },
            "confidence_score": phase_confidence,
            "analysis_type": "manual_fallback",
            "instrument": instrument,
            "timeframe": timeframe,
            "candles_analyzed": len(candles),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_success": True
        }
        
    except Exception as e:
        logger.error(f"Manual fallback analysis failed: {str(e)}")
        return {
            "error": f"Manual analysis failed: {str(e)}",
            "analysis_type": "manual_fallback_failed",
            "instrument": instrument,
            "timeframe": timeframe,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_success": False
        }


# ================================
# CLEANUP AND UTILITY FUNCTIONS
# ================================

def cleanup_async_runner():
    """Cleanup function for graceful shutdown"""
    global async_runner
    if async_runner and hasattr(async_runner, '_executor'):
        async_runner._executor.shutdown(wait=True)


# Export all tools for CrewAI import
__all__ = [
    'get_live_price',
    'get_historical_data', 
    'analyze_wyckoff_patterns',
    'get_account_info',
    'get_portfolio_status',
    'get_open_positions',
    'get_pending_orders',
    'calculate_position_size',
    'execute_market_trade',
    'execute_limit_trade',
    'close_position',
    'cancel_pending_order',
    'cleanup_async_runner'
]
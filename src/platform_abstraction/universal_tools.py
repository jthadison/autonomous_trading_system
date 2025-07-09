"""
Universal Trading Tools for CrewAI Agents (Type-Fixed Version)
File: src/platform_abstraction/universal_tools.py

Updated trading tools that work with the platform abstraction layer.
These tools maintain the same interface that your CrewAI agents already use,
but now route through the universal platform system instead of directly to Oanda.

TYPE FIXES APPLIED:
- Self-contained UniversalAsyncRunner to avoid import conflicts
- Type alias for backward compatibility
- Enhanced error handling and shutdown management
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
import concurrent.futures
import threading
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from crewai.tools import tool
from src.config.logging_config import logger

# Import platform abstraction components
from .router import get_router, PlatformRouter
from .models import (
    TradeParams,
    TradeSide,
    OrderType,
    Platform,
    convert_side_to_enum,
    validate_trade_params
)


# ================================
# SELF-CONTAINED ASYNC RUNNER (TYPE-SAFE)
# ================================

class UniversalAsyncRunner:
    """
    Thread-safe async runner for universal trading tools.
    Self-contained implementation to avoid type conflicts.
    """
    
    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="universal_trading")
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
            logger.error("Universal trading operation timed out after 45 seconds")
            return {"success": False, "error": "Request timed out after 45 seconds"}
        except Exception as e:
            logger.error(f"Universal trading operation failed: {str(e)}")
            return {"success": False, "error": f"Execution failed: {str(e)}"}
    
    def shutdown(self):
        """Shutdown the executor"""
        try:
            self._executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error shutting down async runner: {e}")
ThreadSafeAsyncRunner = UniversalAsyncRunner

# Global async runner instance
async_runner = UniversalAsyncRunner()


# ================================
# HELPER FUNCTIONS
# ================================

def _convert_universal_result_to_legacy_format(universal_result) -> Dict[str, Any]:
    """
    Convert universal result format back to the format your agents expect.
    This ensures backward compatibility with your existing agent logic.
    """
    if hasattr(universal_result, 'success'):
        # Handle UniversalTradeResult
        result = {
            "success": universal_result.success,
            "trade_reference": universal_result.trade_reference,
            "platform_source": universal_result.platform_source.value,
            "timestamp": universal_result.timestamp.isoformat(),
        }
        
        if universal_result.success:
            result.update({
                "order_id": universal_result.order_id,
                "transaction_id": universal_result.transaction_id,
                "instrument": universal_result.instrument,
                "side": universal_result.side.value if universal_result.side else None,
                "units": universal_result.units,
                "execution_price": universal_result.execution_price,
                "status": universal_result.status.value if universal_result.status else None
            })
        else:
            result.update({
                "error": universal_result.error,
                "error_type": universal_result.error_type
            })
            
        # Add any metadata
        if universal_result.metadata:
            result.update(universal_result.metadata)
            
        return result
    
    elif isinstance(universal_result, list):
        # Handle lists (positions, orders, etc.)
        # Return a dict with a key indicating a list result for legacy compatibility
        return {"results": [_convert_universal_item_to_legacy(item) for item in universal_result]}
    
    else:
        # Handle other universal objects
        return _convert_universal_item_to_legacy(universal_result)


def _convert_universal_item_to_legacy(item) -> Dict[str, Any]:
    """Convert individual universal objects to legacy format"""
    if hasattr(item, '__dict__'):
        # Convert dataclass to dict
        result = {}
        for key, value in item.__dict__.items():
            if hasattr(value, 'value'):  # Handle enums
                result[key] = value.value
            elif isinstance(value, datetime):  # Handle datetime
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
    return item


async def _execute_trade_operation_async(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Execute trading operation through platform router
    """
    try:
        router = get_router()
        
        # Route operation to appropriate platform method
        if operation == "execute_market_trade":
            result = await router.execute_market_trade(kwargs["params"])
        elif operation == "execute_limit_trade":
            result = await router.execute_limit_trade(kwargs["params"])
        elif operation == "get_live_price":
            result = await router.get_live_price(kwargs["instrument"])
        elif operation == "get_open_positions":
            result = await router.get_open_positions()
        elif operation == "get_pending_orders":
            result = await router.get_pending_orders()
        elif operation == "close_position":
            result = await router.close_position(
                kwargs["instrument"],
                kwargs.get("units"),
                kwargs.get("reason", "Manual close")
            )
        elif operation == "get_account_info":
            result = await router.get_account_info()
        elif operation == "get_portfolio_status":
            result = await router.get_portfolio_status()
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Convert to legacy format for agent compatibility
        return _convert_universal_result_to_legacy_format(result)
        
    except Exception as e:
        logger.error(f"❌ Platform operation {operation} failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ================================
# CREWAI TOOLS - SAME INTERFACE AS BEFORE
# ================================

@tool
def execute_market_trade(
    instrument: str,
    side: str,  # "buy" or "sell"
    units: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    reason: str = "Wyckoff signal",
    max_slippage: float = 0.001,
    platform_preference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute market order through platform abstraction layer.
    
    This tool maintains the exact same interface your CrewAI agents already use,
    but now routes through the universal platform system instead of directly to Oanda.
    
    Args:
        instrument: Trading instrument (e.g., "EUR_USD", "US30_USD")
        side: Trade side ("buy" or "sell")
        units: Position size in units
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        reason: Reason for the trade
        max_slippage: Maximum acceptable slippage (0.1% default)
        platform_preference: Preferred platform ("oanda", "metatrader5", etc.)
    
    Returns:
        Dict with execution results in same format as before
    """
    try:
        # Convert to universal format
        trade_side = convert_side_to_enum(side)
        platform_pref = None
        if platform_preference:
            try:
                platform_pref = Platform(platform_preference.lower())
            except ValueError:
                logger.warning(f"Unknown platform preference: {platform_preference}")
        
        # Create trade parameters
        params = TradeParams(
            instrument=instrument,
            side=trade_side,
            units=units,
            order_type=OrderType.MARKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_slippage=max_slippage,
            reason=reason,
            platform_preference=platform_pref
        )
        
        # Validate parameters
        validate_trade_params(params)
        
        # Execute through platform router
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="execute_market_trade",
            params=params
        )
        
    except Exception as e:
        logger.error(f"❌ Market trade execution failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "side": side,
            "units": units,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def execute_limit_trade(
    instrument: str,
    side: str,  # "buy" or "sell"
    units: float,
    price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    expiry_time: Optional[str] = None,
    reason: str = "Wyckoff limit order",
    platform_preference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute limit order through platform abstraction layer.
    
    Args:
        instrument: Trading instrument
        side: Trade side ("buy" or "sell")
        units: Position size in units
        price: Limit price
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        expiry_time: Order expiry time (ISO format or "GTC")
        reason: Reason for the trade
        platform_preference: Preferred platform
    
    Returns:
        Dict with order placement results
    """
    try:
        # Convert to universal format
        trade_side = convert_side_to_enum(side)
        platform_pref = None
        if platform_preference:
            try:
                platform_pref = Platform(platform_preference.lower())
            except ValueError:
                logger.warning(f"Unknown platform preference: {platform_preference}")
        
        # Parse expiry time if provided
        expiry_dt = None
        if expiry_time and expiry_time != "GTC":
            try:
                expiry_dt = datetime.fromisoformat(expiry_time)
            except ValueError:
                logger.warning(f"Invalid expiry time format: {expiry_time}")
        
        # Create trade parameters
        params = TradeParams(
            instrument=instrument,
            side=trade_side,
            units=units,
            order_type=OrderType.LIMIT,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expiry_time=expiry_dt,
            reason=reason,
            platform_preference=platform_pref
        )
        
        # Validate parameters
        validate_trade_params(params)
        
        # Execute through platform router
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="execute_limit_trade",
            params=params
        )
        
    except Exception as e:
        logger.error(f"❌ Limit trade execution failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "side": side,
            "units": units,
            "price": price,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_live_price(instrument: str) -> Dict[str, Any]:
    """
    Get current live price through platform abstraction layer.
    
    Args:
        instrument: Trading instrument (e.g., "EUR_USD")
    
    Returns:
        Dict with current bid/ask prices
    """
    try:
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="get_live_price",
            instrument=instrument
        )
    except Exception as e:
        logger.error(f"❌ Failed to get live price for {instrument}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_open_positions() -> Dict[str, Any]:
    """
    Get all open positions through platform abstraction layer.
    
    Returns:
        Dict with list of open positions
    """
    try:
        result = async_runner.run_async(
            _execute_trade_operation_async,
            operation="get_open_positions"
        )
        
        # Wrap in success format for consistency
        if isinstance(result, list):
            return {
                "success": True,
                "positions": result,
                "position_count": len(result),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get open positions: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_pending_orders() -> Dict[str, Any]:
    """
    Get all pending orders through platform abstraction layer.
    
    Returns:
        Dict with list of pending orders
    """
    try:
        result = async_runner.run_async(
            _execute_trade_operation_async,
            operation="get_pending_orders"
        )
        
        # Wrap in success format for consistency
        if isinstance(result, list):
            return {
                "success": True,
                "orders": result,
                "order_count": len(result),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get pending orders: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def close_position(
    instrument: str,
    units: Optional[float] = None,
    reason: str = "Manual close"
) -> Dict[str, Any]:
    """
    Close position through platform abstraction layer.
    
    Args:
        instrument: Trading instrument
        units: Units to close (None for full close)
        reason: Reason for closing
    
    Returns:
        Dict with position close results
    """
    try:
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="close_position",
            instrument=instrument,
            units=units,
            reason=reason
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to close position {instrument}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "units": units,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_account_info() -> Dict[str, Any]:
    """
    Get account information through platform abstraction layer.
    
    Returns:
        Dict with account balance, equity, margin info
    """
    try:
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="get_account_info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get account info: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_portfolio_status() -> Dict[str, Any]:
    """
    Get comprehensive portfolio status through platform abstraction layer.
    
    Returns:
        Dict with account info, positions, and orders
    """
    try:
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="get_portfolio_status"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get portfolio status: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def calculate_position_size(
    instrument: str,
    risk_amount: float,
    stop_loss_distance: float,
    account_balance: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate position size based on risk parameters.
    
    Args:
        instrument: Trading instrument
        risk_amount: Amount to risk (in account currency)
        stop_loss_distance: Distance to stop loss in price units
        account_balance: Account balance (if not provided, will be fetched)
    
    Returns:
        Dict with calculated position size
    """
    try:
        # Basic position size calculation
        if stop_loss_distance <= 0:
            return {
                "success": False,
                "error": "Stop loss distance must be positive",
                "error_type": "validation_error"
            }
        
        # Get account balance if not provided
        if account_balance is None:
            account_info = get_account_info()
            if not account_info.get("success"):
                return {
                    "success": False,
                    "error": "Failed to get account balance for position sizing",
                    "error_type": "account_info_error"
                }
            account_balance = account_info.get("balance", 0)
        
        # Basic calculation - can be enhanced
        position_size = abs(risk_amount / stop_loss_distance)

        # Add some basic validation
        if account_balance is not None:
            max_position_size = account_balance * 0.1  # Max 10% of account per trade
            if position_size > max_position_size:
                position_size = max_position_size

        return {
            "success": True,
            "instrument": instrument,
            "position_size": position_size,
            "risk_amount": risk_amount,
            "stop_loss_distance": stop_loss_distance,
            "account_balance": account_balance,
            "risk_percentage": (risk_amount / account_balance) * 100 if (account_balance is not None and account_balance > 0) else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Position size calculation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ================================
# PLATFORM MANAGEMENT TOOLS
# ================================

@tool
def get_platform_status() -> Dict[str, Any]:
    """
    Get status of all trading platforms.
    
    Returns:
        Dict with platform health and performance information
    """
    try:
        router = get_router()
        return router.get_platform_status()
    except Exception as e:
        logger.error(f"❌ Failed to get platform status: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def switch_primary_platform(platform_name: str) -> Dict[str, Any]:
    """
    Switch primary trading platform.
    
    Args:
        platform_name: Name of platform to switch to ("oanda", "metatrader5", etc.)
    
    Returns:
        Dict with switch results
    """
    try:
        # This would require updating the router configuration
        # For now, return info about current platform capabilities
        return {
            "success": False,
            "error": "Platform switching not yet implemented",
            "info": "This feature will be available when multiple platforms are configured",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Platform switch failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ================================
# INITIALIZATION AND SHUTDOWN
# ================================

def initialize_universal_tools(config_path: Optional[str] = None):
    """
    Initialize the universal trading tools.
    This should be called once when your application starts.
    """
    try:
        from .router import initialize_router
        async_runner.run_async(initialize_router, config_path)
        logger.info("✅ Universal trading tools initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize universal trading tools: {str(e)}")
        return False


def shutdown_universal_tools():
    """
    Shutdown the universal trading tools.
    This should be called when your application shuts down.
    """
    try:
        from .router import shutdown_router
        async_runner.run_async(shutdown_router)
        
        # Shutdown the async runner
        async_runner.shutdown()
        
        logger.info("✅ Universal trading tools shutdown successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to shutdown universal trading tools: {str(e)}")
        return False


# ================================
# BACKWARD COMPATIBILITY EXPORTS
# ================================

# Export all tools for easy importing
__all__ = [
    "execute_market_trade",
    "execute_limit_trade", 
    "get_live_price",
    "get_open_positions",
    "get_pending_orders",
    "close_position",
    "get_account_info",
    "get_portfolio_status",
    "calculate_position_size",
    "get_platform_status",
    "switch_primary_platform",
    "initialize_universal_tools",
    "shutdown_universal_tools",
    "UniversalAsyncRunner",
    "ThreadSafeAsyncRunner"  # Type alias for compatibility
]
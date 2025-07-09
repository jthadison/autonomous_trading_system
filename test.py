"""
Unicode Fix Script for Windows Platform Abstraction
File: fix_unicode_error.py

This script fixes Unicode encoding issues on Windows when working with
the platform abstraction files.
"""

import os
import sys
from pathlib import Path
import re


def fix_unicode_in_file(file_path: str):
    """Fix Unicode encoding issues in a specific file"""
    
    print(f"🔧 Fixing Unicode issues in {file_path}...")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Try reading with UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("✅ File already UTF-8 encoded")
        return True
        
    except UnicodeDecodeError:
        print("⚠️ Unicode decode error detected, attempting to fix...")
        
        # Try different encodings
        encodings_to_try = ['latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                print(f"✅ Successfully read file with {encoding} encoding")
                
                # Clean up problematic Unicode characters
                content = fix_unicode_characters(content)
                
                # Write back as UTF-8
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("✅ File converted to UTF-8 encoding")
                return True
                
            except UnicodeDecodeError:
                continue
        
        print("❌ Could not read file with any encoding")
        return False


def fix_unicode_characters(content: str) -> str:
    """Fix problematic Unicode characters"""
    
    # Replace common problematic characters
    replacements = {
        '\x9d': '',  # Remove undefined character
        '\x93': '"',  # Left double quotation mark
        '\x94': '"',  # Right double quotation mark
        '\x91': "'",  # Left single quotation mark
        '\x92': "'",  # Right single quotation mark
        '\x96': '–',  # En dash
        '\x97': '—',  # Em dash
        '\x85': '...',  # Horizontal ellipsis
    }
    
    for old_char, new_char in replacements.items():
        content = content.replace(old_char, new_char)
    
    # Remove any remaining non-printable characters
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
    
    return content


def create_clean_universal_tools():
    """Create a clean version of universal_tools.py without Unicode issues"""
    
    print("🛠️ Creating clean universal_tools.py...")
    
    # Clean Python code without Unicode issues
    clean_content = '''"""
Universal Trading Tools for CrewAI Agents (Clean Version)
File: src/platform_abstraction/universal_tools.py

Updated trading tools that work with the platform abstraction layer.
These tools maintain the same interface that your CrewAI agents already use,
but now route through the universal platform system instead of directly to Oanda.
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
            self._executor.shutdown(wait=True, timeout=10)
        except Exception as e:
            logger.error(f"Error shutting down async runner: {e}")

# Type alias for backward compatibility
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
        return [_convert_universal_item_to_legacy(item) for item in universal_result]
    
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
    """Execute trading operation through platform router"""
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
        logger.error(f"Platform operation {operation} failed: {str(e)}")
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
        logger.error(f"Market trade execution failed: {str(e)}")
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
    """Execute limit order through platform abstraction layer."""
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
        logger.error(f"Limit trade execution failed: {str(e)}")
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
    """Get current live price through platform abstraction layer."""
    try:
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="get_live_price",
            instrument=instrument
        )
    except Exception as e:
        logger.error(f"Failed to get live price for {instrument}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_open_positions() -> Dict[str, Any]:
    """Get all open positions through platform abstraction layer."""
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
        logger.error(f"Failed to get open positions: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_pending_orders() -> Dict[str, Any]:
    """Get all pending orders through platform abstraction layer."""
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
        logger.error(f"Failed to get pending orders: {str(e)}")
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
    """Close position through platform abstraction layer."""
    try:
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="close_position",
            instrument=instrument,
            units=units,
            reason=reason
        )
        
    except Exception as e:
        logger.error(f"Failed to close position {instrument}: {str(e)}")
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
    """Get account information through platform abstraction layer."""
    try:
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="get_account_info"
        )
        
    except Exception as e:
        logger.error(f"Failed to get account info: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_portfolio_status() -> Dict[str, Any]:
    """Get comprehensive portfolio status through platform abstraction layer."""
    try:
        return async_runner.run_async(
            _execute_trade_operation_async,
            operation="get_portfolio_status"
        )
        
    except Exception as e:
        logger.error(f"Failed to get portfolio status: {str(e)}")
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
    """Calculate position size based on risk parameters."""
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
            "risk_percentage": (risk_amount / account_balance) * 100 if account_balance > 0 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Position size calculation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "instrument": instrument,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@tool
def get_platform_status() -> Dict[str, Any]:
    """Get status of all trading platforms."""
    try:
        router = get_router()
        return router.get_platform_status()
    except Exception as e:
        logger.error(f"Failed to get platform status: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def initialize_universal_tools(config_path: Optional[str] = None):
    """Initialize the universal trading tools."""
    try:
        from .router import initialize_router
        async_runner.run_async(initialize_router, config_path)
        logger.info("Universal trading tools initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize universal trading tools: {str(e)}")
        return False


def shutdown_universal_tools():
    """Shutdown the universal trading tools."""
    try:
        from .router import shutdown_router
        async_runner.run_async(shutdown_router)
        
        # Shutdown the async runner
        async_runner.shutdown()
        
        logger.info("Universal trading tools shutdown successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to shutdown universal trading tools: {str(e)}")
        return False


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
    "initialize_universal_tools",
    "shutdown_universal_tools",
    "UniversalAsyncRunner",
    "ThreadSafeAsyncRunner"
]
'''
    
    # Ensure directory exists
    output_dir = Path("src/platform_abstraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the clean file
    output_path = output_dir / "universal_tools.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(clean_content)
    
    print(f"✅ Created clean universal_tools.py at {output_path}")
    return True


def main():
    """Main function to fix Unicode issues"""
    
    print("🔧 Unicode Fix for Platform Abstraction on Windows")
    print("=" * 55)
    
    # Check if we need to create a clean file
    universal_tools_path = "src/platform_abstraction/universal_tools.py"
    
    if Path(universal_tools_path).exists():
        print("📁 Found existing universal_tools.py")
        success = fix_unicode_in_file(universal_tools_path)
        if not success:
            print("🛠️ Creating clean version...")
            success = create_clean_universal_tools()
    else:
        print("📁 universal_tools.py not found, creating clean version...")
        success = create_clean_universal_tools()
    
    if success:
        print("\n✅ Unicode issues fixed!")
        print("\n📋 Next steps:")
        print("1. Update your crew.py imports:")
        print("   FROM: from src.autonomous_trading_system.tools.trading_execution_tools_sync import (")
        print("   TO:   from src.platform_abstraction.universal_tools import (")
        print("2. Test the import to verify it works")
        print("3. Run your existing backtests")
        
        # Test the import
        try:
            sys.path.append(str(Path.cwd()))
            from src.platform_abstraction.universal_tools import execute_market_trade, UniversalAsyncRunner
            print("\n✅ Import test successful!")
        except Exception as e:
            print(f"\n⚠️ Import test failed: {e}")
            print("Check that all platform abstraction files are in place")
    
    else:
        print("\n❌ Failed to fix Unicode issues")
        print("Please manually create the universal_tools.py file")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
Base Platform Adapter
File: src/platform_abstraction/adapters/base_adapter.py

Base class for all platform adapters with common functionality.
New platform adapters should inherit from this class.
"""

import asyncio
from abc import ABC
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import time

from ..interface import UniversalTradingInterface, TradingPlatformError
from ..models import (
    Platform,
    TradeParams,
    UniversalTradeResult,
    UniversalMarketPrice,
    UniversalAccountInfo,
    UniversalPosition,
    UniversalOrder,
    HistoricalData,
    generate_trade_reference
)
from src.config.logging_config import logger


class BasePlatformAdapter(UniversalTradingInterface):
    """
    Base class for all platform adapters with common functionality
    """
    
    def __init__(self, platform: Platform, config: Dict[str, Any]):
        super().__init__(platform, config)
        self.last_request_time = {}  # Rate limiting tracker
        self.connection_pool = None
        self.session_id = None
        
    # ================================
    # COMMON UTILITY METHODS
    # ================================
    
    async def _rate_limit_check(self, operation: str, min_interval_ms: float = 100):
        """
        Enforce rate limiting between requests
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        last_time = self.last_request_time.get(operation, 0)
        
        time_since_last = current_time - last_time
        if time_since_last < min_interval_ms:
            sleep_time = (min_interval_ms - time_since_last) / 1000
            await asyncio.sleep(sleep_time)
        
        self.last_request_time[operation] = time.time() * 1000
    
    def _validate_instrument(self, instrument: str) -> bool:
        """
        Validate instrument format - override in platform adapters
        """
        return bool(instrument and isinstance(instrument, str))
    
    def _normalize_instrument_name(self, instrument: str) -> str:
        """
        Normalize instrument name for platform - override in platform adapters
        """
        return instrument.upper()
    
    async def _handle_platform_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """
        Common error handling for platform operations
        """
        error_msg = str(error)
        error_type = type(error).__name__
        
        logger.error(f"❌ {self.platform.value} {operation} failed: {error_msg}")
        
        # Create standardized error response
        return {
            "success": False,
            "error": error_msg,
            "error_type": error_type,
            "platform": self.platform.value,
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _create_error_result(self, error: str, error_type: str = "platform_error") -> UniversalTradeResult:
        """
        Create standardized error result
        """
        return UniversalTradeResult.error_result(
            trade_reference=generate_trade_reference(self.platform),
            platform_source=self.platform,
            error=error,
            error_type=error_type
        )
    
    # ================================
    # COMMON VALIDATION METHODS
    # ================================
    
    async def _validate_trade_params(self, params: TradeParams) -> bool:
        """
        Enhanced validation with platform-specific checks
        """
        # Call parent validation first
        await super().validate_trade_params(params)
        
        # Platform-specific validation (override in subclasses)
        return await self._platform_specific_validation(params)
    
    async def _platform_specific_validation(self, params: TradeParams) -> bool:
        """
        Platform-specific validation - override in platform adapters
        """
        return True
    
    # ================================
    # CONNECTION MANAGEMENT HELPERS
    # ================================
    
    async def _test_connection(self) -> bool:
        """
        Test basic connection to platform - override in platform adapters
        """
        try:
            # Basic connection test - should be overridden
            await asyncio.sleep(0.1)  # Simulate connection test
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {self.platform.value}: {e}")
            return False
    
    async def _reconnect(self) -> bool:
        """
        Attempt to reconnect to platform
        """
        try:
            logger.info(f"🔄 Attempting to reconnect to {self.platform.value}")
            
            # Disconnect first
            await self.disconnect()
            await asyncio.sleep(1)  # Brief delay
            
            # Reconnect
            success = await self.connect()
            if success:
                logger.info(f"✅ Successfully reconnected to {self.platform.value}")
            else:
                logger.error(f"❌ Failed to reconnect to {self.platform.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Reconnection failed for {self.platform.value}: {e}")
            return False
    
    # ================================
    # DATA CONVERSION HELPERS
    # ================================
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """
        Convert universal timeframe to platform-specific format
        Override in platform adapters
        """
        # Standard mapping - override in platform adapters
        timeframe_map = {
            "M1": "1m",
            "M5": "5m",
            "M15": "15m",
            "M30": "30m",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d",
            "W1": "1w"
        }
        return timeframe_map.get(timeframe, timeframe)
    
    def _parse_price_data(self, raw_data: Dict[str, Any]) -> UniversalMarketPrice:
        """
        Parse platform-specific price data to universal format
        Override in platform adapters
        """
        # Default implementation - should be overridden
        return UniversalMarketPrice(
            instrument=raw_data.get("instrument", "UNKNOWN"),
            bid=float(raw_data.get("bid", 0)),
            ask=float(raw_data.get("ask", 0)),
            spread=float(raw_data.get("spread", 0)),
            timestamp=datetime.now(timezone.utc),
            platform_source=self.platform
        )
    
    # ================================
    # RETRY LOGIC
    # ================================
    
    async def _execute_with_retry(
        self,
        operation_func,
        operation_name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        *args,
        **kwargs
    ):
        """
        Execute operation with retry logic
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Retry attempt {attempt} for {operation_name} on {self.platform.value}")
                    await asyncio.sleep(retry_delay * attempt)  # Exponential backoff
                
                # Apply rate limiting
                await self._rate_limit_check(operation_name)
                
                # Execute operation
                result = await operation_func(*args, **kwargs)
                
                # Success
                if attempt > 0:
                    logger.info(f"✅ {operation_name} succeeded on retry {attempt}")
                
                return result
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Check if this is a retryable error
                if not self._is_retryable_error(e):
                    logger.error(f"❌ Non-retryable error in {operation_name}: {error_msg}")
                    break
                
                if attempt < max_retries:
                    logger.warning(f"⚠️ {operation_name} failed (attempt {attempt + 1}): {error_msg}")
                else:
                    logger.error(f"❌ {operation_name} failed after {max_retries} retries: {error_msg}")
        
        # All retries failed
        raise TradingPlatformError(
            f"{operation_name} failed after {max_retries} retries: {str(last_error)}",
            self.platform
        )
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable
        Override in platform adapters for platform-specific error handling
        """
        error_msg = str(error).lower()
        
        # Common retryable errors
        retryable_patterns = [
            "timeout", "connection", "network", "temporary", "rate limit",
            "service unavailable", "internal server error", "bad gateway"
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
    
    # ================================
    # PERFORMANCE MONITORING
    # ================================
    
    async def _monitor_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Monitor operation performance
        """
        start_time = time.time()
        
        try:
            result = await operation_func(*args, **kwargs)
            
            # Record successful operation
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.debug(f"✅ {operation_name} on {self.platform.value} completed in {elapsed_time:.1f}ms")
            
            return result
            
        except Exception as e:
            # Record failed operation
            elapsed_time = (time.time() - start_time) * 1000
            logger.error(f"❌ {operation_name} on {self.platform.value} failed after {elapsed_time:.1f}ms: {e}")
            raise
    
    # ================================
    # CONFIGURATION HELPERS
    # ================================
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback
        """
        return self.config.get(key, default)
    
    def _get_timeout(self, operation: str) -> float:
        """
        Get timeout for specific operation
        """
        timeouts = self._get_config_value("timeouts", {})
        return timeouts.get(operation, 30.0)  # Default 30 seconds
    
    def _get_rate_limit(self, operation: str) -> float:
        """
        Get rate limit for specific operation (ms between requests)
        """
        rate_limits = self._get_config_value("rate_limits", {})
        return rate_limits.get(operation, 100.0)  # Default 100ms
    
    # ================================
    # HEALTH CHECK IMPLEMENTATION
    # ================================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Default health check implementation
        """
        health_info = {
            "platform": self.platform.value,
            "connected": self._connected,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Test basic operations
            start_time = time.time()
            
            # Test connection
            connection_ok = await self._test_connection()
            health_info["connection_test"] = connection_ok
            
            # Test account access (if connected)
            if connection_ok and self._connected:
                try:
                    account_info = await self.get_account_info()
                    health_info["account_accessible"] = True
                except Exception as e:
                    health_info["account_accessible"] = False
                    health_info["account_error"] = str(e)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            health_info["response_time_ms"] = response_time
            
            # Determine overall status
            if connection_ok and health_info.get("account_accessible", False):
                health_info["status"] = "healthy"
            elif connection_ok:
                health_info["status"] = "degraded"
            else:
                health_info["status"] = "unhealthy"
            
        except Exception as e:
            health_info.update({
                "status": "error",
                "error": str(e),
                "connection_test": False,
                "account_accessible": False
            })
        
        return health_info
    
    # ================================
    # DEFAULT IMPLEMENTATIONS
    # ================================
    
    async def get_available_instruments(self) -> List[str]:
        """
        Default implementation - should be overridden
        """
        logger.warning(f"get_available_instruments not implemented for {self.platform.value}")
        return []
    
    async def get_platform_info(self) -> Dict[str, Any]:
        """
        Default platform info - should be overridden
        """
        return {
            "platform": self.platform.value,
            "adapter_version": "1.0.0",
            "features": ["basic_trading"],
            "limitations": ["limited_implementation"]
        }


# ================================
# ADAPTER REGISTRY HELPERS
# ================================

def register_adapter(platform: Platform, adapter_class: type):
    """
    Helper function to register platform adapters
    """
    from ..interface import PlatformRegistry
    PlatformRegistry.register(platform, adapter_class)
    logger.info(f"✅ Registered adapter for {platform.value}")


def get_registered_platforms() -> List[Platform]:
    """
    Get list of registered platforms
    """
    from ..interface import PlatformRegistry
    return PlatformRegistry.get_available_platforms()


# ================================
# TESTING UTILITIES
# ================================

async def test_adapter(adapter: UniversalTradingInterface) -> Dict[str, Any]:
    """
    Comprehensive adapter testing
    """
    test_results = {
        "platform": adapter.platform.value,
        "tests": {},
        "overall_status": "unknown"
    }
    
    tests = [
        ("connection", lambda: adapter.connect()),
        ("health_check", lambda: adapter.health_check()),
        ("account_info", lambda: adapter.get_account_info()),
        ("platform_info", lambda: adapter.get_platform_info()),
        ("available_instruments", lambda: adapter.get_available_instruments())
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = await test_func()
            elapsed_time = (time.time() - start_time) * 1000
            
            test_results["tests"][test_name] = {
                "status": "passed",
                "response_time_ms": elapsed_time,
                "result": str(result)[:200]  # Truncate result
            }
            passed_tests += 1
            
        except Exception as e:
            test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Determine overall status
    if passed_tests == len(tests):
        test_results["overall_status"] = "all_passed"
    elif passed_tests > len(tests) // 2:
        test_results["overall_status"] = "mostly_passed"
    elif passed_tests > 0:
        test_results["overall_status"] = "some_passed"
    else:
        test_results["overall_status"] = "all_failed"
    
    # Cleanup
    try:
        await adapter.disconnect()
    except Exception:
        pass
    
    return test_results
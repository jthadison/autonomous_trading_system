"""
Platform Router and Configuration Management
File: src/platform_abstraction/router.py

Intelligent routing system that selects the optimal platform for each trading operation.
Handles platform selection, load balancing, failover, and configuration management.
"""

import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from .interface import UniversalTradingInterface, PlatformRegistry, TradingPlatformError
from .models import (
    Platform,
    TradeParams,
    UniversalTradeResult,
    UniversalMarketPrice,
    UniversalAccountInfo,
    UniversalPosition,
    UniversalOrder,
    UniversalPortfolioStatus,
    HistoricalData
)
from src.config.logging_config import logger


class RoutingStrategy(Enum):
    """Platform routing strategies"""
    PRIMARY_ONLY = "primary_only"  # Use only primary platform
    LOAD_BALANCE = "load_balance"  # Distribute load across platforms
    FAILOVER = "failover"  # Use primary, fallback on failure
    OPERATION_SPECIFIC = "operation_specific"  # Route based on operation type
    PERFORMANCE_BASED = "performance_based"  # Route to best performing platform


@dataclass
class PlatformHealth:
    """Platform health status tracking"""
    platform: Platform
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    response_time_ms: float = 0.0
    uptime_percentage: float = 100.0
    
    def mark_error(self, error: str):
        """Mark platform as having an error"""
        self.error_count += 1
        self.last_error = error
        self.last_check = datetime.now(timezone.utc)
        # Consider unhealthy if 3+ consecutive errors
        if self.error_count >= 3:
            self.is_healthy = False
    
    def mark_success(self, response_time_ms: float = 0.0):
        """Mark platform as successful"""
        self.error_count = 0
        self.last_error = None
        self.last_check = datetime.now(timezone.utc)
        self.is_healthy = True
        self.response_time_ms = response_time_ms


@dataclass
class PlatformConfig:
    """Configuration for individual platforms"""
    platform: Platform
    enabled: bool = True
    weight: float = 1.0  # For load balancing
    max_concurrent_operations: int = 10
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Operation-specific routing
    preferred_operations: List[str] = field(default_factory=list)
    blocked_operations: List[str] = field(default_factory=list)


@dataclass
class RoutingConfig:
    """Main routing configuration"""
    primary_platform: Platform = Platform.OANDA
    strategy: RoutingStrategy = RoutingStrategy.PRIMARY_ONLY
    platforms: Dict[Platform, PlatformConfig] = field(default_factory=dict)
    
    # Routing rules
    operation_routing: Dict[str, Platform] = field(default_factory=dict)
    instrument_routing: Dict[str, Platform] = field(default_factory=dict)
    
    # Health monitoring
    health_check_interval: int = 300  # seconds
    unhealthy_threshold: int = 3  # consecutive failures
    recovery_threshold: int = 2  # consecutive successes to mark healthy
    
    # Performance thresholds
    max_response_time_ms: float = 5000.0
    min_uptime_percentage: float = 95.0


class PlatformRouter:
    """
    Intelligent platform router for trading operations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.adapters: Dict[Platform, UniversalTradingInterface] = {}
        self.health_status: Dict[Platform, PlatformHealth] = {}
        self.performance_metrics: Dict[Platform, Dict[str, float]] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize health status for all platforms
        for platform in self.config.platforms:
            self.health_status[platform] = PlatformHealth(platform)
            self.performance_metrics[platform] = {
                "avg_response_time": 0.0,
                "success_rate": 100.0,
                "total_operations": 0,
                "failed_operations": 0
            }
    
    def _load_config(self, config_path: Optional[str] = None) -> RoutingConfig:
        """Load routing configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return self._parse_config(config_data)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Return default configuration
        return RoutingConfig(
            primary_platform=Platform.OANDA,
            strategy=RoutingStrategy.PRIMARY_ONLY,
            platforms={
                Platform.OANDA: PlatformConfig(
                    platform=Platform.OANDA,
                    enabled=True,
                    weight=1.0,
                    preferred_operations=["market_trade", "limit_trade", "get_positions"]
                )
            }
        )
    
    def _parse_config(self, config_data: Dict[str, Any]) -> RoutingConfig:
        """Parse configuration data into RoutingConfig"""
        # This would parse YAML config into RoutingConfig object
        # Implementation depends on your config file format
        return RoutingConfig()  # Simplified for now
    
    async def initialize(self):
        """Initialize the router and all platform adapters"""
        logger.info("🚀 Initializing Platform Router...")
        
        # Initialize adapters for enabled platforms
        for platform, platform_config in self.config.platforms.items():
            if platform_config.enabled:
                try:
                    adapter = PlatformRegistry.create_adapter(platform, platform_config.config)
                    await adapter.connect()
                    self.adapters[platform] = adapter
                    logger.info(f"✅ Initialized adapter for {platform.value}")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {platform.value}: {e}")
                    self.health_status[platform].mark_error(str(e))
        
        # Start health monitoring
        self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info(f"✅ Platform Router initialized with {len(self.adapters)} platforms")
    
    async def shutdown(self):
        """Shutdown the router and disconnect all adapters"""
        logger.info("🔌 Shutting down Platform Router...")
        
        # Stop health monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all adapters
        for platform, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                logger.info(f"✅ Disconnected from {platform.value}")
            except Exception as e:
                logger.error(f"❌ Error disconnecting from {platform.value}: {e}")
        
        self.adapters.clear()
        logger.info("✅ Platform Router shutdown complete")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring for all platforms"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._check_all_platform_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")
    
    async def _check_all_platform_health(self):
        """Check health of all platforms"""
        for platform, adapter in self.adapters.items():
            try:
                start_time = datetime.now()
                health_result = await adapter.health_check()
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if health_result.get("status") == "healthy":
                    self.health_status[platform].mark_success(response_time)
                else:
                    self.health_status[platform].mark_error(health_result.get("error", "Unknown"))
                    
            except Exception as e:
                self.health_status[platform].mark_error(str(e))
    
    def get_best_platform(
        self,
        operation: str,
        instrument: Optional[str] = None,
        prefer_platform: Optional[Platform] = None
    ) -> Platform:
        """
        Select the best platform for an operation based on routing strategy
        """
        # If specific platform preferred and healthy, use it
        if prefer_platform and prefer_platform in self.adapters:
            if self.health_status[prefer_platform].is_healthy:
                return prefer_platform
        
        # Apply routing strategy
        if self.config.strategy == RoutingStrategy.PRIMARY_ONLY:
            return self._route_primary_only()
        elif self.config.strategy == RoutingStrategy.FAILOVER:
            return self._route_with_failover()
        elif self.config.strategy == RoutingStrategy.OPERATION_SPECIFIC:
            return self._route_by_operation(operation, instrument)
        elif self.config.strategy == RoutingStrategy.PERFORMANCE_BASED:
            return self._route_by_performance(operation)
        else:
            return self.config.primary_platform
    
    def _route_primary_only(self) -> Platform:
        """Route to primary platform only"""
        primary = self.config.primary_platform
        if primary in self.adapters and self.health_status[primary].is_healthy:
            return primary
        
        # Fallback to any healthy platform
        for platform, health in self.health_status.items():
            if health.is_healthy and platform in self.adapters:
                return platform
        
        # Last resort - return primary even if unhealthy
        return self.config.primary_platform
    
    def _route_with_failover(self) -> Platform:
        """Route with failover logic"""
        # Try primary first
        primary = self.config.primary_platform
        if primary in self.adapters and self.health_status[primary].is_healthy:
            return primary
        
        # Try other platforms in order of preference
        healthy_platforms = [
            p for p, h in self.health_status.items()
            if h.is_healthy and p in self.adapters and p != primary
        ]
        
        if healthy_platforms:
            # Sort by weight (higher weight = higher preference)
            healthy_platforms.sort(
                key=lambda p: self.config.platforms.get(p, PlatformConfig(p)).weight,
                reverse=True
            )
            return healthy_platforms[0]
        
        return primary  # Last resort
    
    def _route_by_operation(self, operation: str, instrument: Optional[str] = None) -> Platform:
        """Route based on operation type and instrument"""
        # Check for instrument-specific routing
        if instrument and instrument in self.config.instrument_routing:
            preferred = self.config.instrument_routing[instrument]
            if preferred in self.adapters and self.health_status[preferred].is_healthy:
                return preferred
        
        # Check for operation-specific routing
        if operation in self.config.operation_routing:
            preferred = self.config.operation_routing[operation]
            if preferred in self.adapters and self.health_status[preferred].is_healthy:
                return preferred
        
        # Fallback to general routing
        return self._route_with_failover()
    
    def _route_by_performance(self, operation: str) -> Platform:
        """Route based on performance metrics"""
        # Find platform with best performance for this operation type
        best_platform = None
        best_score = -1
        
        for platform, health in self.health_status.items():
            if not health.is_healthy or platform not in self.adapters:
                continue
            
            metrics = self.performance_metrics[platform]
            # Simple scoring: higher success rate, lower response time = better
            score = metrics["success_rate"] - (metrics["avg_response_time"] / 1000)
            
            if score > best_score:
                best_score = score
                best_platform = platform
        
        return best_platform or self.config.primary_platform
    
    # ================================
    # TRADING OPERATIONS - ROUTE TO BEST PLATFORM
    # ================================
    
    async def execute_market_trade(self, params: TradeParams) -> UniversalTradeResult:
        """Execute market trade on best available platform"""
        platform = self.get_best_platform("market_trade", params.instrument, params.platform_preference)
        return await self._execute_with_monitoring(platform, "execute_market_trade", params)
    
    async def execute_limit_trade(self, params: TradeParams) -> UniversalTradeResult:
        """Execute limit trade on best available platform"""
        platform = self.get_best_platform("limit_trade", params.instrument, params.platform_preference)
        return await self._execute_with_monitoring(platform, "execute_limit_trade", params)
    
    async def get_live_price(self, instrument: str) -> UniversalMarketPrice:
        """Get live price from best available platform"""
        platform = self.get_best_platform("get_live_price", instrument)
        return await self._execute_with_monitoring(platform, "get_live_price", instrument)
    
    async def get_open_positions(self) -> List[UniversalPosition]:
        """Get positions from best available platform"""
        platform = self.get_best_platform("get_positions")
        return await self._execute_with_monitoring(platform, "get_open_positions")
    
    async def get_pending_orders(self) -> List[UniversalOrder]:
        """Get orders from best available platform"""
        platform = self.get_best_platform("get_orders")
        return await self._execute_with_monitoring(platform, "get_pending_orders")
    
    async def close_position(self, instrument: str, units: Optional[float] = None, reason: str = "Manual close") -> UniversalTradeResult:
        """Close position on best available platform"""
        platform = self.get_best_platform("close_position", instrument)
        return await self._execute_with_monitoring(platform, "close_position", instrument, units, reason)
    
    async def get_account_info(self) -> UniversalAccountInfo:
        """Get account info from best available platform"""
        platform = self.get_best_platform("get_account_info")
        return await self._execute_with_monitoring(platform, "get_account_info")
    
    async def get_portfolio_status(self) -> UniversalPortfolioStatus:
        """Get portfolio status from best available platform"""
        platform = self.get_best_platform("get_portfolio_status")
        return await self._execute_with_monitoring(platform, "get_portfolio_status")
    
    async def _execute_with_monitoring(self, platform: Platform, method_name: str, *args, **kwargs):
        """Execute operation with performance monitoring and error handling"""
        if platform not in self.adapters:
            raise TradingPlatformError(f"Platform {platform.value} not available", platform)
        
        adapter = self.adapters[platform]
        start_time = datetime.now()
        
        try:
            # Execute the operation
            method = getattr(adapter, method_name)
            result = await method(*args, **kwargs)
            
            # Record successful operation
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.health_status[platform].mark_success(response_time)
            self._update_performance_metrics(platform, True, response_time)
            
            logger.debug(f"✅ {method_name} executed on {platform.value} in {response_time:.1f}ms")
            return result
            
        except Exception as e:
            # Record failed operation
            self.health_status[platform].mark_error(str(e))
            self._update_performance_metrics(platform, False, 0)
            
            logger.error(f"❌ {method_name} failed on {platform.value}: {e}")
            
            # Try failover if enabled and this isn't already a failover attempt
            if self.config.strategy in [RoutingStrategy.FAILOVER, RoutingStrategy.PERFORMANCE_BASED]:
                # Find alternative platform
                alternative = self._find_alternative_platform(platform, method_name)
                if alternative:
                    logger.info(f"🔄 Attempting failover from {platform.value} to {alternative.value}")
                    return await self._execute_with_monitoring(alternative, method_name, *args, **kwargs)
            
            raise TradingPlatformError(f"Operation {method_name} failed on {platform.value}: {str(e)}", platform)
    
    def _find_alternative_platform(self, failed_platform: Platform, operation: str) -> Optional[Platform]:
        """Find alternative healthy platform for failover"""
        for platform, health in self.health_status.items():
            if platform != failed_platform and health.is_healthy and platform in self.adapters:
                return platform
        return None
    
    def _update_performance_metrics(self, platform: Platform, success: bool, response_time: float):
        """Update performance metrics for platform"""
        metrics = self.performance_metrics[platform]
        metrics["total_operations"] += 1
        
        if not success:
            metrics["failed_operations"] += 1
        
        # Update success rate
        metrics["success_rate"] = ((metrics["total_operations"] - metrics["failed_operations"]) / 
                                   metrics["total_operations"]) * 100
        
        # Update average response time (exponential moving average)
        if response_time > 0:
            alpha = 0.1  # Smoothing factor
            metrics["avg_response_time"] = (alpha * response_time + 
                                            (1 - alpha) * metrics["avg_response_time"])
    
    # ================================
    # STATUS AND MONITORING
    # ================================
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all platforms"""
        return {
            "router_status": {
                "active_platforms": len(self.adapters),
                "healthy_platforms": sum(1 for h in self.health_status.values() if h.is_healthy),
                "primary_platform": self.config.primary_platform.value,
                "routing_strategy": self.config.strategy.value
            },
            "platforms": {
                platform.value: {
                    "connected": platform in self.adapters,
                    "healthy": health.is_healthy,
                    "error_count": health.error_count,
                    "last_error": health.last_error,
                    "response_time_ms": health.response_time_ms,
                    "performance": self.performance_metrics[platform]
                }
                for platform, health in self.health_status.items()
            }
        }
    
    def get_routing_recommendations(self) -> Dict[str, str]:
        """Get recommendations for optimal routing"""
        recommendations = []
        
        # Check for unhealthy platforms
        unhealthy = [p.value for p, h in self.health_status.items() if not h.is_healthy]
        if unhealthy:
            recommendations.append(f"Unhealthy platforms detected: {', '.join(unhealthy)}")
        
        # Check for high response times
        slow_platforms = [
            p.value for p, metrics in self.performance_metrics.items()
            if metrics["avg_response_time"] > self.config.max_response_time_ms
        ]
        if slow_platforms:
            recommendations.append(f"High latency platforms: {', '.join(slow_platforms)}")
        
        # Check success rates
        low_success = [
            p.value for p, metrics in self.performance_metrics.items()
            if metrics["success_rate"] < self.config.min_uptime_percentage
        ]
        if low_success:
            recommendations.append(f"Low success rate platforms: {', '.join(low_success)}")
        return {
            "recommendations": "\n".join(recommendations),
            "overall_health": "healthy" if not unhealthy else "degraded"
        }

# ================================
# GLOBAL ROUTER INSTANCE
# ================================

# Global router instance that can be imported and used
global_router: Optional[PlatformRouter] = None

def get_router() -> PlatformRouter:
    """Get global router instance"""
    global global_router
    if global_router is None:
        global_router = PlatformRouter()
    return global_router

async def initialize_router(config_path: Optional[str] = None) -> PlatformRouter:
    """Initialize global router"""
    global global_router
    global_router = PlatformRouter(config_path)
    await global_router.initialize()
    return global_router

async def shutdown_router():
    """Shutdown global router"""
    global global_router
    if global_router:
        await global_router.shutdown()
        global_router = None
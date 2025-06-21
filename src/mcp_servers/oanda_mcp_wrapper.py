"""
Wrapper for BJLG-92 Oanda MCP Server
Integrates the external Oanda MCP server with our trading system
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
from pathlib import Path 

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
from src.config.logging_config import logger
from src.database.manager import db_manager
from src.database.models import EventLog, LogLevel

class OandaMCPWrapper:
    """Wrapper for BJLG-92 Oanda MCP Server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        # If they deploy their own instance, use that URL
        # Otherwise, we'll use the Railway deployment or run locally
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if Oanda MCP server is healthy"""
        if not self.session:
            raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    return {"status": "healthy", "server": "bjlg-92-oanda-mcp"}
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error("Oanda MCP server health check failed", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.session:
            raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
        try:
            async with self.session.get(f"{self.base_url}/account") as response:
                response.raise_for_status()
                data = await response.json()
                print(f"Raw Account Info: {data}")
                _data = data.get("data")
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="ACCOUNT_INFO_REQUEST",
                    message="Account information retrieved via BJLG-92 MCP",
                    context={"balance": _data.get("balance")}
                )
                
                return _data
        except Exception as e:
            logger.error("Failed to get account info", error=str(e))
            raise
    
    async def get_current_price(self, instrument: str) -> Dict[str, Any]:
        """Get current price for instrument"""
        if not self.session:
            raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
        try:
            async with self.session.get(f"{self.base_url}/price/{instrument}") as response:
                response.raise_for_status()
                data = await response.json()
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="PRICE_REQUEST",
                    message=f"Price retrieved for {instrument}",
                    context={"instrument": instrument, "price": data}
                )
                
                return data.get('data')
        except Exception as e:
            logger.error("Failed to get price", instrument=instrument, error=str(e))
            raise
    
    async def get_historical_data(
        self, 
        instrument: str, 
        timeframe: str = "M1", 
        count: int = 500
    ) -> Dict[str, Any]:
        """Get historical data for instrument"""
        if not self.session:
            raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
        try:
            params = {"timeframe": timeframe, "count": count}
            async with self.session.get(
                f"{self.base_url}/historical/{instrument}", 
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="HISTORICAL_DATA_REQUEST",
                    message=f"Historical data retrieved for {instrument}",
                    context={
                        "instrument": instrument, 
                        "timeframe": timeframe, 
                        "count": len(data.get("data", []))
                    }
                )
                
                return data
        except Exception as e:
            logger.error("Failed to get historical data", 
                        instrument=instrument, error=str(e))
            raise
    
    async def place_market_order(
        self,
        instrument: str,
        units: float,
        side: str  # "buy" or "sell"
    ) -> Dict[str, Any]:
        """Place market order"""
        try:
            order_data = {
                "instrument": instrument,
                "units": abs(units) if side == "buy" else -abs(units)
            }
            if not self.session:
                raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
            
            async with self.session.post(
                f"{self.base_url}/order/market", 
                json=order_data
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="MARKET_ORDER_PLACED",
                    message=f"Market order placed for {instrument}",
                    context={
                        "instrument": instrument,
                        "units": units,
                        "side": side,
                        "order_id": data.get("id")
                    }
                )
                
                return data
        except Exception as e:
            logger.error("Failed to place market order", 
                        instrument=instrument, error=str(e))
            raise
    
    async def place_limit_order(
        self,
        instrument: str,
        units: float,
        price: float,
        side: str
    ) -> Dict[str, Any]:
        """Place limit order"""
        try:
            order_data = {
                "instrument": instrument,
                "units": abs(units) if side == "buy" else -abs(units),
                "price": price
            }
            if not self.session:
                raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")

            async with self.session.post(
                f"{self.base_url}/order/limit",
                json=order_data
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="LIMIT_ORDER_PLACED",
                    message=f"Limit order placed for {instrument}",
                    context={
                        "instrument": instrument,
                        "units": units,
                        "price": price,
                        "side": side,
                        "order_id": data.get("id")
                    }
                )
                
                return data
        except Exception as e:
            logger.error("Failed to place limit order", 
                        instrument=instrument, error=str(e))
            raise
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        try:
            if not self.session:
                raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
            async with self.session.get(f"{self.base_url}/positions") as response:
                response.raise_for_status()
                data = await response.json()
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="POSITIONS_REQUEST",
                    message="Positions retrieved",
                    context={"position_count": len(data.get("positions", []))}
                )
                
                return data
        except Exception as e:
            logger.error("Failed to get positions", error=str(e))
            raise
    
    async def get_orders(self) -> Dict[str, Any]:
        """Get pending orders"""
        try:
            if not self.session:
                raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
            async with self.session.get(f"{self.base_url}/orders") as response:
                response.raise_for_status()
                data = await response.json()
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="ORDERS_REQUEST",
                    message="Orders retrieved",
                    context={"order_count": len(data.get("orders", []))}
                )
                
                return data
        except Exception as e:
            logger.error("Failed to get orders", error=str(e))
            raise
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        try:
            if not self.session:
                raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
            async with self.session.delete(f"{self.base_url}/order/{order_id}") as response:
                response.raise_for_status()
                data = await response.json()
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="ORDER_CANCELLED",
                    message=f"Order {order_id} cancelled",
                    context={"order_id": order_id}
                )
                
                return data
        except Exception as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            raise
    
    async def close_position(self, instrument: str) -> Dict[str, Any]:
        """Close position for instrument"""
        try:
            if not self.session:
                raise RuntimeError("aiohttp session is not initialized. Use 'async with' context.")
            async with self.session.post(f"{self.base_url}/position/close/{instrument}") as response:
                response.raise_for_status()
                data = await response.json()
                
                await self._log_event(
                    level=LogLevel.INFO,
                    event_type="POSITION_CLOSED",
                    message=f"Position closed for {instrument}",
                    context={"instrument": instrument}
                )
                
                return data
        except Exception as e:
            logger.error("Failed to close position", instrument=instrument, error=str(e))
            raise
    
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
                    agent_name="OandaMCPWrapper",
                    event_type=event_type,
                    message=message,
                    context=context or {}
                )
                session.add(event)
                await session.commit()
        except Exception as e:
            logger.error("Failed to log event to database", error=str(e))


# Global wrapper instance
oanda_mcp = OandaMCPWrapper()

# Test function
async def test_bjlg_oanda_integration(symbol_name):
    """Test the BJLG-92 Oanda MCP integration"""
    
    logger.info("🧪 Testing BJLG-92 Oanda MCP Integration...")
    
    async with OandaMCPWrapper() as oanda:
        try:
            # Test health check
            health = await oanda.health_check()
            logger.info("Health check", status=health["status"])
            
            if health["status"] != "healthy":
                logger.error("❌ BJLG-92 server not available")
                logger.info("💡 Make sure to:")
                logger.info("   1. Clone the repository")
                logger.info("   2. Set up environment variables")
                logger.info("   3. Run the server locally or use Railway deployment")
                return False
            
            # Test account info
            account = await oanda.get_account_info()
            logger.info("✅ Account info retrieved", balance=account.get("balance"))
            
            # Test price data
            
            asset_price = await oanda.get_current_price(symbol_name)
            logger.info(f"✅ {symbol_name} price retrieved", price=asset_price)
            
            # Test historical data
            historical = await oanda.get_historical_data(symbol_name, "M1", 10)
            logger.info("✅ Historical data retrieved", 
                       bars=len(historical.get("data", [])))
            
            logger.info("🎉 BJLG-92 Oanda MCP integration successful!")
            return True
            
        except Exception as e:
            logger.error("❌ Integration test failed", error=str(e))
            return False

if __name__ == "__main__":
    symbol_name = "BTC_USD"
    success = asyncio.run(test_bjlg_oanda_integration(symbol_name))
    if success:
        print("✅ Ready to integrate with CrewAI agents!")
    else:
        print("❌ Setup needed - follow setup instructions")
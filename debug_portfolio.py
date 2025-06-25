#!/usr/bin/env python
"""
Debug script to check and fix the get_portfolio_status import issue
"""

import sys
from pathlib import Path

def check_trading_tools_file():
    """Check what's actually in the trading_execution_tools.py file"""
    
    # Find the file
    tools_file = Path("src/autonomous_trading_system/tools/trading_execution_tools.py")
    
    if not tools_file.exists():
        print(f"❌ File not found: {tools_file}")
        return False
    
    print(f"✅ File found: {tools_file}")
    
    try:
        # Read the file content
        with open(tools_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for the function
        if "def get_portfolio_status" in content:
            print("✅ get_portfolio_status function definition found")
        else:
            print("❌ get_portfolio_status function definition NOT found")
        
        # Check for the @tool decorator
        if "@tool\ndef get_portfolio_status" in content.replace(" ", "").replace("\n", ""):
            print("✅ @tool decorator found for get_portfolio_status")
        else:
            print("❌ @tool decorator NOT found for get_portfolio_status")
        
        # Check for syntax errors
        try:
            compile(content, tools_file, 'exec')
            print("✅ No syntax errors found")
        except SyntaxError as e:
            print(f"❌ Syntax error found: {e}")
            print(f"   Line {e.lineno}: {e.text}")
            return False
        
        # Try to import the module
        try:
            sys.path.insert(0, str(Path.cwd() / "src"))
            import autonomous_trading_system.tools.trading_execution_tools as tools_module
            
            if hasattr(tools_module, 'get_portfolio_status'):
                print("✅ get_portfolio_status can be imported successfully")
                return True
            else:
                print("❌ get_portfolio_status attribute not found in module")
                print("Available attributes:", [attr for attr in dir(tools_module) if not attr.startswith('_')])
                return False
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def create_standalone_portfolio_status():
    """Create a standalone get_portfolio_status function"""
    
    portfolio_status_code = '''
@tool
def get_portfolio_status() -> Dict[str, Any]:
    """
    Get comprehensive portfolio status including positions, orders, account info, and risk metrics.
    This is the primary tool for overall portfolio monitoring and risk assessment.
    
    Returns:
        Dict with complete portfolio overview including exposure calculations
    """
    
    async def _get_portfolio_status():
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
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _get_portfolio_status())
                return future.result()
        else:
            return asyncio.run(_get_portfolio_status())
    except Exception as e:
        return {
            "success": False,
            "error": f"Portfolio status framework error: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
'''
    
    return portfolio_status_code

def fix_trading_tools_file():
    """Add the missing get_portfolio_status function to the file"""
    
    tools_file = Path("src/autonomous_trading_system/tools/trading_execution_tools.py")
    
    if not tools_file.exists():
        print(f"❌ File not found: {tools_file}")
        return False
    
    try:
        # Read current content
        with open(tools_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if function already exists
        if "def get_portfolio_status" in content:
            print("⚠️  get_portfolio_status already exists in file")
            return True
        
        # Add the function at the end
        portfolio_function = create_standalone_portfolio_status()
        
        # Append to file
        with open(tools_file, 'a', encoding='utf-8') as f:
            f.write("\n\n")
            f.write(portfolio_function)
        
        print("✅ get_portfolio_status function added to file")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing file: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Debugging get_portfolio_status import issue...")
    print("=" * 50)
    
    # Check the current state
    file_ok = check_trading_tools_file()
    
    if not file_ok:
        print("\n🔧 Attempting to fix the issue...")
        fix_trading_tools_file()
        
        print("\n🔍 Re-checking after fix...")
        check_trading_tools_file()
    
    print("\n✅ Debug complete!")
"""
Real-Time Paper Trading Dashboard
Streamlit dashboard for monitoring live paper trading performance
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional, Union

# Add project root to sys.path - Fix path resolution
current_file = Path(__file__).resolve()
project_root = current_file.parent

# Try multiple possible project structures
possible_roots = [
    project_root,  # Same directory as dashboard
    project_root.parent,  # One level up
    project_root.parent.parent,  # Two levels up
]

for root in possible_roots:
    src_path = root / "src"
    if src_path.exists():
        project_root = root
        break

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Now try imports with error handling
try:
    from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
    OANDA_AVAILABLE = True
    print("OANDA is available")
except ImportError as e:
    st.error(f"⚠️ Oanda MCP wrapper not available: {e}")
    OANDA_AVAILABLE = False
    print("OANDA is NOT available")

try:
    from src.database.manager import get_db_session, db_manager
    from src.database.models import AgentAction, EventLog, LogLevel
    from sqlalchemy import desc, and_
    DATABASE_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Database components not available: {e}")
    DATABASE_AVAILABLE = False

# Configure Streamlit
st.set_page_config(
    page_title="Paper Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .profit {
        color: #00ff00;
        font-weight: bold;
    }
    .loss {
        color: #ff4444;
        font-weight: bold;
    }
    .neutral {
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_live_prices() -> Dict[str, Any]:
    """Get live prices for multiple symbols - WINDOWS COMPATIBLE"""
    if not OANDA_AVAILABLE:
        return {"error": "Oanda MCP not available"}
    
    async def _get_prices():
        symbols = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
        prices = {}
        
        try:
            async with OandaMCPWrapper("http://localhost:8000") as oanda:
                for symbol in symbols:
                    try:
                        price_data = await oanda.get_current_price(symbol)
                        print("*" * 40)
                        print(f"Price Data {price_data}")
                        if isinstance(price_data, dict):
                            prices[symbol] = {
                                'bid': float(price_data.get('bid', 0)),
                                'ask': float(price_data.get('ask', 0)),
                                'spread': float(price_data.get('spread', 0)),
                                'timestamp': datetime.now()
                            }
                        else:
                            prices[symbol] = {
                                'bid': 0.0,
                                'ask': 0.0,
                                'spread': 0.0,
                                'error': f"Invalid price data type: {type(price_data)}",
                                'timestamp': datetime.now()
                            }
                    except Exception as e:
                        prices[symbol] = {
                            'bid': 0.0,
                            'ask': 0.0,
                            'spread': 0.0,
                            'error': str(e),
                            'timestamp': datetime.now()
                        }
        except Exception as e:
            return {"error": f"Connection failed: {e}"}
        
        return prices
    
    try:
        result = asyncio.run(_get_prices())
        return result if isinstance(result, dict) else {"error": "Invalid response type"}
    except Exception as e:
        return {"error": f"Failed to get live prices: {e}"}

# Replace ONLY the get_historical_chart_data function in your dashboard with this enhanced debug version:

@st.cache_data(ttl=10)
def get_historical_chart_data(symbol: str, timeframe: str = "M5", count: int = 100) -> List[Any]:
    """Get historical data for charting - WINDOWS COMPATIBLE VERSION"""
    if not OANDA_AVAILABLE:
        print("ERROR: OANDA not available")
        return []
    
    async def _get_data():
        try:
            async with OandaMCPWrapper("http://localhost:8000") as oanda:
                print(f"INFO: API Call: symbol={symbol}, timeframe={timeframe}, count={count}")
                historical_data = await oanda.get_historical_data(symbol, timeframe, count)
                print(f"INFO: Raw API Response Type: {type(historical_data)}")
                
                if isinstance(historical_data, dict):
                    print(f"INFO: Response Keys: {list(historical_data.keys())}")
                
                return historical_data
        except Exception as e:
            print(f"ERROR: API Error: {e}")
            return {"error": str(e)}
    
    try:
        print(f"\nINFO: Starting historical data request for {symbol}")
        data = asyncio.run(_get_data())
        print(f"SUCCESS: API call completed, processing response...")
        
        # Handle the EXISTING wrapper's response format
        if isinstance(data, dict):
            print("INFO: Response is a dictionary")
            
            if "error" in data:
                error_msg = f"API Error: {data['error']}"
                st.error(f"Failed to get historical data: {error_msg}")
                print(f"ERROR: {error_msg}")
                return []
            
            # Handle the ACTUAL format from your logs: {'success': True, 'data': {'candles': [...]}}
            if "success" in data and data.get("success") and "data" in data:
                nested_data = data["data"]
                print(f"SUCCESS: Found 'success' wrapper! Nested data type: {type(nested_data)}")
                
                if isinstance(nested_data, dict) and "candles" in nested_data:
                    chart_data = nested_data["candles"]
                    print(f"SUCCESS: Found nested 'candles' key! Type: {type(chart_data)}")
                    print(f"INFO: Candles count: {len(chart_data) if isinstance(chart_data, list) else 'Not a list'}")
                    
                    if isinstance(chart_data, list) and len(chart_data) > 0:
                        print(f"INFO: Sample candle: {chart_data[0]}")
                        print(f"SUCCESS: Returning {len(chart_data)} candles from existing wrapper")
                        return chart_data
                    else:
                        print("WARNING: Candles list is empty or invalid")
                        return []
                else:
                    print(f"ERROR: No 'candles' key in nested data. Available keys: {list(nested_data.keys()) if isinstance(nested_data, dict) else 'Not a dict'}")
                    return []
            
            # Fallback for other possible formats
            elif "data" in data:
                chart_data = data.get('data', [])
                if isinstance(chart_data, list):
                    print(f"SUCCESS: Returning {len(chart_data)} items from 'data' key")
                    return chart_data
                elif isinstance(chart_data, dict) and "candles" in chart_data:
                    candles = chart_data["candles"]
                    print(f"SUCCESS: Found candles in data object: {len(candles) if isinstance(candles, list) else 'Not a list'}")
                    return candles if isinstance(candles, list) else []
                else:
                    print(f"ERROR: 'data' key contains unexpected format: {type(chart_data)}")
                    return []
            
            elif "candles" in data:
                chart_data = data.get('candles', [])
                print(f"SUCCESS: Found direct 'candles' key: {len(chart_data) if isinstance(chart_data, list) else 'Not a list'}")
                return chart_data if isinstance(chart_data, list) else []
            
            else:
                available_keys = list(data.keys())
                print(f"ERROR: No expected keys found!")
                print(f"INFO: Available keys: {available_keys}")
                print(f"INFO: Full response (first 500 chars): {str(data)[:500]}...")
                return []
                
        elif isinstance(data, list):
            print(f"INFO: Response is already a list with {len(data)} items")
            return data
            
        else:
            error_msg = f"Unexpected data type: {type(data)}"
            st.error(error_msg)
            print(f"ERROR: {error_msg}")
            return []
            
    except Exception as e:
        error_msg = f"Exception in get_historical_chart_data: {e}"
        st.error(error_msg)
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return []

@st.cache_data(ttl=5)
def get_recent_agent_actions(limit: int = 10) -> List[Any]:
    """Get recent agent actions from database - always returns a list"""
    if not DATABASE_AVAILABLE:
        return []
    
    try:
        session = get_db_session()
        actions = session.query(AgentAction)\
                         .filter(AgentAction.agent_name == "PaperTradingEngine")\
                         .order_by(desc(AgentAction.timestamp))\
                         .limit(limit).all()
        session.close()
        return list(actions) if actions else []
    except Exception as e:
        st.error(f"Failed to get agent actions: {e}")
        return []

@st.cache_data(ttl=5)
def get_recent_events(limit: int = 20) -> List[Any]:
    """Get recent system events - always returns a list"""
    if not DATABASE_AVAILABLE:
        return []
    
    try:
        print("*" * 50)
        print("Getting recent events")
        print("*" * 50)
        session = get_db_session()
        events = session.query(EventLog)\
                        .filter(EventLog.agent_name == "PaperTradingEngine")\
                        .order_by(desc(EventLog.timestamp))\
                        .limit(limit)
        print(f"events sql: {events}")
        events.all()
        session.close()
        return list(events) if events else []
    except Exception as e:
        st.error(f"Failed to get events: {e}")
        return []

def create_price_chart(symbol: str, historical_data: List[Any]) -> Optional[go.Figure]:
    """Create candlestick chart - WINDOWS COMPATIBLE VERSION"""
    if not isinstance(historical_data, list) or not historical_data:
        st.warning(f"No historical data available for {symbol}")
        return None
    
    try:
        processed_data = []
        
        print(f"INFO: Processing {len(historical_data)} candles for {symbol}")
        print(f"INFO: Sample candle structure: {historical_data[0] if historical_data else 'No data'}")
        
        for i, candle in enumerate(historical_data):
            try:
                if isinstance(candle, dict):
                    # Handle the ACTUAL Oanda format from your logs:
                    # {'complete': True, 'volume': 255008, 'time': '2025-04-16T21:00:00.000000000Z', 'mid': {'o': '1.13994', 'h': '1.14094', 'l': '1.13352', 'c': '1.13646'}}
                    
                    if 'mid' in candle and 'time' in candle:
                        mid_data = candle.get('mid', {})
                        processed_data.append({
                            'time': candle.get('time', ''),
                            'open': float(mid_data.get('o', 0)),
                            'high': float(mid_data.get('h', 0)),
                            'low': float(mid_data.get('l', 0)),
                            'close': float(mid_data.get('c', 0)),
                            'volume': int(candle.get('volume', 0))
                        })
                        if i == 0:  # Debug first candle
                            print(f"SUCCESS: First candle processed successfully: {processed_data[-1]}")
                    
                    # Fallback: Handle direct OHLC format
                    elif all(key in candle for key in ['open', 'high', 'low', 'close', 'time']):
                        processed_data.append({
                            'time': candle.get('time', ''),
                            'open': float(candle.get('open', 0)),
                            'high': float(candle.get('high', 0)),
                            'low': float(candle.get('low', 0)),
                            'close': float(candle.get('close', 0)),
                            'volume': int(candle.get('volume', 0))
                        })
                        if i == 0:
                            print(f"SUCCESS: First candle processed (direct format): {processed_data[-1]}")
                    
                    # Fallback: Handle 'bid' format (if it exists)
                    elif 'bid' in candle and 'time' in candle:
                        bid_data = candle.get('bid', {})
                        processed_data.append({
                            'time': candle.get('time', ''),
                            'open': float(bid_data.get('o', 0)),
                            'high': float(bid_data.get('h', 0)),
                            'low': float(bid_data.get('l', 0)),
                            'close': float(bid_data.get('c', 0)),
                            'volume': int(candle.get('volume', 0))
                        })
                        if i == 0:
                            print(f"SUCCESS: First candle processed (bid format): {processed_data[-1]}")
                    
                    else:
                        if i == 0:  # Debug first problematic candle
                            print(f"ERROR: Unrecognized candle structure: {candle}")
                            print(f"INFO: Available keys: {list(candle.keys())}")
                            
            except (KeyError, TypeError, ValueError) as e:
                print(f"ERROR: Error processing candle {i}: {e}")
                continue
        
        print(f"SUCCESS: Successfully processed {len(processed_data)} out of {len(historical_data)} candles")
        
        if not processed_data:
            st.warning(f"Could not process data format for {symbol}")
            st.write("Debug - Sample candle:", str(historical_data[0]) if historical_data else "No data")
            return None
        
        # Create DataFrame from processed data
        df = pd.DataFrame(processed_data)
        
        # Ensure we have required columns
        required_cols = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.warning(f"Missing required data columns for {symbol}: {missing_cols}")
            return None
        
        # Convert time to datetime and ensure numeric columns
        df['time'] = pd.to_datetime(df['time'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if df.empty:
            st.warning(f"No valid data points for {symbol}")
            return None
        
        # Sort by time
        df = df.sort_values('time')
        
        print(f"SUCCESS: Final DataFrame shape: {df.shape}")
        print(f"INFO: Date range: {df['time'].min()} to {df['time'].max()}")
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )])
        
        fig.update_layout(
            title=f"{symbol} Live Chart ({len(df)} candles)",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400,
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Failed to create chart for {symbol}: {e}")
        st.write("Debug - Raw data sample:", str(historical_data[:2]) if len(historical_data) > 0 else "Empty data")
        import traceback
        traceback.print_exc()
        return None

# Windows-compatible debug function
def debug_existing_wrapper():
    """Debug function for your existing wrapper - WINDOWS COMPATIBLE"""
    st.write("### Debug: Existing Wrapper Response")
    
    if st.button("Test Existing Wrapper"):
        with st.spinner("Testing your existing wrapper..."):
            try:
                async def test_wrapper():
                    async with OandaMCPWrapper("http://localhost:8000") as oanda:
                        # Test historical data exactly as your wrapper returns it
                        response = await oanda.get_historical_data("EUR_USD", "M5", 10)
                        return response
                
                result = asyncio.run(test_wrapper())
                
                st.write("**Raw Wrapper Response:**")
                st.write(f"- Type: `{type(result)}`")
                
                if isinstance(result, dict):
                    st.write(f"- Keys: `{list(result.keys())}`")
                    
                    # Analyze the actual structure
                    if "success" in result and "data" in result:
                        data_part = result["data"]
                        st.write(f"- Data type: `{type(data_part)}`")
                        if isinstance(data_part, dict):
                            st.write(f"- Data keys: `{list(data_part.keys())}`")
                            
                            if "candles" in data_part:
                                candles = data_part["candles"]
                                st.write(f"- Candles count: `{len(candles) if isinstance(candles, list) else 'Not a list'}`")
                                
                                if isinstance(candles, list) and len(candles) > 0:
                                    st.write("**Sample candle from your wrapper:**")
                                    st.json(candles[0])
                                    
                                    # Check the actual structure we need to handle
                                    sample_candle = candles[0]
                                    if 'mid' in sample_candle:
                                        st.success("Found 'mid' format - dashboard will handle this!")
                                    elif 'bid' in sample_candle:
                                        st.success("Found 'bid' format - dashboard will handle this!")
                                    elif 'open' in sample_candle:
                                        st.success("Found direct OHLC format - dashboard will handle this!")
                                    else:
                                        st.warning("Unknown format - may need dashboard adjustment")
                
                # Show structure analysis
                with st.expander("Full Response Structure"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"Wrapper test failed: {e}")
                st.write(f"Error type: {type(e).__name__}")

def main() -> None:
    st.title("📈 Real-Time Paper Trading Dashboard")
    st.markdown("---")
    
    # Debug section for existing wrapper
    with st.expander("🔧 Debug Existing Wrapper"):
        debug_existing_wrapper()
    
    # Show system status first
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if OANDA_AVAILABLE:
            st.success("✅ Oanda MCP: Available")
        else:
            st.error("❌ Oanda MCP: Not Available")
    
    with col2:
        if DATABASE_AVAILABLE:
            st.success("✅ Database: Available")
        else:
            st.error("❌ Database: Not Available")
    
    with col3:
        st.info(f"🕐 Current Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Auto-refresh toggle
    seconds_to_refresh = 30
    auto_refresh = st.sidebar.checkbox(f"🔄 Auto Refresh ({seconds_to_refresh}s)", value=True)
    if auto_refresh:
        time.sleep(seconds_to_refresh)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔃 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Trading symbols selection
    st.sidebar.header("📊 Symbols")
    symbols = st.sidebar.multiselect(
        "Select Symbols to Monitor",
        ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"],
        default=["EUR_USD"]  # Start with just EUR_USD for testing
    )
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # Get live prices
    live_prices = get_live_prices()
    
    # Display key metrics
    with col1:
        if "error" not in live_prices:
            active_symbols = len([s for s in symbols if s in live_prices])
            st.metric(
                label="📊 Active Symbols",
                value=active_symbols,
                delta=f"of {len(symbols)} selected"
            )
        else:
            st.metric("📊 Active Symbols", "N/A", "Connection Error")
    
    with col2:
        if "error" not in live_prices and live_prices:
            avg_spread = sum(p.get('spread', 0) for p in live_prices.values()) / len(live_prices)
            st.metric(
                label="📈 Avg Spread",
                value=f"{avg_spread:.5f} pips",
                delta="Live data"
            )
        else:
            st.metric("📈 Avg Spread", "N/A", "No data")
    
    with col3:
        # Get recent agent actions count
        recent_actions = get_recent_agent_actions(limit=1)
        if recent_actions:
            last_action_time = recent_actions[0].timestamp
            time_diff = datetime.now() - last_action_time
            st.metric(
                label="🤖 Last Agent Action",
                value=f"{time_diff.seconds}s ago",
                delta="Active"
            )
        else:
            st.metric("🤖 Last Agent Action", "N/A", "No actions")
    
    with col4:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric(
            label="⏰ Current Time",
            value=current_time,
            delta="Live"
        )
    
    # Live prices table
    st.header("💰 Live Market Prices")
    
    if "error" not in live_prices and live_prices:
        price_data = []
        for symbol, data in live_prices.items():
            if symbol in symbols:
                spread_pips = data.get('spread', 0)
                price_data.append({
                    'Symbol': symbol,
                    'Bid': f"{data.get('bid', 0):.5f}",
                    'Ask': f"{data.get('ask', 0):.5f}",
                    'Spread': f"{spread_pips:.5f} pips",
                    'Last Update': data.get('timestamp', datetime.now()).strftime("%H:%M:%S")
                })
        
        if price_data:
            df_prices = pd.DataFrame(price_data)
            st.dataframe(df_prices, use_container_width=True)
        else:
            st.warning("No price data available for selected symbols")
    else:
        error_msg = live_prices.get("error", "Unknown error") if isinstance(live_prices, dict) else "Connection error"
        st.error(f"❌ Unable to fetch live prices: {error_msg}")
    
    # Charts section
    st.header("📊 Live Charts")
    
    if symbols and OANDA_AVAILABLE:
        # Create tabs for each symbol
        tabs = st.tabs([f"📈 {symbol}" for symbol in symbols])
        
        for i, symbol in enumerate(symbols):
            with tabs[i]:
                if "error" not in live_prices and symbol in live_prices and 'error' not in live_prices[symbol]:
                    
                    # Debug checkbox
                    debug_mode = st.checkbox(f"🔍 Show debug info for {symbol}", key=f"debug_{symbol}")
                    
                    # Get historical data for chart
                    historical_data = get_historical_chart_data(symbol, "M5", 50)
                    
                    if debug_mode:
                        st.write(f"**Debug Info for {symbol}:**")
                        st.write(f"- Data type: {type(historical_data)}")
                        st.write(f"- Data length: {len(historical_data) if historical_data else 0}")
                        if historical_data and len(historical_data) > 0:
                            st.write("- Sample candle:")
                            st.json(historical_data[0])
                    
                    # Ensure we have valid list data before creating chart
                    if isinstance(historical_data, list) and len(historical_data) > 0:
                        # Create and display chart
                        fig = create_price_chart(symbol, historical_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Current price info
                            current_price = live_prices[symbol]
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    f"{symbol} Bid",
                                    f"{current_price.get('bid', 0):.5f}",
                                    delta=None
                                )
                            
                            with col2:
                                st.metric(
                                    f"{symbol} Ask", 
                                    f"{current_price.get('ask', 0):.5f}",
                                    delta=None
                                )
                            
                            with col3:
                                st.metric(
                                    "Spread",
                                    f"{current_price.get('spread', 0):.5f} pips",
                                    delta=None
                                )
                        else:
                            st.error(f"Unable to create chart for {symbol}")
                            st.info("💡 Enable debug info above to see data structure")
                    else:
                        st.warning(f"No historical data available for {symbol}")
                        st.info("💡 Check your Oanda MCP connection and enable debug info")
                else:
                    st.error(f"❌ Price data unavailable for {symbol}")
    else:
        if not OANDA_AVAILABLE:
            st.error("❌ Oanda MCP not available - cannot display charts")
        else:
            st.info("📊 Select symbols from the sidebar to view charts")
    
    # Trading activity section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🤖 Recent Agent Actions")
        
        if DATABASE_AVAILABLE:
            recent_actions = get_recent_agent_actions(10)
            if recent_actions:
                for action in recent_actions:
                    with st.expander(f"🤖 {action.action_type} - {action.timestamp.strftime('%H:%M:%S')}"):
                        st.write(f"**Agent:** {action.agent_name}")
                        st.write(f"**Action:** {action.action_type}")
                        if action.confidence_score is not None:
                            st.write(f"**Confidence:** {action.confidence_score}%")
                        
                        if action.input_data is not None:
                            st.write("**Input Data:**")
                            st.json(action.input_data)
                        
                        if action.output_data is not None:
                            st.write("**Output Data:**")
                            st.json(action.output_data)
            else:
                st.info("No recent agent actions found")
        else:
            st.error("❌ Database not available - cannot show agent actions")
    
    with col2:
        st.header("📊 System Events")
        
        if DATABASE_AVAILABLE:
            recent_events = get_recent_events(10)
            if recent_events:
                for event in recent_events:
                    # Color code by level
                    if hasattr(event.level, 'value'):
                        level_value = event.level.value
                    else:
                        level_value = str(event.level)
                    
                    if level_value == "ERROR":
                        st.error(f"❌ {event.message}")
                    elif level_value == "WARNING":
                        st.warning(f"⚠️ {event.message}")
                    else:
                        st.info(f"ℹ️ {event.message}")
                    
                    st.caption(f"{event.timestamp.strftime('%H:%M:%S')} - {event.event_type}")
            else:
                st.info("No recent events found")
        else:
            st.error("❌ Database not available - cannot show events")
    
    # Paper trading performance section
    st.header("💼 Paper Trading Performance")
    
    if DATABASE_AVAILABLE:
        # Get paper trading metrics from recent events
        recent_events = get_recent_events(50)  # Get more events for analysis
        trading_events = [e for e in recent_events if "position" in e.message.lower() or "trade" in e.message.lower()]
        
        if trading_events:
            # Extract P&L data from events
            pnl_data = []
            for event in trading_events:
                if hasattr(event, 'context') and event.context is not None and 'pnl' in event.context:
                    pnl_data.append({
                        'time': event.timestamp,
                        'pnl': event.context['pnl'],
                        'symbol': event.context.get('symbol', 'Unknown'),
                        'action': event.context.get('side', 'Unknown')
                    })
            
            if pnl_data:
                df_pnl = pd.DataFrame(pnl_data)
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_pnl = df_pnl['pnl'].sum()
                winning_trades = len(df_pnl[df_pnl['pnl'] > 0])
                total_trades = len(df_pnl)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                with col1:
                    st.metric(
                        "Total P&L",
                        f"${total_pnl:.2f}",
                        delta=f"{'+' if total_pnl > 0 else ''}${total_pnl:.2f}"
                    )
                
                with col2:
                    st.metric("Total Trades", total_trades)
                
                with col3:
                    st.metric("Winning Trades", winning_trades)
                
                with col4:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                # P&L chart
                if len(df_pnl) > 1:
                    fig_pnl = px.line(
                        df_pnl, 
                        x='time', 
                        y='pnl', 
                        title="P&L Over Time",
                        labels={'pnl': 'P&L ($)', 'time': 'Time'}
                    )
                    fig_pnl.update_layout(height=300)
                    st.plotly_chart(fig_pnl, use_container_width=True)
            else:
                st.info("No P&L data available yet")
        else:
            st.info("No paper trading activity detected yet")
    else:
        st.error("❌ Database not available - cannot show performance data")
    
    # Connection status footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "error" not in live_prices:
            st.success("✅ Oanda MCP: Connected")
        else:
            st.error("❌ Oanda MCP: Disconnected")
    
    with col2:
        if DATABASE_AVAILABLE:
            try:
                # Test database connection
                session = get_db_session()
                session.close()
                st.success("✅ Database: Connected")
            except:
                st.error("❌ Database: Disconnected")
        else:
            st.error("❌ Database: Not Available")
    
    with col3:
        recent_actions = get_recent_agent_actions(1)
        if recent_actions:
            st.success("✅ Paper Trading: Active")
        else:
            st.warning("⚠️ Paper Trading: No Activity")
    
    # Footer
    st.markdown("---")
    st.markdown("**📈 Real-Time Paper Trading Dashboard** | Built with Streamlit, Oanda MCP, and CrewAI")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
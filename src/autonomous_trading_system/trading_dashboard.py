"""
Simple Trading Dashboard - Real-time monitoring of your autonomous trading system
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
from src.database.manager import get_db_session
from src.database.models import AgentAction, EventLog
from sqlalchemy import desc

# Configure Streamlit
st.set_page_config(
    page_title="Autonomous Trading System Dashboard",
    page_icon="🚀",
    layout="wide"
)

symbol_name = "BTC_USD"

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_live_price():
    """Get live BTC/USD price"""
    async def _get_price():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            return await oanda.get_current_price(symbol_name)
    
    try:
        return asyncio.run(_get_price())
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=30)
def get_account_info():
    """Get account information"""
    async def _get_account():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            return await oanda.get_account_info()
    
    try:
        return asyncio.run(_get_account())
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=60)
def get_historical_data():
    """Get historical {symbol_name} data for chart"""
    async def _get_historical():
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            return await oanda.get_historical_data(symbol_name, "M1", 100)
    
    try:
        return asyncio.run(_get_historical())
    except Exception as e:
        return {"error": str(e)}

def get_recent_agent_actions():
    """Get recent agent actions from database"""
    try:
        session = get_db_session()
        actions = session.query(AgentAction)\
            .order_by(desc(AgentAction.timestamp))\
            .limit(10)\
            .all()
        session.close()
        return actions
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

def get_recent_events():
    """Get recent system events"""
    try:
        session = get_db_session()
        events = session.query(EventLog)\
            .order_by(desc(EventLog.timestamp))\
            .limit(20)\
            .all()
        session.close()
        return events
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🚀 Autonomous Trading System Dashboard")
    st.markdown("### Real-time monitoring of your AI trading agents")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    
    if auto_refresh:
        # Auto-refresh every 30 seconds
        import time
        placeholder = st.empty()
        
        # This will refresh the page every 30 seconds
        st.markdown("""
        <script>
        setTimeout(function() {
            window.location.reload();
        }, 30000);
        </script>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Live Price Display
    with col1:
        st.subheader(f"💰 Live {symbol_name}")
        price_data = get_live_price()
        print(f"PriceDate: {price_data}")
        if "error" not in price_data:
            bid = price_data.get("bid", 0)
            ask = price_data.get("ask", 0)
            spread = price_data.get("spread", 0)
            
            st.metric("Bid", f"{float(bid):.5f}")
            st.metric("Ask", f"{float(ask):.5f}")
            st.metric("Spread", f"{float(spread):.5f}")
            
            # Price indicator
            mid_price = (float(bid) + float(ask)) / 2
            st.markdown(f"**Mid Price: {float(mid_price):.5f}**")
        else:
            st.error("❌ Unable to fetch live price")
    
    # Account Information
    with col2:
        st.subheader("💳 Account Status")
        account_data = get_account_info()
        print(f"AccountData:  {account_data}")
        if "error" not in account_data:
            balance = account_data.get("balance", 0)
            currency = account_data.get("currency", "USD")
            margin_used = account_data.get("margin_used", 0)
            margin_available = account_data.get("margin_available", 0)
            
            st.metric("Balance", f"{float(balance):,.2f} {currency}")
            st.metric("Margin Used", f"{float(margin_used):,.2f}")
            st.metric("Margin Available", f"{float(margin_available):,.2f}")
            
            # Account health indicator
            if float(margin_available) > float(balance) * 0.5:
                st.success("✅ Account Healthy")
            else:
                st.warning("⚠️ Monitor Margin")
        else:
            st.error("❌ Unable to fetch account info")
    
    # System Status
    with col3:
        st.subheader("🤖 Agent Status")
        
        # Check if agents are active (recent actions)
        recent_actions = get_recent_agent_actions()
        
        if recent_actions:
            latest_action = recent_actions[0]
            time_diff = datetime.utcnow() - latest_action.timestamp
            
            st.metric("Last Analysis", f"{time_diff.seconds // 60} min ago")
            st.metric("Agent Actions", len(recent_actions))
            
            if time_diff.seconds < 300:  # 5 minutes
                st.success("✅ Agents Active")
            else:
                st.warning("⚠️ Agents Idle")
        else:
            st.info("🔄 No recent agent activity")
        
        # Manual analysis trigger
        if st.button("🧠 Run Analysis Now"):
            st.info("Analysis would be triggered here...")
    
    # Price Chart
    st.subheader(f"📈 {symbol_name} Price Chart (Last 100 Minutes)")
    
    historical_data = get_historical_data()
    print(f"Historical data structure: {historical_data}")
    
    if "error" not in historical_data and historical_data.get("data"):
        try:
            # Handle nested data structure from Oanda
            data = historical_data["data"]
            
            # Check if data has 'candles' key (Oanda format)
            if isinstance(data, dict) and "candles" in data:
                candles = data["candles"]
                df = pd.DataFrame(candles)
                
                # Extract OHLC data from 'mid' object if it exists
                if 'mid' in df.columns:
                    df['open'] = df['mid'].apply(lambda x: float(x.get('o', 0)) if x else 0)
                    df['high'] = df['mid'].apply(lambda x: float(x.get('h', 0)) if x else 0)
                    df['low'] = df['mid'].apply(lambda x: float(x.get('l', 0)) if x else 0)
                    df['close'] = df['mid'].apply(lambda x: float(x.get('c', 0)) if x else 0)
                    df['timestamp'] = pd.to_datetime(df['time'])
                else:
                    # Direct OHLC columns
                    df['timestamp'] = pd.to_datetime(df['time'])
                    df['open'] = df['open'].astype(float)
                    df['high'] = df['high'].astype(float)
                    df['low'] = df['low'].astype(float)
                    df['close'] = df['close'].astype(float)
            else:
                # Direct DataFrame creation
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['time'])
            
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame shape: {df.shape}")
            
            # Create candlestick chart
            fig = go.Figure(data=go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'], 
                low=df['low'],
                close=df['close'],
                name=symbol_name
            ))
            
            fig.update_layout(
                title=f"{symbol_name} 1-Minute Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error processing chart data: {e}")
            st.write("Raw data structure:", historical_data)
    else:
        st.error("❌ Unable to load chart data")
        if "error" in historical_data:
            st.write("Error details:", historical_data["error"])
    
    # Recent Agent Actions
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🤖 Recent Agent Actions")
        
        recent_actions = get_recent_agent_actions()
        if recent_actions:
            for action in recent_actions[:5]:
                with st.expander(f"🤖 {action.agent_name} - {action.action_type}"):
                    st.write(f"**Time:** {action.timestamp}")
                    st.write(f"**Agent:** {action.agent_name}")
                    st.write(f"**Action:** {action.action_type}")
                    if action.confidence_score is not None:
                        st.write(f"**Confidence:** {action.confidence_score}%")
                    
                    if action.output_data is not None:
                        st.json(action.output_data)
        else:
            st.info("No recent agent actions found")
    
    # Recent System Events
    with col2:
        st.subheader("📊 System Events")
        
        recent_events = get_recent_events()
        if recent_events:
            for event in recent_events[:5]:
                
                # Color code by level
                if str(event.level) == "ERROR":
                    st.error(f"❌ {event.message}")
                elif str(event.level) == "WARNING":
                    st.warning(f"⚠️ {event.message}")
                else:
                    st.info(f"ℹ️ {event.message}")
                
                st.caption(f"{event.timestamp} - {event.event_type}")
        else:
            st.info("No recent events found")
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Autonomous Trading System** | Built with CrewAI, Oanda MCP, and Streamlit")
    st.markdown(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
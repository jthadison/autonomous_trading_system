"""
Real-Time Paper Trading Dashboard
Streamlit dashboard for monitoring live paper trading performance
Updated to use Direct Oanda API instead of MCP wrapper
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

# UPDATED: Import Direct API instead of MCP wrapper
try:
    from src.mcp_servers.oanda_direct_api import OandaDirectAPI
    OANDA_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Oanda Direct API not available: {e}")
    OANDA_AVAILABLE = False

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
    """Get live prices for multiple symbols using Direct API"""
    if not OANDA_AVAILABLE:
        return {"error": "Oanda Direct API not available"}
    
    async def _get_prices():
        symbols = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
        prices = {}
        
        try:
            async with OandaDirectAPI() as oanda:
                for symbol in symbols:
                    try:
                        price_data = await oanda.get_current_price(symbol)
                        prices[symbol] = {
                            "bid": price_data.get("bid", 0),
                            "ask": price_data.get("ask", 0),
                            "spread": price_data.get("ask", 0) - price_data.get("bid", 0),
                            "timestamp": datetime.now().isoformat()
                        }
                    except Exception as e:
                        prices[symbol] = {"error": str(e)}
                        
        except Exception as e:
            return {"error": f"Direct API connection failed: {e}"}
        
        return prices
    
    try:
        return asyncio.run(_get_prices())
    except Exception as e:
        return {"error": f"Failed to get live prices: {e}"}

@st.cache_data(ttl=10)  # Cache for 10 seconds
def get_chart_data(symbol: str = "EUR_USD") -> Dict[str, Any]:
    """Get historical data for chart using Direct API"""
    if not OANDA_AVAILABLE:
        return {"error": "Oanda Direct API not available"}
    
    async def _get_chart():
        try:
            async with OandaDirectAPI() as oanda:
                historical = await oanda.get_historical_data(symbol, "M5", 100)
                return historical
        except Exception as e:
            return {"error": str(e)}
    
    try:
        return asyncio.run(_get_chart())
    except Exception as e:
        return {"error": f"Failed to get chart data: {e}"}

@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_recent_agent_actions(limit: int = 10) -> List[Dict]:
    """Get recent agent actions from database"""
    if not DATABASE_AVAILABLE:
        return []
    
    try:
        session = get_db_session()
        actions = session.query(AgentAction)\
            .filter(AgentAction.agent_name == "PaperTradingEngine")\
            .order_by(AgentAction.timestamp.desc())\
            .limit(limit).all()
        
        result = []
        for action in actions:
            result.append({
                "timestamp": action.timestamp,
                "action_type": action.action_type,
                "input_data": action.input_data,
                "output_data": action.output_data
            })
        
        session.close()
        return result
        
    except Exception as e:
        st.error(f"Database query failed: {e}")
        return []

def main():
    """Main dashboard function"""
    
    # Header
    st.title("📈 Real-Time Paper Trading Dashboard")
    st.markdown("**Live monitoring of autonomous trading system with Direct Oanda API**")
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5s)", value=True)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Current time
    st.sidebar.markdown(f"**🕐 Current Time:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Live Market Data")
        
        # Get live prices
        live_prices = get_live_prices()
        
        if "error" not in live_prices:
            # Create price table
            price_data = []
            for symbol, data in live_prices.items():
                if "error" not in data:
                    price_data.append({
                        "Symbol": symbol,
                        "Bid": f"{data['bid']:.5f}",
                        "Ask": f"{data['ask']:.5f}",
                        "Spread": f"{data['spread']:.5f}",
                        "Time": data['timestamp'].split('T')[1][:8]
                    })
            
            if price_data:
                df_prices = pd.DataFrame(price_data)
                st.dataframe(df_prices, use_container_width=True)
            else:
                st.warning("No price data available")
        else:
            st.error(f"❌ {live_prices['error']}")
    
    with col2:
        st.header("⚡ Quick Stats")
        
        # System status metrics
        if "error" not in live_prices:
            st.metric("📡 Data Feed", "🟢 Connected")
            st.metric("📊 Symbols", len(live_prices))
        else:
            st.metric("📡 Data Feed", "🔴 Disconnected")
            st.metric("📊 Symbols", "0")
        
        # Last update time
        st.metric("🕐 Last Update", datetime.now().strftime("%H:%M:%S"))
    
    # Live Price Chart
    st.header("📈 Live Price Chart")
    
    # Symbol selector
    chart_symbol = st.selectbox("Select Symbol", ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"])
    
    # Get chart data
    chart_data = get_chart_data(chart_symbol)
    
    if "error" not in chart_data and "data" in chart_data:
        candles = chart_data["data"]
        
        if candles:
            # Convert to DataFrame
            df_chart = pd.DataFrame(candles)
            
            # Ensure we have the right columns
            if all(col in df_chart.columns for col in ['timestamp', 'open', 'high', 'low', 'close']):
                # Convert timestamp
                df_chart['timestamp'] = pd.to_datetime(df_chart['timestamp'])
                
                # Create candlestick chart
                fig = go.Figure(data=go.Candlestick(
                    x=df_chart['timestamp'],
                    open=df_chart['open'],
                    high=df_chart['high'],
                    low=df_chart['low'],
                    close=df_chart['close'],
                    name=chart_symbol
                ))
                
                fig.update_layout(
                    title=f"{chart_symbol} - 5 Minute Chart",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=400,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Invalid chart data format for {chart_symbol}")
        else:
            st.warning(f"No chart data available for {chart_symbol}")
    else:
        st.error(f"❌ Chart data error: {chart_data.get('error', 'Unknown error')}")
    
    # Trading Activity Section
    st.header("🤖 Trading Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Agent Actions")
        
        recent_actions = get_recent_agent_actions(5)
        
        if recent_actions:
            for action in recent_actions:
                with st.expander(f"{action['action_type']} - {action['timestamp'].strftime('%H:%M:%S')}"):
                    st.write(f"**Type:** {action['action_type']}")
                    st.write(f"**Time:** {action['timestamp']}")
                    
                    if action['input_data']:
                        try:
                            input_data = json.loads(action['input_data'])
                            st.write("**Input:**")
                            st.json(input_data)
                        except:
                            st.write(f"**Input:** {action['input_data']}")
                    
                    if action['output_data']:
                        try:
                            output_data = json.loads(action['output_data'])
                            st.write("**Output:**")
                            st.json(output_data)
                        except:
                            st.write(f"**Output:** {action['output_data']}")
        else:
            st.info("No recent trading activity")
    
    with col2:
        st.subheader("📊 Performance Summary")
        
        # Get trading statistics from recent actions
        trade_actions = [a for a in recent_actions if a['action_type'] in ['TRADE_EXECUTED', 'POSITION_CLOSED']]
        
        if trade_actions:
            # Calculate basic stats
            total_trades = len([a for a in trade_actions if a['action_type'] == 'TRADE_EXECUTED'])
            closed_positions = len([a for a in trade_actions if a['action_type'] == 'POSITION_CLOSED'])
            
            st.metric("📈 Total Trades", total_trades)
            st.metric("📊 Closed Positions", closed_positions)
            
            # Calculate P&L from closed positions
            total_pnl = 0
            winning_trades = 0
            
            for action in trade_actions:
                if action['action_type'] == 'POSITION_CLOSED':
                    try:
                        output_data = json.loads(action['output_data'])
                        pnl = output_data.get('realized_pnl', 0)
                        total_pnl += pnl
                        if pnl > 0:
                            winning_trades += 1
                    except:
                        pass
            
            if closed_positions > 0:
                win_rate = (winning_trades / closed_positions) * 100
            else:
                win_rate = 0
            
            st.metric("💰 Total P&L", f"${total_pnl:.2f}", delta=f"${total_pnl:.2f}")
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
            
            # P&L breakdown
            if closed_positions > 0:
                col_win, col_loss = st.columns(2)
                with col_win:
                    st.metric("✅ Winning", winning_trades)
                with col_loss:
                    st.metric("❌ Losing", closed_positions - winning_trades)
        else:
            st.info("No trading statistics available yet")
    
    # Performance Analytics
    if DATABASE_AVAILABLE:
        st.header("📊 Performance Analytics")
        
        # Get extended trading history for analysis
        extended_actions = get_recent_agent_actions(50)
        closed_trades = [a for a in extended_actions if a['action_type'] == 'POSITION_CLOSED']
        
        if closed_trades:
            # Create P&L timeline
            pnl_data = []
            cumulative_pnl = 0
            
            for action in reversed(closed_trades):  # Reverse to get chronological order
                try:
                    output_data = json.loads(action['output_data'])
                    pnl = output_data.get('realized_pnl', 0)
                    cumulative_pnl += pnl
                    
                    pnl_data.append({
                        'time': action['timestamp'],
                        'pnl': pnl,
                        'cumulative_pnl': cumulative_pnl
                    })
                except:
                    pass
            
            if pnl_data:
                df_pnl = pd.DataFrame(pnl_data)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_pnl = df_pnl['pnl'].sum()
                total_trades = len(df_pnl)
                winning_trades = len(df_pnl[df_pnl['pnl'] > 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
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
                        y='cumulative_pnl', 
                        title="Cumulative P&L Over Time",
                        labels={'cumulative_pnl': 'Cumulative P&L ($)', 'time': 'Time'}
                    )
                    fig_pnl.update_layout(height=300)
                    st.plotly_chart(fig_pnl, use_container_width=True)
            else:
                st.info("No P&L data available yet")
        else:
            st.info("No trading performance data available yet")
    else:
        st.error("❌ Database not available - cannot show performance data")
    
    # Connection status footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "error" not in live_prices:
            st.success("✅ Oanda Direct API: Connected")
        else:
            st.error("❌ Oanda Direct API: Disconnected")
    
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
    st.markdown("**📈 Real-Time Paper Trading Dashboard** | Built with Streamlit, Direct Oanda API, and CrewAI")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
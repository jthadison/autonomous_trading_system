# 📈 Paper Trading System - Setup & Usage Guide

## 🎯 Overview

Your **Real-Time Paper Trading System** integrates:
- **Live Oanda market data** via MCP wrapper
- **CrewAI agents** for Wyckoff analysis
- **Virtual trading engine** with no real money risk
- **Real-time Streamlit dashboard** for monitoring
- **Database logging** for performance tracking

---

## 🛠️ Installation & Setup

### **Step 1: Prerequisites**

Ensure you have the paper trading files in your project:
```
├── paper_trading_system.py          # Core trading engine
├── paper_trading_dashboard.py       # Streamlit dashboard  
├── paper_trading_launcher.py        # Control script
└── README_paper_trading.md          # This guide
```

### **Step 2: Install Dependencies**

```bash
# Core dependencies
pip install streamlit plotly pandas numpy
pip install crewai langchain-anthropic
pip install aiohttp sqlalchemy asyncio

# Optional: for better charts
pip install plotly-dash
```

### **Step 3: Environment Variables**

Create or update your `.env` file:
```bash
# Required for CrewAI agents
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional fallback
OPENAI_API_KEY=your_openai_api_key_here

# Database (if using custom)
DATABASE_URL=your_database_url_here
```

### **Step 4: Start Oanda MCP Server**

Ensure your BJLG-92 Oanda MCP server is running:
```bash
# In a separate terminal
cd oanda-mcp-server
python server.py
# Should run on http://localhost:8000
```

### **Step 5: Verify Setup**

```bash
python paper_trading_launcher.py check
```

---

## 🚀 Quick Start

### **Option 1: Complete System (Recommended)**
```bash
python paper_trading_launcher.py start
```
This starts:
- Paper trading engine with live analysis
- Real-time dashboard at http://localhost:8501

### **Option 2: Dashboard Only**
```bash
python paper_trading_launcher.py dashboard
```
Useful for monitoring without active trading.

### **Option 3: Interactive Mode**
```bash
python paper_trading_launcher.py
```
Follow the menu prompts.

---

## 📊 Dashboard Features

### **Live Market Data**
- Real-time prices for EUR_USD, GBP/USD, USD/JPY, AUD/USD
- Live candlestick charts with 5-minute timeframe
- Bid/Ask spreads and market connectivity status

### **Trading Activity Monitor**
- Recent CrewAI agent decisions
- Paper trade executions and closures
- Position management and P&L tracking

### **Performance Analytics**
- Total P&L and win rate
- Trade history and statistics
- Real-time account equity tracking

### **System Status**
- Oanda MCP connection health
- Database connectivity
- Agent activity monitoring

---

## 🤖 Trading Engine Features

### **Automated Analysis**
- **Market Monitor Agent**: Identifies trends and patterns
- **Wyckoff Specialist**: Applies Wyckoff methodology
- **Risk Manager**: Calculates position sizing
- **Trading Coordinator**: Makes final decisions

### **Risk Management**
- 2% risk per trade maximum
- Automatic stop-loss and take-profit
- Position size based on account balance
- Maximum margin utilization limits

### **Trading Logic**
```python
# Example trading decision flow:
1. Get live market data every 60 seconds
2. Analyze with CrewAI agents
3. Generate trading signal (BUY/SELL/HOLD)
4. Execute paper trade if confidence > 75%
5. Monitor position and manage exits
6. Log all activity to database
```

---

## 📋 Monitoring & Control

### **Real-Time Status**
```bash
python paper_trading_launcher.py status
```

### **Stop System**
```bash
python paper_trading_launcher.py stop
# Or press Ctrl+C in the running terminal
```

### **Dashboard Access**
- **URL**: http://localhost:8501
- **Auto-refresh**: Every 5 seconds
- **Manual refresh**: Use sidebar button

---

## 🔧 Configuration Options

### **Trading Parameters** (in `paper_trading_system.py`)
```python
# Modify these settings:
initial_balance = 100000.0        # Starting capital
analysis_interval = 60            # Analysis frequency (seconds)
trading_symbols = ["EUR_USD", "GBP_USD", "USD_JPY"]
risk_per_trade = 0.02             # 2% risk per trade
confidence_threshold = 75         # Minimum signal confidence
```

### **Dashboard Settings** (in `paper_trading_dashboard.py`)
```python
# Cache durations
live_price_cache = 5              # Seconds
chart_data_cache = 10             # Seconds
agent_action_cache = 5            # Seconds
```

---

## 📈 Understanding the Output

### **Console Output**
```
🚀 Starting Paper Trading Engine...
✅ Oanda MCP connection established
📊 Processing bar 50/200 - Price: 1.10250
🎯 Trade: buy @ 1.10250 (confidence: 82%)
📊 Position closed: EUR_USD BUY P&L: $45.50 (take_profit)

📊 PAPER TRADING STATUS - 14:30:15
   💰 Balance: $100,045.50
   📈 Equity: $100,045.50
   📊 Unrealized P&L: $0.00
   🔢 Open Positions: 0
   📈 Total Trades: 1
   🎯 Win Rate: 100.0%
   💹 Total P&L: $45.50
```

### **Dashboard Metrics**
- **Balance**: Virtual account balance
- **Equity**: Balance + unrealized P&L
- **Win Rate**: Percentage of profitable trades
- **Drawdown**: Maximum equity decline
- **Sharpe Ratio**: Risk-adjusted returns

---

## 🔍 Troubleshooting

### **Common Issues**

**1. "Oanda MCP server not running"**
```bash
# Check if BJLG-92 server is running
curl http://localhost:8000
# Should return server status
```

**2. "No LLM available"**
```bash
# Check environment variables
echo $ANTHROPIC_API_KEY
# Ensure it's set and valid
```

**3. "Database connection failed"**
```bash
# Check database setup
python -c "from src.database.manager import db_manager; print('DB OK')"
```

**4. "Dashboard won't start"**
```bash
# Check Streamlit installation
streamlit --version
# Ensure port 8501 is available
```

**5. "No trading signals generated"**
```bash
# Check agent configuration and market data
# Verify symbols are available from Oanda MCP
```

### **Debug Mode**

Enable detailed logging:
```python
# In paper_trading_system.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Manual Testing**

Test individual components:
```bash
# Test Oanda connection
python -c "
import asyncio
from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
async def test():
    async with OandaMCPWrapper() as oanda:
        price = await oanda.get_current_price('EUR_USD')
        print(price)
asyncio.run(test())
"

# Test CrewAI agents
python paper_trading_system.py
# Choose option 2: Test components only
```

---

## 📊 Performance Analysis

### **Key Metrics to Monitor**

1. **Win Rate**: Aim for >60%
2. **Profit Factor**: Should be >1.5
3. **Maximum Drawdown**: Keep <10%
4. **Sharpe Ratio**: Target >1.0
5. **Average Trade Duration**: Monitor for optimization

### **Database Queries**

Access trade data directly:
```python
from src.database.manager import get_db_session
from src.database.models import AgentAction, EventLog

session = get_db_session()

# Get recent trades
trades = session.query(AgentAction)\
    .filter(AgentAction.action_type == "TRADE_EXECUTED")\
    .order_by(AgentAction.timestamp.desc())\
    .limit(10).all()

for trade in trades:
    print(f"Trade: {trade.output_data}")
```

---

## 🎯 Next Steps

### **Optimization Opportunities**

1. **Strategy Refinement**
   - Adjust Wyckoff analysis parameters
   - Fine-tune confidence thresholds
   - Optimize position sizing

2. **Risk Management Enhancement**
   - Implement correlation limits
   - Add volatility-based position sizing
   - Dynamic stop-loss adjustment

3. **Multi-Timeframe Analysis**
   - Combine M5, M15, H1 signals
   - Trend confirmation across timeframes
   - Better entry timing

4. **Portfolio Management**
   - Multiple symbol allocation
   - Currency exposure limits
   - Drawdown protection

### **Advanced Features to Add**

- **Email/SMS alerts** for significant events
- **Performance reporting** (daily/weekly summaries)
- **Strategy comparison** (A/B testing)
- **Machine learning integration** for signal enhancement
- **Risk budgeting** across multiple strategies

---

## 🆘 Support & Resources

### **Documentation**
- [CrewAI Documentation](https://docs.crewai.com/)
- [Oanda API Reference](https://developer.oanda.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### **Wyckoff Resources**
- **Books**: "The Wyckoff Methodology in Depth" by Rubén Villahermosa
- **Patterns**: Accumulation, Distribution, Springs, Upthrusts
- **Volume Analysis**: VPOC, VAH, VAL concepts

### **Getting Help**
1. Check the troubleshooting section above
2. Review console output for error messages
3. Test individual components separately
4. Verify all prerequisites are met

---

## ⚠️ Important Disclaimers

- **Paper Trading Only**: This system uses virtual money only
- **Educational Purpose**: For learning and strategy development
- **No Financial Advice**: Not intended as investment advice
- **Market Risk**: Real trading involves substantial risk
- **Data Dependency**: Requires reliable internet and data feeds

---

## 🎉 Congratulations!

You now have a **sophisticated paper trading system** that:
- ✅ Trades with live market data
- ✅ Uses AI agents for analysis
- ✅ Manages risk automatically
- ✅ Provides real-time monitoring
- ✅ Tracks performance comprehensively

**Happy paper trading!** 📈🚀
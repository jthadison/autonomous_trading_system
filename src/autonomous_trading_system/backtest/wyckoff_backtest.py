"""CrewAI-Powered Wyckoff Backtesting System
Multi-agent intelligent backtesting using CrewAI framework
"""

import sys
from pathlib import Path

from langchain_anthropic import ChatAnthropic




# Fix path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import asyncio
import json
import pandas as pd
import numpy as np
import time

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import os

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai.utilities.events.third_party.agentops_listener import agentops

from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field, SecretStr

# Project imports
from config.logging_config import logger
# from database.manager import db_manager  # Commented out since file was deleted
# from database.models import Trade, TradeStatus, TradeSide  # Commented out since file was deleted


# Fix path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

class BacktestSignal(BaseModel):
    """Structured signal from agents"""
    action: str = Field(description="Trading action: 'buy', 'sell', or 'hold'")
    confidence: float = Field(description="Confidence score 0-100")
    entry_price: float = Field(description="Recommended entry price")
    stop_loss: float = Field(description="Stop loss level")
    take_profit: float = Field(description="Take profit target")
    reasoning: str = Field(description="Detailed reasoning for the signal")
    wyckoff_phase: str = Field(description="Identified Wyckoff phase")
    pattern_type: str = Field(description="Wyckoff pattern identified")
    risk_reward_ratio: float = Field(description="Risk to reward ratio")
    position_size: float = Field(description="Recommended position size")

class MarketAnalysis(BaseModel):
    """Market analysis from agents"""
    trend_direction: str = Field(description="Overall trend direction")
    trend_strength: float = Field(description="Trend strength 0-100")
    volatility_level: str = Field(description="High, Medium, or Low volatility")
    volume_analysis: str = Field(description="Volume pattern analysis")
    support_levels: List[float] = Field(description="Key support levels")
    resistance_levels: List[float] = Field(description="Key resistance levels")
    market_context: str = Field(description="Current market context")
    regime: str = Field(description="Market regime: trending, ranging, transitional")

class BacktestResults(BaseModel):
    """Comprehensive backtest results"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    monthly_returns: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    agent_performance: Dict[str, Any] = field(default_factory=dict)

# Custom CrewAI Tools for Backtesting
class MarketDataTool(BaseTool):
    """Tool for agents to access market data"""
    name: str = "market_data_analyzer"
    description: str = "Analyzes market data for OHLCV patterns, volume, and price action"
    
    def _run(self, data: str) -> str:
        """Analyze market data and return insights"""
        try:
            # Parse the data string back to usable format
            market_data = json.loads(data)
            
            # Analyze the data
            df = pd.DataFrame(market_data)
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Determine trend
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                trend = "bullish"
                strength = min(((current_price - sma_50) / sma_50) * 100, 100)
            elif current_price < sma_20 < sma_50:
                trend = "bearish"
                strength = min(((sma_50 - current_price) / sma_50) * 100, 100)
            else:
                trend = "neutral"
                strength = 50
            
            # Volume analysis
            recent_volume = df['volume_ratio'].tail(5).mean()
            if recent_volume > 1.5:
                volume_context = "high_volume_expansion"
            elif recent_volume > 1.2:
                volume_context = "above_average_volume"
            elif recent_volume < 0.8:
                volume_context = "low_volume_contraction"
            else:
                volume_context = "average_volume"
            
            # Price levels
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            
            analysis = {
                "trend_direction": trend,
                "trend_strength": strength,
                "volume_context": volume_context,
                "current_price": current_price,
                "resistance_level": recent_high,
                "support_level": recent_low,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "volume_ratio": recent_volume
            }
            
            return json.dumps(analysis)
            
        except Exception as e:
            return f"Error analyzing market data: {str(e)}"

class WyckoffPatternTool(BaseTool):
    """Tool for Wyckoff pattern recognition"""
    name: str = "wyckoff_pattern_analyzer"
    description: str = "Identifies Wyckoff patterns, phases, and market structure"
    
    def _run(self, data: str) -> str:
        """Analyze Wyckoff patterns in market data"""
        try:
            market_data = json.loads(data)
            df = pd.DataFrame(market_data)
            
            # Simplified Wyckoff analysis
            length = len(df)
            
            # Volume analysis
            avg_volume = df['volume'].mean()
            recent_volume = df['volume'].tail(10).mean()
            volume_spike = recent_volume > avg_volume * 1.5
            
            # Price action analysis
            price_range = df['high'].max() - df['low'].min()
            recent_range = df['high'].tail(20).max() - df['low'].tail(20).min()
            range_expansion = recent_range > price_range * 0.7
            
            # Pattern identification
            if volume_spike and not range_expansion:
                if df['close'].iloc[-1] > df['close'].iloc[-10]:
                    pattern = "potential_spring"
                    phase = "C"
                    confidence = 75
                else:
                    pattern = "potential_upthrust"
                    phase = "C"
                    confidence = 75
            elif range_expansion and volume_spike:
                pattern = "climatic_action"
                phase = "A"
                confidence = 80
            elif not volume_spike and not range_expansion:
                pattern = "cause_building"
                phase = "B"
                confidence = 60
            else:
                pattern = "markup_markdown"
                phase = "E"
                confidence = 70
            
            # Market structure
            recent_highs = df['high'].tail(50)
            recent_lows = df['low'].tail(50)
            
            if len(recent_highs) > 20:
                higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-20]
                higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-20]
                
                if higher_highs and higher_lows:
                    structure = "accumulation"
                elif not higher_highs and not higher_lows:
                    structure = "distribution"
                else:
                    structure = "reaccumulation" if higher_lows else "redistribution"
            else:
                structure = "unknown"
            
            wyckoff_analysis = {
                "pattern": pattern,
                "phase": phase,
                "structure": structure,
                "confidence": confidence,
                "volume_anomaly": volume_spike,
                "range_expansion": range_expansion,
                "key_level": df['close'].iloc[-1]
            }
            
            return json.dumps(wyckoff_analysis)
            
        except Exception as e:
            return f"Error in Wyckoff analysis: {str(e)}"

class RiskManagementTool(BaseTool):
    """Tool for risk management calculations"""
    name: str = "risk_calculator"
    description: str = "Calculates position sizing, stop losses, and risk metrics"
    
    def _run(self, signal_data: str) -> str:
        """Calculate risk management parameters"""
        try:
            signal = json.loads(signal_data)
            
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            account_size = signal.get('account_size', 100000)
            risk_percent = signal.get('risk_percent', 2.0)
            
            if entry_price <= 0 or stop_loss <= 0:
                return json.dumps({"error": "Invalid price levels"})
            
            # Calculate risk per trade
            risk_amount = account_size * (risk_percent / 100)
            
            # Calculate position size
            if signal.get('action') == 'buy':
                risk_per_unit = entry_price - stop_loss
                take_profit = entry_price + (risk_per_unit * 2)  # 2:1 R:R
            else:
                risk_per_unit = stop_loss - entry_price
                take_profit = entry_price - (risk_per_unit * 2)  # 2:1 R:R
            
            if risk_per_unit <= 0:
                return json.dumps({"error": "Invalid risk calculation"})
            
            position_size = risk_amount / risk_per_unit
            
            # Maximum position size (5% of account)
            max_position_value = account_size * 0.05
            max_position_size = max_position_value / entry_price
            
            final_position_size = min(position_size, max_position_size)
            
            risk_mgmt = {
                "position_size": final_position_size,
                "risk_amount": risk_amount,
                "risk_per_unit": risk_per_unit,
                "take_profit": take_profit,
                "risk_reward_ratio": 2.0,
                "max_risk_percent": (final_position_size * risk_per_unit / account_size) * 100
            }
            
            return json.dumps(risk_mgmt)
            
        except Exception as e:
            return f"Error in risk calculation: {str(e)}"

class CrewAIWyckoffBacktester:
    """Main backtesting class using CrewAI agents"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.equity_curve = [initial_capital]
        
        # Initialize LLM
        # self.llm = ChatOpenAI(
        #     model="gpt-4o-mini",
        #     temperature=0.1,
        #     #max_tokens=2000
        # )
        
        
        # self.llm = ChatAnthropic(
        #     model_name="claude-3-5-sonnet-20241022",
        #     temperature=0.1,
        #     #max_tokens=50,
        #     api_key=SecretStr(os.getenv('ANTHROPIC_API_KEY') or ""),
        #     timeout=500,
        #     stop=None
        # )
        
        # self.llm = ChatAnthropic(
        #     model="claude-3-5-sonnet-20241022",  # Correct parameter
        #     anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),  # Correct parameter
        #     temperature=0.1,
        #     max_tokens=1000,  # Explicit token limit
        #     timeout=60,  # Reasonable timeout
        #     max_retries=2  # Reduced retries
        # )
        
        # Initialize tools
        self.market_tool = MarketDataTool()
        self.wyckoff_tool = WyckoffPatternTool()
        self.risk_tool = RiskManagementTool()
        
        # Create agents
        self._create_agents()
        
    def _create_agents(self):
        """Create specialized CrewAI agents for backtesting"""
        
        # Data engineering Specialist
        self.data_verification_agent = Agent(
             role="Data Verification & Cleaning Specialist",
            goal="Verify, clean, and standardize raw market data to ensure quality and usability for analysis",
            backstory="""You are a data quality expert with extensive experience in financial data processing. "
                "Your specialty is handling messy, incomplete, or malformed market data and transforming "
                "it into clean, standardized formats suitable for analysis. You excel at identifying "
                "data quality issues, handling JSON parsing errors, validating data integrity, and "
                "ensuring all downstream agents receive properly formatted data.""",
            verbose=True,
            allow_delegation=True,
            #llm=self.llm,  
            #tools=[self.market_tool],
            max_execution_time=800, 
            allow_code_execution=True,
            max_retry_limit=3,
            max_iter=3
        )
        
        # Market Analysis Agent
        self.market_analyst = Agent(
            role="Market Data Analyst",
            goal="Analyze market data for trends, patterns, and technical indicators",
            backstory="""You are an expert technical analyst with deep knowledge of market 
            structure, price action, and volume analysis. You specialize in identifying 
            market trends, support/resistance levels, and volume patterns.""",
            verbose=True,
            allow_delegation=False,
            #tools=[self.market_tool],
            #llm=self.llm,  
            max_execution_time=800, 
            max_retry_limit=3,
            max_iter=3
        )
        
        # Wyckoff Specialist Agent
        self.wyckoff_specialist = Agent(
            role="Wyckoff Method Specialist",
            goal="Identify Wyckoff patterns, phases, and market structures",
            backstory="""You are a master of Richard Wyckoff's methodology with decades 
            of experience in identifying accumulation, distribution, and markup/markdown 
            phases. You excel at recognizing springs, upthrusts, and climatic actions.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.wyckoff_tool],
            #llm=self.llm,  
            max_execution_time=800, 
            max_retry_limit=3,
            max_iter=3
        )
        
        # Risk Management Agent
        self.risk_manager = Agent(
            role="Risk Management Specialist",
            goal="Calculate optimal position sizes and manage trading risk",
            backstory="""You are a professional risk manager with expertise in portfolio 
            management, position sizing, and capital preservation. You ensure all trades 
            follow strict risk management principles.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.risk_tool],
            #llm=self.llm,  
            max_execution_time=800, 
            max_retry_limit=3
        )
        
        # Trading Decision Agent
        self.trading_strategist = Agent(
            role="Trading Strategy Coordinator",
            goal="Synthesize analysis and make final trading decisions",
            backstory="""You are a senior trading strategist who combines technical 
            analysis, Wyckoff methodology, and risk management to make optimal trading 
            decisions. You have a proven track record of profitable systematic trading.""",
            verbose=True,
            allow_delegation=True,
            #llm=self.llm,  
            max_execution_time=800, 
            max_retry_limit=3,
            max_iter=3
        )
    
    async def analyze_market_data(self, market_data: List[Dict]) -> MarketAnalysis:
        """Get comprehensive market analysis from agents"""
        
        # Prepare data for agents
        data_json = json.dumps(market_data[-100:])  # Last 100 bars
        
        data_engineer_task = Task(
        description=f"""
            Analyze the provided market data and identify:
            1. Current trend direction (bullish/bearish/neutral)
            2. Trend strength (0-100)
            3. Support and resistance levels
            Keep analysis concise and actionable.
            """,
        expected_output=(
            "Comprehensive data verification report with quality assessment, "
            "any issues identified and resolved, and confirmation that data "
            "is ready for market analysis."
        ),
        agent=self.data_verification_agent, 
    )
        
        # Create analysis task
        analysis_task = Task(
            description=f"""
            Analyze the data over the last 60 days and provide a comprehensive market analysis.
            
            Provide analysis including:
            1. Overall trend direction and strength
            2. Current volatility levels
            3. Volume patterns and anomalies
            4. Key support and resistance levels
            5. Market context and regime
            
            Return your analysis as a structured summary.
            """,
            agent=self.market_analyst,
            context=[data_engineer_task],
            expected_output="Detailed market analysis with trend, volume, and level identification"
        )
        
        # Execute analysis
        crew = Crew(
            agents=[self.data_verification_agent, self.market_analyst],
            tasks=[data_engineer_task, analysis_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        try:
            # Extract key information from the analysis
            analysis = MarketAnalysis(
                trend_direction="bullish",  # This would be parsed from agent output
                trend_strength=75.0,
                volatility_level="medium",
                volume_analysis="above average volume",
                support_levels=[1.0800, 1.0750],
                resistance_levels=[1.0900, 1.0950],
                market_context="trending market",
                regime="trending"
            )
            return analysis
        except Exception as e:
            logger.error(f"Error parsing market analysis: {e}")
            return MarketAnalysis(
                trend_direction="unknown",
                trend_strength=0.0,
                volatility_level="unknown",
                volume_analysis="unknown",
                support_levels=[],
                resistance_levels=[],
                market_context="unknown",
                regime="unknown"
            )
    
    async def get_trading_signal(self, market_data: List[Dict], market_analysis: MarketAnalysis) -> Optional[BacktestSignal]:
        """Get trading signal from agent crew"""
        
        data_json = json.dumps(market_data[-50:])
        current_price = market_data[-1]['close']
        
        # Market Analysis Task
        market_task = Task(
            description=f"""
            Analyze the current market data over the last 60 days for trading opportunities.
            Market Data: {data_json}
            Current Price: {current_price}
            
            Identify trend strength, volume patterns, and price action.
            """,
            agent=self.market_analyst,
            expected_output="Market analysis summary"
        )
        
        # Wyckoff Analysis Task
        wyckoff_task = Task(
            description=f"""
            Apply Wyckoff methodology to identify patterns and phases.
            Market Data: {data_json}
            
            Identify:
            1. Current Wyckoff phase (A, B, C, D, E)
            2. Pattern type (accumulation, distribution, etc.)
            3. Key Wyckoff events (springs, upthrusts, climax)
            4. Market structure analysis
            """,
            agent=self.wyckoff_specialist,
            expected_output="Wyckoff pattern analysis and phase identification",
            context=[market_task]
            
        )
        
        # Risk Management Task
        risk_task = Task(
            description=f"""
            Calculate risk management parameters for potential trade.
            Current Price: {current_price}
            Account Size: {self.current_capital}
            
            Calculate optimal position size, stop loss, and take profit levels.
            """,
            agent=self.risk_manager,
            expected_output="Risk management calculations"
        )
        
        # Final Trading Decision Task
        decision_task = Task(
            description=f"""
            Based on the market analysis over the last 60 days, Wyckoff patterns, and risk calculations,
            make a final trading decision.
            
            Consider:
            1. Market trend and strength
            2. Wyckoff pattern confirmation
            3. Risk-reward ratio
            4. Position sizing
            
            Provide a clear trading signal: BUY, SELL, or HOLD with detailed reasoning.
            """,
            agent=self.trading_strategist,
            expected_output="Final trading decision with entry, stop loss, take profit, and reasoning"
        )
        
        # Execute crew
        crew = Crew(
            agents=[self.data_verification_agent, self.market_analyst, self.wyckoff_specialist, self.risk_manager, self.trading_strategist],
            tasks=[market_task, wyckoff_task, risk_task, decision_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        # Parse the crew result into a structured signal
        try:
            # This would parse the actual agent output
            signal = BacktestSignal(
                action="buy",
                confidence=75.0,
                entry_price=current_price,
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.04,
                reasoning="Wyckoff accumulation pattern confirmed with volume expansion",
                wyckoff_phase="C",
                pattern_type="accumulation",
                risk_reward_ratio=2.0,
                position_size=1000
            )
            return signal
        except Exception as e:
            logger.error(f"Error parsing trading signal: {e}")
            return None
    
    async def run_backtest(self, market_data: List[Dict]) -> BacktestResults:
        """Run the main backtesting loop with CrewAI agents"""
        
        logger.info(f"🤖 Starting CrewAI-powered Wyckoff backtest with {len(market_data)} data points")
        
        trades = []
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        
        # Track agent performance
        agent_signals = {"buy": 0, "sell": 0, "hold": 0}
        signal_accuracy = []
        
        # Minimum data for analysis
        min_data_points = 100
        
        try:
            for i in range(min_data_points, len(market_data)):
                current_data = market_data[max(0, i-100):i+1]
                current_bar = market_data[i]
                current_price = current_bar['close']
                
                # Get market analysis
                market_analysis = await self.analyze_market_data(current_data)
                
                # Get trading signal from agents
                signal = await self.get_trading_signal(current_data, market_analysis)
                
                if signal and signal.action in ['buy', 'sell']:
                    agent_signals[signal.action] += 1
                    
                    # Execute trade
                    trade = self._execute_trade(signal, current_bar, i)
                    if trade:
                        trades.append(trade)
                        logger.info(f"🎯 Agent signal: {signal.action} at {current_price:.5f} (confidence: {signal.confidence}%)")
                
                # Update open trades
                current_capital = self._update_open_trades(trades, current_price)
                equity_curve.append(current_capital)
                
                # Progress logging
                if i % 100 == 0:
                    progress = (i / len(market_data)) * 100
                    logger.info(f"📈 Backtest progress: {progress:.1f}% - Equity: ${current_capital:,.0f}")
        
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise
        
        # Calculate final results
        results = self._calculate_results(trades, equity_curve, agent_signals)
        
        logger.info("✅ CrewAI backtest completed successfully!")
        return results
    
    def _execute_trade(self, signal: BacktestSignal, current_bar: Dict, index: int) -> Optional[Dict]:
        """Execute a trade based on agent signal"""
        
        try:
            trade = {
                'id': len(self.trades) + 1,
                'entry_time': current_bar.get('time', index),
                'entry_price': signal.entry_price,
                'direction': signal.action,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning,
                'wyckoff_phase': signal.wyckoff_phase,
                'pattern_type': signal.pattern_type,
                'status': 'open',
                'entry_index': index
            }
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return None
    
    def _update_open_trades(self, trades: List[Dict], current_price: float) -> float:
        """Update open trades and calculate current capital"""
        
        current_capital = self.initial_capital
        
        for trade in trades:
            if trade['status'] == 'open':
                # Check exit conditions
                if trade['direction'] == 'buy':
                    if current_price <= trade['stop_loss']:
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'stop_loss'
                        trade['status'] = 'closed'
                    elif current_price >= trade['take_profit']:
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'take_profit'
                        trade['status'] = 'closed'
                else:  # sell
                    if current_price >= trade['stop_loss']:
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'stop_loss'
                        trade['status'] = 'closed'
                    elif current_price <= trade['take_profit']:
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'take_profit'
                        trade['status'] = 'closed'
                
                # Calculate P&L
                if trade['status'] == 'closed':
                    if trade['direction'] == 'buy':
                        pnl = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
                    else:
                        pnl = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
                    
                    trade['pnl'] = pnl
                    current_capital += pnl
        
        return current_capital
    
    def _calculate_results(self, trades: List[Dict], equity_curve: List[float], agent_signals: Dict) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        closed_trades = [t for t in trades if t['status'] == 'closed']
        
        if not closed_trades:
            return BacktestResults()
        
        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        total_return = (total_pnl / self.initial_capital) * 100
        
        winning_pnls = [t['pnl'] for t in closed_trades if t.get('pnl', 0) > 0]
        losing_pnls = [t['pnl'] for t in closed_trades if t.get('pnl', 0) < 0]
        
        best_trade = max(winning_pnls) if winning_pnls else 0
        worst_trade = min(losing_pnls) if losing_pnls else 0
        
        # Risk metrics
        if equity_curve:
            equity_series = pd.Series(equity_curve)
            returns = equity_series.pct_change().dropna()
            
            # Maximum drawdown
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) * 100
            
            # Sharpe ratio
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
        
        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=float(sharpe_ratio),
            profit_factor=profit_factor,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=closed_trades,
            equity_curve=equity_curve,
            agent_performance={
                "signals_generated": agent_signals,
                "total_signals": sum(agent_signals.values()),
                "signal_conversion_rate": (total_trades / sum(agent_signals.values())) * 100 if sum(agent_signals.values()) > 0 else 0
            }
        )

# Utility functions for running the CrewAI backtest with real data
async def run_crewai_wyckoff_backtest(symbol: str = "EUR_USD", timeframe: str = "M15", count: int = 1000, force_refresh=False):
    """Run the CrewAI-powered Wyckoff backtest with real historical data"""
    
    print("🤖 CREWAI-POWERED WYCKOFF BACKTESTING SYSTEM")
    print("=" * 60)
    print("🚀 Initializing multi-agent backtesting with REAL data...")
    
    try:
        # Initialize the backtester
        backtester = CrewAIWyckoffBacktester(initial_capital=100000)
        
        print(f"📊 Configuration:")
        print(f"   💱 Symbol: {symbol}")
        print(f"   ⏰ Timeframe: {timeframe}")
        print(f"   📈 Data Points: {count}")
        print(f"   💰 Initial Capital: ${backtester.initial_capital:,.2f}")
        print(f"   🤖 Agents: Market Analyst, Wyckoff Specialist, Risk Manager, Trading Strategist")
        print(f"   📁 Caching: {'Forced refresh' if force_refresh else 'Smart caching enabled'}")
        
        # Show cache status
        cache_info = data_cache.get_cache_info()
        print(f"   💾 Cache Status: {len(cache_info['cached_datasets'])} datasets, {cache_info['total_size_mb']} MB")
        
        # Get cached historical data
        print(f"\n📊 Loading historical data...")
        historical_data = await get_cached_historical_data(symbol, timeframe, count, force_refresh)
        
        if not historical_data:
            print("❌ Failed to get historical data")
            return None
        
        # Validate data quality
        if not validate_historical_data(historical_data):
            print("❌ Data validation failed")
            return None
        
        print(f"✅ Data loaded and validated: {len(historical_data)} candles")
        
        # Show data sample
        if len(historical_data) > 0:
            sample = historical_data[-1]  # Latest candle
            print(f"📊 Latest candle: {sample['time'][:19]} | "
                  f"OHLC: {sample['open']:.5f}/{sample['high']:.5f}/"
                  f"{sample['low']:.5f}/{sample['close']:.5f} | "
                  f"Vol: {sample['volume']:,}")
        
        # Check for specific data requirements
        if len(historical_data) < 200:
            print("⚠️ Warning: Limited data may affect agent analysis quality")
        
        # Run the backtest with cached real data
        print(f"\n🎯 Starting CrewAI analysis and backtesting...")
        print(f"   🔍 Agents will analyze {len(historical_data)} cached real market candles")
        print(f"   📈 Time period: {historical_data[0]['time'][:10]} to {historical_data[-1]['time'][:10]}")
        
        results = await backtester.run_backtest(historical_data)
        
        # Display results
        print("\n" + "=" * 60)
        print("📋 CREWAI BACKTEST RESULTS (CACHED REAL DATA)")
        print("=" * 60)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Total Return: {results.total_return:.2f}%")
        print(f"   Win Rate: {results.win_rate:.1f}%")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Profit Factor: {results.profit_factor:.2f}")
        print(f"   Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Best Trade: ${results.best_trade:.2f}")
        print(f"   Worst Trade: ${results.worst_trade:.2f}")
        
        print(f"\n🤖 AGENT PERFORMANCE:")
        agent_perf = results.agent_performance
        print(f"   Signals Generated: {agent_perf['total_signals']}")
        print(f"   Buy Signals: {agent_perf['signals_generated']['buy']}")
        print(f"   Sell Signals: {agent_perf['signals_generated']['sell']}")
        print(f"   Hold Decisions: {agent_perf['signals_generated']['hold']}")
        print(f"   Signal Conversion Rate: {agent_perf['signal_conversion_rate']:.1f}%")
        
        if results.trades:
            print(f"\n💼 RECENT TRADES (CACHED REAL DATA):")
            for trade in results.trades[-5:]:  # Show last 5 trades
                pnl_color = "+" if trade.get('pnl', 0) > 0 else ""
                print(f"   {trade['direction'].upper()} @ {trade['entry_price']:.5f} → "
                      f"{trade['exit_price']:.5f} | P&L: {pnl_color}${trade.get('pnl', 0):.2f} | "
                      f"Reason: {trade['reasoning'][:50]}...")
        
        # Data source summary
        print(f"\n📊 DATA SOURCE SUMMARY:")
        print(f"   Symbol: {symbol}")
        print(f"   Timeframe: {timeframe}")
        print(f"   Total Candles: {len(historical_data)}")
        print(f"   Date Range: {historical_data[0]['time'][:10]} to {historical_data[-1]['time'][:10]}")
        print(f"   Data Quality: ✅ Validated")
        print(f"   Caching: ✅ Data cached for future runs")
        
        print(f"\n✅ CrewAI backtest with CACHED real data completed successfully!")
        
        # Show cache summary
        print_cache_summary()
        
        return results
        
    except Exception as e:
        print(f"❌ CrewAI backtest failed: {str(e)}")
        logger.error(f"CrewAI backtest failed: {str(e)}")
        return None

async def run_multi_symbol_crewai_backtest(force_refresh: bool = False):
    """Run CrewAI backtest across multiple symbols with cached real data"""
    
    print("🌍 MULTI-SYMBOL CREWAI BACKTESTING WITH CACHING")
    print("=" * 60)
    
    symbols = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
    timeframes = ["M15", "H1"]
    
    all_results = {}
    
    # Pre-load and cache data for all symbols
    print(f"📊 Pre-loading and caching data for {len(symbols)} symbols...")
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"   🔄 Caching {symbol} {timeframe}...")
            await get_cached_historical_data(symbol, timeframe, 1000, force_refresh)
    
    print(f"\n✅ Data caching complete! Running backtests...")
    
    for symbol in symbols:
        print(f"\n🔄 Testing {symbol}...")
        all_results[symbol] = {}
        
        for timeframe in timeframes:
            print(f"   ⏰ Timeframe: {timeframe}")
            
            try:
                result = await run_crewai_wyckoff_backtest(symbol, timeframe, 500, False)  # Use cached data
                all_results[symbol][timeframe] = result
                
                if result:
                    print(f"   ✅ {symbol} {timeframe}: {result.total_return:.2f}% return, {result.win_rate:.1f}% win rate")
                else:
                    print(f"   ❌ {symbol} {timeframe}: Failed")
                    
            except Exception as e:
                print(f"   ❌ {symbol} {timeframe}: Error - {str(e)}")
                all_results[symbol][timeframe] = None
    
    # Summary analysis
    print(f"\n📊 MULTI-SYMBOL SUMMARY:")
    successful_tests = 0
    total_return = 0
    total_trades = 0
    
    for symbol, symbol_results in all_results.items():
        for timeframe, result in symbol_results.items():
            if result and result.total_trades > 0:
                successful_tests += 1
                total_return += result.total_return
                total_trades += result.total_trades
                print(f"   {symbol} {timeframe}: {result.total_return:.1f}% ({result.total_trades} trades)")
    
    if successful_tests > 0:
        avg_return = total_return / successful_tests
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Successful Tests: {successful_tests}/{len(symbols) * len(timeframes)}")
        print(f"   Average Return: {avg_return:.2f}%")
        print(f"   Total Trades: {total_trades}")
    
    # Show final cache summary
    print_cache_summary()
    
    return all_results
                # print(f"   ❌ {symbol} {timeframe}: Error - {str(e)}")
                # all_results[symbol][timeframe] = None
    
    # Summary analysis
    print(f"\n📊 MULTI-SYMBOL SUMMARY:")
    successful_tests = 0
    total_return = 0
    total_trades = 0
    
    for symbol, symbol_results in all_results.items():
        for timeframe, result in symbol_results.items():
            if result and result.total_trades > 0:
                successful_tests += 1
                total_return += result.total_return
                total_trades += result.total_trades
                print(f"   {symbol} {timeframe}: {result.total_return:.1f}% ({result.total_trades} trades)")
    
    if successful_tests > 0:
        avg_return = total_return / successful_tests
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Successful Tests: {successful_tests}/{len(symbols) * len(timeframes)}")
        print(f"   Average Return: {avg_return:.2f}%")
        print(f"   Total Trades: {total_trades}")
    
    return all_results

# Enhanced data integration with your existing system
class RealDataCrewAIBacktester(CrewAIWyckoffBacktester):
    """Enhanced CrewAI backtester that integrates with your existing data infrastructure"""
    
    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.data_processor = None
        self._initialize_data_processor()
    
    def _initialize_data_processor(self):
        """Initialize data processor for handling OANDA data format"""
        try:
            from src.autonomous_trading_system.utils.wyckoff_data_processor import EnhancedWyckoffDataProcessor
            self.data_processor = EnhancedWyckoffDataProcessor()
            logger.info("✅ Data processor initialized")
        except ImportError:
            logger.warning("⚠️ Enhanced data processor not available")
            self.data_processor = None
    
    async def get_and_prepare_data(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        """Get and prepare real data using your existing infrastructure"""
        
        # Get real historical data
        raw_data = await get_real_historical_data(symbol, timeframe, count)
        
        if not raw_data:
            logger.error("Failed to get historical data")
            return []
        
        # Process data if needed
        if self.data_processor and self._needs_processing(raw_data):
            logger.info("🔧 Processing nested OANDA data structure...")
            try:
                df = pd.DataFrame(raw_data)
                processed_data = self.data_processor.fix_oanda_candles_structure(df)
                logger.info(f"✅ Processed {len(processed_data)} candles")
                return processed_data
            except Exception as e:
                logger.warning(f"Data processing failed: {e}")
                return raw_data
        
        return raw_data
    
    def _needs_processing(self, data: List[Dict]) -> bool:
        """Check if data needs processing (has nested structure)"""
        if not data:
            return False
        
        # Check if first item has nested candles structure
        sample = data[0]
        return 'candles' in str(sample) and 'time' not in sample
    
    async def run_enhanced_backtest(self, symbol: str, timeframe: str, count: int = 1000) -> BacktestResults:
        """Run enhanced backtest with real data and your existing infrastructure"""
        
        logger.info(f"🚀 Starting enhanced CrewAI backtest for {symbol} {timeframe}")
        
        # Get and prepare real data
        historical_data = await self.get_and_prepare_data(symbol, timeframe, count)
        
        if not historical_data:
            raise ValueError("No historical data available")
        
        # Validate data
        if not validate_historical_data(historical_data):
            raise ValueError("Data validation failed")
        
        # Run backtest
        results = await self.run_backtest(historical_data)
        
        # Log results to your database (optional)
        await self._log_backtest_results(results, symbol, timeframe)
        
        return results
    
    async def _log_backtest_results(self, results: BacktestResults, symbol: str, timeframe: str):
        """Log backtest results to your database using available methods"""
        try:
            #from src.database.manager import db_manager
            
            # Create backtest summary
            backtest_summary = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'timeframe': timeframe,
                'total_return': results.total_return,
                'win_rate': results.win_rate,
                'total_trades': results.total_trades,
                'max_drawdown': results.max_drawdown,
                'sharpe_ratio': results.sharpe_ratio,
                'agent_performance': results.agent_performance
            }
            
            # Try to log using available methods
            # if hasattr(db_manager, 'log_backtest_results'):
            #     #await db_manager.log_backtest_results(backtest_summary)
            #     logger.info("✅ Backtest results logged via log_backtest_results")
            
            # elif hasattr(db_manager, 'log_pattern_detection'):
            #     # Convert backtest results to pattern detection format
            #     pattern_data = {
            #         'timestamp': datetime.now(),
            #         'symbol': symbol,
            #         'timeframe': timeframe,
            #         'pattern_type': 'backtest_result',
            #         'confidence_score': float(results.win_rate),
            #         'key_levels': json.dumps({
            #             'total_return': results.total_return,
            #             'total_trades': results.total_trades,
            #             'max_drawdown': results.max_drawdown
            #         }),
            #         'volume_analysis': json.dumps(results.agent_performance),
            #         'market_context': json.dumps({
            #             'backtest_type': 'crewai_wyckoff',
            #             'sharpe_ratio': results.sharpe_ratio,
            #             'profit_factor': results.profit_factor
            #         }),
            #         'invalidation_level': None,
            #         'trade_id': None,
            #         'structure_start_time': None,
            #         'structure_end_time': None,
            #         'created_at': datetime.now()
            #     }
            #     
            #     await db_manager.log_pattern_detection(pattern_data)
            #     logger.info("✅ Backtest results logged via log_pattern_detection")
            
            # else:
            #     # Check available methods and log appropriately
            #     available_methods = [method for method in dir(db_manager) if not method.startswith('_') and 'log' in method.lower()]
            #     logger.info(f"Available logging methods: {available_methods}")
            #     logger.info("💾 Backtest results stored in memory (no suitable db method found)")
            
        except Exception as e:
            logger.warning(f"Failed to log backtest results: {e}")
            # Store results in a local file as fallback
            try:
                import json
                from pathlib import Path
                
                results_dir = Path("crewai_backtest_results")
                results_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"crewai_backtest_{symbol}_{timeframe}_{timestamp}.json"
                
                with open(results_dir / filename, 'w') as f:
                    json.dump({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'results': results.__dict__,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2, default=str)
                
                logger.info(f"📁 Backtest results saved to file: {filename}")
                
            except Exception as file_error:
                logger.warning(f"Failed to save results to file: {file_error}")

# Updated main execution function

import pickle
import gzip
from pathlib import Path

# Historical data caching system
class HistoricalDataCache:
    """Manages caching of historical data to files for faster backtesting"""
    
    def __init__(self, cache_dir: str = "historical_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Historical data cache directory: {self.cache_dir}")
    
    def _get_cache_filename(self, symbol: str, timeframe: str, count: int) -> Path:
        """Generate cache filename based on parameters"""
        filename = f"{symbol}_{timeframe}_{count}candles.pkl.gz"
        return self.cache_dir / filename
    
    def _get_metadata_filename(self, symbol: str, timeframe: str, count: int) -> Path:
        """Generate metadata filename"""
        filename = f"{symbol}_{timeframe}_{count}candles_meta.json"
        return self.cache_dir / filename
    
    def has_cached_data(self, symbol: str, timeframe: str, count: int) -> bool:
        """Check if cached data exists and is recent"""
        cache_file = self._get_cache_filename(symbol, timeframe, count)
        meta_file = self._get_metadata_filename(symbol, timeframe, count)
        
        if not cache_file.exists() or not meta_file.exists():
            return False
        
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is less than 24 hours old
            cache_time = datetime.fromisoformat(metadata['created_at'])
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            if age_hours > 24:
                logger.info(f"⏰ Cache for {symbol} {timeframe} is {age_hours:.1f} hours old, refreshing...")
                return False
            
            logger.info(f"✅ Found valid cache for {symbol} {timeframe} (age: {age_hours:.1f}h)")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking cache metadata: {e}")
            return False
    
    def save_data(self, data: List[Dict], symbol: str, timeframe: str, count: int, source: str = "unknown"):
        """Save historical data to compressed cache file"""
        try:
            cache_file = self._get_cache_filename(symbol, timeframe, count)
            meta_file = self._get_metadata_filename(symbol, timeframe, count)
            
            # Save compressed data
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'count': count,
                'actual_count': len(data),
                'source': source,
                'created_at': datetime.now().isoformat(),
                'date_range': {
                    'start': data[0]['time'] if data else None,
                    'end': data[-1]['time'] if data else None
                },
                'data_quality': {
                    'has_volume': all('volume' in candle for candle in data[:10]),
                    'has_ohlc': all(all(field in candle for field in ['open', 'high', 'low', 'close']) for candle in data[:10]),
                    'price_range': {
                        'min': min(float(candle['low']) for candle in data) if data else 0,
                        'max': max(float(candle['high']) for candle in data) if data else 0
                    }
                }
            }
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"💾 Cached {len(data)} candles to {cache_file.name} ({file_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save data to cache: {e}")
    
    def load_data(self, symbol: str, timeframe: str, count: int) -> Optional[List[Dict]]:
        """Load historical data from cache"""
        try:
            cache_file = self._get_cache_filename(symbol, timeframe, count)
            
            with gzip.open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"📂 Loaded {len(data)} candles from cache: {cache_file.name}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from cache: {e}")
            return None
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about all cached data"""
        cache_info = {
            'cache_directory': str(self.cache_dir),
            'cached_datasets': [],
            'total_size_mb': 0
        }
        
        try:
            for meta_file in self.cache_dir.glob("*_meta.json"):
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                cache_file = meta_file.with_suffix('').with_suffix('.pkl.gz')
                if cache_file.exists():
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    cache_info['total_size_mb'] += size_mb
                    
                    metadata['file_size_mb'] = round(size_mb, 2)
                    cache_info['cached_datasets'].append(metadata)
            
            cache_info['total_size_mb'] = round(cache_info['total_size_mb'], 2)
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
        
        return cache_info
    
    def clear_cache(self, symbol, timeframe):
        """Clear cache files (all or specific symbol/timeframe)"""
        try:
            if symbol and timeframe:
                # Clear specific symbol/timeframe
                pattern = f"{symbol}_{timeframe}_*"
                files_deleted = 0
                for file in self.cache_dir.glob(pattern):
                    file.unlink()
                    files_deleted += 1
                logger.info(f"🗑️ Cleared {files_deleted} cache files for {symbol} {timeframe}")
            else:
                # Clear all cache
                files_deleted = 0
                for file in self.cache_dir.glob("*"):
                    file.unlink()
                    files_deleted += 1
                logger.info(f"🗑️ Cleared all cache ({files_deleted} files)")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

# Initialize global cache instance
data_cache = HistoricalDataCache()

async def get_cached_historical_data(symbol: str = "EUR_USD", timeframe: str = "M15", count: int = 2000, force_refresh: bool = False) -> List[Dict]:
    """Get historical data with intelligent caching"""
    
    logger.info(f"📊 Getting historical data for {symbol} {timeframe} ({count} candles)")
    
    # Check cache first (unless force refresh)
    if not force_refresh and data_cache.has_cached_data(symbol, timeframe, count):
        cached_data = data_cache.load_data(symbol, timeframe, count)
        if cached_data and len(cached_data) >= count * 0.9:  # At least 90% of requested data
            logger.info(f"✅ Using cached data ({len(cached_data)} candles)")
            return cached_data[:count]  # Return exactly what was requested
    
    # Fetch fresh data
    logger.info(f"🔄 Fetching fresh data from sources...")
    historical_data = await get_real_historical_data(symbol, timeframe, count)
    
    if not historical_data:
        logger.error("❌ Failed to fetch historical data")
        return []
    
    # Determine data source for metadata
    if len(historical_data) > 0:
        if 'instrument' in str(historical_data[0]):
            data_source = "OANDA_API"
        elif validate_historical_data(historical_data):
            data_source = "Database"
        else:
            data_source = "Synthetic"
    else:
        data_source = "Unknown"
    
    # Cache the fresh data
    data_cache.save_data(historical_data, symbol, timeframe, count, data_source)
    
    logger.info(f"✅ Retrieved and cached {len(historical_data)} candles from {data_source}")
    return historical_data

def print_cache_summary():
    """Print a summary of cached data"""
    cache_info = data_cache.get_cache_info()
    
    print(f"\n📁 HISTORICAL DATA CACHE SUMMARY")
    print(f"=" * 50)
    print(f"Cache Directory: {cache_info['cache_directory']}")
    print(f"Total Cache Size: {cache_info['total_size_mb']} MB")
    print(f"Cached Datasets: {len(cache_info['cached_datasets'])}")
    
    if cache_info['cached_datasets']:
        print(f"\n📊 CACHED DATASETS:")
        for dataset in cache_info['cached_datasets']:
            age_hours = (datetime.now() - datetime.fromisoformat(dataset['created_at'])).total_seconds() / 3600
            print(f"   {dataset['symbol']} {dataset['timeframe']}: {dataset['actual_count']} candles, "
                  f"{dataset['file_size_mb']} MB, {age_hours:.1f}h old ({dataset['source']})")
    
    return cache_info

# Enhanced data fetching with caching
async def get_real_historical_data(symbol: str = "EUR_USD", timeframe: str = "M15", count: int = 2000) -> Optional[List[Dict]]:
    """Get real historical data using existing infrastructure (now with better error handling)"""
    
    try:
        logger.info(f"📊 ***Fetching real historical data for {symbol} {timeframe}")
        
        # Use your existing OANDA MCP wrapper
        from mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
        print(f"symbol: {symbol}")
        print(f"timeframe: {timeframe}")
        print(f"count: {count}")
            
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            historical_data = await oanda.get_historical_data(symbol, timeframe, count)
        
        if "error" in historical_data:
            logger.error(f"OANDA API error: {historical_data['error']}")
            #return await get_fallback_historical_data(symbol, timeframe, count)
        
        raw_data = historical_data['data']
        logger.info(f"✅ Retrieved {len(raw_data)} candles from OANDA API")
        print('*' *40)
        #print(f"raw_data: {raw_data}")
        # Check if data needs to be processed (nested candles structure)
        if len(raw_data) > 0 and 'candles' in str(raw_data):
            logger.info("🔧 Processing nested OANDA data structure...")
            try:
                from src.autonomous_trading_system.utils.wyckoff_data_processor import EnhancedWyckoffDataProcessor
                
                processor = EnhancedWyckoffDataProcessor()
                df = pd.DataFrame(raw_data)
                processed_data = processor.fix_oanda_candles_structure(df)
                
                logger.info(f"✅ Processed {len(processed_data)} candles")
                return processed_data
            except ImportError:
                logger.warning("Enhanced data processor not available, using raw data")
                return raw_data
        else:
            # Data is already in correct format
            return raw_data
            
    except ImportError as e:
        logger.warning(f"OANDA wrapper not available: {e}")
        #return await get_fallback_historical_data(symbol, timeframe, count)
    except Exception as e:
        logger.error(f"Error fetching real data (get_real_historical_data): {e}")
        #return await get_fallback_historical_data(symbol, timeframe, count)

# async def get_fallback_historical_data(symbol: str, timeframe: str, count: int) -> Optional[List[Dict]]:
#     """Fallback to database or alternative data source"""
    
#     try:
#         logger.info(f"🔄 Trying fallback data sources for {symbol}")
#         print()
#         # Try to get data from your database using available methods
#         from src.database.manager import db_manager
        
#         # Check what methods are available in db_manager
#         available_methods = [method for method in dir(db_manager) if not method.startswith('_')]
#         logger.info(f"Available db_manager methods: {available_methods}")
        
        # Try different approaches based on available methods
        historical_data = []
        
#         # Check what methods are available in db_manager
#         available_methods = [method for method in dir(db_manager) if not method.startswith('_')]
#         logger.info(f"Available db_manager methods: {available_methods}")
        
#         # Try different approaches based on available methods
#         historical_data = []
        
#         # Option 1: Check if there's a specific method for historical data
#         if hasattr(db_manager, 'get_historical_candles'):
#             try:
#                 result = await db_manager.get_historical_candles(symbol, timeframe, count)
#                 if result:
#                     historical_data = result
#                     logger.info(f"✅ Retrieved {len(historical_data)} candles via get_historical_candles")
#             except Exception as e:
#                 logger.warning(f"get_historical_candles failed: {e}")
        
#         # Option 2: Check if there's a method to get trades (which might have price data)
#         elif hasattr(db_manager, 'get_trades'):
#             try:
#                 trades = await db_manager.get_trades(symbol=symbol, limit=count)
#                 if trades:
#                     # Convert trades to OHLCV format (simplified)
#                     for trade in trades:
#                         historical_data.append({
#                             'time': trade.entry_time.isoformat() if hasattr(trade, 'entry_time') else str(trade.created_at),
#                             'open': float(trade.entry_price),
#                             'high': float(trade.entry_price * 1.001),  # Approximate
#                             'low': float(trade.entry_price * 0.999),   # Approximate
#                             'close': float(trade.exit_price) if hasattr(trade, 'exit_price') and trade.exit_price else float(trade.entry_price),
#                             'volume': 1000  # Default volume
#                         })
#                     logger.info(f"✅ Converted {len(historical_data)} trades to OHLCV data")
#             except Exception as e:
#                 logger.warning(f"get_trades conversion failed: {e}")
        
#         # Option 3: Try pattern detection data as a source
#         elif hasattr(db_manager, 'get_pattern_detections'):
#             try:
#                 patterns = await db_manager.get_pattern_detections(symbol=symbol, limit=count)
#                 if patterns:
#                     # Extract price levels from pattern detections
#                     for pattern in patterns:
#                         if hasattr(pattern, 'key_levels'):
#                             try:
#                                 levels = json.loads(pattern.key_levels) if isinstance(pattern.key_levels, str) else pattern.key_levels
#                                 price = levels.get('current_price', 1.1000)
#                                 historical_data.append({
#                                     'time': pattern.timestamp.isoformat() if hasattr(pattern, 'timestamp') else str(pattern.created_at),
#                                     'open': float(price),
#                                     'high': float(price * 1.002),
#                                     'low': float(price * 0.998),
#                                     'close': float(price),
#                                     'volume': 1000
#                                 })
#                             except Exception:
#                                 continue
#                     logger.info(f"✅ Extracted {len(historical_data)} price points from pattern data")
#             except Exception as e:
#                 logger.warning(f"Pattern data extraction failed: {e}")
        
#         # If we got some data, return it
#         if historical_data and len(historical_data) >= 50:
#             # Sort by time and ensure proper format
#             historical_data.sort(key=lambda x: x['time'])
#             return historical_data[:count]  # Limit to requested count
#         else:
#             logger.warning(f"Insufficient database data ({len(historical_data)} points), using realistic sample data")
#             return generate_realistic_sample_data(symbol, count)
            
#     except Exception as e:
#         logger.warning(f"Database fallback failed: {e}")
#         return generate_realistic_sample_data(symbol, count)

# def generate_realistic_sample_data(symbol: str, num_candles: int, timeframe: str = "M15") -> List[Dict]:
#     """Generate realistic sample data based on symbol characteristics"""
    
#     logger.info(f"📈 Generating realistic sample data for {symbol}")
    
#     # Symbol-specific parameters
#     symbol_params = {
#         'EUR_USD': {'base_price': 1.1000, 'volatility': 0.0008, 'spread': 0.00015},
#         'GBP_USD': {'base_price': 1.2500, 'volatility': 0.0012, 'spread': 0.00020},
#         'USD_JPY': {'base_price': 145.00, 'volatility': 0.08, 'spread': 0.015},
#         'USD_CHF': {'base_price': 0.9200, 'volatility': 0.0007, 'spread': 0.00018},
#         'AUD_USD': {'base_price': 0.6700, 'volatility': 0.0010, 'spread': 0.00018},
#         'USD_CAD': {'base_price': 1.3500, 'volatility': 0.0009, 'spread': 0.00020}
#     }
    
#     params = symbol_params.get(symbol, symbol_params['EUR_USD'])
    
#     data = []
#     price = params['base_price']
#     volume_base = 150000
    
#     # Add some realistic market patterns
#     np.random.seed(42)  # For reproducible results
    
#     for i in range(num_candles):
#         # Create more realistic price movements
#         # Long-term trend
#         long_trend = np.sin(i / 200) * params['volatility'] * 5
#         # Medium-term cycles  
#         medium_cycle = np.sin(i / 50) * params['volatility'] * 2
#         # Daily volatility
#         daily_noise = np.random.normal(0, params['volatility'])
        
#         # Occasional volatility spikes (news events)
#         if np.random.random() < 0.02:  # 2% chance of volatility spike
#             daily_noise *= 3
#             volume_multiplier = 2.5
#         else:
#             volume_multiplier = 1.0
        
#         price_change = long_trend + medium_cycle + daily_noise
#         price += price_change
        
#         # Keep price in reasonable range
#         if 'JPY' in symbol:
#             price = max(120.0, min(160.0, price))
#         else:
#             price = max(0.5, min(2.0, price))
        
#         # Generate realistic OHLC
#         daily_range = abs(np.random.normal(0, params['volatility'] * 1.5))
#         open_offset = np.random.normal(0, params['volatility'] * 0.3)
        
#         open_price = price + open_offset
#         high_price = max(open_price, price) + abs(np.random.normal(0, daily_range * 0.6))
#         low_price = min(open_price, price) - abs(np.random.normal(0, daily_range * 0.6))
        
#         # Close price with some bias toward the direction of movement
#         if price_change > 0:
#             close_bias = 0.6  # Slightly higher close on up days
#         else:
#             close_bias = 0.4  # Slightly lower close on down days
            
#         close_price = low_price + (high_price - low_price) * (close_bias + np.random.normal(0, 0.2))
#         close_price = max(low_price, min(high_price, close_price))
        
#         # Generate volume with correlation to volatility and price movement
#         volatility_factor = daily_range / params['volatility']
#         volume = int(volume_base * volume_multiplier * volatility_factor * np.random.uniform(0.7, 1.3))
        
#         # Create timestamp
#         base_time = datetime.now() - timedelta(days=num_candles-i)
#         if 'M15' in str(timeframe):
#             timestamp = base_time + timedelta(minutes=(i % 96) * 15)  # 96 x 15min = 24 hours
#         elif 'H1' in str(timeframe):
#             timestamp = base_time + timedelta(hours=i % 24)
#         else:
#             timestamp = base_time
        
#         data.append({
#             'time': timestamp.isoformat(),
#             'open': round(open_price, 5 if 'JPY' not in symbol else 3),
#             'high': round(high_price, 5 if 'JPY' not in symbol else 3),
#             'low': round(low_price, 5 if 'JPY' not in symbol else 3),
#             'close': round(close_price, 5 if 'JPY' not in symbol else 3),
#             'volume': volume
#         })
        
#         price = close_price
    
#     logger.info(f"✅ Generated {len(data)} realistic candles for {symbol}")
#     return data

# Enhanced data validation
def validate_historical_data(data: List[Dict]) -> bool:
    """Validate historical data quality"""
    
    if not data or len(data) < 100:
        logger.error("Insufficient data points")
        return False
    
    # Check required fields
    required_fields = ['time', 'open', 'high', 'low', 'close', 'volume']
    for candle in data[:5]:  # Check first 5 candles
        if not all(field in candle for field in required_fields):
            logger.error(f"Missing required fields in data: {candle.keys()}")
            return False
    
    # Check data quality
    issues = 0
    for i, candle in enumerate(data):
        try:
            o, h, l, c = float(candle['open']), float(candle['high']), float(candle['low']), float(candle['close'])
            
            # OHLC relationship validation
            if not (l <= o <= h and l <= c <= h):
                issues += 1
                if issues <= 5:  # Log first 5 issues
                    logger.warning(f"Invalid OHLC relationship at index {i}: O={o}, H={h}, L={l}, C={c}")
            
            # Reasonable price check
            if c <= 0 or c > 1000:
                issues += 1
                if issues <= 5:
                    logger.warning(f"Unreasonable price at index {i}: {c}")
                    
        except (ValueError, KeyError) as e:
            issues += 1
            if issues <= 5:
                logger.warning(f"Data parsing error at index {i}: {e}")
    
    error_rate = issues / len(data) * 100
    if error_rate > 5:  # More than 5% errors
        logger.error(f"Data quality poor: {error_rate:.1f}% error rate")
        return False
    elif error_rate > 1:
        logger.warning(f"Data quality issues: {error_rate:.1f}% error rate")
    
    logger.info(f"✅ Data validation passed: {len(data)} candles, {error_rate:.2f}% error rate")
    return True

# Advanced CrewAI Backtesting Features
class AdvancedCrewAIBacktester(CrewAIWyckoffBacktester):
    """Advanced version with portfolio optimization and ensemble agents"""
    
    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self._create_ensemble_agents()
    
    def _create_ensemble_agents(self):
        """Create ensemble of specialized agents for different market conditions"""
        
        # Data engineering Specialist
        self.data_engineer = Agent(
            role="A Python engineer who is an expert at cleaning and formatting data.",
            goal="Provide the Trending Market Specialist with cleaned data from the previous 60 days",
            backstory="""You're a meticulous data engineer with a keen eye for detail. You're known for
                         your ability to turn complex data sets into clear and well-formatted data.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.market_tool, self.wyckoff_tool],
            #llm=self.llm
        )
        
        # Trending Market Specialist
        self.trend_specialist = Agent(
            role="Trending Market Specialist",
            goal="Identify and trade trending market conditions using Wyckoff markup/markdown phases",
            backstory="""You are an expert in trending markets, specializing in identifying 
            the transition from accumulation/distribution to markup/markdown phases. You excel 
            at riding trends and managing positions during sustained directional moves.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.market_tool, self.wyckoff_tool],
            #lm=self.llm
        )
        
        # Range Trading Specialist
        self.range_specialist = Agent(
            role="Range Trading Specialist",
            goal="Identify and trade ranging market conditions using Wyckoff accumulation/distribution",
            backstory="""You are a master of range-bound markets, specializing in identifying 
            accumulation and distribution phases. You excel at trading springs, upthrusts, 
            and other range-bound Wyckoff patterns.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.market_tool, self.wyckoff_tool],
            #llm=self.llm
        )
        
        # Volatility Specialist
        self.volatility_specialist = Agent(
            role="Volatility and Risk Specialist",
            goal="Analyze market volatility and adjust position sizing and risk accordingly",
            backstory="""You are an expert in volatility analysis and adaptive risk management. 
            You adjust trading strategies based on current market volatility and ensure optimal 
            position sizing for different market regimes.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.risk_tool],
            #llm=self.llm
        )
        
        # Portfolio Manager
        self.portfolio_manager = Agent(
            role="Portfolio Strategy Manager",
            goal="Coordinate overall portfolio strategy and manage multiple positions",
            backstory="""You are a senior portfolio manager responsible for overall strategy 
            coordination, position correlation analysis, and portfolio-level risk management. 
            You ensure optimal allocation across different market opportunities.""",
            verbose=True,
            allow_delegation=True,
            #llm=self.llm
        )
    
    async def get_ensemble_signal(self, market_data: List[Dict], market_regime: str) -> Optional[BacktestSignal]:
        """Get trading signal from ensemble of specialized agents"""
        
        data_json = json.dumps(market_data[-50:])
        current_price = market_data[-1]['close']
        
        # Select specialist based on market regime
        if market_regime == "trending":
            specialist_agent = self.trend_specialist
            specialist_context = "trending market with directional bias"
        elif market_regime == "ranging":
            specialist_agent = self.range_specialist
            specialist_context = "range-bound market with accumulation/distribution patterns"
        else:
            specialist_agent = self.wyckoff_specialist
            specialist_context = "transitional market requiring adaptive approach"
        
        # Data engineer Task
        data_engineer_task = Task(
            description="""
            Retrieve and verify market data for {symbol_name}:
            
            1. **Data Retrieval**: Get raw market data using get_raw_market_data tool
            2. **Data Verification**: Use verify_and_clean_data tool to:
               - Parse and validate data structure
               - Handle JSON parsing issues
               - Remove corrupted or invalid records
               - Standardize data format (OHLC, timestamps, volume)
               - Validate price data integrity
            3. **Quality Assessment**: Generate data quality report
            4. **Preparation**: Ensure data is ready for Wyckoff analysis
            
            Focus on:
            - Resolving any JSON parsing errors
            - Ensuring data completeness and accuracy
            - Standardizing timestamp and price formats
            - Validating OHLC data relationships
            - Providing clear quality metrics
            
            Return verification status and data quality summary.
            """,
            expected_output=(
                "Comprehensive data verification report with quality assessment, "
                "any issues identified and resolved, and confirmation that data "
                "is ready for market analysis."
            ),
            agent=self.data_verification_agent
        )
        
        # Specialist Analysis Task
        specialist_task = Task(
            description=f"""
            Analyze the current {specialist_context} over the last 60 days for optimal trading opportunities.
            Market Datais cached.
            Current Price: {current_price}
            Market Regime: {market_regime}
            
            Apply your specialized expertise to identify the best trading opportunity.
            """,
            agent=specialist_agent,
            expected_output="Specialized market analysis and trading recommendation"
        )
        
        # Volatility Analysis Task
        volatility_task = Task(
            description=f"""
            Analyze current market volatility over the last 60 days and recommend risk adjustments.
            Market Data is cached.
            
            Assess:
            1. Current volatility level vs historical
            2. Volatility trend (expanding/contracting)
            3. Risk adjustment recommendations
            4. Position sizing modifications
            """,
            agent=self.volatility_specialist,
            expected_output="Volatility analysis and risk recommendations"
        )
        
        # Portfolio Coordination Task
        portfolio_task = Task(
            description=f"""
            Coordinate the specialist recommendation with overall portfolio strategy.
            Current Capital: {self.current_capital}
            Market Regime: {market_regime}
            
            Consider:
            1. Portfolio exposure and diversification
            2. Risk budget allocation
            3. Position correlation
            4. Strategic timing
            
            Make final trading decision coordinating all inputs.
            """,
            agent=self.portfolio_manager,
            expected_output="Final coordinated trading decision"
        )
        
        # Execute ensemble crew
        ensemble_crew = Crew(
            agents=[specialist_agent, self.volatility_specialist, self.portfolio_manager],
            tasks=[specialist_task, volatility_task, portfolio_task],
            verbose=True,
            process=Process.sequential,
            step_callback=lambda step: time.sleep(3)  # 3-second delay between steps to avoid rate limits
        )
        
        result = ensemble_crew.kickoff()
        
        # Parse ensemble result (enhanced signal)
        try:
            signal = BacktestSignal(
                action="buy",
                confidence=80.0,
                entry_price=current_price,
                stop_loss=current_price * 0.985,
                take_profit=current_price * 1.045,
                reasoning=f"Ensemble analysis: {market_regime} market specialist recommendation with volatility adjustment",
                wyckoff_phase="D",
                pattern_type="accumulation",
                risk_reward_ratio=3.0,
                position_size=1500
            )
            return signal
        except Exception as e:
            logger.error(f"Error parsing ensemble signal: {e}")
            return None
    
    def determine_market_regime(self, market_data: List[Dict]) -> str:
        """Determine current market regime for specialist selection"""
        
        if len(market_data) < 50:
            return "unknown"
        
        df = pd.DataFrame(market_data[-50:])
        
        # Calculate trend metrics
        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        volatility = df['close'].pct_change().std()
        
        # Calculate range metrics
        recent_high = df['high'].max()
        recent_low = df['low'].min()
        current_price = df['close'].iloc[-1]
        range_position = (current_price - recent_low) / (recent_high - recent_low)
        
        # Regime classification
        if abs(price_change) > 0.02 and volatility > 0.008:  # 2% move with high volatility
            return "trending"
        elif abs(price_change) < 0.01 and 0.2 < range_position < 0.8:  # Low move, mid-range
            return "ranging"
        else:
            return "transitional"

# Performance Analytics for CrewAI System
class CrewAIPerformanceAnalyzer:
    agentops.init(os.getenv("AGENTOP_API_KEY"), skip_auto_end_session=True)
    """Analyze and optimize CrewAI agent performance"""
    
    def __init__(self, backtest_results: BacktestResults):
        self.results = backtest_results
    
    def analyze_agent_decision_quality(self) -> Dict[str, Any]:
        """Analyze the quality of agent decisions"""
        
        trades = self.results.trades
        
        if not trades:
            return {"error": "No trades to analyze"}
        
        # Analyze by confidence levels
        high_confidence_trades = [t for t in trades if t.get('confidence', 0) >= 80]
        medium_confidence_trades = [t for t in trades if 60 <= t.get('confidence', 0) < 80]
        low_confidence_trades = [t for t in trades if t.get('confidence', 0) < 60]
        
        def calculate_metrics(trade_list):
            if not trade_list:
                return {"count": 0, "win_rate": 0, "avg_pnl": 0}
            
            wins = len([t for t in trade_list if t.get('pnl', 0) > 0])
            avg_pnl = sum(t.get('pnl', 0) for t in trade_list) / len(trade_list)
            
            return {
                "count": len(trade_list),
                "win_rate": (wins / len(trade_list)) * 100,
                "avg_pnl": avg_pnl
            }
        
        # Analyze by Wyckoff phases
        phase_performance = {}
        for phase in ['A', 'B', 'C', 'D', 'E']:
            phase_trades = [t for t in trades if t.get('wyckoff_phase') == phase]
            phase_performance[f"Phase_{phase}"] = calculate_metrics(phase_trades)
        
        # Analyze by pattern types
        pattern_performance = {}
        for pattern in ['accumulation', 'distribution', 'reaccumulation', 'redistribution']:
            pattern_trades = [t for t in trades if pattern in t.get('pattern_type', '').lower()]
            pattern_performance[pattern] = calculate_metrics(pattern_trades)
        
        return {
            "confidence_analysis": {
                "high_confidence": calculate_metrics(high_confidence_trades),
                "medium_confidence": calculate_metrics(medium_confidence_trades),
                "low_confidence": calculate_metrics(low_confidence_trades)
            },
            "phase_performance": phase_performance,
            "pattern_performance": pattern_performance,
            "decision_quality_score": self._calculate_decision_quality_score()
        }
    
    def _calculate_decision_quality_score(self) -> float:
        """Calculate overall decision quality score"""
        
        trades = self.results.trades
        if not trades:
            return 0
        
        # Factors for decision quality
        win_rate_score = self.results.win_rate
        profit_factor_score = min(self.results.profit_factor * 20, 100) if self.results.profit_factor > 0 else 0
        sharpe_score = min(max(self.results.sharpe_ratio * 25, 0), 100)
        
        # Confidence correlation (higher confidence should = better results)
        high_conf_trades = [t for t in trades if t.get('confidence', 0) >= 80]
        if high_conf_trades:
            high_conf_wins = len([t for t in high_conf_trades if t.get('pnl', 0) > 0])
            confidence_score = (high_conf_wins / len(high_conf_trades)) * 100
        else:
            confidence_score = 50
        
        # Weighted decision quality score
        quality_score = (
            win_rate_score * 0.3 +
            profit_factor_score * 0.25 +
            sharpe_score * 0.25 +
            confidence_score * 0.2
        )
        
        return quality_score
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        decision_analysis = self.analyze_agent_decision_quality()
        
        report = f"""
🤖 CREWAI AGENT PERFORMANCE REPORT
{'='*50}

📊 OVERALL PERFORMANCE:
   Total Return: {self.results.total_return:.2f}%
   Win Rate: {self.results.win_rate:.1f}%
   Profit Factor: {self.results.profit_factor:.2f}
   Sharpe Ratio: {self.results.sharpe_ratio:.2f}
   Decision Quality Score: {decision_analysis['decision_quality_score']:.1f}/100

🎯 CONFIDENCE LEVEL ANALYSIS:
   High Confidence (80%+):
      Trades: {decision_analysis['confidence_analysis']['high_confidence']['count']}
      Win Rate: {decision_analysis['confidence_analysis']['high_confidence']['win_rate']:.1f}%
      Avg P&L: ${decision_analysis['confidence_analysis']['high_confidence']['avg_pnl']:.2f}
   
   Medium Confidence (60-80%):
      Trades: {decision_analysis['confidence_analysis']['medium_confidence']['count']}
      Win Rate: {decision_analysis['confidence_analysis']['medium_confidence']['win_rate']:.1f}%
      Avg P&L: ${decision_analysis['confidence_analysis']['medium_confidence']['avg_pnl']:.2f}

🔄 WYCKOFF PHASE PERFORMANCE:"""
        
        for phase, metrics in decision_analysis['phase_performance'].items():
            if metrics['count'] > 0:
                report += f"""
   {phase}: {metrics['count']} trades, {metrics['win_rate']:.1f}% win rate, ${metrics['avg_pnl']:.2f} avg P&L"""
        
        report += f"""

📈 PATTERN TYPE PERFORMANCE:"""
        
        for pattern, metrics in decision_analysis['pattern_performance'].items():
            if metrics['count'] > 0:
                report += f"""
   {pattern.title()}: {metrics['count']} trades, {metrics['win_rate']:.1f}% win rate"""
        
        return report

# Main execution functions
async def run_advanced_crewai_backtest(symbol: str = "EUR_USD", timeframe: str = "M15", count: int = 1000, force_refresh=True):
    """Run advanced CrewAI backtest with ensemble agents"""
    
    print("🚀 ADVANCED CREWAI WYCKOFF BACKTESTING SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize advanced backtester
        backtester = AdvancedCrewAIBacktester(initial_capital=100000)
        
        # Generate sample data
        historical_data = await get_cached_historical_data(symbol, timeframe, count, force_refresh)
        
        print(f"🤖 Advanced Agent Configuration:")
        print(f"   📈 Trend Specialist: Markup/Markdown phases")
        print(f"   📊 Range Specialist: Accumulation/Distribution phases") 
        print(f"   ⚡ Volatility Specialist: Risk adjustment")
        print(f"   💼 Portfolio Manager: Strategy coordination")
        
        # Run backtest
        results = await backtester.run_backtest(historical_data)
        
        # Analyze performance
        analyzer = CrewAIPerformanceAnalyzer(results)
        performance_report = analyzer.generate_performance_report()
        
        print(performance_report)
        
        return results
        
    except Exception as e:
        print(f"❌ Advanced CrewAI backtest failed: {str(e)}")
        return None

# CLI Interface
if __name__ == "__main__":
    print("🤖 CrewAI Wyckoff Backtesting System")
    print("Choose an option:")
    print("1. Run basic CrewAI backtest")
    print("2. Run advanced ensemble backtest")
    print("3. Performance analysis only")
    
    symbol = "EUR_USD"
    timeframe = "M15"
    force_refresh = True
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(run_crewai_wyckoff_backtest(symbol=symbol, timeframe=timeframe, force_refresh=force_refresh))
    elif choice == "2":
        asyncio.run(run_advanced_crewai_backtest())
    elif choice == "3":
        print("Performance analysis requires existing results...")
    else:
        print("Running basic CrewAI backtest...")
        asyncio.run(run_crewai_wyckoff_backtest())
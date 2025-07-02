"""
Monitored Backtesting Integration
Combines your existing backtesting system with enhanced performance monitoring
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Your existing imports
from src.monitoring.agent_performance_monitor import PerformanceMonitor, PerformanceDashboard
from src.monitoring.monitored_crew import MonitoredAutonomousTradingSystem
from src.backtesting.enhanced_agent_backtest import EnhancedAgentBacktester
from src.config.logging_config import logger

class MonitoredBacktester(EnhancedAgentBacktester):
    """Enhanced backtester with comprehensive agent performance monitoring"""
    
    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        
        # Initialize monitoring for backtesting
        self.performance_monitor = PerformanceMonitor(db_path="backtest_performance.db")
        self.monitored_system = None
        self.backtest_session_id = None
        
        # Backtest-specific tracking
        self.agent_decisions_count = {'market_analyst': 0, 'risk_manager': 0, 'trading_coordinator': 0}
        self.market_regime_performance = {}
        
    async def run_monitored_backtest(
        self,
        historical_data: List[Dict],
        initial_balance: float,
        symbol: str,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run backtest with full agent performance monitoring"""
        
        logger.info("🚀 Starting MONITORED BACKTEST with enhanced tracking...")
        
        # Start monitoring
        await self.performance_monitor.start_monitoring()
        self.backtest_session_id = self.performance_monitor.session_id
        
        # Initialize monitored trading system
        self.monitored_system = MonitoredAutonomousTradingSystem()
        await self.monitored_system._ensure_monitoring_started()
        
        # Apply agent configuration if provided
        if agent_config:
            await self._apply_agent_configuration(agent_config)
        
        start_time = time.time()
        
        try:
            # Update monitoring with historical price data
            self.performance_monitor.update_price_data(historical_data)
            
            # Run enhanced backtest with monitoring
            results = await self._execute_monitored_backtest_loop(
                historical_data, initial_balance, symbol
            )
            
            execution_time = time.time() - start_time
            
            # Generate comprehensive monitoring report
            monitoring_report = await self._generate_backtest_monitoring_report(results)
            
            # Combine backtest results with monitoring insights
            final_results = {
                **results,
                'monitoring_insights': monitoring_report,
                'execution_time_seconds': execution_time,
                'backtest_session_id': self.backtest_session_id,
                'agent_config_tested': agent_config or {},
                'recommendations': monitoring_report.get('optimization_recommendations', [])
            }
            
            logger.info(f"✅ Monitored backtest completed in {execution_time:.2f} seconds")
            logger.info(f"📊 Agent decisions tracked: {sum(self.agent_decisions_count.values())}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Monitored backtest failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'partial_monitoring_data': self.performance_monitor.get_real_time_dashboard_data()
            }
        
        finally:
            # Cleanup
            await self._cleanup_backtest_monitoring()
    
    # Add these methods to your MonitoredBacktester class

    def _update_positions(self, current_bar: Dict):
        """Update existing positions with current market prices"""
        
        current_price = current_bar['close']
        current_time = current_bar.get('timestamp', datetime.now().isoformat())
        
        # Update open trades
        for trade in self.trades:
            if not trade.is_closed:
                # Update unrealized P&L
                if trade.action == 'buy':
                    unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
                else:  # sell
                    unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
                
                # Use setattr to avoid attribute assignment issues
                setattr(trade, 'unrealized_pnl', unrealized_pnl)
                
                # Check stop loss
                if trade.stop_loss:
                    if ((trade.action == 'buy' and current_price <= trade.stop_loss) or
                        (trade.action == 'sell' and current_price >= trade.stop_loss)):
                        
                        # Close trade at stop loss
                        trade.exit_price = trade.stop_loss
                        trade.exit_timestamp = current_time
                        trade.exit_reason = 'stop_loss'
                        trade.is_closed = True
                        trade.pnl = trade.stop_loss - trade.entry_price if trade.action == 'buy' else trade.entry_price - trade.stop_loss
                        trade.pnl *= trade.quantity
                        
                        # Update account balance
                        self.current_balance += trade.pnl
                        
                        logger.debug(f"💥 Stop loss hit: {trade.id}, P&L: ${trade.pnl:.2f}")
                
                # Check take profit
                if trade.take_profit:
                    if ((trade.action == 'buy' and current_price >= trade.take_profit) or
                        (trade.action == 'sell' and current_price <= trade.take_profit)):
                        
                        # Close trade at take profit
                        trade.exit_price = trade.take_profit
                        trade.exit_timestamp = current_time
                        trade.exit_reason = 'take_profit'
                        trade.is_closed = True
                        trade.pnl = trade.take_profit - trade.entry_price if trade.action == 'buy' else trade.entry_price - trade.take_profit
                        trade.pnl *= trade.quantity
                        
                        # Update account balance
                        self.current_balance += trade.pnl
                        
                        logger.debug(f"🎯 Take profit hit: {trade.id}, P&L: ${trade.pnl:.2f}")

    def _calculate_final_backtest_results(self) -> Dict[str, Any]:
        """Calculate final backtest results and statistics"""
        
        # Basic statistics
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.is_closed]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        # Calculate returns
        total_pnl = sum(t.pnl for t in closed_trades)
        total_return_pct = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # Win rate
        win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
        
        # Average wins/losses
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Maximum drawdown
        running_balance = self.initial_capital
        peak_balance = self.initial_capital
        max_drawdown = 0
        
        for trade in closed_trades:
            running_balance += trade.pnl
            if running_balance > peak_balance:
                peak_balance = running_balance
            
            current_drawdown = (peak_balance - running_balance) / peak_balance * 100
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
        
        # Sharpe ratio (simplified)
        if closed_trades:
            returns = [t.pnl / self.initial_capital for t in closed_trades]
            avg_return = sum(returns) / len(returns)
            return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'success': True,
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_balance,
            'trades': [self._trade_to_dict(t) for t in self.trades]
        }

    def _trade_to_dict(self, trade) -> Dict[str, Any]:
        """Convert trade object to dictionary for JSON serialization"""
        
        return {
            'id': trade.id,
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'action': trade.action,
            'entry_price': float(trade.entry_price) if trade.entry_price else None,
            'exit_price': float(trade.exit_price) if trade.exit_price else None,
            'quantity': float(trade.quantity) if trade.quantity else None,
            'pnl': float(trade.pnl) if trade.pnl else 0.0,
            'confidence': float(trade.confidence) if trade.confidence else 0.0,
            'wyckoff_phase': trade.wyckoff_phase,
            'pattern_type': trade.pattern_type,
            'reasoning': trade.reasoning,
            'agent_name': trade.agent_name,
            'is_closed': trade.is_closed,
            'exit_reason': trade.exit_reason
        }

    async def _update_decision_outcomes_for_closed_trades(self):
        """Update monitoring outcomes for trades that just closed"""
        
        for trade in self.trades:
            if (trade.is_closed and 
                hasattr(trade, 'decision_id') and 
                getattr(trade, 'decision_id', None) and
                not hasattr(trade, 'outcome_updated')):
            
                try:
                    # Update the decision outcome
                    decision_id = getattr(trade, 'decision_id', None)
                    if decision_id is not None:
                        await self.performance_monitor.update_decision_outcome(
                            decision_id=decision_id,
                            outcome_positive=trade.pnl > 0,
                            outcome_value=trade.pnl
                        )
                    # Mark as updated to avoid double-updating
                    setattr(trade, 'outcome_updated', True)
                    
                    logger.debug(f"📊 Updated decision outcome: {getattr(trade, 'decision_id', 'unknown')} -> P&L: ${trade.pnl:.2f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update decision outcome for {getattr(trade, 'decision_id', 'unknown')}: {e}")

    def _close_remaining_trades(self, final_bar: Dict):
        """Close any remaining open trades at the end of backtest"""
        final_price = final_bar['close']
        final_time = final_bar.get('timestamp', datetime.now().isoformat())
        
        for trade in self.trades:
            if not trade.is_closed:
                # Close at final price
                trade.exit_price = final_price
                trade.exit_timestamp = final_time
                trade.exit_reason = 'backtest_end'
                trade.is_closed = True
                
                # Calculate P&L
                if trade.action == 'buy':
                    trade.pnl = (final_price - trade.entry_price) * trade.quantity
                else:  # sell
                    trade.pnl = (trade.entry_price - final_price) * trade.quantity
                
                # Update balance
                self.current_balance += trade.pnl
        
    
    async def _execute_monitored_backtest_loop(
        self,
        historical_data: List[Dict],
        initial_balance: float,
        symbol: str
    ) -> Dict[str, Any]:
        """Execute backtest loop with agent decision monitoring"""
        
        self.initial_capital = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        
        logger.info(f"📊 Processing {len(historical_data)} historical bars...")
        
        for i, current_bar in enumerate(historical_data):
            try:
                # Simulate realistic market context for this bar
                market_context = await self._create_market_context(
                    current_bar, historical_data[:i+1], symbol
                )
                
                # Execute agents with monitoring for this bar
                agent_decisions = await self._execute_monitored_agents(
                    current_bar, market_context, symbol, i
                )
                
                # Process any trading decisions
                for decision in agent_decisions:
                    if decision.get('action') in ['buy', 'sell']:
                        await self._execute_monitored_trade(decision, current_bar, i)
                
                # Update existing positions
                self._update_positions(current_bar)
                
                # Update decision outcomes for closed trades
                await self._update_decision_outcomes_for_closed_trades()
                
                # Log progress every 50 bars
                if (i + 1) % 50 == 0:
                    progress = ((i + 1) / len(historical_data)) * 100
                    decisions_tracked = sum(self.agent_decisions_count.values())
                    logger.info(f"📈 Progress: {progress:.1f}% ({decisions_tracked} decisions tracked)")
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing bar {i}: {e}")
                continue
        
        final_bar = historical_data[-1]
        self._close_remaining_trades(final_bar)
        # Calculate final results
        final_results = self._calculate_final_backtest_results()
        
        return final_results
    
    async def _execute_monitored_agents(
        self,
        current_bar: Dict,
        market_context: Dict,
        symbol: str,
        bar_index: int
    ) -> List[Dict[str, Any]]:
        """Execute agents and monitor their decisions"""
        
        agent_decisions = []
        
        # 1. Market Analyst Decision
        analyst_decision = await self._execute_monitored_market_analyst(
            current_bar, market_context, symbol, bar_index
        )
        if analyst_decision:
            agent_decisions.append(analyst_decision)
            self.agent_decisions_count['market_analyst'] += 1
        
        # 2. Risk Manager Decision (if analyst found a signal)
        if analyst_decision and analyst_decision.get('signal_detected'):
            risk_decision = await self._execute_monitored_risk_manager(
                analyst_decision, current_bar, market_context, symbol, bar_index
            )
            if risk_decision:
                agent_decisions.append(risk_decision)
                self.agent_decisions_count['risk_manager'] += 1
        
        # 3. Trading Coordinator Decision (if risk approved)
        if len(agent_decisions) >= 2 and agent_decisions[-1].get('risk_approved'):
            coordinator_decision = await self._execute_monitored_trading_coordinator(
                agent_decisions, current_bar, market_context, symbol, bar_index
            )
            if coordinator_decision:
                agent_decisions.append(coordinator_decision)
                self.agent_decisions_count['trading_coordinator'] += 1
        
        return agent_decisions
    
    async def _execute_monitored_market_analyst(
        self,
        current_bar: Dict,
        market_context: Dict,
        symbol: str,
        bar_index: int
    ) -> Optional[Dict[str, Any]]:
        """Execute market analyst with monitoring"""
        
        start_time = time.time()
        
        # Simulate market analysis decision
        analysis_input = {
            'current_price': current_bar['close'],
            'historical_data': market_context.get('recent_bars', []),
            'market_regime': market_context.get('regime_info', {}),
            'symbol': symbol,
            'bar_index': bar_index
        }
        
        # Simulate Wyckoff analysis (replace with actual agent logic)
        wyckoff_analysis = self._simulate_wyckoff_analysis(analysis_input)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Track this decision with monitoring
        decision_id = await self.performance_monitor.track_agent_decision(
            agent_name="wyckoff_market_analyst",
            decision_type="market_analysis",
            input_data=analysis_input,
            output_data=wyckoff_analysis,
            confidence=wyckoff_analysis.get('confidence', 50.0),
            tools_used=['wyckoff_analyzer', 'pattern_detection'],
            execution_time_ms=execution_time,
            market_context=market_context
        )
        
        # Return decision for next agent
        return {
            'agent': 'market_analyst',
            'decision_id': decision_id,
            'signal_detected': wyckoff_analysis.get('signal_strength', 0) > 60,
            'analysis': wyckoff_analysis,
            'confidence': wyckoff_analysis.get('confidence', 50.0)
        }
    
    async def _execute_monitored_risk_manager(
        self,
        analyst_decision: Dict,
        current_bar: Dict,
        market_context: Dict,
        symbol: str,
        bar_index: int
    ) -> Optional[Dict[str, Any]]:
        """Execute risk manager with monitoring"""
        
        start_time = time.time()
        
        # Simulate risk assessment
        risk_input = {
            'signal': analyst_decision['analysis'],
            'current_price': current_bar['close'],
            'account_balance': self.current_balance,
            'existing_positions': len(self.trades),
            'market_volatility': market_context.get('volatility_level', 0.5)
        }
        
        # Simulate risk calculation
        risk_assessment = self._simulate_risk_assessment(risk_input)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Track this decision
        decision_id = await self.performance_monitor.track_agent_decision(
            agent_name="wyckoff_risk_manager",
            decision_type="risk_assessment",
            input_data=risk_input,
            output_data=risk_assessment,
            confidence=risk_assessment.get('confidence', 50.0),
            tools_used=['position_sizing', 'risk_calculator'],
            execution_time_ms=execution_time,
            market_context=market_context
        )
        
        return {
            'agent': 'risk_manager',
            'decision_id': decision_id,
            'risk_approved': risk_assessment.get('risk_score', 0) <= 0.02,  # 2% max risk
            'position_size': risk_assessment.get('position_size', 0),
            'stop_loss': risk_assessment.get('stop_loss', 0),
            'confidence': risk_assessment.get('confidence', 50.0)
        }
    
    async def _execute_monitored_trading_coordinator(
        self,
        prior_decisions: List[Dict],
        current_bar: Dict,
        market_context: Dict,
        symbol: str,
        bar_index: int
    ) -> Optional[Dict[str, Any]]:
        """Execute trading coordinator with monitoring"""
        
        start_time = time.time()
        
        # Get analysis and risk decisions
        analysis = prior_decisions[0]['analysis']
        risk_params = prior_decisions[1]
        
        # Simulate trading decision
        trading_input = {
            'market_analysis': analysis,
            'risk_parameters': risk_params,
            'current_price': current_bar['close'],
            'market_conditions': market_context
        }
        
        # Simulate trading decision
        trading_decision = self._simulate_trading_decision(trading_input)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Track this decision
        decision_id = await self.performance_monitor.track_agent_decision(
            agent_name="wyckoff_trading_coordinator",
            decision_type="trading_decision",
            input_data=trading_input,
            output_data=trading_decision,
            confidence=trading_decision.get('confidence', 50.0),
            tools_used=['trade_execution', 'order_management'],
            execution_time_ms=execution_time,
            market_context=market_context
        )
        
        return {
            'agent': 'trading_coordinator',
            'decision_id': decision_id,
            'action': trading_decision.get('action'),  # 'buy', 'sell', or 'wait'
            'entry_price': current_bar['close'],
            'position_size': risk_params['position_size'],
            'stop_loss': risk_params['stop_loss'],
            'confidence': trading_decision.get('confidence', 50.0)
        }
    
    async def _execute_monitored_trade(
        self,
        decision: Dict,
        current_bar: Dict,
        bar_index: int
    ):
        """Execute trade and update decision outcomes"""
        
        # Create trade
        trade = self._create_backtest_trade(decision, current_bar, bar_index)
        self.trades.append(trade)
        
        # Update decision outcome when trade is created
        if 'decision_id' in decision:
            await self.performance_monitor.update_decision_outcome(
                decision_id=decision['decision_id'],
                outcome_positive=True,  # Initial outcome (trade executed)
                outcome_value=0.0  # Will be updated when trade closes
            )
    
    def _create_backtest_trade(self, decision: Dict, current_bar: Dict, bar_index: int):
        """Create a backtest trade from agent decision"""
        from src.backtesting.enhanced_agent_backtest import BacktestTrade
        
        return BacktestTrade(
            id=f"backtest_{bar_index}_{decision['agent']}",
            timestamp=current_bar.get('timestamp', datetime.now().isoformat()),
            symbol='EUR_USD',  # Or from context
            action=decision['action'],
            entry_price=decision['entry_price'],
            quantity=decision['position_size'],
            stop_loss=decision.get('stop_loss', 0.0),
            take_profit=decision.get('take_profit', 0.0),
            confidence=decision['confidence'],
            wyckoff_phase='backtest',
            pattern_type='backtest_pattern',
            reasoning=f"Backtest decision from {decision['agent']}",
            agent_name=decision['agent']
        )
    
    async def _create_market_context(
    self,
    current_bar: Dict,
    historical_bars: List[Dict],
    symbol: str
) -> Dict[str, Any]:
        """Create market context for agent decisions - FIXED VERSION"""
        
        # Ensure current_bar is a dictionary
        if isinstance(current_bar, str):
            logger.warning(f"⚠️ current_bar is string, skipping context creation")
            return {
                'current_bar': {'close': 1.0, 'timestamp': datetime.now().isoformat()},
                'recent_bars': [],
                'regime_info': {},
                'symbol': symbol,
                'volatility_level': 0.5,
                'session_time': 'unknown'
            }
        
        if not isinstance(current_bar, dict):
            logger.warning(f"⚠️ current_bar is not dict: {type(current_bar)}")
            current_bar = {'close': 1.0, 'timestamp': datetime.now().isoformat()}
        
        # Ensure historical_bars are dictionaries
        valid_historical_bars = []
        for bar in historical_bars:
            if isinstance(bar, dict) and 'close' in bar:
                valid_historical_bars.append(bar)
            elif isinstance(bar, str):
                logger.debug(f"⚠️ Skipping string bar in historical data")
            else:
                logger.debug(f"⚠️ Skipping invalid bar type: {type(bar)}")
        
        # Get regime detection if available
        regime_info = {}
        if (hasattr(self.performance_monitor, 'regime_detector') and 
            len(valid_historical_bars) > 20 and 
            'close' in current_bar):
            
            try:
                regime_info = self.performance_monitor.regime_detector.detect_current_regime(
                    valid_historical_bars[-50:], current_bar['close']
                )
            except Exception as e:
                logger.debug(f"Regime detection failed: {e}")
                regime_info = {'primary_regime': 'unknown'}
        
        return {
            'current_bar': current_bar,
            'recent_bars': valid_historical_bars[-20:],  # Last 20 valid bars
            'regime_info': regime_info,
            'symbol': symbol,
            'volatility_level': self._calculate_volatility(valid_historical_bars[-14:] if len(valid_historical_bars) >= 14 else valid_historical_bars),
            'session_time': self._get_session_time(current_bar)
        }
    
    def _simulate_wyckoff_analysis(self, input_data: Dict) -> Dict[str, Any]:
        """Simulate Wyckoff analysis (replace with actual agent logic)"""
        import random
        
        # Simulate realistic Wyckoff analysis
        patterns = ['accumulation', 'distribution', 'spring', 'upthrust', 'none']
        phases = ['A', 'B', 'C', 'D', 'E']
        
        signal_strength = random.uniform(30, 90)
        confidence = random.uniform(40, 95)
        
        return {
            'pattern': random.choice(patterns),
            'phase': random.choice(phases),
            'signal_strength': signal_strength,
            'confidence': confidence,
            'support_level': input_data['current_price'] * 0.995,
            'resistance_level': input_data['current_price'] * 1.005,
            'target_price': input_data['current_price'] * (1.01 if signal_strength > 60 else 0.99)
        }
    
    def _simulate_risk_assessment(self, input_data: Dict) -> Dict[str, Any]:
        """Simulate risk assessment (replace with actual agent logic)"""
        import random
        
        # Simulate risk calculation
        signal_confidence = input_data['signal']['confidence']
        risk_score = (100 - signal_confidence) / 100 * 0.03  # Higher confidence = lower risk
        
        # Position sizing based on risk
        max_position = input_data['account_balance'] * 0.1  # Max 10% of balance
        position_size = max_position * (signal_confidence / 100)
        
        return {
            'risk_score': risk_score,
            'position_size': position_size,
            'stop_loss': input_data['current_price'] * 0.99,
            'take_profit': input_data['current_price'] * 1.03,
            'confidence': min(signal_confidence + random.uniform(-10, 10), 95)
        }
    
    def _simulate_trading_decision(self, input_data: Dict) -> Dict[str, Any]:
        """Simulate trading decision (replace with actual agent logic)"""
        import random
        
        # Simulate trading decision based on analysis
        analysis = input_data['market_analysis']
        
        if analysis['signal_strength'] > 70:
            action = 'buy' if analysis['pattern'] in ['accumulation', 'spring'] else 'sell'
        elif analysis['signal_strength'] > 50:
            action = random.choice(['buy', 'sell', 'wait'])
        else:
            action = 'wait'
        
        return {
            'action': action,
            'reasoning': f"Signal strength {analysis['signal_strength']:.1f}, pattern: {analysis['pattern']}",
            'confidence': analysis['confidence'] * random.uniform(0.9, 1.1)
        }
    
    async def _generate_backtest_monitoring_report(self, backtest_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive monitoring report for backtest"""
        
        # Get monitoring dashboard data
        dashboard_data = self.performance_monitor.get_real_time_dashboard_data()
        
        # Analyze agent performance
        agent_performance = {}
        for agent_name, metrics in dashboard_data.get('agents', {}).items():
            agent_performance[agent_name] = {
                'accuracy': metrics.get('accuracy_score', 0),
                'confidence_calibration': metrics.get('confidence_calibration', 0),
                'total_decisions': metrics.get('total_decisions', 0),
                'avg_confidence': self._calculate_avg_confidence(agent_name),
                'decision_speed': self._calculate_avg_decision_time(agent_name)
            }
        
        # Market regime analysis
        regime_performance = self._analyze_regime_performance()
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            agent_performance, regime_performance, backtest_results
        )
        
        return {
            'backtest_session_id': self.backtest_session_id,
            'total_decisions_tracked': sum(self.agent_decisions_count.values()),
            'agent_performance': agent_performance,
            'regime_performance': regime_performance,
            'optimization_recommendations': recommendations,
            'confidence_vs_outcomes': self._analyze_confidence_vs_outcomes(),
            'tool_efficiency': dashboard_data.get('agents', {}),
            'session_summary': {
                'duration_bars': len(backtest_results.get('trades', [])),
                'total_return': backtest_results.get('total_return_pct', 0),
                'decisions_per_trade': sum(self.agent_decisions_count.values()) / max(len(backtest_results.get('trades', [])), 1)
            }
        }
    
    def _generate_optimization_recommendations(
        self,
        agent_performance: Dict,
        regime_performance: Dict,
        backtest_results: Dict
    ) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        
        # Agent-specific recommendations
        for agent_name, metrics in agent_performance.items():
            if metrics['accuracy'] < 60:
                recommendations.append(f"🔴 {agent_name}: Low accuracy ({metrics['accuracy']:.1f}%) - Review prompt engineering")
            
            if abs(metrics['confidence_calibration']) < 0.3:
                recommendations.append(f"🟡 {agent_name}: Poor confidence calibration - Adjust confidence scoring")
            
            if metrics['decision_speed'] > 3000:  # > 3 seconds
                recommendations.append(f"⏱️ {agent_name}: Slow decisions ({metrics['decision_speed']:.1f}ms) - Optimize processing")
        
        # Overall performance recommendations
        if backtest_results.get('total_return_pct', 0) < 5:
            recommendations.append("📈 Low overall returns - Consider lowering confidence thresholds")
        
        if backtest_results.get('max_drawdown', 0) > 15:
            recommendations.append("🛡️ High drawdown - Improve risk management parameters")
        
        # Market regime recommendations
        worst_regime = min(regime_performance.items(), key=lambda x: x[1].get('win_rate', 0), default=(None, {}))
        if worst_regime[0]:
            recommendations.append(f"🌍 Poor performance in {worst_regime[0]} markets - Adapt strategy")
        
        return recommendations
    
    def _calculate_avg_confidence(self, agent_name: str) -> float:
        """Calculate average confidence for an agent"""
        agent_decisions = [d for d in self.performance_monitor.decisions.values() 
                      if d.agent_name == agent_name and hasattr(d, 'confidence')]
        if not agent_decisions:
            return 0.0
        
        confidences = [d.confidence for d in agent_decisions if d.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _calculate_avg_decision_time(self, agent_name: str) -> float:
        """Calculate average decision time for an agent"""
        agent_decisions = [d for d in self.performance_monitor.decisions.values() if d.agent_name == agent_name]
        if not agent_decisions:
            return 0.0
        return sum(d.execution_time_ms for d in agent_decisions) / len(agent_decisions)
    
    def _analyze_regime_performance(self) -> Dict[str, Dict]:
        """Analyze performance by market regime"""
        regime_performance = {}
        
        for decision in self.performance_monitor.decisions.values():
            if decision.outcome_known:
                regime = decision.market_context.get('regime_info', {}).get('primary_regime', 'unknown')
                
                if regime not in regime_performance:
                    regime_performance[regime] = {
                        'total_decisions': 0,
                        'positive_outcomes': 0,
                        'total_pnl': 0,
                        'win_rate': 0
                    }
                
                regime_performance[regime]['total_decisions'] += 1
                if decision.outcome_positive:
                    regime_performance[regime]['positive_outcomes'] += 1
                regime_performance[regime]['total_pnl'] += decision.outcome_value
        
        # Calculate win rates
        for regime_data in regime_performance.values():
            if regime_data['total_decisions'] > 0:
                regime_data['win_rate'] = (regime_data['positive_outcomes'] / regime_data['total_decisions']) * 100
        
        return regime_performance
    
    def _analyze_confidence_vs_outcomes(self) -> Dict[str, float]:
        """Analyze correlation between confidence and outcomes"""
        
        confidences = []
        outcomes = []
        
        for decision in self.performance_monitor.decisions.values():
            if (decision.outcome_known and 
                hasattr(decision, 'confidence') and 
                decision.confidence is not None and
                hasattr(decision, 'outcome_positive')):
                
                confidences.append(decision.confidence)
                outcomes.append(1 if decision.outcome_positive else 0)
        
        if len(confidences) < 2:
            return {
                'correlation': 0.0, 
                'sample_size': len(confidences),
                'avg_confidence': 0.0,
                'success_rate': 0.0
            }
        
        import numpy as np
        correlation = np.corrcoef(confidences, outcomes)[0, 1] if len(confidences) > 1 else 0.0
        
        return {
            'correlation': float(correlation if not np.isnan(correlation) else 0.0),
            'sample_size': float(len(confidences)),
            'avg_confidence': float(np.mean(confidences)),
            'success_rate': float(np.mean(outcomes) * 100)
        }
    
    def _calculate_volatility(self, bars: List[Dict]) -> float:
        """Calculate simple volatility measure"""
        if len(bars) < 2:
            return 0.5
        
        price_changes = []
        for i in range(1, len(bars)):
            change = abs(bars[i]['close'] - bars[i-1]['close']) / bars[i-1]['close']
            price_changes.append(change)
        
        return sum(price_changes) / len(price_changes) if price_changes else 0.5
    
    def _get_session_time(self, bar: Dict) -> str:
        """Determine trading session from timestamp"""
        try:
            timestamp = datetime.fromisoformat(bar.get('timestamp', ''))
            hour = timestamp.hour
            
            if 7 <= hour < 15:
                return 'london_session'
            elif 13 <= hour < 21:
                return 'ny_session'
            elif 22 <= hour or hour < 7:
                return 'asia_session'
            else:
                return 'off_hours'
        except:
            return 'unknown'
    
    async def _apply_agent_configuration(self, config: Dict[str, Any]):
        """Apply configuration changes to agents for A/B testing"""
        
        # This would modify agent parameters for testing
        # Example configurations:
        if 'confidence_threshold' in config:
            logger.info(f"📊 Testing confidence threshold: {config['confidence_threshold']}")
        
        if 'temperature_settings' in config:
            logger.info(f"🌡️ Testing temperature settings: {config['temperature_settings']}")
        
        if 'prompt_variations' in config:
            logger.info(f"💬 Testing prompt variations: {list(config['prompt_variations'].keys())}")
    
    async def _cleanup_backtest_monitoring(self):
        """Clean up monitoring resources"""
        try:
            if self.monitored_system:
                await self.monitored_system.shutdown_monitoring()
            await self.performance_monitor.stop_monitoring()
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")


# A/B Testing Framework for Agent Optimization
class AgentOptimizationTester:
    """Framework for A/B testing different agent configurations"""
    
    def __init__(self):
        self.test_results = {}
    
    async def run_optimization_tests(
        self,
        historical_data: List[Dict],
        test_configurations: Dict[str, Dict],
        initial_balance: float = 100000
    ) -> Dict[str, Any]:
        """Run A/B tests on different agent configurations"""
        
        logger.info(f"🧪 Starting A/B tests with {len(test_configurations)} configurations...")
        
        test_results = {}
        
        for config_name, config in test_configurations.items():
            logger.info(f"🔬 Testing configuration: {config_name}")
            
            # Run monitored backtest with this configuration
            backtester = MonitoredBacktester()
            
            result = await backtester.run_monitored_backtest(
                historical_data=historical_data,
                initial_balance=initial_balance,
                symbol='EUR_USD',
                agent_config=config
            )
            
            test_results[config_name] = result
            
            # Log quick summary
            if result.get('success'):
                total_return = result.get('total_return_pct', 0)
                decisions_tracked = result.get('monitoring_insights', {}).get('total_decisions_tracked', 0)
                logger.info(f"✅ {config_name}: {total_return:+.2f}% return, {decisions_tracked} decisions")
            else:
                logger.warning(f"❌ {config_name}: Test failed")
        
        # Compare results
        comparison = self._compare_test_results(test_results)
        
        return {
            'individual_results': test_results,
            'comparison': comparison,
            'recommendation': self._generate_optimization_recommendation(comparison)
        }
    
    def _compare_test_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare A/B test results"""
        
        comparison = {
            'performance_ranking': [],
            'key_metrics': {},
            'statistical_significance': {}
        }
        
        # Rank by total return
        valid_results = {k: v for k, v in results.items() if v.get('success')}
        
        if valid_results:
            ranked = sorted(
                valid_results.items(),
                key=lambda x: x[1].get('total_return_pct', 0),
                reverse=True
            )
            
            comparison['performance_ranking'] = [
                {
                    'config': name,
                    'return_pct': result.get('total_return_pct', 0),
                    'decisions_tracked': result.get('monitoring_insights', {}).get('total_decisions_tracked', 0),
                    'avg_accuracy': self._get_avg_accuracy(result)
                }
                for name, result in ranked
            ]
        
        return comparison
    
    def _get_avg_accuracy(self, result: Dict) -> float:
        """Get average accuracy across all agents"""
        monitoring = result.get('monitoring_insights', {})
        agent_performance = monitoring.get('agent_performance', {})
        
        if not agent_performance:
            return 0.0
        
        accuracies = [metrics.get('accuracy', 0) for metrics in agent_performance.values()]
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    def _generate_optimization_recommendation(self, comparison: Dict) -> str:
        """Generate recommendation based on test results"""
        
        ranking = comparison.get('performance_ranking', [])
        
        if not ranking:
            return "No valid test results to compare"
        
        best_config = ranking[0]
        
        if len(ranking) == 1:
            return f"Only one configuration tested: {best_config['config']}"
        
        second_best = ranking[1]
        improvement = best_config['return_pct'] - second_best['return_pct']
        
        if improvement > 1.0:  # More than 1% improvement
            return f"✅ Recommend {best_config['config']} - {improvement:+.2f}% better than {second_best['config']}"
        elif improvement > 0.5:
            return f"⚠️ Marginal improvement with {best_config['config']} - consider more testing"
        else:
            return f"🤔 No clear winner - results within margin of error"


# Example usage
async def example_monitored_backtest():
    """Example of running monitored backtest with optimization"""
    
    # 1. Run baseline backtest with monitoring
    backtester = MonitoredBacktester()
    
    # Generate or load historical data
    historical_data = []  # Your historical data here
    
    baseline_result = await backtester.run_monitored_backtest(
        historical_data=historical_data,
        initial_balance=100000,
        symbol='EUR_USD',
        agent_config=None  # Default configuration
    )
    
    logger.info("📊 Baseline Backtest Results:")
    logger.info(f"   Return: {baseline_result.get('total_return_pct', 0):+.2f}%")
    logger.info(f"   Decisions: {baseline_result.get('monitoring_insights', {}).get('total_decisions_tracked', 0)}")
    
    # 2. Run A/B tests on optimizations
    optimizer = AgentOptimizationTester()
    
    test_configs = {
        'baseline': {},
        'lower_confidence': {'confidence_threshold': 65},
        'higher_creativity': {'temperature_settings': {'market_analyst': 0.25}},
        'conservative_risk': {'risk_multiplier': 0.8}
    }
    
    optimization_results = await optimizer.run_optimization_tests(
        historical_data=historical_data,
        test_configurations=test_configs,
        initial_balance=100000
    )
    
    logger.info("🏆 Optimization Results:")
    for rank, result in enumerate(optimization_results['comparison']['performance_ranking'], 1):
        logger.info(f"   {rank}. {result['config']}: {result['return_pct']:+.2f}% ({result['decisions_tracked']} decisions)")
    
    logger.info(f"📝 Recommendation: {optimization_results['recommendation']}")
    
    return optimization_results

if __name__ == "__main__":
    asyncio.run(example_monitored_backtest())
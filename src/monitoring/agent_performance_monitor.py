"""
Advanced Agent Performance Monitoring System
Real-time tracking and analysis of CrewAI agent performance for optimization
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
import threading
from collections import defaultdict, deque

# Assuming these are available from your existing system
from src.config.logging_config import logger
from src.database.manager import db_manager
from src.database.models import AgentAction, LogLevel
from src.monitoring.market_regime_detector import MarketRegimeDetector, MarketRegime as EnhancedMarketRegime
from src.monitoring.decision_scorer import EnhancedDecisionScorer, DecisionContext

class DecisionQuality(Enum):
    """Quality rating for agent decisions"""
    EXCELLENT = "excellent"  # >80% confidence, positive outcome
    GOOD = "good"           # >60% confidence, positive outcome
    POOR = "poor"           # High confidence, negative outcome
    UNCERTAIN = "uncertain"  # Low confidence, any outcome

class MarketRegime(Enum):
    """Market condition types for performance analysis"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NEWS_EVENT = "news_event"

@dataclass
class AgentDecision:
    """Detailed tracking of individual agent decisions"""
    decision_id: str
    agent_name: str
    timestamp: datetime
    decision_type: str  # 'analysis', 'risk_assessment', 'trade_decision'
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    tools_used: List[str]
    execution_time_ms: float
    market_context: Dict[str, Any]
    session_id: str
    
    # Outcome tracking (filled later)
    outcome_known: bool = False
    outcome_positive: bool = False
    outcome_value: float = 0.0  # P&L or accuracy score
    outcome_timestamp: Optional[datetime] = None
    decision_quality: Optional[DecisionQuality] = None

@dataclass
class ToolUsageMetrics:
    """Metrics for tool usage efficiency"""
    tool_name: str
    total_uses: int = 0
    successful_uses: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    usage_trend: List[int] = field(default_factory=list)  # Usage count per hour

@dataclass
class AgentPerformanceSnapshot:
    """Real-time performance snapshot for an agent"""
    agent_name: str
    timestamp: datetime
    
    # Decision metrics
    total_decisions: int = 0
    high_confidence_decisions: int = 0
    decisions_last_hour: int = 0
    
    # Quality metrics
    accuracy_score: float = 0.0  # % of decisions with positive outcomes
    confidence_calibration: float = 0.0  # Correlation between confidence and success
    consistency_score: float = 0.0  # How consistent decisions are
    
    # Tool usage
    tool_efficiency: Dict[str, float] = field(default_factory=dict)
    most_used_tool: str = ""
    
    # Performance trends
    performance_trend_1h: float = 0.0  # % change in last hour
    performance_trend_24h: float = 0.0  # % change in last 24 hours
    
    # Market adaptation
    current_market_regime: MarketRegime = MarketRegime.RANGING
    adaptation_speed: float = 0.0  # How quickly agent adapts to regime changes

class PerformanceMonitor:
    """Real-time agent performance monitoring system"""
    
    def __init__(self, db_path: str = "agent_performance.db"):
        self.db_path = db_path
        self.decisions: Dict[str, AgentDecision] = {}
        self.agent_snapshots: Dict[str, AgentPerformanceSnapshot] = {}
        self.tool_metrics: Dict[str, ToolUsageMetrics] = {}
        self.market_regime_history: deque = deque(maxlen=1000)
        
        #Monitoring
        self.regime_detector = MarketRegimeDetector()
        self.decision_scorer = EnhancedDecisionScorer()
        self.price_history = []  # Store recent price data for regime detection
        
        # Performance tracking
        self.session_id = str(uuid.uuid4())
        self.monitoring_active = False
        self.alert_callbacks: List[Callable[[str], Awaitable[None]]] = []
        
        # Initialize database
        self._init_monitoring_db()
        
    def _init_monitoring_db(self):
        """Initialize SQLite database for performance monitoring"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    decision_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    decision_type TEXT,
                    confidence REAL,
                    tools_used TEXT,
                    execution_time_ms REAL,
                    market_regime TEXT,
                    outcome_positive BOOLEAN,
                    outcome_value REAL,
                    session_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    accuracy_score REAL,
                    confidence_calibration REAL,
                    consistency_score REAL,
                    decisions_count INTEGER,
                    session_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_usage (
                    tool_name TEXT,
                    agent_name TEXT,
                    timestamp TEXT,
                    success BOOLEAN,
                    execution_time_ms REAL,
                    session_id TEXT
                )
            """)
    
    async def start_monitoring(self):
        """Start the performance monitoring system"""
        self.monitoring_active = True
        logger.info("🔍 Agent Performance Monitoring Started")
        
        # Start background tasks
        asyncio.create_task(self._snapshot_loop())
        asyncio.create_task(self._alert_loop())
        
    async def stop_monitoring(self):
        """Stop monitoring and save final reports"""
        self.monitoring_active = False
        await self._generate_final_report()
        logger.info("📊 Agent Performance Monitoring Stopped")
    
    async def track_agent_decision(
        self, 
        agent_name: str,
        decision_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence: float,
        tools_used: List[str],
        execution_time_ms: float,
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track a new agent decision"""
        
        decision_id = f"{agent_name}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        
        decision = AgentDecision(
            decision_id=decision_id,
            agent_name=agent_name,
            timestamp=datetime.now(),
            decision_type=decision_type,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            tools_used=tools_used,
            execution_time_ms=execution_time_ms,
            market_context=market_context or {},
            session_id=self.session_id
        )
        
        # Store decision
        self.decisions[decision_id] = decision
        
        if market_context is None:
            market_context = {}
        
        # ADD ENHANCED REGIME DETECTION:
        if 'price_data' in input_data and len(self.price_history) > 0:
            # Use enhanced regime detection
            current_price = input_data.get('current_price', 0)
            regime_info = self.regime_detector.detect_current_regime(
                self.price_history, current_price
            )
            market_context.update(regime_info)
        
        # Update tool usage metrics
        await self._update_tool_metrics(agent_name, tools_used, execution_time_ms)
        
        # Save to database
        await self._save_decision_to_db(decision)
        
        # Update real-time snapshot
        await self._update_agent_snapshot(agent_name)
        
        logger.debug(f"📝 Tracked decision: {agent_name} -> {decision_type} (confidence: {confidence:.1f}%)")
        
        return decision_id
    
    
    def update_price_data(self, price_data: List[Dict[str, Any]]):
        """Update price history for regime detection"""
        self.price_history = price_data[-100:]  # Keep last 100 bars
        
    async def update_decision_outcome(
        self,
        decision_id: str,
        outcome_positive: bool,
        outcome_value: float = 0.0
    ):
        """Update the outcome of a previous decision"""
        
        if decision_id not in self.decisions:
            logger.warning(f"⚠️ Decision {decision_id} not found for outcome update")
            return
        
        decision = self.decisions[decision_id]
        decision.outcome_known = True
        decision.outcome_positive = outcome_positive
        decision.outcome_value = outcome_value
        decision.outcome_timestamp = datetime.now()
        
        # Determine decision quality
        decision.decision_quality = self._assess_decision_quality(decision)
        
        # Update agent performance
        await self._update_agent_snapshot(decision.agent_name)
        
        logger.debug(f"✅ Updated outcome: {decision_id} -> {'positive' if outcome_positive else 'negative'} ({outcome_value:.2f})")
    
    def _assess_decision_quality(self, decision) -> DecisionQuality:
        """Enhanced decision quality assessment"""
        
        # Get market context for scoring
        context = self._build_decision_context(decision)
        
        # Use enhanced scorer
        scoring_result = self.decision_scorer.score_decision(
            confidence=decision.confidence,
            outcome_positive=decision.outcome_positive,
            outcome_value=decision.outcome_value,
            context=context,
            execution_time_ms=decision.execution_time_ms
        )
        
        # Store additional scoring data
        decision.scoring_breakdown = scoring_result['component_scores']
        decision.improvement_recommendations = scoring_result['recommendations']
        decision.scoring_explanation = scoring_result['scoring_explanation']
        
        return scoring_result['quality_rating']
    
    def _build_decision_context(self, decision) -> DecisionContext:
        """Build enhanced context for decision scoring"""
        
        # Get recent agent performance
        agent_decisions = [d for d in self.decisions.values() 
                          if d.agent_name == decision.agent_name and d.outcome_known]
        
        recent_performance = 50.0  # Default
        if agent_decisions:
            recent_positive = len([d for d in agent_decisions[-10:] if d.outcome_positive])
            recent_performance = (recent_positive / min(len(agent_decisions), 10)) * 100
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        for d in reversed(agent_decisions):
            if d.outcome_positive:
                consecutive_wins += 1
                break
            else:
                consecutive_losses += 1
        
        # Extract market regime from decision context
        market_regime = decision.market_context.get('regime', 'unknown')
        if isinstance(market_regime, EnhancedMarketRegime):
            market_regime = market_regime.value
        
        return DecisionContext(
            market_regime=market_regime,
            volatility_level=decision.market_context.get('volatility_percentile', 50) / 100,
            session_time=decision.market_context.get('session_time', {}).get('current_session', 'unknown'),
            recent_performance=recent_performance,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            time_pressure=min(decision.execution_time_ms / 5000, 1.0),  # Normalize to 0-1
            complexity_score=len(decision.tools_used) / 10  # Normalize by tool count
        )
        
    
    async def _update_tool_metrics(self, agent_name: str, tools_used: List[str], execution_time: float):
        """Update tool usage metrics"""
        
        for tool in tools_used:
            key = f"{agent_name}_{tool}"
            
            if key not in self.tool_metrics:
                self.tool_metrics[key] = ToolUsageMetrics(tool_name=tool)
            
            metrics = self.tool_metrics[key]
            metrics.total_uses += 1
            metrics.last_used = datetime.now()
            
            # Update average execution time
            if metrics.avg_execution_time == 0:
                metrics.avg_execution_time = execution_time
            else:
                metrics.avg_execution_time = (metrics.avg_execution_time + execution_time) / 2
    
    async def _update_agent_snapshot(self, agent_name: str):
        """Update real-time performance snapshot for an agent"""
        
        now = datetime.now()
        
        # Get agent's decisions
        agent_decisions = [d for d in self.decisions.values() if d.agent_name == agent_name]
        recent_decisions = [d for d in agent_decisions if (now - d.timestamp).total_seconds() < 3600]  # Last hour
        decisions_with_outcomes = [d for d in agent_decisions if d.outcome_known]
        
        if not agent_decisions:
            return
        
        # Calculate metrics
        total_decisions = len(agent_decisions)
        high_confidence_decisions = len([d for d in agent_decisions if d.confidence >= 75])
        decisions_last_hour = len(recent_decisions)
        
        # Accuracy score
        accuracy_score = 0.0
        if decisions_with_outcomes:
            positive_outcomes = len([d for d in decisions_with_outcomes if d.outcome_positive])
            accuracy_score = (positive_outcomes / len(decisions_with_outcomes)) * 100
        
        # Confidence calibration
        confidence_calibration = 0.0
        if len(decisions_with_outcomes) > 1:
            confidences = [d.confidence for d in decisions_with_outcomes]
            outcomes = [1 if d.outcome_positive else 0 for d in decisions_with_outcomes]
            correlation = np.corrcoef(confidences, outcomes)[0, 1]
            confidence_calibration = correlation if not np.isnan(correlation) else 0.0
        
        # Tool efficiency
        agent_tools = {key.split('_', 1)[1]: metrics for key, metrics in self.tool_metrics.items() 
                      if key.startswith(f"{agent_name}_")}
        tool_efficiency = {tool: metrics.success_rate for tool, metrics in agent_tools.items()}
        most_used_tool = max(agent_tools.keys(), key=lambda t: agent_tools[t].total_uses) if agent_tools else ""
        
        # Create/update snapshot
        snapshot = AgentPerformanceSnapshot(
            agent_name=agent_name,
            timestamp=now,
            total_decisions=total_decisions,
            high_confidence_decisions=high_confidence_decisions,
            decisions_last_hour=decisions_last_hour,
            accuracy_score=accuracy_score,
            confidence_calibration=confidence_calibration,
            tool_efficiency=tool_efficiency,
            most_used_tool=most_used_tool
        )
        
        self.agent_snapshots[agent_name] = snapshot
        
        # Save to database
        await self._save_snapshot_to_db(snapshot)
    
    async def _save_decision_to_db(self, decision: AgentDecision):
        """Save decision to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO agent_decisions 
                    (decision_id, agent_name, timestamp, decision_type, confidence, 
                     tools_used, execution_time_ms, market_regime, outcome_positive, 
                     outcome_value, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.decision_id,
                    decision.agent_name,
                    decision.timestamp.isoformat(),
                    decision.decision_type,
                    decision.confidence,
                    json.dumps(decision.tools_used),
                    decision.execution_time_ms,
                    decision.market_context.get('regime', 'unknown'),
                    decision.outcome_positive if decision.outcome_known else None,
                    decision.outcome_value if decision.outcome_known else None,
                    decision.session_id
                ))
        except Exception as e:
            logger.error(f"❌ Failed to save decision to DB: {e}")
    
    async def _save_snapshot_to_db(self, snapshot: AgentPerformanceSnapshot):
        """Save performance snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_snapshots 
                    (snapshot_id, agent_name, timestamp, accuracy_score, 
                     confidence_calibration, consistency_score, decisions_count, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    snapshot.agent_name,
                    snapshot.timestamp.isoformat(),
                    snapshot.accuracy_score,
                    snapshot.confidence_calibration,
                    snapshot.consistency_score,
                    snapshot.total_decisions,
                    self.session_id
                ))
        except Exception as e:
            logger.error(f"❌ Failed to save snapshot to DB: {e}")
    
    async def _snapshot_loop(self):
        """Background loop to update performance snapshots"""
        while self.monitoring_active:
            try:
                for agent_name in set(d.agent_name for d in self.decisions.values()):
                    await self._update_agent_snapshot(agent_name)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"❌ Snapshot loop error: {e}")
                await asyncio.sleep(30)
    
    async def _alert_loop(self):
        """Background loop to check for performance alerts"""
        while self.monitoring_active:
            try:
                await self._check_performance_alerts()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Alert loop error: {e}")
                await asyncio.sleep(60)
    
    async def _check_performance_alerts(self):
        """Check for performance issues and trigger alerts"""
        
        for agent_name, snapshot in self.agent_snapshots.items():
            alerts = []
            
            # Low accuracy alert
            if snapshot.accuracy_score < 40 and snapshot.total_decisions > 10:
                alerts.append(f"🔴 {agent_name}: Low accuracy ({snapshot.accuracy_score:.1f}%)")
            
            # Poor confidence calibration
            if snapshot.confidence_calibration < -0.3:
                alerts.append(f"⚠️ {agent_name}: Poor confidence calibration ({snapshot.confidence_calibration:.2f})")
            
            # No recent decisions
            if snapshot.decisions_last_hour == 0 and snapshot.total_decisions > 0:
                alerts.append(f"⏰ {agent_name}: No decisions in last hour")
            
            # Tool efficiency issues
            inefficient_tools = [tool for tool, eff in snapshot.tool_efficiency.items() if eff < 0.3]
            if inefficient_tools:
                alerts.append(f"🔧 {agent_name}: Inefficient tools: {', '.join(inefficient_tools)}")
            
            # Trigger alerts
            for alert in alerts:
                await self._trigger_alert(alert)
    
    async def _trigger_alert(self, message: str):
        """Trigger a performance alert"""
        logger.warning(f"🚨 PERFORMANCE ALERT: {message}")
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"❌ Alert callback failed: {e}")

    def add_alert_callback(self, callback):
        """Add a callback function for performance alerts"""
        self.alert_callbacks.append(callback)

    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get current performance data for dashboard display"""
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "monitoring_active": self.monitoring_active,
            "agents": {},
            "overall_metrics": {},
            "alerts": []
        }
        
        dashboard_data["enhanced_metrics"] = {
            "current_market_regime": getattr(self.regime_detector, 'current_regime', 'unknown'),
            "regime_strength": getattr(self.regime_detector, 'regime_strength', 0),
            "decision_quality_distribution": self._get_quality_distribution(),
            "top_improvement_recommendations": self._get_top_recommendations()
        }
        
        # Agent-specific data
        for agent_name, snapshot in self.agent_snapshots.items():
            dashboard_data["agents"][agent_name] = {
                "accuracy_score": snapshot.accuracy_score,
                "confidence_calibration": snapshot.confidence_calibration,
                "total_decisions": snapshot.total_decisions,
                "decisions_last_hour": snapshot.decisions_last_hour,
                "most_used_tool": snapshot.most_used_tool,
                "tool_efficiency": snapshot.tool_efficiency
            }
        
        # Overall metrics
        if self.agent_snapshots:
            overall_accuracy = np.mean([s.accuracy_score for s in self.agent_snapshots.values()])
            overall_calibration = np.mean([s.confidence_calibration for s in self.agent_snapshots.values()])
            total_decisions = sum(s.total_decisions for s in self.agent_snapshots.values())
            
            dashboard_data["overall_metrics"] = {
                "overall_accuracy": overall_accuracy,
                "overall_calibration": overall_calibration,
                "total_decisions": total_decisions,
                "active_agents": len(self.agent_snapshots)
            }
        
        return dashboard_data
    
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of decision qualities"""
        from collections import defaultdict
        distribution = defaultdict(int)
        
        for decision in self.decisions.values():
            if hasattr(decision, 'decision_quality') and decision.decision_quality:
                distribution[decision.decision_quality.value] += 1
        
        return dict(distribution)
    
    def _get_top_recommendations(self) -> List[str]:
        """Get most common improvement recommendations"""
        from collections import Counter
        all_recommendations = []
        
        for decision in self.decisions.values():
            improvement_recommendations = getattr(decision, 'improvement_recommendations', None)
            if improvement_recommendations:
                all_recommendations.extend(improvement_recommendations)

        # Return top 3 most common recommendations
        return [rec for rec, count in Counter(all_recommendations).most_common(3)]
    async def _generate_final_report(self):
        """Generate comprehensive performance report"""
        
        report_path = Path(f"performance_reports/session_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(exist_ok=True)
        
        report_data = {
            "session_summary": {
                "session_id": self.session_id,
                "start_time": min(d.timestamp for d in self.decisions.values()).isoformat() if self.decisions else None,
                "end_time": datetime.now().isoformat(),
                "total_decisions": len(self.decisions),
                "agents_monitored": list(self.agent_snapshots.keys())
            },
            "agent_performance": {},
            "tool_analysis": {},
            "recommendations": []
        }
        
        # Agent performance summary
        for agent_name, snapshot in self.agent_snapshots.items():
            agent_decisions = [d for d in self.decisions.values() if d.agent_name == agent_name]
            
            report_data["agent_performance"][agent_name] = {
                "final_accuracy": snapshot.accuracy_score,
                "confidence_calibration": snapshot.confidence_calibration,
                "total_decisions": len(agent_decisions),
                "average_confidence": np.mean([d.confidence for d in agent_decisions]) if agent_decisions else 0,
                "average_execution_time": np.mean([d.execution_time_ms for d in agent_decisions]) if agent_decisions else 0,
                "decision_quality_distribution": self._get_quality_distribution(),
                "most_used_tools": snapshot.tool_efficiency
            }
        
        # Generate recommendations
        report_data["recommendations"] = self._generate_recommendations()
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"📊 Performance report saved: {report_path}")
    
    # def _get_quality_distribution(self, decisions: List[AgentDecision]) -> Dict[str, int]:
    #     """Get distribution of decision qualities"""
    #     distribution = defaultdict(int)
        
    #     for decision in decisions:
    #         if decision.decision_quality:
    #             distribution[decision.decision_quality.value] += 1
        
    #     return dict(distribution)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data"""
        recommendations = []
        
        for agent_name, snapshot in self.agent_snapshots.items():
            # Low accuracy recommendations
            if snapshot.accuracy_score < 60:
                recommendations.append(f"Consider adjusting {agent_name} confidence thresholds or prompt engineering")
            
            # Poor calibration recommendations
            if snapshot.confidence_calibration < 0:
                recommendations.append(f"{agent_name} shows poor confidence calibration - review confidence scoring logic")
            
            # Tool efficiency recommendations
            inefficient_tools = [tool for tool, eff in snapshot.tool_efficiency.items() if eff < 0.5]
            if inefficient_tools:
                recommendations.append(f"Review tool usage for {agent_name}: {', '.join(inefficient_tools)} showing low efficiency")
        
        return recommendations

# Example integration with existing CrewAI system
class MonitoredAgent:
    """Wrapper to add performance monitoring to existing agents"""
    
    def __init__(self, agent, monitor: PerformanceMonitor, agent_name: str):
        self.agent = agent
        self.monitor = monitor
        self.agent_name = agent_name

    async def execute_with_monitoring(self, task_description: str, context: Dict[str, Any] = {}):
        """Execute agent task with performance monitoring"""

        start_time = time.time()
        tools_used = []
        
        try:
            # Execute the actual agent task
            result = await self.agent.execute(task_description)
            
            # Track the decision
            execution_time = (time.time() - start_time) * 1000
            
            # Extract confidence from result (you'll need to adapt this to your agent output format)
            confidence = self._extract_confidence(result)
            
            decision_id = await self.monitor.track_agent_decision(
                agent_name=self.agent_name,
                decision_type="task_execution",
                input_data={"task": task_description, "context": context or {}},
                output_data={"result": str(result)},
                confidence=confidence,
                tools_used=tools_used,
                execution_time_ms=execution_time,
                market_context=context
            )
            
            return result, decision_id
            
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")
            raise
    
    def _extract_confidence(self, result) -> float:
        """Extract confidence score from agent result"""
        # This is a placeholder - you'll need to adapt this to your actual result format
        if hasattr(result, 'confidence'):
            return result.confidence
        elif isinstance(result, dict) and 'confidence' in result:
            return result['confidence']
        else:
            return 50.0  # Default confidence

# Performance Dashboard (simple console version)
class PerformanceDashboard:
    """Real-time performance dashboard"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.running = False
    
    async def start_dashboard(self, update_interval: int = 30):
        """Start the real-time dashboard"""
        self.running = True
        
        while self.running:
            self._clear_screen()
            self._display_dashboard()
            await asyncio.sleep(update_interval)
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        self.running = False
    
    def _clear_screen(self):
        """Clear the console screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _display_dashboard(self):
        """Display the current performance dashboard"""
        data = self.monitor.get_real_time_dashboard_data()
        
        print("🚀 AUTONOMOUS TRADING SYSTEM - AGENT PERFORMANCE DASHBOARD")
        print("=" * 70)
        print(f"📅 {data['timestamp']}")
        print(f"🔍 Monitoring: {'ACTIVE' if data['monitoring_active'] else 'INACTIVE'}")
        print()
        
        if data['overall_metrics']:
            metrics = data['overall_metrics']
            print("📊 OVERALL PERFORMANCE")
            print("-" * 30)
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.1f}%")
            print(f"Confidence Calibration: {metrics['overall_calibration']:.2f}")
            print(f"Total Decisions: {metrics['total_decisions']}")
            print(f"Active Agents: {metrics['active_agents']}")
            print()
        
        print("🤖 AGENT PERFORMANCE")
        print("-" * 40)
        
        for agent_name, agent_data in data['agents'].items():
            print(f"\n{agent_name.upper()}:")
            print(f"  Accuracy: {agent_data['accuracy_score']:.1f}%")
            print(f"  Calibration: {agent_data['confidence_calibration']:.2f}")
            print(f"  Decisions (Total/Hour): {agent_data['total_decisions']}/{agent_data['decisions_last_hour']}")
            print(f"  Most Used Tool: {agent_data['most_used_tool']}")
            
            if agent_data['tool_efficiency']:
                print(f"  Tool Efficiency: {', '.join([f'{k}:{v:.1%}' for k, v in agent_data['tool_efficiency'].items()])}")

# Usage Example
async def example_integration():
    """Example of how to integrate the performance monitoring system"""
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    await monitor.start_monitoring()
    
    # Add alert callback
    async def performance_alert(message: str):
        print(f"🚨 ALERT: {message}")
    
    monitor.add_alert_callback(performance_alert)
    
    # Example of tracking a decision
    decision_id = await monitor.track_agent_decision(
        agent_name="wyckoff_market_analyst",
        decision_type="pattern_analysis",
        input_data={"symbol": "EUR_USD", "timeframe": "M15"},
        output_data={"pattern": "accumulation", "phase": "C"},
        confidence=85.0,
        tools_used=["get_historical_data", "wyckoff_analyzer"],
        execution_time_ms=1250.0,
        market_context={"regime": "ranging", "volatility": "medium"}
    )
    
    # Later, update the outcome
    await monitor.update_decision_outcome(
        decision_id=decision_id,
        outcome_positive=True,
        outcome_value=150.0  # P&L from trade
    )
    
    # Start dashboard
    dashboard = PerformanceDashboard(monitor)
    # await dashboard.start_dashboard()  # Uncomment to run dashboard
    
    # Stop monitoring
    await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(example_integration())
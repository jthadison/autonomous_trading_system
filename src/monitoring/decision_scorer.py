"""
Enhanced Decision Quality Scoring System
Provides more nuanced agent decision evaluation for optimization
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

class DecisionQuality(Enum):
    """Enhanced decision quality ratings"""
    EXCELLENT = "excellent"        # High confidence + positive outcome + good timing
    GOOD = "good"                 # Medium confidence + positive outcome
    ACCEPTABLE = "acceptable"      # Low confidence + positive outcome OR high confidence + neutral
    POOR = "poor"                 # High confidence + negative outcome
    TERRIBLE = "terrible"         # High confidence + very negative outcome
    UNCERTAIN = "uncertain"       # Low confidence + any outcome
    LUCKY = "lucky"               # Low confidence + very positive outcome

@dataclass
class DecisionContext:
    """Enhanced context for decision evaluation"""
    market_regime: str
    volatility_level: float
    session_time: str
    recent_performance: float  # Agent's recent performance score
    consecutive_wins: int
    consecutive_losses: int
    time_pressure: float       # 0-1, how urgent the decision was
    complexity_score: float    # 0-1, how complex the analysis was

class EnhancedDecisionScorer:
    """Advanced decision quality scoring system"""
    
    def __init__(self):
        self.scoring_weights = {
            'confidence_accuracy': 0.3,    # How well confidence matched outcome
            'timing_quality': 0.2,         # How good was the timing
            'context_adaptation': 0.2,     # How well adapted to market context
            'risk_reward': 0.15,           # Risk/reward ratio quality
            'consistency': 0.15            # Consistency with recent decisions
        }
    
    def score_decision(
        self, 
        confidence: float,
        outcome_positive: bool,
        outcome_value: float,
        context: DecisionContext,
        execution_time_ms: float
    ) -> Dict[str, Any]:
        """
        Comprehensive decision scoring
        
        Returns:
            Dict with quality rating, score breakdown, and improvement recommendations
        """
        
        scores = {}
        
        # 1. Confidence Accuracy Score (0-100)
        scores['confidence_accuracy'] = self._score_confidence_accuracy(
            confidence, outcome_positive, outcome_value
        )
        
        # 2. Timing Quality Score (0-100)
        scores['timing_quality'] = self._score_timing_quality(
            outcome_value, context, execution_time_ms
        )
        
        # 3. Context Adaptation Score (0-100)
        scores['context_adaptation'] = self._score_context_adaptation(
            confidence, context
        )
        
        # 4. Risk/Reward Score (0-100)
        scores['risk_reward'] = self._score_risk_reward(
            confidence, outcome_value, context
        )
        
        # 5. Consistency Score (0-100)
        scores['consistency'] = self._score_consistency(
            confidence, outcome_positive, context
        )
        
        # Calculate weighted overall score
        overall_score = sum(
            scores[component] * self.scoring_weights[component] 
            for component in scores
        )
        
        # Determine quality rating
        quality_rating = self._determine_quality_rating(
            confidence, outcome_positive, outcome_value, overall_score, context
        )
        
        # Generate improvement recommendations
        recommendations = self._generate_recommendations(scores, context)
        
        return {
            'overall_score': overall_score,
            'quality_rating': quality_rating,
            'component_scores': scores,
            'recommendations': recommendations,
            'scoring_explanation': self._explain_scoring(scores, quality_rating)
        }
    
    def _score_confidence_accuracy(self, confidence: float, outcome_positive: bool, outcome_value: float) -> float:
        """Score how well confidence matched the actual outcome"""
        
        # Normalize confidence to 0-1
        conf_normalized = confidence / 100.0
        
        if outcome_positive:
            # Good outcome: high confidence should score high
            if outcome_value > 0:
                # Scale by outcome magnitude
                outcome_factor = min(abs(outcome_value) / 100, 2.0)  # Cap at 2x
                base_score = conf_normalized * 100
                return min(base_score * outcome_factor, 100)
            else:
                return conf_normalized * 100
        else:
            # Bad outcome: low confidence should score high (good risk awareness)
            if abs(outcome_value) > 50:  # Significant loss
                # Heavily penalize high confidence with bad outcomes
                penalty_factor = abs(outcome_value) / 100
                return max(100 - (conf_normalized * 100 * penalty_factor), 0)
            else:
                # Minor loss: moderate penalty
                return max(100 - (conf_normalized * 50), 20)
    
    def _score_timing_quality(self, outcome_value: float, context: DecisionContext, execution_time_ms: float) -> float:
        """Score the timing quality of the decision"""
        
        base_score = 50  # Neutral starting point
        
        # Execution speed factor
        if execution_time_ms < 1000:  # Fast execution (< 1 second)
            speed_bonus = 20
        elif execution_time_ms < 5000:  # Normal execution (< 5 seconds)
            speed_bonus = 10
        else:  # Slow execution
            speed_bonus = -10
        
        # Market condition timing
        session_timing_bonus = 0
        if context.session_time in ['london_session', 'ny_session']:
            session_timing_bonus = 10  # Active session bonus
        elif context.session_time == 'market_open':
            session_timing_bonus = 15  # Market open bonus
        
        # Volatility timing
        volatility_bonus = 0
        if context.volatility_level > 0.8:  # High volatility
            if abs(outcome_value) > 50:  # Significant move captured
                volatility_bonus = 20
            else:
                volatility_bonus = -10  # Failed to capitalize on volatility
        
        # Time pressure handling
        pressure_bonus = 0
        if context.time_pressure > 0.7:  # High pressure decision
            if outcome_value > 0:
                pressure_bonus = 15  # Good decision under pressure
            else:
                pressure_bonus = -15  # Poor decision under pressure
        
        total_score = base_score + speed_bonus + session_timing_bonus + volatility_bonus + pressure_bonus
        return max(0, min(100, total_score))
    
    def _score_context_adaptation(self, confidence: float, context: DecisionContext) -> float:
        """Score how well the decision adapted to market context"""
        
        base_score = 50
        
        # Market regime adaptation
        regime_bonus = 0
        if context.market_regime in ['trending_bullish', 'trending_bearish']:
            # Trending markets: higher confidence should be rewarded
            if confidence > 75:
                regime_bonus = 20
            elif confidence < 50:
                regime_bonus = -10
        elif context.market_regime in ['ranging_tight', 'ranging_wide']:
            # Ranging markets: moderate confidence is appropriate
            if 50 <= confidence <= 75:
                regime_bonus = 15
            elif confidence > 85:
                regime_bonus = -15  # Over-confident in ranging market
        elif context.market_regime == 'high_volatility':
            # High volatility: lower confidence is prudent
            if confidence < 60:
                regime_bonus = 20
            elif confidence > 80:
                regime_bonus = -20
        
        # Recent performance adaptation
        performance_bonus = 0
        if context.recent_performance < 40:  # Recent poor performance
            if confidence < 60:  # Appropriately cautious
                performance_bonus = 15
            elif confidence > 80:  # Overconfident despite poor performance
                performance_bonus = -20
        elif context.recent_performance > 80:  # Recent good performance
            if confidence > 70:  # Confident based on good track record
                performance_bonus = 10
        
        # Consecutive loss adaptation
        loss_adaptation = 0
        if context.consecutive_losses > 2:
            if confidence < 50:  # Appropriately cautious after losses
                loss_adaptation = 15
            elif confidence > 75:  # Overconfident after losses
                loss_adaptation = -25
        
        total_score = base_score + regime_bonus + performance_bonus + loss_adaptation
        return max(0, min(100, total_score))
    
    def _score_risk_reward(self, confidence: float, outcome_value: float, context: DecisionContext) -> float:
        """Score the risk/reward quality of the decision"""
        
        # Base score depends on absolute outcome
        if abs(outcome_value) < 10:  # Small outcome
            base_score = 40
        elif abs(outcome_value) < 50:  # Medium outcome
            base_score = 60
        else:  # Large outcome
            base_score = 80
        
        # Confidence vs outcome relationship
        conf_outcome_bonus = 0
        if outcome_value > 0:  # Positive outcome
            if confidence > 75 and outcome_value > 50:
                conf_outcome_bonus = 20  # High confidence, high reward
            elif confidence < 50 and outcome_value > 100:
                conf_outcome_bonus = -10  # Low confidence but missed bigger opportunity
        else:  # Negative outcome
            if confidence < 50:
                conf_outcome_bonus = 15  # Low confidence limited loss
            elif confidence > 80:
                conf_outcome_bonus = -30  # High confidence but lost
        
        # Market regime risk appropriateness
        regime_risk_bonus = 0
        if context.market_regime == 'high_volatility':
            if abs(outcome_value) < 30:  # Conservative in volatile market
                regime_risk_bonus = 10
        elif context.market_regime in ['trending_bullish', 'trending_bearish']:
            if outcome_value > 50:  # Captured trend move
                regime_risk_bonus = 15
        
        total_score = base_score + conf_outcome_bonus + regime_risk_bonus
        return max(0, min(100, total_score))
    
    def _score_consistency(self, confidence: float, outcome_positive: bool, context: DecisionContext) -> float:
        """Score decision consistency with recent performance"""
        
        base_score = 50
        
        # Confidence consistency
        confidence_bonus = 0
        if context.consecutive_wins > 2:
            if confidence > 70:  # Confident during winning streak
                confidence_bonus = 15
        elif context.consecutive_losses > 2:
            if confidence < 60:  # Cautious during losing streak
                confidence_bonus = 15
            elif confidence > 80:  # Overconfident during losing streak
                confidence_bonus = -20
        
        # Performance pattern consistency
        pattern_bonus = 0
        if context.recent_performance > 70:  # Recently performing well
            if outcome_positive:
                pattern_bonus = 10  # Maintaining good performance
            else:
                pattern_bonus = -5   # Breaking good streak
        elif context.recent_performance < 40:  # Recently performing poorly
            if outcome_positive:
                pattern_bonus = 20  # Breaking out of poor performance
            else:
                pattern_bonus = -10  # Continuing poor performance
        
        total_score = base_score + confidence_bonus + pattern_bonus
        return max(0, min(100, total_score))
    
    def _determine_quality_rating(
        self, 
        confidence: float, 
        outcome_positive: bool, 
        outcome_value: float, 
        overall_score: float,
        context: DecisionContext
    ) -> DecisionQuality:
        """Determine overall decision quality rating"""
        
        # Excellent: High score + high confidence + good outcome
        if overall_score > 85 and confidence > 75 and outcome_positive and outcome_value > 50:
            return DecisionQuality.EXCELLENT
        
        # Terrible: High confidence + very bad outcome
        elif confidence > 80 and not outcome_positive and abs(outcome_value) > 100:
            return DecisionQuality.TERRIBLE
        
        # Lucky: Low confidence + very good outcome
        elif confidence < 40 and outcome_positive and outcome_value > 100:
            return DecisionQuality.LUCKY
        
        # Poor: High confidence + negative outcome
        elif confidence > 70 and not outcome_positive:
            return DecisionQuality.POOR
        
        # Good: Generally positive with reasonable confidence
        elif overall_score > 70 and outcome_positive:
            return DecisionQuality.GOOD
        
        # Acceptable: Moderate score or mixed results
        elif overall_score > 50 or (confidence < 60 and outcome_positive):
            return DecisionQuality.ACCEPTABLE
        
        # Uncertain: Low confidence decisions
        elif confidence < 50:
            return DecisionQuality.UNCERTAIN
        
        # Default to poor for remaining cases
        else:
            return DecisionQuality.POOR
    
    def _generate_recommendations(self, scores: Dict[str, float], context: DecisionContext) -> List[str]:
        """Generate specific improvement recommendations"""
        
        recommendations = []
        
        # Confidence accuracy recommendations
        if scores['confidence_accuracy'] < 60:
            recommendations.append("Improve confidence calibration - review confidence scoring logic")
            if context.consecutive_losses > 2:
                recommendations.append("Consider reducing confidence after consecutive losses")
        
        # Timing recommendations
        if scores['timing_quality'] < 50:
            recommendations.append("Improve decision timing - consider market session awareness")
            if context.volatility_level > 0.8:
                recommendations.append("Be more decisive in high volatility periods")
        
        # Context adaptation recommendations
        if scores['context_adaptation'] < 60:
            recommendations.append(f"Better adapt to {context.market_regime} market conditions")
            if context.recent_performance < 40:
                recommendations.append("Reduce confidence when recent performance is poor")
        
        # Risk/reward recommendations
        if scores['risk_reward'] < 50:
            recommendations.append("Improve risk/reward ratio - consider position sizing")
            recommendations.append("Review stop loss and take profit levels")
        
        # Consistency recommendations
        if scores['consistency'] < 60:
            recommendations.append("Improve decision consistency across market conditions")
            if context.consecutive_losses > 3:
                recommendations.append("Consider taking a break after extended losing streaks")
        
        return recommendations
    
    def _explain_scoring(self, scores: Dict[str, float], quality_rating: DecisionQuality) -> str:
        """Provide human-readable explanation of the scoring"""
        
        best_aspect = max(scores.keys(), key=lambda k: scores[k])
        worst_aspect = min(scores.keys(), key=lambda k: scores[k])
        
        explanation = f"Decision rated as {quality_rating.value.upper()}. "
        explanation += f"Strongest aspect: {best_aspect} ({scores[best_aspect]:.1f}/100). "
        explanation += f"Weakest aspect: {worst_aspect} ({scores[worst_aspect]:.1f}/100). "
        
        if quality_rating in [DecisionQuality.EXCELLENT, DecisionQuality.GOOD]:
            explanation += "Continue this approach."
        elif quality_rating in [DecisionQuality.POOR, DecisionQuality.TERRIBLE]:
            explanation += "Significant improvement needed."
        else:
            explanation += "Mixed results - focus on consistency."
        
        return explanation
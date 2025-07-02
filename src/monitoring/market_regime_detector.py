"""
Enhanced Market Regime Detection for Agent Performance Optimization
Adds sophisticated market condition analysis to improve agent decision tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

class MarketRegime(Enum):
    """Enhanced market regime classification"""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_TIGHT = "ranging_tight"  # Low volatility range
    RANGING_WIDE = "ranging_wide"    # High volatility range
    BREAKOUT_UPWARD = "breakout_upward"
    BREAKOUT_DOWNWARD = "breakout_downward"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NEWS_EVENT = "news_event"
    MARKET_OPEN = "market_open"      # First hour of session
    MARKET_CLOSE = "market_close"    # Last hour of session

class MarketRegimeDetector:
    """Advanced market regime detection for performance optimization"""
    
    def __init__(self):
        self.regime_history = []
        self.volatility_window = 20  # bars for volatility calculation
        self.trend_window = 50       # bars for trend detection
        
    def detect_current_regime(self, price_data: List[Dict], current_price: float) -> Dict[str, Any]:
        """
        Detect current market regime based on price data
        
        Args:
            price_data: List of OHLCV dictionaries
            current_price: Current market price
            
        Returns:
            Dict with regime information for performance tracking
        """
        
        if len(price_data) < self.trend_window:
            return self._default_regime()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate technical indicators
        regime_data = self._calculate_regime_indicators(df, current_price)
        
        # Determine primary regime
        primary_regime = self._classify_primary_regime(regime_data)
        
        # Determine secondary conditions
        secondary_conditions = self._detect_secondary_conditions(regime_data, df)
        
        # Calculate regime strength and confidence
        regime_strength = self._calculate_regime_strength(regime_data)
        
        regime_info = {
            'primary_regime': primary_regime,
            'secondary_conditions': secondary_conditions,
            'regime_strength': regime_strength,
            'volatility_percentile': regime_data['volatility_percentile'],
            'trend_strength': regime_data['trend_strength'],
            'range_efficiency': regime_data['range_efficiency'],
            'session_time': self._get_session_info(),
            'regime_duration': self._calculate_regime_duration(primary_regime),
            'breakout_probability': regime_data['breakout_probability'],
            'mean_reversion_probability': regime_data['mean_reversion_probability']
        }
        
        # Update regime history
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': primary_regime,
            'strength': regime_strength
        })
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.regime_history = [r for r in self.regime_history if r['timestamp'] > cutoff_time]
        
        return regime_info
    
    def _calculate_regime_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Calculate technical indicators for regime detection"""
        
        # Price-based indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['atr'] = self._calculate_atr(df, 14)
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df, 20, 2)
        
        # Current values
        latest = df.iloc[-1]
        
        # Trend indicators
        trend_strength = (current_price - latest['sma_50']) / latest['sma_50'] * 100
        sma_slope = (latest['sma_20'] - df['sma_20'].iloc[-5]) / 5  # 5-bar slope
        
        # Volatility indicators
        current_atr = latest['atr']
        atr_percentile = (current_atr > df['atr'].quantile(0.8)) * 100
        
        # Range efficiency (how much price moves vs how much it could move)
        high_low_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        close_move = abs(df['close'].iloc[-1] - df['close'].iloc[-20])
        range_efficiency = (close_move / high_low_range.iloc[-1]) * 100 if high_low_range.iloc[-1] > 0 else 0
        
        # Breakout indicators
        bb_position = (current_price - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        breakout_probability = 0
        if bb_position > 0.95:
            breakout_probability = 75  # High probability of upward breakout
        elif bb_position < 0.05:
            breakout_probability = 75  # High probability of downward breakout
        else:
            breakout_probability = 25  # Low breakout probability
        
        # Mean reversion probability (inverse of breakout)
        mean_reversion_probability = 100 - breakout_probability
        
        return {
            'trend_strength': trend_strength,
            'sma_slope': sma_slope,
            'volatility_percentile': atr_percentile,
            'range_efficiency': range_efficiency,
            'bb_position': bb_position,
            'breakout_probability': breakout_probability,
            'mean_reversion_probability': mean_reversion_probability,
            'current_atr': current_atr
        }
    
    def _classify_primary_regime(self, indicators: Dict[str, float]) -> MarketRegime:
        """Classify the primary market regime"""
        
        trend_strength = indicators['trend_strength']
        volatility_percentile = indicators['volatility_percentile']
        range_efficiency = indicators['range_efficiency']
        breakout_probability = indicators['breakout_probability']
        
        # Strong trend conditions
        if abs(trend_strength) > 2.0 and range_efficiency > 60:
            if trend_strength > 0:
                return MarketRegime.TRENDING_BULLISH
            else:
                return MarketRegime.TRENDING_BEARISH
        
        # Breakout conditions
        elif breakout_probability > 60:
            if trend_strength > 0:
                return MarketRegime.BREAKOUT_UPWARD
            else:
                return MarketRegime.BREAKOUT_DOWNWARD
        
        # High volatility ranging
        elif volatility_percentile > 70 and range_efficiency < 40:
            return MarketRegime.RANGING_WIDE
        
        # Low volatility ranging
        elif volatility_percentile < 30 and range_efficiency < 40:
            return MarketRegime.RANGING_TIGHT
        
        # High volatility without clear direction
        elif volatility_percentile > 80:
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility default
        elif volatility_percentile < 20:
            return MarketRegime.LOW_VOLATILITY
        
        # Default to tight ranging
        else:
            return MarketRegime.RANGING_TIGHT
    
    def _detect_secondary_conditions(self, indicators: Dict[str, float], df: pd.DataFrame) -> List[str]:
        """Detect secondary market conditions"""
        conditions = []
        
        # Session timing
        session_info = self._get_session_info()
        if session_info['is_open']:
            conditions.append("market_open")
        if session_info['is_close']:
            conditions.append("market_close")
        
        # Volume analysis (if available)
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].mean()
            if recent_volume > avg_volume * 1.5:
                conditions.append("high_volume")
            elif recent_volume < avg_volume * 0.5:
                conditions.append("low_volume")
        
        # Price action patterns
        if indicators['bb_position'] > 0.9:
            conditions.append("near_resistance")
        elif indicators['bb_position'] < 0.1:
            conditions.append("near_support")
        
        return conditions
    
    def _calculate_regime_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate regime strength/confidence (0-100)"""
        
        # Base strength on multiple factors
        trend_component = min(abs(indicators['trend_strength']) * 10, 40)  # Max 40 points
        efficiency_component = indicators['range_efficiency'] * 0.3  # Max 30 points
        volatility_component = min(indicators['volatility_percentile'] * 0.3, 30)  # Max 30 points
        
        total_strength = trend_component + efficiency_component + volatility_component
        return min(total_strength, 100)
    
    def _get_session_info(self) -> Dict[str, Any]:
        """Get current trading session information"""
        now = datetime.now()
        hour = now.hour
        
        # Define session times (UTC)
        sessions = {
            'tokyo': (22, 7),      # 22:00-07:00 UTC
            'london': (7, 15),     # 07:00-15:00 UTC  
            'new_york': (13, 21)   # 13:00-21:00 UTC
        }
        
        current_session = "off_hours"
        is_open = False
        is_close = False
        
        for session, (start, end) in sessions.items():
            if start <= end:  # Same day session
                if start <= hour < end:
                    current_session = session
                    is_open = hour == start
                    is_close = hour == end - 1
                    break
            else:  # Cross-midnight session (Tokyo)
                if hour >= start or hour < end:
                    current_session = session
                    is_open = hour == start
                    is_close = hour == end - 1
                    break
        
        return {
            'current_session': current_session,
            'is_open': is_open,
            'is_close': is_close,
            'hour': hour
        }
    
    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long the current regime has been active (in periods)"""
        if not self.regime_history:
            return 1
        
        duration = 1
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i]['regime'] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return pd.Series(true_range.rolling(period).mean())
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band
    
    def _default_regime(self) -> Dict[str, Any]:
        """Return default regime when insufficient data"""
        return {
            'primary_regime': MarketRegime.RANGING_TIGHT,
            'secondary_conditions': [],
            'regime_strength': 50.0,
            'volatility_percentile': 50.0,
            'trend_strength': 0.0,
            'range_efficiency': 50.0,
            'session_time': self._get_session_info(),
            'regime_duration': 1,
            'breakout_probability': 25.0,
            'mean_reversion_probability': 75.0
        }

    def get_regime_based_recommendations(self, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent optimization recommendations based on current regime"""
        
        regime = regime_info['primary_regime']
        strength = regime_info['regime_strength']
        
        recommendations = {
            'confidence_adjustment': 0,  # -20 to +20
            'tool_priority': [],
            'risk_adjustment': 0,        # -10 to +10
            'execution_speed': 'normal'  # 'fast', 'normal', 'slow'
        }
        
        # Regime-specific recommendations
        if regime in [MarketRegime.TRENDING_BULLISH, MarketRegime.TRENDING_BEARISH]:
            recommendations['confidence_adjustment'] = +10
            recommendations['tool_priority'] = ['trend_analysis', 'momentum_indicators']
            recommendations['risk_adjustment'] = +5
            recommendations['execution_speed'] = 'fast'
            
        elif regime in [MarketRegime.RANGING_TIGHT, MarketRegime.RANGING_WIDE]:
            recommendations['confidence_adjustment'] = -5
            recommendations['tool_priority'] = ['support_resistance', 'mean_reversion']
            recommendations['risk_adjustment'] = -5
            recommendations['execution_speed'] = 'normal'
            
        elif regime in [MarketRegime.BREAKOUT_UPWARD, MarketRegime.BREAKOUT_DOWNWARD]:
            recommendations['confidence_adjustment'] = +15
            recommendations['tool_priority'] = ['breakout_analysis', 'volume_analysis']
            recommendations['risk_adjustment'] = +10
            recommendations['execution_speed'] = 'fast'
            
        elif regime == MarketRegime.HIGH_VOLATILITY:
            recommendations['confidence_adjustment'] = -10
            recommendations['tool_priority'] = ['risk_management', 'volatility_analysis']
            recommendations['risk_adjustment'] = -10
            recommendations['execution_speed'] = 'slow'
            
        elif regime == MarketRegime.LOW_VOLATILITY:
            recommendations['confidence_adjustment'] = +5
            recommendations['tool_priority'] = ['pattern_analysis', 'accumulation_detection']
            recommendations['risk_adjustment'] = 0
            recommendations['execution_speed'] = 'normal'
        
        # Adjust based on regime strength
        strength_multiplier = strength / 100
        recommendations['confidence_adjustment'] = int(recommendations['confidence_adjustment'] * strength_multiplier)
        recommendations['risk_adjustment'] = int(recommendations['risk_adjustment'] * strength_multiplier)
        
        return recommendations
"""
Wyckoff Pattern Analyzer
Analyzes price action for Wyckoff accumulation/distribution patterns
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow running this script directly
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.config.logging_config import logger
from src.database.manager import db_manager
from src.database.models import PatternDetection, MarketContext, WyckoffPattern, MarketRegime

class WyckoffPhase(Enum):
    """Wyckoff phases for accumulation/distribution"""
    PHASE_A = "phase_a"  # Stopping action
    PHASE_B = "phase_b"  # Building cause
    PHASE_C = "phase_c"  # Test (Spring/Upthrust)
    PHASE_D = "phase_d"  # Evidence of supply/demand
    PHASE_E = "phase_e"  # Markup/Markdown

class StructureType(Enum):
    """Types of Wyckoff structures"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    REACCUMULATION = "reaccumulation"
    REDISTRIBUTION = "redistribution"
    TRENDING = "trending"
    UNKNOWN = "unknown"

@dataclass
class WyckoffStructure:
    """Wyckoff structure data class"""
    structure_type: StructureType
    phase: WyckoffPhase
    confidence: float
    start_time: datetime
    end_time: datetime
    key_levels: Dict[str, float]
    volume_analysis: Dict[str, Any]
    spring_upthrust: Optional[Dict[str, Any]] = None
    cause_measurement: Optional[float] = None
    target_projection: Optional[float] = None

@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    vpoc: float  # Value Point of Control
    value_area_high: float
    value_area_low: float
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]
    profile_shape: str  # "normal", "b_shape", "p_shape", "double_distribution"

class WyckoffAnalyzer:
    """Advanced Wyckoff pattern analysis engine"""
    
    def __init__(self):
        self.min_structure_bars = 20  # Minimum bars for structure identification
        self.volume_threshold = 1.5   # Volume spike threshold (1.5x average)
        self.spring_threshold = 0.02  # 2% threshold for spring identification
        
    async def analyze_market_data(self, 
                          price_data: List[Dict[str, Any]], 
                          timeframe: str = "15m") -> Dict[str, Any]:
        """
        Complete Wyckoff analysis of market data
        
        Args:
            price_data: List of OHLCV data
            timeframe: Timeframe being analyzed
            
        Returns:
            Complete Wyckoff analysis results
        """
        
        if len(price_data) < self.min_structure_bars:
            return {"error": "Insufficient data for Wyckoff analysis"}
        
        try:
            # Convert to DataFrame for analysis
            df = self._prepare_dataframe(price_data)
            
            # Core Wyckoff analysis components
            structure_analysis = self._identify_structures(df)
            volume_analysis = self._analyze_volume_profile(df)
            spring_upthrust_analysis = self._detect_springs_upthrusts(df)
            market_regime = self._determine_market_regime(df)
            cause_effect = self._calculate_cause_effect(df, structure_analysis)
            
            # Combine all analyses
            wyckoff_analysis = {
                "timestamp": datetime.utcnow().isoformat(),
                "timeframe": timeframe,
                "data_points": len(price_data),
                "structure_analysis": structure_analysis,
                "volume_profile": volume_analysis,
                "spring_upthrust": spring_upthrust_analysis,
                "market_regime": market_regime,
                "cause_effect": cause_effect,
                "trading_recommendations": self._generate_trading_recommendations(
                    structure_analysis, volume_analysis, spring_upthrust_analysis
                ),
                "confidence_score": self._calculate_overall_confidence(
                    structure_analysis, volume_analysis, spring_upthrust_analysis
                )
            }
            
            # Log the analysis
            await self._log_pattern_detection(wyckoff_analysis)
            
            return wyckoff_analysis
            
        except Exception as e:
            logger.error("Wyckoff analysis failed", error=str(e))
            return {"error": str(e)}
    
    def _prepare_dataframe(self, price_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare DataFrame with technical indicators for analysis"""
        
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate essential indicators
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['range'] = df['high'] - df['low']
        
        # Volume analysis (using tick volume for forex)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['high_volume'] = df['volume_ratio'] > self.volume_threshold
        
        # Price action indicators
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Trend and momentum
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['trend_direction'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
        
        return df
    
    def _identify_structures(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify Wyckoff accumulation/distribution structures"""
        
        structure_analysis = {
            "current_structure": StructureType.UNKNOWN.value,
            "phase": None,
            "confidence": 0.0,
            "key_levels": {},
            "structure_bounds": {},
            "time_in_structure": 0
        }
        
        try:
            # Identify potential trading ranges
            recent_high = df['high'].tail(50).max()
            recent_low = df['low'].tail(50).min()
            range_size = recent_high - recent_low
            current_price = df['close'].iloc[-1]
            
            # Calculate position within range
            range_position = (current_price - recent_low) / range_size if range_size > 0 else 0.5
            
            # Analyze price action characteristics
            consolidation_bars = self._count_consolidation_bars(df, float(recent_high), float(recent_low))
            volume_characteristics = self._analyze_structure_volume(df)
            
            # Determine structure type based on price action and volume
            if consolidation_bars >= self.min_structure_bars:
                if self._is_accumulation_pattern(df, volume_characteristics):
                    structure_analysis["current_structure"] = StructureType.ACCUMULATION.value
                    structure_analysis["phase"] = self._determine_accumulation_phase(df, range_position)
                elif self._is_distribution_pattern(df, volume_characteristics):
                    structure_analysis["current_structure"] = StructureType.DISTRIBUTION.value
                    structure_analysis["phase"] = self._determine_distribution_phase(df, range_position)
                else:
                    structure_analysis["current_structure"] = StructureType.TRENDING.value
            
            # Set key levels
            structure_analysis["key_levels"] = {
                "resistance": recent_high,
                "support": recent_low,
                "midpoint": (recent_high + recent_low) / 2,
                "current_price": current_price
            }
            
            # Calculate confidence
            structure_analysis["confidence"] = self._calculate_structure_confidence(
                consolidation_bars, volume_characteristics, range_position
            )
            
            structure_analysis["structure_bounds"] = {
                "high": recent_high,
                "low": recent_low,
                "range_size": range_size,
                "position_in_range": range_position
            }
            
            structure_analysis["time_in_structure"] = consolidation_bars
            
        except Exception as e:
            logger.error("Structure identification failed", error=str(e))
        
        return structure_analysis
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile and identify key levels"""
        
        try:
            # Create price bins for volume profile
            price_range = df['high'].max() - df['low'].min()
            num_bins = min(50, len(df) // 2)  # Reasonable number of bins
            
            price_bins = np.linspace(df['low'].min(), df['high'].max(), num_bins)
            volume_at_price = np.zeros(len(price_bins) - 1)
            
            # Distribute volume across price bins
            for _, row in df.iterrows():
                # Simple volume distribution across the bar's range
                bar_prices = np.linspace(row['low'], row['high'], 10)
                bar_volume = row['volume'] / 10
                
                for price in bar_prices:
                    bin_idx = np.digitize(price, price_bins) - 1
                    if 0 <= bin_idx < len(volume_at_price):
                        volume_at_price[bin_idx] += bar_volume
            
            # Find VPOC (Value Point of Control)
            vpoc_idx = np.argmax(volume_at_price)
            vpoc_price = (price_bins[vpoc_idx] + price_bins[vpoc_idx + 1]) / 2
            
            # Calculate value area (70% of volume)
            total_volume = np.sum(volume_at_price)
            sorted_indices = np.argsort(volume_at_price)[::-1]
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_at_price[idx]
                value_area_indices.append(idx)
                if cumulative_volume >= total_volume * 0.7:
                    break
            
            # Convert generator expressions to lists to fix type issues
            value_area_prices = [float(price_bins[i]) for i in value_area_indices]
            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
            
            # Identify high and low volume nodes
            volume_threshold = np.percentile(volume_at_price, 75)
            high_volume_nodes = []
            low_volume_nodes = []
            
            for i, volume in enumerate(volume_at_price):
                price = (price_bins[i] + price_bins[i + 1]) / 2
                if volume > volume_threshold:
                    high_volume_nodes.append(price)
                elif volume < np.percentile(volume_at_price, 25):
                    low_volume_nodes.append(price)
            
            return {
                "vpoc": vpoc_price,
                "value_area_high": value_area_high,
                "value_area_low": value_area_low,
                "high_volume_nodes": high_volume_nodes[:5],  # Top 5
                "low_volume_nodes": low_volume_nodes[:5],   # Top 5
                "profile_shape": self._determine_profile_shape(volume_at_price),
                "volume_distribution": {
                    "balanced": abs(vpoc_price - (value_area_high + value_area_low) / 2) < (value_area_high - value_area_low) * 0.1,
                    "developing_direction": "higher" if vpoc_price > df['close'].tail(10).mean() else "lower"
                }
            }
            
        except Exception as e:
            logger.error("Volume profile analysis failed", error=str(e))
            return {"error": str(e)}
    
    def _detect_springs_upthrusts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Wyckoff springs and upthrusts"""
        
        springs_upthrusts = {
            "springs_detected": [],
            "upthrusts_detected": [],
            "latest_signal": None,
            "signal_confidence": 0.0
        }
        
        try:
            # Look for springs (downward breaks that fail)
            recent_lows = df['low'].rolling(window=10).min()
            for i in range(20, len(df)):
                current_low = df.iloc[i]['low']
                previous_support = recent_lows.iloc[i-10]
                
                # Check for spring conditions
                if current_low < previous_support * (1 - self.spring_threshold):
                    # Look for recovery (effort vs result divergence)
                    recovery_bars = df.iloc[i:i+5]
                    if len(recovery_bars) > 0:
                        volume_increase = recovery_bars['volume'].mean() > df['volume'].iloc[i-10:i].mean()
                        price_recovery = recovery_bars['close'].iloc[-1] > previous_support
                        
                        if volume_increase and price_recovery:
                            spring = {
                                "type": "spring",
                                "timestamp": df.iloc[i]['timestamp'],
                                "price": current_low,
                                "support_level": previous_support,
                                "recovery_strength": volume_increase,
                                "confidence": 75 if volume_increase and price_recovery else 50
                            }
                            springs_upthrusts["springs_detected"].append(spring)
            
            # Look for upthrusts (upward breaks that fail)
            recent_highs = df['high'].rolling(window=10).max()
            for i in range(20, len(df)):
                current_high = df.iloc[i]['high']
                previous_resistance = recent_highs.iloc[i-10]
                
                # Check for upthrust conditions
                if current_high > previous_resistance * (1 + self.spring_threshold):
                    # Look for failure (effort vs result divergence)
                    failure_bars = df.iloc[i:i+5]
                    if len(failure_bars) > 0:
                        volume_climax = failure_bars['volume'].max() > df['volume'].iloc[i-10:i].mean() * 1.5
                        price_failure = failure_bars['close'].iloc[-1] < previous_resistance
                        
                        if volume_climax and price_failure:
                            upthrust = {
                                "type": "upthrust",
                                "timestamp": df.iloc[i]['timestamp'],
                                "price": current_high,
                                "resistance_level": previous_resistance,
                                "failure_confirmation": price_failure,
                                "confidence": 75 if volume_climax and price_failure else 50
                            }
                            springs_upthrusts["upthrusts_detected"].append(upthrust)
            
            # Determine latest signal
            all_signals = springs_upthrusts["springs_detected"] + springs_upthrusts["upthrusts_detected"]
            if all_signals:
                latest_signal = max(all_signals, key=lambda x: x['timestamp'])
                springs_upthrusts["latest_signal"] = latest_signal
                springs_upthrusts["signal_confidence"] = latest_signal.get("confidence", 0)
            
        except Exception as e:
            logger.error("Spring/upthrust detection failed", error=str(e))
        
        return springs_upthrusts
    
    def _determine_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine current market regime"""
        
        try:
            # Calculate volatility
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].tail(20).std() * np.sqrt(20)  # 20-period volatility
            
            # Calculate trend strength
            trend_strength = abs(df['ema_20'].iloc[-1] - df['ema_50'].iloc[-1]) / df['close'].iloc[-1]
            
            # Determine regime
            if volatility > 0.02:  # High volatility threshold
                regime = MarketRegime.VOLATILE.value
            elif trend_strength > 0.01:  # Strong trend threshold
                regime = MarketRegime.TRENDING.value
            else:
                regime = MarketRegime.RANGING.value
            
            return {
                "regime": regime,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "trend_direction": "bullish" if df['trend_direction'].iloc[-1] > 0 else "bearish",
                "regime_confidence": min(95, max(50, (volatility + trend_strength) * 1000))
            }
            
        except Exception as e:
            logger.error("Market regime determination failed", error=str(e))
            return {"regime": "unknown", "error": str(e)}
    
    def _calculate_cause_effect(self, df: pd.DataFrame, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Wyckoff cause and effect relationships"""
        
        try:
            structure_time = structure_analysis.get("time_in_structure", 0)
            range_size = structure_analysis.get("structure_bounds", {}).get("range_size", 0)
            
            if structure_time == 0 or range_size == 0:
                return {"cause": 0, "projected_effect": 0, "target_levels": []}
            
            # Wyckoff point and figure counting method (simplified)
            # Cause = time spent in accumulation/distribution
            # Effect = projected price movement
            
            cause_measurement = structure_time * range_size  # Simplified cause calculation
            projected_move = cause_measurement * 0.5  # Conservative projection
            
            current_price = df['close'].iloc[-1]
            structure_type = structure_analysis.get("current_structure", "unknown")
            
            target_levels = []
            if structure_type == "accumulation":
                target_levels = [
                    current_price + projected_move * 0.5,  # Conservative target
                    current_price + projected_move,        # Primary target
                    current_price + projected_move * 1.5   # Extended target
                ]
            elif structure_type == "distribution":
                target_levels = [
                    current_price - projected_move * 0.5,  # Conservative target
                    current_price - projected_move,        # Primary target
                    current_price - projected_move * 1.5   # Extended target
                ]
            
            return {
                "cause": cause_measurement,
                "projected_effect": projected_move,
                "target_levels": target_levels,
                "cause_effect_ratio": projected_move / cause_measurement if cause_measurement > 0 else 0
            }
            
        except Exception as e:
            logger.error("Cause-effect calculation failed", error=str(e))
            return {"cause": 0, "projected_effect": 0, "target_levels": []}
    
    def _generate_trading_recommendations(self, 
                                        structure_analysis: Dict[str, Any],
                                        volume_analysis: Dict[str, Any], 
                                        spring_upthrust_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Wyckoff-based trading recommendations"""
        
        recommendations = {
            "action": "wait",
            "confidence": 0,
            "reasoning": [],
            "entry_levels": [],
            "stop_loss": None,
            "targets": []
        }
        
        try:
            structure_type = structure_analysis.get("current_structure", "unknown")
            latest_signal = spring_upthrust_analysis.get("latest_signal")
            vpoc = volume_analysis.get("vpoc", 0)
            current_price = structure_analysis.get("key_levels", {}).get("current_price", 0)
            
            # Accumulation-based recommendations
            if structure_type == "accumulation":
                if latest_signal and latest_signal.get("type") == "spring":
                    recommendations["action"] = "prepare_long"
                    recommendations["confidence"] = latest_signal.get("confidence", 0)
                    recommendations["reasoning"].append("Spring detected in accumulation structure")
                    recommendations["entry_levels"].append(current_price)
                    recommendations["stop_loss"] = latest_signal.get("price") * 0.999  # Below spring low
                
            # Distribution-based recommendations
            elif structure_type == "distribution":
                if latest_signal and latest_signal.get("type") == "upthrust":
                    recommendations["action"] = "prepare_short"
                    recommendations["confidence"] = latest_signal.get("confidence", 0)
                    recommendations["reasoning"].append("Upthrust detected in distribution structure")
                    recommendations["entry_levels"].append(current_price)
                    recommendations["stop_loss"] = latest_signal.get("price") * 1.001  # Above upthrust high
            
            # Volume-based recommendations
            if abs(current_price - vpoc) / current_price < 0.001:  # Near VPOC
                recommendations["reasoning"].append("Price near Value Point of Control")
                if recommendations["action"] == "wait":
                    recommendations["action"] = "monitor_closely"
                    recommendations["confidence"] = 60
            
            # Add targets if action is not wait
            if recommendations["action"] != "wait":
                structure_bounds = structure_analysis.get("structure_bounds", {})
                if structure_bounds:
                    if recommendations["action"] == "prepare_long":
                        recommendations["targets"] = [
                            structure_bounds.get("high", current_price * 1.01),
                            structure_bounds.get("high", current_price * 1.01) * 1.01
                        ]
                    elif recommendations["action"] == "prepare_short":
                        recommendations["targets"] = [
                            structure_bounds.get("low", current_price * 0.99),
                            structure_bounds.get("low", current_price * 0.99) * 0.99
                        ]
            
        except Exception as e:
            logger.error("Trading recommendation generation failed", error=str(e))
            recommendations["reasoning"].append(f"Analysis error: {str(e)}")
        
        return recommendations
    
    def _calculate_overall_confidence(self, 
                                    structure_analysis: Dict[str, Any],
                                    volume_analysis: Dict[str, Any],
                                    spring_upthrust_analysis: Dict[str, Any]) -> float:
        """Calculate overall analysis confidence score"""
        
        try:
            structure_confidence = structure_analysis.get("confidence", 0)
            signal_confidence = spring_upthrust_analysis.get("signal_confidence", 0)
            
            # Volume profile adds to confidence if well-formed
            volume_confidence = 70 if volume_analysis.get("vpoc") else 30
            
            # Weighted average
            overall_confidence = (
                structure_confidence * 0.5 + 
                signal_confidence * 0.3 + 
                volume_confidence * 0.2
            )
            
            return min(95, max(20, overall_confidence))
            
        except Exception:
            return 50  # Default moderate confidence
    
    # Helper methods for structure identification
    def _count_consolidation_bars(self, df: pd.DataFrame, high: float, low: float) -> int:
        """Count bars in consolidation range"""
        range_size = high - low
        tolerance = range_size * 0.1  # 10% tolerance
        
        count = 0
        for _, row in df.tail(100).iterrows():  # Look at last 100 bars
            if (low - tolerance) <= row['low'] and row['high'] <= (high + tolerance):
                count += 1
        
        return count
    
    def _analyze_structure_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume characteristics within structure"""
        recent_volume = df['volume'].tail(50)
        volume_trend = "increasing" if recent_volume.iloc[-10:].mean() > recent_volume.iloc[-20:-10].mean() else "decreasing"
        
        return {
            "volume_trend": volume_trend,
            "avg_volume": recent_volume.mean(),
            "volume_spikes": len(recent_volume[recent_volume > recent_volume.mean() * 1.5])
        }
    
    def _is_accumulation_pattern(self, df: pd.DataFrame, volume_chars: Dict[str, Any]) -> bool:
        """Determine if pattern shows accumulation characteristics"""
        # Simplified: decreasing volume on declines, increasing on advances
        return volume_chars.get("volume_trend") == "increasing"
    
    def _is_distribution_pattern(self, df: pd.DataFrame, volume_chars: Dict[str, Any]) -> bool:
        """Determine if pattern shows distribution characteristics"""
        # Simplified: increasing volume on advances (selling into strength)
        return volume_chars.get("volume_spikes", 0) > 3
    
    def _determine_accumulation_phase(self, df: pd.DataFrame, range_position: float) -> str:
        """Determine phase of accumulation"""
        if range_position < 0.3:
            return WyckoffPhase.PHASE_A.value
        elif range_position < 0.5:
            return WyckoffPhase.PHASE_B.value
        elif range_position < 0.7:
            return WyckoffPhase.PHASE_C.value
        else:
            return WyckoffPhase.PHASE_D.value
    
    def _determine_distribution_phase(self, df: pd.DataFrame, range_position: float) -> str:
        """Determine phase of distribution"""
        if range_position > 0.7:
            return WyckoffPhase.PHASE_A.value
        elif range_position > 0.5:
            return WyckoffPhase.PHASE_B.value
        elif range_position > 0.3:
            return WyckoffPhase.PHASE_C.value
        else:
            return WyckoffPhase.PHASE_D.value
    
    def _calculate_structure_confidence(self, consolidation_bars: int, volume_chars: Dict[str, Any], range_position: float) -> float:
        """Calculate confidence in structure identification"""
        confidence = 0
        
        # Time factor
        if consolidation_bars >= self.min_structure_bars:
            confidence += min(40, consolidation_bars * 2)
        
        # Volume factor
        if volume_chars.get("volume_spikes", 0) > 0:
            confidence += 20
        
        # Position factor (middle of range is more confident)
        position_confidence = 100 - abs(range_position - 0.5) * 100
        confidence += position_confidence * 0.3
        
        return min(95, max(20, confidence))
    
    def _determine_profile_shape(self, volume_at_price: np.ndarray) -> str:
        """Determine volume profile shape"""
        peak_count = len([i for i in range(1, len(volume_at_price)-1) 
                         if volume_at_price[i] > volume_at_price[i-1] and volume_at_price[i] > volume_at_price[i+1]])
        
        if peak_count == 1:
            return "normal"
        elif peak_count == 2:
            return "double_distribution"
        else:
            return "complex"
    
    async def _log_pattern_detection(self, analysis_result: Dict[str, Any]):
        """Log pattern detection to database"""
        try:
            pattern_type = analysis_result.get("structure_analysis", {}).get("current_structure", "unknown")
            confidence = analysis_result.get("confidence_score", 0)
            
            # Map string to enum
            wyckoff_pattern = None
            if pattern_type == "accumulation":
                wyckoff_pattern = WyckoffPattern.ACCUMULATION
            elif pattern_type == "distribution":
                wyckoff_pattern = WyckoffPattern.DISTRIBUTION
            elif pattern_type == "reaccumulation":
                wyckoff_pattern = WyckoffPattern.REACCUMULATION
            elif pattern_type == "redistribution":
                wyckoff_pattern = WyckoffPattern.REDISTRIBUTION
            
            if wyckoff_pattern:
                async with db_manager.get_async_session() as session:
                    detection = PatternDetection(
                        symbol="EUR_USD",
                        timeframe=analysis_result.get("timeframe", "15m"),
                        pattern_type=wyckoff_pattern,
                        confidence_score=confidence,
                        key_levels=analysis_result.get("structure_analysis", {}).get("key_levels", {}),
                        volume_analysis=analysis_result.get("volume_profile", {}),
                        market_context=analysis_result
                    )
                    session.add(detection)
                    await session.commit()
                    
        except Exception as e:
            logger.error("Failed to log pattern detection", error=str(e))


# Global analyzer instance
wyckoff_analyzer = WyckoffAnalyzer()

# Test function
async def test_wyckoff_analysis(symbol_name):
    """Test Wyckoff analysis with live data"""
    
    print("🧠 WYCKOFF INTELLIGENCE TEST")
    print("=" * 40)
    
    try:
        # Import our Oanda wrapper
        from src.mcp_servers.oanda_mcp_wrapper import OandaMCPWrapper
        
        # Get historical data for analysis
        async with OandaMCPWrapper("http://localhost:8000") as oanda:
            historical_data = await oanda.get_historical_data(symbol_name, "M15", 200)
        
        if "error" in historical_data:
            print(f"❌ Failed to get data: {historical_data['error']}")
            return False
        
        print(f"📊 Analyzing {len(historical_data['data'])} bars of {symbol_name} 15-minute data...")
        
        # Run Wyckoff analysis
        analysis_result = await wyckoff_analyzer.analyze_market_data(historical_data['data'], "M15")
        
        if "error" in analysis_result:
            print(f"❌ Analysis failed: {analysis_result['error']}")
            return False
        
        # Display results
        print("\n🎯 WYCKOFF ANALYSIS RESULTS")
        print("=" * 40)
        
        # Structure Analysis
        structure = analysis_result.get("structure_analysis", {})
        print(f"📐 Structure Type: {structure.get('current_structure', 'Unknown').title()}")
        print(f"🔄 Phase: {structure.get('phase', 'Unknown').title()}")
        print(f"📊 Confidence: {structure.get('confidence', 0):.1f}%")
        print(f"⏱️  Time in Structure: {structure.get('time_in_structure', 0)} bars")
        
        # Key Levels
        levels = structure.get("key_levels", {})
        if levels:
            print(f"\n🎯 Key Levels:")
            print(f"   Resistance: {levels.get('resistance', 0):.5f}")
            print(f"   Support: {levels.get('support', 0):.5f}")
            print(f"   Current: {levels.get('current_price', 0):.5f}")
        
        # Volume Profile
        volume_profile = analysis_result.get("volume_profile", {})
        if volume_profile and "error" not in volume_profile:
            print(f"\n📊 Volume Profile:")
            print(f"   VPOC: {volume_profile.get('vpoc', 0):.5f}")
            print(f"   Value Area: {volume_profile.get('value_area_low', 0):.5f} - {volume_profile.get('value_area_high', 0):.5f}")
            print(f"   Profile Shape: {volume_profile.get('profile_shape', 'Unknown').title()}")
        
        # Springs/Upthrusts
        signals = analysis_result.get("spring_upthrust", {})
        springs = signals.get("springs_detected", [])
        upthrusts = signals.get("upthrusts_detected", [])
        latest_signal = signals.get("latest_signal")
        
        print(f"\n⚡ Springs & Upthrusts:")
        print(f"   Springs Detected: {len(springs)}")
        print(f"   Upthrusts Detected: {len(upthrusts)}")
        
        if latest_signal:
            print(f"   Latest Signal: {latest_signal.get('type', 'Unknown').title()}")
            print(f"   Signal Confidence: {latest_signal.get('confidence', 0)}%")
        
        # Market Regime
        regime = analysis_result.get("market_regime", {})
        print(f"\n🌊 Market Regime:")
        print(f"   Current Regime: {regime.get('regime', 'Unknown').title()}")
        print(f"   Trend Direction: {regime.get('trend_direction', 'Unknown').title()}")
        print(f"   Volatility: {regime.get('volatility', 0):.4f}")
        
        # Trading Recommendations
        recommendations = analysis_result.get("trading_recommendations", {})
        print(f"\n💡 Trading Recommendations:")
        print(f"   Action: {recommendations.get('action', 'Wait').title()}")
        print(f"   Confidence: {recommendations.get('confidence', 0)}%")
        
        reasoning = recommendations.get("reasoning", [])
        if reasoning:
            print(f"   Reasoning:")
            for reason in reasoning:
                print(f"     • {reason}")
        
        # Overall Confidence
        overall_confidence = analysis_result.get("confidence_score", 0)
        print(f"\n🎯 Overall Analysis Confidence: {overall_confidence:.1f}%")
        
        print("\n✅ Wyckoff analysis complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    symbol_name = "EUR_USD"
    success = asyncio.run(test_wyckoff_analysis(symbol_name))
    if success:
        print("\n🎉 Wyckoff Intelligence ready for integration!")
    else:
        print("\n🔧 Check the setup and try again")
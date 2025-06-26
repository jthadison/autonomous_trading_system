"""
Enhanced CrewAI Result Parser - FIXED VERSION
Handles type validation and parsing errors for trading signals
"""

import re
import json
import numpy as np
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Robust trading signal with type validation"""
    action: str
    symbol: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    wyckoff_phase: str
    pattern_type: str
    timestamp: datetime
    
    def __post_init__(self):
        """Validate and sanitize data after initialization"""
        self.action = self._validate_action(self.action)
        self.confidence = self._validate_float(self.confidence, 0, 100, 75.0)
        self.entry_price = self._validate_float(self.entry_price, 0.001, 100000, 1.0)
        self.stop_loss = self._validate_float(self.stop_loss, 0.001, 100000, self.entry_price * 0.99)
        self.take_profit = self._validate_float(self.take_profit, 0.001, 100000, self.entry_price * 1.01)
        self.position_size = self._validate_float(self.position_size, 1, 1000000, 1000.0)
        self.reasoning = str(self.reasoning)[:500]  # Limit length
        self.wyckoff_phase = self._validate_wyckoff_phase(self.wyckoff_phase)
        self.pattern_type = self._validate_pattern_type(self.pattern_type)
    
    def _validate_action(self, action: Any) -> str:
        """Validate trading action"""
        if isinstance(action, str):
            action = action.lower().strip()
            if action in ['buy', 'sell', 'hold']:
                return action
        return 'hold'  # Default safe action
    
    def _validate_float(self, value: Any, min_val: float, max_val: float, default: float) -> float:
        """Validate and clamp float values"""
        try:
            if isinstance(value, (int, float)):
                float_val = float(value)
                if min_val <= float_val <= max_val:
                    return float_val
            elif isinstance(value, str):
                # Try to extract number from string
                numbers = re.findall(r'[\d.]+', value)
                if numbers:
                    float_val = float(numbers[0])
                    if min_val <= float_val <= max_val:
                        return float_val
        except (ValueError, TypeError, IndexError):
            pass
        
        logger.warning(f"Invalid float value {value}, using default {default}")
        return default
    
    def _validate_wyckoff_phase(self, phase: Any) -> str:
        """Validate Wyckoff phase"""
        valid_phases = ['A', 'B', 'C', 'D', 'E']
        if isinstance(phase, str) and phase.upper() in valid_phases:
            return phase.upper()
        return 'C'  # Default to Phase C
    
    def _validate_pattern_type(self, pattern: Any) -> str:
        """Validate pattern type"""
        valid_patterns = ['accumulation', 'distribution', 'reaccumulation', 'redistribution', 'spring', 'upthrust']
        if isinstance(pattern, str) and pattern.lower() in valid_patterns:
            return pattern.lower()
        return 'accumulation'  # Default pattern


class CrewResultParser:
    """Enhanced parser for CrewAI results with comprehensive error handling"""
    
    def __init__(self):
        self.default_confidence = 75.0
        self.default_position_size = 1000.0
        self.default_stop_distance = 0.01  # 1%
        self.default_take_profit_distance = 0.02  # 2%
    
    def parse_crew_result(self, crew_result: Any, symbol: str, current_price: float) -> Optional[TradingSignal]:
        """Parse CrewAI result into structured trading signal with robust error handling"""
        try:
            # Convert result to string for parsing
            result_text = self._normalize_result_text(crew_result)
            
            # Extract trading action
            action = self._extract_action(result_text)
            
            # Extract confidence
            confidence = self._extract_confidence(result_text)
            
            # Extract price levels
            entry_price = self._extract_entry_price(result_text, current_price)
            stop_loss = self._extract_stop_loss(result_text, entry_price, action)
            take_profit = self._extract_take_profit(result_text, entry_price, action)
            
            # Extract position size
            position_size = self._extract_position_size(result_text)
            
            # Extract Wyckoff information
            wyckoff_phase = self._extract_wyckoff_phase(result_text)
            pattern_type = self._extract_pattern_type(result_text)
            
            # Create signal with validation
            signal = TradingSignal(
                action=action,
                symbol=symbol,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=result_text[:200],  # First 200 chars as reasoning
                wyckoff_phase=wyckoff_phase,
                pattern_type=pattern_type,
                timestamp=datetime.now()
            )
            
            logger.info(f"✅ Parsed trading signal: {action} {symbol} @ {entry_price:.5f}")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Failed to parse crew result: {e}")
            return self._create_default_signal(symbol, current_price)
    
    def _normalize_result_text(self, result: Any) -> str:
        """Convert various result types to normalized text"""
        try:
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return json.dumps(result, default=str)
            elif hasattr(result, 'raw'):
                return str(result.raw)
            elif hasattr(result, '__dict__'):
                return str(result.__dict__)
            else:
                return str(result)
        except Exception as e:
            logger.warning(f"Failed to normalize result: {e}")
            return "hold"
    
    def _extract_action(self, text: str) -> str:
        """Extract trading action from text"""
        text_lower = text.lower()
        
        # Look for explicit action words
        if any(word in text_lower for word in ['buy', 'long', 'bullish', 'accumulate']):
            return 'buy'
        elif any(word in text_lower for word in ['sell', 'short', 'bearish', 'distribute']):
            return 'sell'
        else:
            return 'hold'
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        try:
            # Look for percentage patterns
            confidence_patterns = [
                r'confidence[:\s]*(\d+(?:\.\d+)?)%?',
                r'(\d+(?:\.\d+)?)%\s*confidence',
                r'score[:\s]*(\d+(?:\.\d+)?)',
            ]
            
            for pattern in confidence_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    confidence = float(matches[0])
                    # If it's already a percentage (0-100), use it
                    if 0 <= confidence <= 100:
                        return confidence
                    # If it's a decimal (0-1), convert to percentage
                    elif 0 <= confidence <= 1:
                        return confidence * 100
            
        except (ValueError, IndexError):
            pass
        
        return self.default_confidence
    
    def _extract_entry_price(self, text: str, current_price: float) -> float:
        """Extract entry price from text"""
        try:
            # Look for price patterns
            price_patterns = [
                r'entry[:\s]*(\d+\.\d+)',
                r'price[:\s]*(\d+\.\d+)',
                r'@\s*(\d+\.\d+)',
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    price = float(matches[0])
                    # Validate price is reasonable (within 10% of current)
                    if 0.9 * current_price <= price <= 1.1 * current_price:
                        return price
            
        except (ValueError, IndexError):
            pass
        
        return current_price
    
    def _extract_stop_loss(self, text: str, entry_price: float, action: str) -> float:
        """Extract stop loss from text"""
        try:
            # Look for stop loss patterns
            stop_patterns = [
                r'stop[:\s]*(\d+\.\d+)',
                r'sl[:\s]*(\d+\.\d+)',
                r'stop.loss[:\s]*(\d+\.\d+)',
            ]
            
            for pattern in stop_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    stop_price = float(matches[0])
                    # Validate stop is on correct side
                    if action == 'buy' and stop_price < entry_price:
                        return stop_price
                    elif action == 'sell' and stop_price > entry_price:
                        return stop_price
            
        except (ValueError, IndexError):
            pass
        
        # Default stop loss calculation
        if action == 'buy':
            return entry_price * (1 - self.default_stop_distance)
        else:
            return entry_price * (1 + self.default_stop_distance)
    
    def _extract_take_profit(self, text: str, entry_price: float, action: str) -> float:
        """Extract take profit from text"""
        try:
            # Look for take profit patterns
            tp_patterns = [
                r'target[:\s]*(\d+\.\d+)',
                r'tp[:\s]*(\d+\.\d+)',
                r'take.profit[:\s]*(\d+\.\d+)',
                r'profit[:\s]*(\d+\.\d+)',
            ]
            
            for pattern in tp_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    tp_price = float(matches[0])
                    # Validate TP is on correct side
                    if action == 'buy' and tp_price > entry_price:
                        return tp_price
                    elif action == 'sell' and tp_price < entry_price:
                        return tp_price
            
        except (ValueError, IndexError):
            pass
        
        # Default take profit calculation
        if action == 'buy':
            return entry_price * (1 + self.default_take_profit_distance)
        else:
            return entry_price * (1 - self.default_take_profit_distance)
    
    def _extract_position_size(self, text: str) -> float:
        """Extract position size from text with type safety"""
        try:
            # Look for position size patterns
            size_patterns = [
                r'size[:\s]*(\d+(?:\.\d+)?)',
                r'position[:\s]*(\d+(?:\.\d+)?)',
                r'quantity[:\s]*(\d+(?:\.\d+)?)',
                r'units[:\s]*(\d+(?:\.\d+)?)',
            ]
            
            for pattern in size_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    size_str = matches[0]
                    # Ensure we have a valid numeric string
                    if isinstance(size_str, str) and size_str.replace('.', '').isdigit():
                        size = float(size_str)
                        if 1 <= size <= 1000000:  # Reasonable bounds
                            return size
            
        except (ValueError, IndexError, TypeError) as e:
            logger.debug(f"Position size extraction failed: {e}")
        
        return self.default_position_size
    
    def _extract_wyckoff_phase(self, text: str) -> str:
        """Extract Wyckoff phase from text"""
        text_upper = text.upper()
        
        # Look for phase indicators
        for phase in ['A', 'B', 'C', 'D', 'E']:
            if f'PHASE {phase}' in text_upper or f'PHASE_{phase}' in text_upper:
                return phase
        
        return 'C'  # Default phase
    
    def _extract_pattern_type(self, text: str) -> str:
        """Extract Wyckoff pattern type from text"""
        text_lower = text.lower()
        
        patterns = {
            'accumulation': ['accumulation', 'accumulating', 'buying'],
            'distribution': ['distribution', 'distributing', 'selling'],
            'spring': ['spring', 'false breakdown'],
            'upthrust': ['upthrust', 'false breakout'],
            'reaccumulation': ['reaccumulation', 'reaccumulating'],
            'redistribution': ['redistribution', 'redistributing']
        }
        
        for pattern_type, keywords in patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return pattern_type
        
        return 'accumulation'  # Default pattern
    
    def _create_default_signal(self, symbol: str, current_price: float) -> TradingSignal:
        """Create a safe default signal when parsing fails"""
        return TradingSignal(
            action='hold',
            symbol=symbol,
            confidence=50.0,
            entry_price=current_price,
            stop_loss=current_price * 0.99,
            take_profit=current_price * 1.01,
            position_size=self.default_position_size,
            reasoning="Default signal due to parsing error",
            wyckoff_phase='C',
            pattern_type='accumulation',
            timestamp=datetime.now()
        )


# Safe position size calculator with type validation
def safe_calculate_position_size(
    balance: Union[str, int, float], 
    risk_percent: Union[str, int, float], 
    stop_distance: Union[str, int, float],
    price: Union[str, int, float] = 1.0
) -> float:
    """Calculate position size with comprehensive type safety"""
    try:
        # Convert all inputs to float with validation
        balance_float = _safe_float_convert(balance, 100000.0)
        risk_float = _safe_float_convert(risk_percent, 0.02)
        stop_float = _safe_float_convert(stop_distance, 0.01)
        price_float = _safe_float_convert(price, 1.0)
        
        # Ensure no division by zero
        if stop_float <= 0 or price_float <= 0:
            logger.warning("Invalid stop distance or price for position sizing")
            return 1000.0
        
        # Calculate risk amount
        risk_amount = balance_float * risk_float
        
        # Calculate position size
        position_size = risk_amount / (stop_float * price_float)
        
        # Apply reasonable bounds
        min_size, max_size = 1.0, balance_float * 0.1 / price_float
        position_size = max(min_size, min(position_size, max_size))
        
        return round(position_size, 0)
        
    except Exception as e:
        logger.error(f"Position size calculation error: {e}")
        return 1000.0  # Safe default


def _safe_float_convert(value: Any, default: float) -> float:
    """Safely convert any value to float"""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Remove any non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.]', '', value)
            if cleaned and cleaned.count('.') <= 1:
                return float(cleaned)
    except (ValueError, TypeError):
        pass
    
    logger.debug(f"Could not convert {value} to float, using default {default}")
    return default


# Global parser instance
crew_result_parser = CrewResultParser()

# Convenience function
def parse_trading_signal(crew_result: Any, symbol: str, current_price: float) -> Optional[TradingSignal]:
    """Parse crew result into trading signal (convenience function)"""
    return crew_result_parser.parse_crew_result(crew_result, symbol, current_price)
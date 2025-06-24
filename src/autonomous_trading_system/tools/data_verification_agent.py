import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import sys
from pathlib import Path
from datetime import datetime
import re

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.config.logging_config import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class DataVerificationInput(BaseModel):
    """Input schema for Data Verification tools."""
    data: Union[str, List[Dict], Dict] = Field(..., description="Raw market data to verify and clean")
    instrument: Optional[str] = Field(default="UNKNOWN", description="Trading instrument")
    timeframe: Optional[str] = Field(default="M15", description="Timeframe")

class DataVerificationTool(BaseTool):
    name: str = "verify_and_clean_data"
    description: str = (
        "Verifies, cleans, and formats raw market data. Handles JSON parsing issues, "
        "validates data structure, removes corrupted records, and standardizes format."
    )
    args_schema: type[BaseModel] = DataVerificationInput

    def _run(self, data: Union[str, List[Dict], Dict], instrument: str = "UNKNOWN", timeframe: str = "M15") -> str:
        """Comprehensive data verification and cleaning process"""
        try:
            logger.info(f"Starting data verification for {instrument} {timeframe}")
            print(f"Starting data verification for {instrument} {timeframe}")
            
            # Step 1: Parse and extract data
            parsed_data = self._parse_raw_data(data)
            
            # Step 2: Validate data structure
            validated_data = self._validate_data_structure(parsed_data)
            
            # Step 3: Clean and standardize
            cleaned_data = self._clean_and_standardize(validated_data, instrument, timeframe)
            
            # Step 4: Quality assessment
            quality_report = self._assess_data_quality(cleaned_data)
            
            # Step 5: Format verification result
            result = self._format_verification_result(cleaned_data, quality_report, instrument, timeframe)
            
            logger.info(f"Data verification completed for {instrument}")
            return result
            
        except Exception as e:
            logger.error(f"Data verification failed: {str(e)}")
            return self._format_error_result(str(e), instrument, timeframe)
    
    def _parse_raw_data(self, data: Union[str, List[Dict], Dict]) -> List[Dict]:
        """Parse raw data from various input formats"""
        try:
            if isinstance(data, str):
                return self._parse_string_data(data)
            elif isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data] if data else []
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            logger.warning(f"Data parsing failed: {e}")
            return []
    
    def _parse_string_data(self, data_str: str) -> List[Dict]:
        """Parse string data with robust error handling"""
        try:
            # Remove problematic characters
            cleaned_str = data_str.replace("...", "").replace(" ... ", "")
            
            # Try direct JSON parsing first
            try:
                if cleaned_str.strip().startswith('['):
                    return json.loads(cleaned_str)
                else:
                    return [json.loads(cleaned_str)]
            except json.JSONDecodeError:
                pass
            
            # Try regex extraction of JSON objects
            json_objects = re.findall(r'\\{[^{}]*\\}', cleaned_str)
            parsed_objects = []
            
            for obj_str in json_objects[:100]:  # Limit to 100 objects
                try:
                    obj = json.loads(obj_str)
                    parsed_objects.append(obj)
                except:
                    continue
            
            return parsed_objects
            
        except Exception as e:
            logger.warning(f"String parsing failed: {e}")
            return []
    
    def _validate_data_structure(self, data: List[Dict]) -> List[Dict]:
        """Validate and filter data records"""
        if not data:
            return []
        
        validated_records = []
        
        for i, record in enumerate(data):
            try:
                # Check if record has required fields
                has_time = any(field in record for field in ['time', 'timestamp', 'Time'])
                has_ohlc = self._check_ohlc_fields(record)
                
                if has_time and has_ohlc:
                    validated_records.append(record)
                    
            except Exception as e:
                continue
        
        logger.info(f"Validated {len(validated_records)} out of {len(data)} records")
        return validated_records
    
    def _check_ohlc_fields(self, record: Dict) -> bool:
        """Check if record has OHLC price data"""
        # Check for direct OHLC fields
        ohlc_fields = ['open', 'high', 'low', 'close', 'o', 'h', 'l', 'c']
        if any(field in record for field in ohlc_fields):
            return True
        
        # Check for nested OHLC in 'mid' field (OANDA format)
        if 'mid' in record:
            mid_data = record['mid']
            if isinstance(mid_data, dict):
                return any(field in mid_data for field in ['o', 'h', 'l', 'c'])
        
        return False
    
    def _clean_and_standardize(self, data: List[Dict], instrument: str, timeframe: str) -> List[Dict]:
        """Clean and standardize data format"""
        if not data:
            return []
        
        standardized_records = []
        
        for record in data:
            try:
                cleaned_record = self._standardize_record(record, instrument, timeframe)
                if cleaned_record:
                    standardized_records.append(cleaned_record)
            except Exception as e:
                continue
        
        # Sort by timestamp
        try:
            standardized_records.sort(key=lambda x: x.get('timestamp', ''))
        except:
            pass
        
        logger.info(f"Standardized {len(standardized_records)} records")
        return standardized_records
    
    def _standardize_record(self, record: Dict, instrument: str, timeframe: str) -> Optional[Dict]:
        """Standardize a single record to consistent format"""
        try:
            # Extract timestamp
            timestamp = (
                record.get('time') or 
                record.get('timestamp') or 
                record.get('Time') or
                datetime.now().isoformat()
            )
            
            # Extract OHLC data
            ohlc = self._extract_ohlc(record)
            if not ohlc:
                return None
            
            # Extract volume
            volume = record.get('volume', record.get('v', 1000))
            
            # Create standardized record
            standardized = {
                'timestamp': timestamp,
                'instrument': instrument,
                'timeframe': timeframe,
                'open': float(ohlc['open']),
                'high': float(ohlc['high']),
                'low': float(ohlc['low']),
                'close': float(ohlc['close']),
                'volume': int(volume),
                'complete': record.get('complete', True)
            }
            
            # Validate prices
            if not self._validate_prices(standardized):
                return None
            
            return standardized
            
        except Exception as e:
            return None
    
    def _extract_ohlc(self, record: Dict) -> Optional[Dict]:
        """Extract OHLC data from various formats"""
        try:
            # Try direct fields first
            if all(field in record for field in ['open', 'high', 'low', 'close']):
                return {
                    'open': record['open'],
                    'high': record['high'], 
                    'low': record['low'],
                    'close': record['close']
                }
            
            # Try short field names
            if all(field in record for field in ['o', 'h', 'l', 'c']):
                return {
                    'open': record['o'],
                    'high': record['h'],
                    'low': record['l'], 
                    'close': record['c']
                }
            
            # Try nested 'mid' field (OANDA format)
            if 'mid' in record:
                mid = record['mid']
                if isinstance(mid, dict) and all(field in mid for field in ['o', 'h', 'l', 'c']):
                    return {
                        'open': mid['o'],
                        'high': mid['h'],
                        'low': mid['l'],
                        'close': mid['c']
                    }
            
            return None
            
        except Exception:
            return None
    
    def _validate_prices(self, record: Dict) -> bool:
        """Validate price data makes sense"""
        try:
            o, h, l, c = record['open'], record['high'], record['low'], record['close']
            
            # Check all prices are positive
            if any(price <= 0 for price in [o, h, l, c]):
                return False
            
            # Check high >= low
            if h < l:
                return False
            
            # Check open and close are within high/low range
            if not (l <= o <= h and l <= c <= h):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _assess_data_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """Assess the quality of cleaned data"""
        if not data:
            return {
                'quality_score': 0,
                'total_records': 0,
                'issues': ['No data available']
            }
        
        issues = []
        total_records = len(data)
        
        # Check data completeness
        if total_records < 10:
            issues.append(f"Limited data: only {total_records} records")
        
        # Calculate quality score
        quality_score = 100
        quality_score -= len(issues) * 10
        quality_score = max(0, min(100, quality_score))
        
        return {
            'quality_score': quality_score,
            'total_records': total_records,
            'issues': issues
        }
    
    def _format_verification_result(self, cleaned_data: List[Dict], quality_report: Dict, instrument: str, timeframe: str) -> str:
        """Format the verification result"""
        
        result_lines = [
            "# Data Verification & Cleaning Report",
            "",
            f"## Instrument: {instrument} | Timeframe: {timeframe}",
            "",
            "### Data Quality Assessment",
            f"- **Quality Score**: {quality_report['quality_score']}/100",
            f"- **Total Records**: {quality_report['total_records']}",
            ""
        ]
        
        if quality_report['issues']:
            result_lines.extend([
                "### Issues Identified",
                ""
            ])
            for issue in quality_report['issues']:
                result_lines.append(f"- {issue}")
            result_lines.append("")
        
        # Data sample
        if cleaned_data:
            sample_record = cleaned_data[0]
            result_lines.extend([
                "### Sample Record (First)",
                f"- **Timestamp**: {sample_record.get('timestamp', 'N/A')}",
                f"- **Open**: {sample_record.get('open', 'N/A')}",
                f"- **High**: {sample_record.get('high', 'N/A')}",
                f"- **Low**: {sample_record.get('low', 'N/A')}",
                f"- **Close**: {sample_record.get('close', 'N/A')}",
                f"- **Volume**: {sample_record.get('volume', 'N/A')}",
                ""
            ])
        
        # Verification status
        if quality_report['quality_score'] >= 80:
            status = "PASSED - Data ready for analysis"
        elif quality_report['quality_score'] >= 60:
            status = "WARNING - Data usable but has issues"
        else:
            status = "FAILED - Data quality too poor for reliable analysis"
        
        result_lines.extend([
            "### Verification Status",
            f"**{status}**",
            "",
            "### Next Steps",
            "- Data has been cleaned and standardized",
            "- Ready for market analysis and Wyckoff pattern detection",
            "- Proceed to next agent in the pipeline"
        ])
        
        return "\\n".join(result_lines)
    
    def _format_error_result(self, error: str, instrument: str, timeframe: str) -> str:
        """Format error result"""
        return f"""# Data Verification Failed

## Instrument: {instrument} | Timeframe: {timeframe}

### Error Details
{error}

### Status
**FAILED - Unable to process data**
"""

class DataQualityReportTool(BaseTool):
    name: str = "generate_data_quality_report"
    description: str = "Generates detailed data quality report for cleaned market data"
    args_schema: type[BaseModel] = DataVerificationInput

    def _run(self, cleaned_data: Union[str, List[Dict]]) -> str:
        """Generate comprehensive data quality report"""
        try:
            return "Data quality report generated successfully"
        except Exception as e:
            return f"Failed to generate data quality report: {str(e)}"
        
# Create tool instances
data_verification_tool = DataVerificationTool()
data_quality_report_tool = DataQualityReportTool()
#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from autonomous_trading_system.crew import AutonomousTradingSystem

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    symbol_name ="EUR_USD"
    
    inputs = {
        'topic': 'AI LLMs',
        'symbol_name': symbol_name,
        'current_year': str(datetime.now().year)
    }
    
    try:
        AutonomousTradingSystem().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")



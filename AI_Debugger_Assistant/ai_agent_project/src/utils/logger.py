# Path: AI_Debugger_Assistant/ai_agent_project/src/utils/logger.py
# Custom logging setup for the project

import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger('AI_Debugger_Assistant')

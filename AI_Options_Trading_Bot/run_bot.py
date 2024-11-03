# Path: AI_Options_Trading_Bot/run_bot.py
# Main script to start the agent dispatcher and execute tasks

from src.agents.AgentDispatcher import AgentDispatcher

if __name__ == '__main__':
    dispatcher = AgentDispatcher(config='config/config.yaml')
    dispatcher.dispatch()

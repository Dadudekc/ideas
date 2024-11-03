# Path: Dating_Assistant_Bot/run_dating_assistant.py
# Main script to initiate the Dating Assistant Bot

from src.agent.DatingAssistantAgent import DatingAssistantAgent

if __name__ == '__main__':
    # Initialize and start the Dating Assistant Bot
    bot = DatingAssistantAgent(config='config/config.yaml')
    bot.start_conversation()

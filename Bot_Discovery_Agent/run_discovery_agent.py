# Path: Bot_Discovery_Agent/run_discovery_agent.py
# Main script to initiate bot discovery and evaluation

from src.agent.BotDiscoveryAgent import BotDiscoveryAgent

if __name__ == '__main__':
    # Initialize and start the Bot Discovery Agent
    bot_agent = BotDiscoveryAgent(config='config/config.yaml')
    bot_agent.discover_bots()

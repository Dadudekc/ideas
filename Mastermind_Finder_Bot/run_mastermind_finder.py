# Path: Mastermind_Finder_Bot/run_mastermind_finder.py
# Main script to initiate the Mastermind Finder Bot

from src.agent.MastermindFinderAgent import MastermindFinderAgent

if __name__ == '__main__':
    # Initialize and start the Mastermind Finder Bot
    bot = MastermindFinderAgent(config='config/config.yaml')
    bot.source_candidates()

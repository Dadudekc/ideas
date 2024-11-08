# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import TaskBoard
from models.database import init_db, get_session, Project
from integrations.alpaca_integration import AlpacaIntegration
from utils.data_fetcher import DataFetcher
from utils.logger import logger
from models.database import Insight

def ensure_project_exists(session):
    """Ensures that a default project exists in the database."""
    project = session.query(Project).filter_by(name="TSLA Monitoring").first()
    if not project:
        project = Project(name="TSLA Monitoring", description="Monitor daily price action of TSLA.")
        session.add(project)
        session.commit()
        logger.info("Default project 'TSLA Monitoring' created.")

def main():
    """Main entry point for the application."""
    # Initialize the database
    init_db()

    # Create a new database session
    session = get_session()

    # Ensure a default project exists
    ensure_project_exists(session)

    # Initialize Alpaca Integration
    alpaca = AlpacaIntegration()

    # Initialize PyQt5 Application
    app = QApplication(sys.argv)
    task_board = TaskBoard(session, alpaca)
    task_board.setWindowTitle("TSLA Price Monitor")
    task_board.setGeometry(100, 100, 1200, 800)
    task_board.show()

    # Initialize Data Fetcher
    data_fetcher = DataFetcher()

    # Define a timer to fetch and log data periodically (e.g., every hour)
    from PyQt5.QtCore import QTimer

    def fetch_and_log_data():
        """Fetches TSLA data and logs insights."""
        data = data_fetcher.fetch_daily_price()
        if data:
            insight_content = (
                f"TSLA closed at ${data['close']:.2f} on {data['date']}. "
                f"High: ${data['high']:.2f}, Low: ${data['low']:.2f}, Volume: {data['volume']}"
            )
            # Log insight to database
            insight = Insight(content=insight_content, project_id=1)  # Assuming project_id=1
            session.add(insight)
            session.commit()
            logger.info(f"Logged insight: {insight_content}")
        else:
            logger.warning("No data fetched to log insight.")

    # Set up the timer (e.g., every hour: 3600000 milliseconds)
    timer = QTimer()
    timer.timeout.connect(fetch_and_log_data)
    timer.start(3600000)  # 1 hour

    # Fetch data immediately on startup
    fetch_and_log_data()

    # Execute the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

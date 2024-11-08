# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/Scripts/Utilities/data/data_store.py
# Description:
#     Manages storage and retrieval of financial data, with primary support
#     for SQL database (PostgreSQL) and optional CSV support. This script
#     merges functionalities from postgres_data_store.py into data_store.py,
#     providing a comprehensive DataStore class capable of handling various
#     data operations including loading, saving, updating, and deleting stock
#     data, as well as managing model data.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Section 1: Imports
# -------------------------------------------------------------------
import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from datetime import datetime
import pandas as pd
import pickle

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Date,
    UniqueConstraint,
    text
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

from dotenv import load_dotenv

# -------------------------------------------------------------------
# Section 2: Project Path Setup
# -------------------------------------------------------------------

# Dynamically set the project root based on the current file's location
project_root = Path(__file__).resolve().parents[3]  # Adjusted to match your project structure
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# -------------------------------------------------------------------
# Section 3: Import ConfigManager and Logging Setup Using Absolute Imports
# -------------------------------------------------------------------
from Scripts.Utilities.config_handling.config_manager import ConfigManager
from Scripts.Utilities.config_handling.logging_setup import setup_logging
from Scripts.Utilities.data.data_store_interface import DataStoreInterface

# -------------------------------------------------------------------
# Section 4: Load Environment Variables
# -------------------------------------------------------------------
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    logging.warning(f"No .env file found at {env_path}. Environment variables will be used as-is.")

# -------------------------------------------------------------------
# Section 5: Logging Configuration
# -------------------------------------------------------------------
log_dir = project_root / 'logs' / 'Utilities'
log_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logging(
    script_name="data_store",
    log_dir=log_dir,
    max_log_size=5 * 1024 * 1024,  # 5 MB
    backup_count=3,
    console_log_level=logging.DEBUG,  # Set to DEBUG for detailed logs
    file_log_level=logging.DEBUG,
    feedback_loop_enabled=True
)

# -------------------------------------------------------------------
# Section 6: SQLAlchemy Setup
# -------------------------------------------------------------------
Base = declarative_base()

# PostgreSQL configuration from environment variables
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_DBNAME = os.getenv('POSTGRES_DBNAME', 'trading_robot_plug')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

# Construct PostgreSQL connection string
DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DBNAME}"
)
logger.info(f"DATABASE_URL: {DATABASE_URL}")  # Log the DATABASE_URL for debugging

def get_db_engine() -> Engine:
    """
    Returns a SQLAlchemy engine for PostgreSQL with connection pooling.

    Returns:
        SQLAlchemy Engine instance.

    Raises:
        SQLAlchemyError: If the engine creation fails.
    """
    try:
        database_url = DATABASE_URL
        logger.info(f"Connecting to PostgreSQL database using URL: {database_url}")

        engine = create_engine(database_url, pool_size=10, max_overflow=20)
        Base.metadata.create_all(engine)
        logger.info("Successfully connected to PostgreSQL database.")
        return engine
    except SQLAlchemyError as e:
        logger.error(f"Failed to create engine: {e}")
        raise

# -------------------------------------------------------------------
# Section 7: Database Models
# -------------------------------------------------------------------
class StockData(Base):
    __tablename__ = 'stock_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True, nullable=False)
    Date = Column(Date, index=True, nullable=False)
    Open = Column(Float, nullable=True)
    High = Column(Float, nullable=True)
    Low = Column(Float, nullable=True)
    Close = Column(Float, nullable=True)
    Volume = Column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint('symbol', 'Date', name='uix_symbol_date'),
    )

class ModelData(Base):
    """
    Stores information about machine learning models, including configuration,
    training parameters, and performance metrics.
    """
    __tablename__ = 'model_data'
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False, index=True, unique=True)
    model_type = Column(String, nullable=False)
    training_date = Column(Date, nullable=False)
    hyperparameters = Column(String)  # JSON string
    metrics = Column(String)          # JSON string
    description = Column(String)      # Optional description of the model

    __table_args__ = (
        UniqueConstraint('model_name', name='unique_model_name'),
    )

# -------------------------------------------------------------------
# Section 8: Database Connection Class
# -------------------------------------------------------------------
class DatabaseHandler:
    def __init__(self, logger: logging.Logger) -> None:
        """
        Initializes the DatabaseHandler with a SQLAlchemy engine and session maker.

        Args:
            logger (logging.Logger): Logger instance for logging.
        """
        self.logger = logger
        try:
            self.engine = get_db_engine()
            # Create a sessionmaker instance
            self.Session = sessionmaker(bind=self.engine)
            self.logger.info("DatabaseHandler initialized successfully.")
        except SQLAlchemyError as e:
            self.logger.error(f"DatabaseHandler initialization failed: {e}")
            raise

    def get_session(self):
        """
        Creates and returns a new SQLAlchemy session instance.

        Returns:
            SQLAlchemy session instance.
        """
        try:
            session = self.Session()
            self.logger.debug("New database session created.")
            return session
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to create a database session: {e}")
            raise

    def close(self):
        """
        Closes the SQLAlchemy engine and disposes of all connections.
        """
        try:
            self.engine.dispose()
            self.logger.info("Database engine disposed and connection closed.")
        except Exception as e:
            self.logger.error(f"Error closing the database engine: {e}")
            raise

# -------------------------------------------------------------------
# Section 9: DataStore Class
# -------------------------------------------------------------------
class DataStore(DataStoreInterface):
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None,
        use_csv: bool = False,
        csv_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initializes the DataStore class. Supports both SQL and CSV storage modes.

        Args:
            config (ConfigManager, optional): Configuration manager instance.
            config_manager (ConfigManager, optional): Alias for config to maintain backward compatibility.
            logger (logging.Logger, optional): Logger instance for logging.
            use_csv (bool): Use CSV instead of a database. Defaults to False.
            csv_dir (str or Path, optional): Directory path for CSV files. Defaults to None.
        """
        # Use config_manager if config is not provided (for backward compatibility)
        if config is None:
            config = config_manager

        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.use_csv = use_csv

        if self.use_csv:
            self.csv_dir = Path(csv_dir) if csv_dir else self._get_csv_dir()
            self.csv_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initialized for CSV mode with directory: {self.csv_dir}")
        else:
            self.csv_dir = None
            self.db_handler = DatabaseHandler(logger=self.logger)
            self.logger.info(f"Initialized for SQL mode with DB URL: {self.db_handler.engine.url}")

    def _get_csv_dir(self) -> Path:
        """Retrieve or create the CSV directory."""
        csv_dir = Path(os.getenv('CSV_DIR', project_root / 'data' / 'csv'))
        csv_dir.mkdir(parents=True, exist_ok=True)
        return csv_dir

    # -------------------------------------------------------------------
    # General Data Handling
    # -------------------------------------------------------------------
    def load_data(self, symbol: str, start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Loads data for a given symbol from SQL or CSV based on the mode.
        Supports optional start_date filtering for SQL mode.

        Args:
            symbol (str): The stock symbol to load data for.
            start_date (str, optional): The start date for data retrieval in 'YYYY-MM-DD' format.
                - If not provided, it loads all available data for the symbol.

        Returns:
            Optional[pd.DataFrame]: The DataFrame containing the stock data, or None if no data is found.
        """
        if self.use_csv:
            return self.load_data_from_csv(symbol)
        else:
            return self.load_data_from_sql(symbol, start_date)

    def save_data(self, data: pd.DataFrame, symbol: str, overwrite: bool = False) -> None:
        """
        Saves data for a given symbol to SQL or CSV based on the mode.

        Args:
            data (pd.DataFrame): The DataFrame containing the stock data.
            symbol (str): The stock symbol.
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.
        """
        if self.use_csv:
            self.save_data_to_csv(data, symbol, overwrite=overwrite)
        else:
            self.save_data_to_sql(data, symbol, overwrite=overwrite)

    def update_data(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Update existing stock data with new entries from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing updated stock data.
            symbol (str): Stock symbol for the data.
        """
        self.logger.info(f"Updating data for symbol: {symbol}")
        self.save_data(df, symbol, overwrite=True)
        self.logger.info(f"Data for symbol {symbol} updated successfully.")

    def delete_data(self, symbol: str, date: str) -> None:
        """
        Delete stock data for a given symbol and date.

        Args:
            symbol (str): Stock symbol.
            date (str): Date of the stock data to delete in 'YYYY-MM-DD' format.
        """
        if self.use_csv:
            self.delete_data_from_csv(symbol, date)
        else:
            self.delete_data_from_sql(symbol, date)

    # -------------------------------------------------------------------
    # Section 9.1: SQL Data Handling with Upsert (Insert or Update)
    # -------------------------------------------------------------------
    def save_data_to_sql(self, data: pd.DataFrame, symbol: str, overwrite: bool = False) -> None:
        """
        Saves the provided DataFrame to the SQL database using an upsert operation.

        Args:
            data (pd.DataFrame): The DataFrame containing the stock data.
            symbol (str): The stock symbol.
            overwrite (bool): If True, existing data will be overwritten.
        """
        session = self.db_handler.get_session()
        try:
            self.logger.info(f"Saving data for {symbol} to SQL database using upsert.")

            # Standardize column names to match database schema (case-insensitive)
            column_mapping = {
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            }
            data = data.rename(columns={
                col.lower(): col_name for col, col_name in column_mapping.items() if col.lower() in data.columns
            })

            # Drop 'id' column if present to prevent interference
            if 'id' in data.columns:
                data = data.drop(columns=['id'])
                self.logger.debug("'id' column dropped before saving to SQL.")

            # Add the 'symbol' column to the DataFrame
            data['symbol'] = symbol
            self.logger.debug(f"Data columns: {data.columns}")
            self.logger.debug(f"Data shape for {symbol}: {data.shape}")

            # **Logging Before Conversion**
            self.logger.debug(f"DataFrame 'Date' column dtype before conversion: {data['Date'].dtype}")
            self.logger.debug(f"Sample 'Date' values before conversion:\n{data['Date'].head()}")

            # Check if 'Date' column is already in datetime format
            if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                self.logger.warning(f"'Date' column is not in datetime format. Attempting conversion.")
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.date
            else:
                self.logger.info(f"'Date' column is already in datetime format. Skipping conversion.")

            # **Logging After Conversion**
            self.logger.debug(f"DataFrame 'Date' column dtype after conversion: {data['Date'].dtype}")
            self.logger.debug(f"Sample 'Date' values after conversion:\n{data['Date'].head()}")

            # Drop rows with NaT in 'Date' column
            initial_length = len(data)
            data = data.dropna(subset=['Date'])
            dropped_rows = initial_length - len(data)
            if dropped_rows > 0:
                self.logger.warning(f"Dropped {dropped_rows} rows with invalid or missing dates for {symbol}.")

            # If the DataFrame is empty after dropping NaNs, log and return
            if data.empty:
                self.logger.warning(f"No valid data available to save for {symbol} after cleaning. Skipping save operation.")
                return

            # Prepare data records
            data_records = data.to_dict(orient='records')

            # Log first 5 records for debugging
            self.logger.debug(f"First 5 records to upsert for {symbol}: {data_records[:5]}")

            # Perform upsert operation
            for record in data_records:
                stmt = text("""
                    INSERT INTO stock_data (symbol, "Date", "Open", "High", "Low", "Close", "Volume")
                    VALUES (:symbol, :Date, :Open, :High, :Low, :Close, :Volume)
                    ON CONFLICT (symbol, "Date") DO 
                    {update_clause}
                """.format(
                    update_clause=(
                        'UPDATE SET "Open" = EXCLUDED."Open", "High" = EXCLUDED."High", "Low" = EXCLUDED."Low", '
                        '"Close" = EXCLUDED."Close", "Volume" = EXCLUDED."Volume"' if overwrite else 'NOTHING'
                    )
                ))

                # Execute the upsert statement with the record values
                session.execute(
                    stmt,
                    {
                        "symbol": record.get("symbol"),
                        "Date": record.get("Date"),
                        "Open": record.get("Open"),
                        "High": record.get("High"),
                        "Low": record.get("Low"),
                        "Close": record.get("Close"),
                        "Volume": record.get("Volume"),
                    }
                )

            session.commit()
            self.logger.info(f"Data for {symbol} upserted to SQL database successfully.")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error saving data for {symbol} to SQL database: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def load_data_from_sql(self, symbol: str, start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Loads data for a given symbol from the SQL database.
        Supports optional start_date filtering.

        Args:
            symbol (str): The stock symbol to load data for.
            start_date (str, optional): The start date for data retrieval in 'YYYY-MM-DD' format.
                - If not provided, it loads all available data for the symbol.

        Returns:
            Optional[pd.DataFrame]: The DataFrame containing the stock data, or None if no data is found.
        """
        session = self.db_handler.get_session()
        try:
            self.logger.info(f"Loading data for {symbol} from SQL database.")
            query = session.query(StockData).filter(StockData.symbol == symbol)

            if start_date:
                try:
                    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
                    query = query.filter(StockData.Date >= start_date_obj)
                    self.logger.info(f"Filtering data from {start_date_obj}.")
                except ValueError:
                    self.logger.error(f"Invalid start_date format: {start_date}. Expected 'YYYY-MM-DD'.")
                    return None

            query = query.order_by(StockData.Date.asc())
            df = pd.read_sql_query(query.statement, self.db_handler.engine)

            if df.empty:
                self.logger.warning(f"No data found for symbol: {symbol}")
                return None

            self.logger.info(f"Loaded {len(df)} records for symbol: {symbol}")
            return df
        except SQLAlchemyError as e:
            self.logger.error(f"Error loading data for {symbol} from SQL database: {e}", exc_info=True)
            return None
        finally:
            session.close()

    def delete_data_from_sql(self, symbol: str, date: str) -> None:
        """
        Delete stock data for a given symbol and date from the SQL database.

        Args:
            symbol (str): Stock symbol.
            date (str): Date of the stock data to delete in 'YYYY-MM-DD' format.
        """
        session = self.db_handler.get_session()
        try:
            self.logger.info(f"Deleting data for symbol: {symbol} on date: {date}")
            date_obj = datetime.strptime(date, '%Y-%m-%d').date()
            stock_data = session.query(StockData).filter(
                StockData.symbol == symbol,
                StockData.Date == date_obj
            ).first()
            if stock_data:
                session.delete(stock_data)
                session.commit()
                self.logger.info(f"Deleted data for {symbol} on {date}.")
            else:
                self.logger.warning(f"No data found for {symbol} on {date} to delete.")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error deleting data for symbol {symbol} on {date}: {e}", exc_info=True)
            raise
        finally:
            session.close()

    # -------------------------------------------------------------------
    # Section 9.2: Model Data Handling
    # -------------------------------------------------------------------
    def save_model_data(
        self,
        model_name: str,
        model_type: str,
        training_date: datetime,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        description: Optional[str] = None
    ) -> None:
        """
        Saves a record of the trained model's data to the database using an upsert operation.

        Args:
            model_name (str): Name of the model.
            model_type (str): Type of the model.
            training_date (datetime): Date when the model was trained.
            hyperparameters (Dict[str, Any]): Hyperparameters used for training.
            metrics (Dict[str, Any]): Performance metrics of the model.
            description (Optional[str], optional): Optional description of the model.
        """
        session = self.db_handler.get_session()
        try:
            self.logger.info(f"Saving model data for model: {model_name}")

            # Define the upsert statement using raw SQL.
            upsert_sql = text("""
                INSERT INTO model_data (model_name, model_type, training_date, hyperparameters, metrics, description)
                VALUES (:model_name, :model_type, :training_date, :hyperparameters, :metrics, :description)
                ON CONFLICT (model_name) DO UPDATE SET
                    model_type = EXCLUDED.model_type,
                    training_date = EXCLUDED.training_date,
                    hyperparameters = EXCLUDED.hyperparameters,
                    metrics = EXCLUDED.metrics,
                    description = EXCLUDED.description;
            """)

            # Execute the upsert statement with the provided parameters.
            session.execute(
                upsert_sql,
                {
                    'model_name': model_name,
                    'model_type': model_type,
                    'training_date': training_date.date(),
                    'hyperparameters': json.dumps(hyperparameters),
                    'metrics': json.dumps(metrics),
                    'description': description
                }
            )
            session.commit()
            self.logger.info(f"Model data for {model_name} upserted successfully.")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error saving model data for {model_name}: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def load_model_data(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a record of the model data from the database by model name.

        Args:
            model_name (str): The name of the model to retrieve data for.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing model data or None if not found.
        """
        session = self.db_handler.get_session()
        try:
            self.logger.info(f"Loading model data for model: {model_name}")
            model_data = session.query(ModelData).filter(ModelData.model_name == model_name).one_or_none()
            if model_data:
                self.logger.info(f"Loaded model data for {model_name}.")
                return {
                    'model_name': model_data.model_name,
                    'model_type': model_data.model_type,
                    'training_date': model_data.training_date,
                    'hyperparameters': json.loads(model_data.hyperparameters) if model_data.hyperparameters else {},
                    'metrics': json.loads(model_data.metrics) if model_data.metrics else {},
                    'description': model_data.description
                }
            else:
                self.logger.warning(f"No model data found for {model_name}.")
                return None
        except SQLAlchemyError as e:
            self.logger.error(f"Error loading model data for {model_name}: {e}", exc_info=True)
            return None
        finally:
            session.close()

    # -------------------------------------------------------------------
    # Section 9.3: CSV Data Handling
    # -------------------------------------------------------------------
    def save_data_to_csv(self, data: pd.DataFrame, symbol: str, overwrite: bool = False) -> None:
        """
        Saves the provided DataFrame as a CSV file.

        Args:
            data (pd.DataFrame): The DataFrame containing the stock data.
            symbol (str): The stock symbol.
            overwrite (bool, optional): Whether to overwrite existing CSV. Defaults to False.
        """
        try:
            file_path = self.csv_dir / f"{symbol}.csv"
            if not overwrite and file_path.exists():
                self.logger.warning(f"CSV file for {symbol} already exists and overwrite is set to False.")
                return

            data.to_csv(file_path, index=False)
            self.logger.info(f"Data for {symbol} saved to CSV at {file_path}.")
        except Exception as e:
            self.logger.error(f"Error saving data to CSV for {symbol}: {e}", exc_info=True)

    def load_data_from_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Loads data for a given symbol from a CSV file.

        Args:
            symbol (str): The stock symbol to load data for.

        Returns:
            Optional[pd.DataFrame]: The DataFrame containing the stock data, or None if no data is found.
        """
        try:
            file_path = self.csv_dir / f"{symbol}.csv"
            if not file_path.exists():
                self.logger.warning(f"CSV file for {symbol} not found at {file_path}.")
                return None

            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.date
            self.logger.info(f"Loaded {len(data)} rows of data for {symbol} from CSV.")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data from CSV for {symbol}: {e}", exc_info=True)
            return None

    def delete_data_from_csv(self, symbol: str, date: str) -> None:
        """
        Delete stock data for a given symbol and date from the CSV file.

        Args:
            symbol (str): Stock symbol.
            date (str): Date of the stock data to delete in 'YYYY-MM-DD' format.
        """
        try:
            file_path = self.csv_dir / f"{symbol}.csv"
            if not file_path.exists():
                self.logger.warning(f"CSV file for {symbol} not found at {file_path}.")
                return

            data = pd.read_csv(file_path)
            initial_length = len(data)
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.date
            date_to_delete = datetime.strptime(date, '%Y-%m-%d').date()
            data = data[data['Date'] != date_to_delete]

            if len(data) < initial_length:
                data.to_csv(file_path, index=False)
                self.logger.info(f"Deleted data for {symbol} on {date} from CSV.")
            else:
                self.logger.warning(f"No data found for {symbol} on {date} in CSV.")
        except Exception as e:
            self.logger.error(f"Error deleting data from CSV for {symbol} on {date}: {e}", exc_info=True)

    def query_database(self, query, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executes a database query and returns the results as a list of dictionaries."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, params or {})
                records = [dict(row) for row in result]
                self.logger.info(f"Query executed successfully: {query}")
                return records
        except Exception as e:
            self.logger.error(f"Failed to execute query: {query}. Error: {e}")
            return []
        
# -------------------------------------------------------------------
# Section 10: Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    from Scripts.Utilities.config_handling.config_manager import ConfigManager

    # Initialize ConfigManager (replace with actual config as needed)
    config_manager = ConfigManager()

    # Initialize DataStore for SQL mode (ensure .env has correct PostgreSQL details)
    data_store = DataStore(
        config_manager=config_manager,
        logger=logger,
        use_csv=False
    )

    # Example: Save stock data for AAPL
    try:
        stock_data = pd.DataFrame({
            'Date': [datetime(2024, 10, 20), datetime(2024, 10, 21)],
            'Open': [150.0, 152.5],
            'High': [155.0, 157.0],
            'Low': [149.0, 151.5],
            'Close': [153.0, 156.0],
            'Volume': [1000000, 1100000]
        })

        data_store.save_data(data=stock_data, symbol='AAPL', overwrite=False)
        print("Stock data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save stock data: {e}", exc_info=True)

    # Example: Load stock data for AAPL
    try:
        loaded_data = data_store.load_data(symbol='AAPL')  # Without start_date for backward compatibility
        if loaded_data is not None:
            print("Loaded Data:")
            print(loaded_data.head())
        else:
            print("No data loaded for AAPL.")
    except Exception as e:
        logger.error(f"Failed to load stock data: {e}", exc_info=True)

    # Example: Update stock data for AAPL
    try:
        update_data = pd.DataFrame({
            'Date': [datetime(2024, 10, 22)],
            'Open': [154.0],
            'High': [158.0],
            'Low': [152.0],
            'Close': [157.5],
            'Volume': [1200000]
        })
        data_store.update_data(df=update_data, symbol='AAPL')
        print("Stock data updated successfully.")
    except Exception as e:
        logger.error(f"Failed to update stock data: {e}", exc_info=True)

    # Example: Delete stock data for AAPL on a specific date
    try:
        data_store.delete_data(symbol='AAPL', date='2024-10-20')
        print("Stock data for AAPL on 2024-10-20 deleted successfully.")
    except Exception as e:
        logger.error(f"Failed to delete stock data: {e}", exc_info=True)

    # Example: Save model data for a trained model
    try:
        model_name = 'TRP'
        model_type = 'GradientBoostingRegressor'
        training_date = datetime.now()
        hyperparameters = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
        metrics = {'accuracy': 0.95, 'loss': 0.05}
        description = 'Proprietary Gradient Boosting model for predicting stock prices.'

        data_store.save_model_data(
            model_name=model_name,
            model_type=model_type,
            training_date=training_date,
            hyperparameters=hyperparameters,
            metrics=metrics,
            description=description
        )
        print("Model data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model data: {e}", exc_info=True)

# -------------------------------------------------------------------
# Section 11: Future Improvements
# -------------------------------------------------------------------
# - Add comprehensive error handling for all database operations.
# - Implement bulk operations for saving and updating large datasets.
# - Enhance logging to include more detailed information and timestamps.
# - Integrate asynchronous database operations for better performance.
# - Develop unit tests to ensure the reliability of each method.
# - Expand the interface to support additional databases like MySQL or SQLite.
# - Incorporate caching mechanisms to reduce database load for frequently accessed data.
# - Implement data validation and sanitization to ensure data integrity.
# - Add support for transaction management to handle complex operations.
# - Consider refactoring code to further modularize and improve maintainability.

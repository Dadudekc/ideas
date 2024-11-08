# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/Scripts/Utilities/data_store_interface.py
# Description:
#     Defines the DataStoreInterface abstract base class, which serves
#     as a blueprint for all data storage implementations within the
#     TradingRobotPlug project. This interface enforces the implementation
#     of essential methods for loading and saving data.
# -------------------------------------------------------------------

from abc import ABC, abstractmethod
import pandas as pd

class DataStoreInterface(ABC):
    @abstractmethod
    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load data for a given symbol.

        Args:
            symbol (str): The stock symbol to load data for.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """
        pass

    @abstractmethod
    def save_data(self, df: pd.DataFrame, symbol: str, overwrite: bool = False) -> None:
        """
        Save data for a given symbol.

        Args:
            df (pd.DataFrame): The DataFrame containing data to save.
            symbol (str): The stock symbol for which data is being saved.
            overwrite (bool): If True, existing data will be overwritten.
        """
        pass

# -------------------------------------------------------------------
# Future Improvements:
# - Add methods for updating and deleting data entries.
# - Incorporate asynchronous method definitions for non-blocking operations.
# - Extend the interface to support batch operations for improved performance.
# - Implement error handling protocols within abstract methods.
# -------------------------------------------------------------------

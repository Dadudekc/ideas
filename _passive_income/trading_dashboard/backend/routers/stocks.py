# backend/routers/stocks.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

import pandas as pd

from utils.dependencies import get_current_user, get_db
from data_store import DataStore
from models.user import User

router = APIRouter()

# Initialize DataStore (adjust parameters as needed)
data_store = DataStore(
    config_manager=None,  # Assuming ConfigManager is handled inside data_store.py
    logger=None,          # Replace with actual logger if needed
    use_csv=False         # Using SQL database
)

@router.get("/{symbol}", response_model=dict)
def get_stock_data(symbol: str, start_date: Optional[str] = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Retrieve stock data for a given symbol, optionally filtering by start_date.
    """
    try:
        df = data_store.load_data(symbol=symbol, start_date=start_date)
        if df is None:
            raise HTTPException(status_code=404, detail="Stock data not found")
        # Convert DataFrame to dictionary
        data = df.to_dict(orient="records")
        return {"symbol": symbol, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

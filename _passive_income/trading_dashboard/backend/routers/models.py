# backend/routers/models.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any

from utils.dependencies import get_current_user, get_db
from data_store import DataStore
from models.user import User

router = APIRouter()

# Initialize DataStore
data_store = DataStore(
    config_manager=None,
    logger=None,
    use_csv=False
)

@router.post("/predict", response_model=dict)
def predict_stock(symbol: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Make a prediction for a given stock symbol using saved models.
    """
    try:
        # Load stock data
        df = data_store.load_data(symbol=symbol)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available for prediction")

        # Load the model (assuming a model named 'TRP' exists)
        model_data = data_store.load_model_data(model_name='TRP')
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")

        # Deserialize the model (assuming it's saved using pickle)
        model = pickle.loads(model_data['metrics'].get('model_pickle'))

        # Prepare data for prediction (this will depend on your model's requirements)
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Example features
        predictions = model.predict(features)

        # Return predictions
        return {"symbol": symbol, "predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

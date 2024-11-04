import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yfinance as yf  # For data fetching, replaceable with other APIs
import ta  # For technical indicators

class DataFetcher:
    """Fetches and preprocesses historical stock data."""
    
    def fetch_data(self, symbol, start="2020-01-01", end="2023-12-31"):
        """Fetch historical stock data from Yahoo Finance."""
        data = yf.download(symbol, start=start, end=end)
        data = self.add_technical_indicators(data)
        return data

    def add_technical_indicators(self, data):
        """Add technical indicators like RSI, MACD, Bollinger Bands, VWAP, and support/resistance levels."""
        # Add RSI, MACD, and Bollinger Bands
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['Bollinger_High'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
        data['Bollinger_Low'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
        
        # Calculate VWAP
        data['VWAP'] = self.calculate_vwap(data)
        
        # Calculate support and resistance levels
        data['Support'], data['Resistance'] = self.calculate_support_resistance(data)

        data.dropna(inplace=True)  # Drop rows with NaN values
        return data

    def calculate_vwap(self, data):
        """Calculate VWAP (Volume Weighted Average Price) based on high, low, close prices, and volume."""
        price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap

    def calculate_support_resistance(self, data, window=14):
        """Calculate rolling support and resistance levels based on rolling minimum/maximum."""
        support = data['Low'].rolling(window=window).min()
        resistance = data['High'].rolling(window=window).max()
        return support, resistance


class TradingModel:
    """Handles training, saving, loading, and predicting with the trading model."""
    
    def __init__(self, model_path="trading_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def train_model(self, data, target_column="Profitable"):
        """Train a model with historical data and save it."""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Encoding and scaling
        y = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        
        # Save model
        self.save_model()
        
    def save_model(self):
        """Save the model, scaler, and label encoder to disk."""
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder
            }, f)

    def load_model(self):
        """Load the model, scaler, and label encoder from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.label_encoder = data["label_encoder"]
        else:
            raise FileNotFoundError("Model file not found. Train and save the model first.")

    def predict(self, features):
        """Make a prediction based on input features."""
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)
        return self.label_encoder.inverse_transform(prediction)[0]


class FeedbackManager:
    """Manages user feedback on predictions for model improvement."""
    
    def __init__(self, feedback_path="feedback.json"):
        self.feedback_path = feedback_path

    def save_feedback(self, prediction, actual):
        """Save feedback data for future model improvement."""
        feedback_data = {"prediction": prediction, "actual": actual}
        with open(self.feedback_path, "a") as f:
            json.dump(feedback_data, f)
            f.write("\n")


# Utility Function for Standalone Calculation of Technical Indicators
def calculate_technical_indicators(data):
    """Calculate various technical indicators and return a DataFrame."""
    indicators = pd.DataFrame(index=data.index)
    indicators['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    indicators['MACD'] = ta.trend.MACD(data['Close']).macd()
    indicators['Bollinger_High'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
    indicators['Bollinger_Low'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
    indicators['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3 * data['Volume'].cumsum() / data['Volume'].cumsum()
    indicators['Support'] = data['Low'].rolling(window=14).min()
    indicators['Resistance'] = data['High'].rolling(window=14).max()
    indicators.dropna(inplace=True)
    return indicators

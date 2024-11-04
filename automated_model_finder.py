import os
import time
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Mock DataStore class
class DataStore:
    """Simple mock of a data store that loads data for a given symbol and date range."""
    def __init__(self, db_uri, logger=None):
        self.db_uri = db_uri
        self.logger = logger

    def load_data(self, symbol, start_date):
        # This would normally pull from a database; here we mock with sample data
        data = pd.DataFrame({
            'date': pd.date_range(start_date, periods=100),
            'open': np.random.random(100) * 100,
            'high': np.random.random(100) * 100,
            'low': np.random.random(100) * 100,
            'close': np.random.random(100) * 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        return data

# Mock ModelManager class
class ModelManager:
    """Simple mock of a model manager to save trained models."""
    def __init__(self, logger=None):
        self.logger = logger

    def save_lstm_model(self, model, symbol, best_params, metrics=None, target_scaler=None):
        model.save(f"{symbol}_trained_model.h5")
        if self.logger:
            self.logger.info(f"Model saved for symbol: {symbol}")

# Utility function for preprocessing data
def preprocess_data_for_lstm(data, target_column='close', time_steps=10, features=None, test_size=0.2):
    # Select the target column
    y = data[target_column].values
    X = data[features].values

    # Scale features
    feature_scaler = MinMaxScaler()
    X = feature_scaler.fit_transform(X)

    # Scale target
    target_scaler = MinMaxScaler()
    y = target_scaler.fit_transform(y.reshape(-1, 1))

    # Reshape for LSTM input
    X_lstm = []
    y_lstm = []
    for i in range(time_steps, len(X)):
        X_lstm.append(X[i-time_steps:i])
        y_lstm.append(y[i])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    # Train-test split
    split_idx = int(len(X_lstm) * (1 - test_size))
    return X_lstm[:split_idx], X_lstm[split_idx:], y_lstm[:split_idx], y_lstm[split_idx:], feature_scaler, target_scaler

# LSTM Model creation function
def create_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Main AutomatedModelFinder class
class AutomatedModelFinder:
    def __init__(self, db_uri, logger, config, cache_dir=None):
        self.db_uri = db_uri
        self.logger = logger
        self.config = config
        self.data_store = DataStore(self.db_uri, logger=self.logger)
        self.model_manager = ModelManager(logger)
        self.cache_dir = Path(cache_dir or "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.target_column = self.config.get('target_column', 'close')
        self.time_steps = self.config.get('time_steps', 10)

        # Hardware Configuration
        self._check_hardware_and_configure()

    def _check_hardware_and_configure(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            self.logger.info(f"{len(gpus)} GPU(s) found. Enabling GPU.")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.backend.clear_session()
        else:
            self.logger.info("No GPU found. Using CPU.")

    def run(self, symbols, start_date, end_date):
        try:
            for symbol in symbols:
                self.logger.info(f"Processing symbol: {symbol}")

                # Load data and add technical indicators
                data = self.data_store.load_data(symbol, start_date)
                if data is None or data.empty:
                    self.logger.error(f"No data available for {symbol}. Skipping.")
                    continue

                # Preprocess and cache data
                cache_file = self.cache_dir / f'{symbol}_preprocessed_data.pkl'
                X_train, X_val, y_train, y_val, feature_scaler, target_scaler = self._load_or_preprocess_data(
                    symbol, data, cache_file
                )

                # Model training and evaluation
                input_shape = (X_train.shape[1], X_train.shape[2])
                model, best_params = self._load_or_tune_model(symbol, input_shape, X_train, y_train, X_val, y_val)
                train_dataset, val_dataset = self._prepare_datasets(X_train, y_train, X_val, y_val, best_params['batch_size'])
                self._train_model(model, train_dataset, val_dataset, symbol, best_params)
                self._evaluate_and_save_model(model, symbol, X_val, y_val, target_scaler, best_params)

        except Exception as e:
            self.logger.error(f"An error occurred: {e}")

    def _load_or_preprocess_data(self, symbol, data, cache_file):
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                X_train, X_val, y_train, y_val, feature_scaler, target_scaler = pickle.load(f)
                self.logger.info(f"Loaded cached preprocessed data for {symbol}.")
        else:
            features = [col for col in data.columns if col != self.target_column]
            X_train, X_val, y_train, y_val, feature_scaler, target_scaler = preprocess_data_for_lstm(
                data,
                target_column=self.target_column,
                time_steps=self.time_steps,
                features=features
            )
            with open(cache_file, 'wb') as f_out:
                pickle.dump((X_train, X_val, y_train, y_val, feature_scaler, target_scaler), f_out)
            self.logger.info(f"Cached preprocessed data for {symbol}.")
        return X_train, X_val, y_train, y_val, feature_scaler, target_scaler

    def _load_or_tune_model(self, symbol, input_shape, X_train, y_train, X_val, y_val):
        model = create_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001)
        best_params = {'batch_size': 64, 'epochs': 10}
        return model, best_params

    def _prepare_datasets(self, X_train, y_train, X_val, y_val, batch_size):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_dataset, val_dataset

    def _train_model(self, model, train_dataset, val_dataset, symbol, best_params):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5)
        
        self.logger.info(f"Starting training for {symbol}. Epochs: {best_params['epochs']}, Batch Size: {best_params['batch_size']}")
        model.fit(
            train_dataset,
            epochs=best_params['epochs'],
            validation_data=val_dataset,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

    def _evaluate_and_save_model(self, model, symbol, X_val, y_val, target_scaler, best_params):
        y_pred = model.predict(X_val)
        y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))

        if self.model_manager:
            self.model_manager.save_lstm_model(model, symbol, best_params)


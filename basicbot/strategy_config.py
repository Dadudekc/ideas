# strategy_config.py

from pydantic import BaseModel, Field

class StrategyConfig(BaseModel):
    ema_length: int = Field(default=8, ge=1, description="Length of EMA calculation.")
    rsi_length: int = Field(default=14, ge=1, description="Length of RSI calculation.")
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0, description="RSI overbought threshold.")
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0, description="RSI oversold threshold.")
    macd_fast_window: int = Field(default=12, ge=1, description="Fast window for MACD calculation.")
    macd_slow_window: int = Field(default=26, ge=1, description="Slow window for MACD calculation.")
    macd_signal_window: int = Field(default=9, ge=1, description="Signal window for MACD calculation.")
    adx_window: int = Field(default=14, ge=1, description="ADX window for ADX calculation.")
    vwap_window: int = Field(default=14, ge=1, description="VWAP window for VWAP calculation.")
    bollinger_window: int = Field(default=20, ge=1, description="Window for Bollinger Bands calculation.")
    atr_window: int = Field(default=14, ge=1, description="ATR window for ATR calculation.")
    risk_percent: float = Field(default=0.5, ge=0.1, le=100.0, description="Risk percent per trade.")
    profit_target: float = Field(default=15.0, ge=1.0, le=100.0, description="Profit target percent.")
    stop_loss_multiplier: float = Field(default=2.0, ge=1.0, description="Stop loss multiplier based on ATR.")

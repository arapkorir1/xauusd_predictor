import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class FeaturePipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates stationary financial features from raw price history."""
        logger.info("Computing stationary financial features...")
        
        # Ensure data is sorted sequentially by date
        df = df.sort_values("date").reset_index(drop=True)
        
        # 1. Log Returns: Stationary representation of price changes
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        
        # 2. Rolling Volatility: Standard deviation of log returns (5 and 21 day windows)
        df["volatility_5d"] = df["log_return"].rolling(window=5).std()
        df["volatility_21d"] = df["log_return"].rolling(window=21).std()
        
        # 3. Average True Range (ATR): Standard quantitative volatility indicator
        # ATR measures the typical trading range (high to low) adjusted for overnight gaps.
        high_low = df["high"] - df["low"]
        high_close_prev = (df["high"] - df["close"].shift(1)).abs()
        low_close_prev = (df["low"] - df["close"].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df["atr_14d"] = true_range.rolling(window=14).mean()
        # Scale ATR by close price to keep it stationary (percentage-based ATR)
        df["atr_ratio"] = df["atr_14d"] / df["close"]

        # 4. Relative Strength Index (RSI - 14 Days): Momentum oscillator
        change = df["close"].diff()
        gain = change.mask(change < 0, 0)
        loss = -change.mask(change > 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / (avg_loss + 1e-9) # Avoid division by zero
        df["rsi_14d"] = 100 - (100 / (1 + rs))

        # 5. Volume Trend: Volume relative to its 20-day moving average
        df["volume_ratio"] = df["volume"] / (df["volume"].rolling(window=20).mean() + 1e-9)
        
        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates the binary target variable for volatility regimes.
        
        We define 'High Volatility' (1) as when the next day's 5-day rolling volatility
        is above the 60th percentile of historical volatility. Otherwise normal (0).
        This makes it a predictive binary classification task.
        """
        logger.info("Generating predictive labels (volatility regimes)...")
        
        # Calculate the historical 60th percentile threshold of 5-day volatility
        threshold = df["volatility_5d"].quantile(0.60)
        
        # We want to predict *tomorrow's* volatility regime using *today's* indicators
        # Shift volatility back by 1 day to represent the target label
        df["target_volatility"] = df["volatility_5d"].shift(-1)
        
        # Label as 1 if high volatility, 0 if normal
        df["target"] = (df["target_volatility"] > threshold).astype(int)
        
        # Drop rows where we can't look forward (the very last row of the dataset)
        # Also drop early rows where rolling window indicators are still NaN
        df = df.dropna().reset_index(drop=True)
        
        return df

    def run_pipeline(self) -> None:
        """Loads raw data, processes indicators, builds targets, and saves the output."""
        raw_path = self.config["data"]["raw_data_path"]
        processed_path = self.config["data"]["processed_data_path"]
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data not found at {raw_path}. Run ingestion first!")
            
        # Load raw data
        df = pd.read_parquet(raw_path)
        
        # Apply pipeline transformations
        df = self.compute_technical_indicators(df)
        df = self.create_labels(df)
        
        # Ensure output folder exists and write to Parquet
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_parquet(processed_path, index=False)
        
        logger.info(f"Feature engineering pipeline run complete. Processed shape: {df.shape}")
        logger.info(f"Successfully cached features to {processed_path}")

if __name__ == "__main__":
    try:
        pipeline = FeaturePipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.critical(f"Feature pipeline run failed: {e}")
        sys.exit(1)

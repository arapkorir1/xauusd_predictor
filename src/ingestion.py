import os
import sys
import logging
import yaml
import yfinance as yf
import pandas as pd

# Setup robust logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class DataIngestor:
    """Handles parsing configurations, fetching historical market data, and local caching."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Loads and parses the YAML configuration file."""
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found at {self.config_path}")
            raise FileNotFoundError(f"Missing config at {self.config_path}")
        
        with open(self.config_path, "r") as f:
            try:
                config = yaml.safe_load(f)
                logger.info("Configuration file loaded successfully.")
                return config
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML config: {e}")
                raise e

    def fetch_data(self) -> pd.DataFrame:
        """Downloads historical data based on the config parameters."""
        data_cfg = self.config["data"]
        ticker = data_cfg["ticker"]
        start_date = data_cfg["start_date"]
        end_date = data_cfg["end_date"]
        
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date} via Yahoo Finance...")
        
        try:
            # Fetch data with auto-adjust to correct stock splits/dividends (essential for accurate backtesting)
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                raise ValueError(f"Downloaded DataFrame is empty. Check ticker '{ticker}' or date range.")
            
            # Reset index to turn 'Date' into a normal column
            df = df.reset_index()
            
            # Normalize column names to lowercase for consistency
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
            
            logger.info(f"Successfully fetched {len(df)} rows of raw data.")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise e

    def save_data(self, df: pd.DataFrame) -> None:
        """Saves the DataFrame as a Parquet file, creating destination folders if missing."""
        raw_path = self.config["data"]["raw_data_path"]
        
        # Ensure target directories exist
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        
        try:
            logger.info(f"Writing raw data to production storage at {raw_path}...")
            df.to_parquet(raw_path, index=False)
            logger.info("Data storage operation completed successfully.")
        except Exception as e:
            logger.error(f"Failed to write Parquet file to {raw_path}: {e}")
            raise e

if __name__ == "__main__":
    # This block allows running the ingestion pipeline directly
    try:
        ingestor = DataIngestor()
        raw_df = ingestor.fetch_data()
        ingestor.save_data(raw_df)
    except Exception as e:
        logger.critical(f"Data ingestion step failed: {e}")
        sys.exit(1)

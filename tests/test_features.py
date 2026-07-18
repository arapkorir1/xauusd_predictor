import pytest
import pandas as pd
import numpy as np
from src.features import FeaturePipeline

def test_compute_technical_indicators():
    """Verifies that the technical indicator calculations build the required mathematical features."""
    # Create a simple synthetic 30-day market dataset to test our rolling windows safely
    dates = pd.date_range(start="2026-01-01", periods=30)
    mock_data = pd.DataFrame({
        "date": dates,
        "open": np.linspace(100, 115, 30),
        "high": np.linspace(102, 117, 30),
        "low": np.linspace(99, 114, 30),
        "close": np.linspace(101, 116, 30),
        "volume": np.random.randint(1000, 5000, 30)
    })
    
    # Initialize our pipeline using our existing config structure
    pipeline = FeaturePipeline(config_path="config/config.yaml")
    
    # Run the feature calculations
    processed_df = pipeline.compute_technical_indicators(mock_data)
    
    # Assertions: Verify expected production columns exist in output
    expected_features = ["log_return", "volatility_5d", "volatility_21d", "atr_ratio", "rsi_14d", "volume_ratio"]
    for feature in expected_features:
        assert feature in processed_df.columns, f"Feature column '{feature}' was missing from output."
        
    # Verify the chronological sequence was preserved
    assert processed_df["date"].is_monotonic_increasing, "The output dataset dates are not sorted chronologically."

import os
import sys
import logging
import joblib
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="XAU/USD Volatility Regime Predictor",
    description="A production-grade API predicting gold market volatility regimes (High vs. Normal Volatility).",
    version="1.0.0"
)

# Load configuration to know where the model lives and which features it expects
CONFIG_PATH = "config/config.yaml"
if not os.path.exists(CONFIG_PATH):
    logger.error(f"Config not found at {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["model_save_path"]
FEATURES = config["model"]["features"]

# Global variable to hold our loaded model
model = None

@app.on_event("startup")
def load_model():
    """Loads the serialized model artifact into memory on server startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}. Did you run training first?")
        # We don't crash the server immediately, but we log the issue
        model = None
    else:
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Machine learning model loaded successfully into memory.")
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
            model = None

# Define the Pydantic schema for input data validation
class PredictionRequest(BaseModel):
    log_return: float = Field(..., description="Today's logarithmic price return.", examples=[0.0012])
    volatility_5d: float = Field(..., description="5-day rolling standard deviation of returns.", examples=[0.0085])
    volatility_21d: float = Field(..., description="21-day rolling standard deviation of returns.", examples=[0.0102])
    atr_ratio: float = Field(..., description="ATR (14-day) scaled by the current close price.", examples=[0.0078])
    rsi_14d: float = Field(..., description="14-day Relative Strength Index (0-100).", ge=0.0, le=100.0, examples=[55.4])
    volume_ratio: float = Field(..., description="Current volume relative to 20-day moving average.", examples=[1.25])

# Define the Pydantic schema for response structure
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted volatility regime (1 = High Volatility, 0 = Normal Volatility)")
    probability_high_volatility: float = Field(..., description="Model confidence score for High Volatility regime.")
    status: str = Field(..., description="Request execution status message.")

@app.get("/")
def read_root():
    """Health check endpoint to ensure API service is live."""
    return {"status": "healthy", "service": "XAU/USD Volatility Regime Predictor"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Processes features, validates inputs, runs predictions, and returns target labels."""
    global model
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Prediction service unavailable. Model artifact is missing or failed to load."
        )
    
    try:
        # Convert incoming JSON payload to a pandas DataFrame representing a single row
        input_data = pd.DataFrame([{
            "log_return": request.log_return,
            "volatility_5d": request.volatility_5d,
            "volatility_21d": request.volatility_21d,
            "atr_ratio": request.atr_ratio,
            "rsi_14d": request.rsi_14d,
            "volume_ratio": request.volume_ratio
        }])
        
        # Ensure features are in the exact sequence expected by the trained XGBoost model
        input_data = input_data[FEATURES]
        
        # Execute model prediction
        pred_class = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        prob_high = float(probabilities[1]) # Class 1 probability
        
        return PredictionResponse(
            prediction=pred_class,
            probability_high_volatility=prob_high,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing prediction request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during inference step.")

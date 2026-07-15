import os
import sys
import logging
import yaml
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def prepare_data(self) -> tuple:
        """Loads features and performs a chronological (non-random) split to prevent leakage."""
        processed_path = self.config["data"]["processed_data_path"]
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data missing at {processed_path}. Run features pipeline first!")

        df = pd.read_parquet(processed_path)
        
        # Sort chronologically by date to be 100% sure
        df = df.sort_values("date").reset_index(drop=True)
        
        # Select features and target from config
        features = self.config["model"]["features"]
        target = self.config["model"]["target"]
        
        X = df[features]
        y = df[target]
        
        # Chronological Split (e.g., last 20% of data is used for testing)
        test_size = self.config["model"]["test_size"]
        split_idx = int(len(df) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Temporal Split executed. Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
        logger.info(f"Training features list: {features}")
        
        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """Trains the XGBoost classifier model."""
        logger.info("Initializing and training XGBoost Classifier...")
        
        # Pull parameters dynamically from YAML config
        hp = self.config["model"]["hyperparameters"]
        random_state = self.config["model"]["random_state"]
        
        model = XGBClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            subsample=hp["subsample"],
            eval_metric=hp["eval_metric"],
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        return model

    def evaluate(self, model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluates the model on unseen future test data using core classification metrics."""
        logger.info("Evaluating model on test dataset...")
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "f1_score": float(f1_score(y_test, predictions)),
            "auc_roc": float(roc_auc_score(y_test, probabilities))
        }
        
        # Display nicely formatted logs
        for metric, val in metrics.items():
            logger.info(f"Test Set {metric.upper()}: {val:.4f}")
            
        return metrics

    def save_artifacts(self, model: XGBClassifier, metrics: dict) -> None:
        """Saves the trained model and metrics registry files to the project path."""
        model_path = self.config["model"]["model_save_path"]
        metrics_path = self.config["model"]["metrics_save_path"]
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model using joblib
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics as structural JSON
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Model performance metrics saved to {metrics_path}")

    def run(self) -> None:
        """Executes the complete training, evaluation, and saving workflow."""
        X_train, X_test, y_train, y_test = self.prepare_data()
        model = self.train(X_train, y_train)
        metrics = self.evaluate(model, X_test, y_test)
        self.save_artifacts(model, metrics)

if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        trainer.run()
    except Exception as e:
        logger.critical(f"Model training process failed: {e}")
        sys.exit(1)

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def load_models(models_dir="models"):
    """Load all .pkl models from the models/ directory."""
    models = {}
    model_files = ["LinearRegression.pkl", "Lasso.pkl", "Ridge.pkl", "RandomForest.pkl"]
    
    try:
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
            with open(model_path, "rb") as f:
                model_name = model_file.replace(".pkl", "")
                models[model_name] = pickle.load(f)
            print(f"Loaded {model_name} from {model_path}")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    
    return models

def generate_test_data(n_samples=100, n_features=5, scaler=None):
    """Generate synthetic test data with 5 features, optionally scaled."""
    # Generate synthetic regression data
    X, _ = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    
    # Create DataFrame with feature names matching processed_data.csv
    feature_names = [f"feature_{i}" for i in range(n_features)]  # Adjust as needed
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Scale features if scaler is provided
    if scaler:
        X_scaled = scaler.transform(X_df)
        X_df = pd.DataFrame(X_scaled, columns=feature_names)
        print("Test data scaled using provided scaler")
    
    return X_df

def make_predictions(models, X_test, output_file="predictions.csv"):
    """Make predictions with all models and save to CSV."""
    predictions = {}
    
    try:
        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            print(f"Generated predictions for {name}")
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame(predictions)
        
        # Save to CSV
        pred_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise
    
    return pred_df

def main():
    # Load scaler from training pipeline (assume it's saved or retrievable)
    # For simplicity, we'll fit a new scaler to match training data structure
    try:
        # Load processed_data.csv to get feature names and fit scaler
        if not os.path.isfile("processed_data/pro_data.csv"):
            raise FileNotFoundError("processed_data.csv not found. Run data_preprocessing.py first.")
        processed_df = pd.read_csv("processed_data/pro_data.csv")
        feature_names = processed_df.drop(columns=['target']).columns
        scaler = StandardScaler()
        scaler.fit(processed_df[feature_names])  # Fit to match training scale
        print("Scaler fitted using processed_data.csv")
    except Exception as e:
        print(f"Error setting up scaler: {e}")
        raise
    
    # Load models
    models = load_models()
    
    # Generate test data
    X_test = generate_test_data(n_samples=100, n_features=len(feature_names), scaler=scaler)
    
    # Make predictions
    make_predictions(models, X_test)

if __name__ == "__main__":
    main()
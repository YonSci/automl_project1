import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_and_evaluate(input_file="processed_data/pro_data.csv"):
    # Ensure directories exist
    try:
        os.makedirs("plots", exist_ok=True)
        print("Checked/created plots/ directory")
        os.makedirs("models", exist_ok=True)
        print("Checked/created models/ directory")
        # Verify models directory exists
        if os.path.isdir("models"):
            print("Confirmed models/ directory exists")
        else:
            raise FileNotFoundError("Failed to create models/ directory")
    except Exception as e:
        print(f"Error creating directories: {e}")
        raise
    
    # Load processed data
    try:
        df = pd.read_csv(input_file)
        X = df.drop(columns=['target'])
        y = df['target']
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Ensure data_preprocessing.py has run.")
        raise
    except KeyError:
        print("Error: 'target' column missing in data. Check data_preprocessing.py output.")
        raise
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Store metrics
    metrics = {}
    
    # Train, evaluate, and save each model
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics[name] = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        }
        
        # Save model to .pkl file
        model_filename = os.path.join("models", f"{name}.pkl")
        try:
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {model_filename}")
        except Exception as e:
            print(f"Error saving {name} to {model_filename}: {e}")
            raise
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{name}: Actual vs Predicted")
        plt.savefig(os.path.join("plots", f"{name}_actual_vs_predicted.png"))
        plt.close()
        
        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title(f"{name}: Residual Plot")
        plt.savefig(os.path.join("plots", f"{name}_residual_plot.png"))
        plt.close()
    
    # Save metrics to JSON
    try:
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("Metrics saved to metrics.json")
    except Exception as e:
        print(f"Error saving metrics.json: {e}")
        raise
    
    print("Plots saved to plots/ directory")

if __name__ == "__main__":
    train_and_evaluate()
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def train_and_evaluate(input_file="processed_data/pro_data.csv"):
    # Create directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Load processed data
    df = pd.read_csv(input_file)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False,
    )
    
    # Define models
    models = {
        "LinearRegression": LinearRegression(fit_intercept=True),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Store metrics
    metrics = {}
    
    # Train and evaluate each model
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
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{name}: Actual vs Predicted")
        plt.savefig(f"plots/{name}_actual_vs_predicted.png")
        plt.close()
        
        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title(f"{name}: Residual Plot")
        plt.savefig(f"plots/{name}_residual_plot.png")
        plt.close()
    
    # Save metrics to JSON
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")
    
    print("Plots saved to plots/ directory")

if __name__ == "__main__":
    train_and_evaluate()
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    X, _ = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    if scaler:
        X_scaled = scaler.transform(X_df)
        X_df = pd.DataFrame(X_scaled, columns=feature_names)
        print("Test data scaled using provided scaler")
    
    return X_df

def make_predictions(models, X_test, output_file="predictions.csv"):
    """Make predictions, save to CSV, and plot them."""
    predictions = {}
    
    try:
        # Create plots directory
        os.makedirs("plots", exist_ok=True)
        print("Checked/created plots/ directory")
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            print(f"Generated predictions for {name}")
            
            # Individual plot for this model
            plt.figure(figsize=(8, 6))
            plt.plot(y_pred, label=f"{name} Predictions", alpha=0.7)
            plt.xlabel("Sample Index")
            plt.ylabel("Predicted Value")
            plt.title(f"Predictions: {name}")
            plt.legend()
            plt.grid(True)
            individual_plot_file = os.path.join("plots", f"predictions_{name}.png")
            plt.savefig(individual_plot_file)
            plt.close()
            print(f"Saved individual plot to {individual_plot_file}")
        
        # Combined plot for all models
        plt.figure(figsize=(10, 6))
        for name, y_pred in predictions.items():
            plt.plot(y_pred, label=name, alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Predicted Value")
        plt.title("Predictions: All Models")
        plt.legend()
        plt.grid(True)
        combined_plot_file = os.path.join("plots", "predictions_all_models.png")
        plt.savefig(combined_plot_file)
        plt.close()
        print(f"Saved combined plot to {combined_plot_file}")
        
        # Save predictions to CSV
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
    except Exception as e:
        print(f"Error making predictions or plotting: {e}")
        raise
    
    return pred_df

def main():
    try:
        if not os.path.isfile("processed_data/pro_data.csv"):
            raise FileNotFoundError("processed_data.csv not found. Run data_preprocessing.py first.")
        processed_df = pd.read_csv("processed_data/pro_data.csv")
        feature_names = processed_df.drop(columns=['target']).columns
        scaler = StandardScaler()
        scaler.fit(processed_df[feature_names])
        print("Scaler fitted using processed_data.csv")
    except Exception as e:
        print(f"Error setting up scaler: {e}")
        raise
    
    models = load_models()
    X_test = generate_test_data(n_samples=100, n_features=len(feature_names), scaler=scaler)
    make_predictions(models, X_test)

if __name__ == "__main__":
    main()
    

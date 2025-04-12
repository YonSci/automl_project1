import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def preprocess_data(input_file="raw_data/data.csv", output_file="processed_data/pro_data.csv"):
    # Load data
    df = pd.read_csv(input_file)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Compute feature importance using Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    
    # Select top 5 features
    feature_importance = pd.Series(importances, index=X.columns)
    top_features = feature_importance.nlargest(5).index
    print(f"Top 5 features: {top_features.tolist()}")
    
    # Create processed DataFrame with top 5 features and target
    processed_df = X_scaled_df[top_features].copy()
    processed_df['target'] = y
    
    # Save processed data
    processed_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data()
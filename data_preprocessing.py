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
    
    
    X_scaled_df['target'] = y
    
    # Save processed data
    X_scaled_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data()
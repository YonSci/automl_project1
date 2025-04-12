from sklearn.datasets import make_regression
import pandas as pd

def generate_data():
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=100,    # 100 rows
        n_features=10,    # 10 features
        noise=0.1,        # Add some noise for realism
        random_state=42   # Reproducible results
    )
    
    # Create DataFrame with feature names and target
    feature_names = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    df.to_csv("raw_data/data.csv", index=False)
    print("Generated data saved to data.csv")

if __name__ == "__main__":
    generate_data()
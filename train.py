#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():
    # 1) Load the data from CSV.
    data = pd.read_csv("data/wine_quality.csv")
    
    # Assuming that the target variable is 'quality'
    X = data.drop(columns=['quality'])
    y = data['quality']
    
    # 2) Split the data into training and test sets (80% training, 20% testing).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3) Initialize and fit the Random Forest Regressor on the training data.
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 4) Make predictions on both training and testing sets.
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # 5) Report the training and test set scores (using R² score).
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # print(f"Training Score: {train_score:.4f}")
    # print(f"Test Score: {test_score:.4f}")
    
    # 6) Write the scores into a single file called metrics.txt.
    with open("output/metrics.txt", "w") as f:
        f.write(f"Training Score: {train_score:.4f}\n")
        f.write(f"Test Score: {test_score:.4f}\n")
    
    # 7) Plot feature importance.
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("output/feature_importances.png")
    plt.close()
    
    # 8) Plot residuals: difference between actual and predicted test target values.
    residuals = y_test - test_pred
    plt.figure()
    plt.scatter(test_pred, residuals)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals Plot")
    plt.tight_layout()
    plt.savefig("output/residuals_plot.png")
    plt.close()

if __name__ == "__main__":
    main()

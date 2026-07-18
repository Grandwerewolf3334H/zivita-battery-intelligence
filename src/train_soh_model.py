"""
train_soh_model.py

Author: Zivita Research Team

Description
-----------
Trains a Random Forest Regression model to estimate
Battery State of Health (SOH) using features extracted
from the NASA Battery Aging Dataset.

Input:
    data/processed/battery_dataset.csv

Outputs:
    models/random_forest_soh.pkl
    results/metrics.txt
    results/prediction_vs_actual.png
    results/feature_importance.png
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def main():

    print("=" * 60)
    print("ZIVITA SOH MODEL TRAINING")
    print("=" * 60)

    # ----------------------------------------------------
    # Create output folders
    # ----------------------------------------------------

    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    # ----------------------------------------------------
    # Load Dataset
    # ----------------------------------------------------

    dataset_path = "../data/processed/battery_dataset.csv"

    df = pd.read_csv(dataset_path)

    # Remove rows containing missing values
    df = df.dropna()

    print(f"\nDataset Loaded Successfully")
    print(f"Total Samples : {len(df)}")

    # ----------------------------------------------------
    # Feature Selection
    # ----------------------------------------------------

    feature_columns = [

        "Cycle",
        "AvgVoltage",
        "AvgCurrent",
        "AvgTemp",
        "MaxTemp",
        "Duration",
        "Re",
        "Rct"

    ]

    X = df[feature_columns]

    y = df["SOH"]

    # ----------------------------------------------------
    # Train-Test Split
    # ----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.20,
        random_state=42

    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # ----------------------------------------------------
    # Model Training
    # ----------------------------------------------------

    print("\nTraining Random Forest Regressor...\n")

    model = RandomForestRegressor(

        n_estimators=300,
        random_state=42

    )

    model.fit(X_train, y_train)

    # ----------------------------------------------------
    # Prediction
    # ----------------------------------------------------

    predictions = model.predict(X_test)

    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print("=" * 60)

    print(f"Mean Absolute Error (MAE) : {mae:.3f}")

    print(f"R² Score                  : {r2:.3f}")

    print("=" * 60)

    # ----------------------------------------------------
    # Save Model
    # ----------------------------------------------------

    model_path = "../models/random_forest_soh.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel Saved : {model_path}")

    # ----------------------------------------------------
    # Save Metrics
    # ----------------------------------------------------

    metrics_path = "../results/metrics.txt"

    with open(metrics_path, "w") as f:

        f.write("ZIVITA SOH MODEL RESULTS\n")
        f.write("=" * 40 + "\n\n")

        f.write("Model\n")
        f.write("Random Forest Regressor\n\n")

        f.write(f"Training Samples : {len(X_train)}\n")
        f.write(f"Testing Samples  : {len(X_test)}\n\n")

        f.write(f"Mean Absolute Error : {mae:.4f}\n")
        f.write(f"R² Score            : {r2:.4f}\n")

    print(f"Metrics Saved : {metrics_path}")

    # ----------------------------------------------------
    # Prediction Plot
    # ----------------------------------------------------

    plt.figure(figsize=(7,7))

    plt.scatter(y_test, predictions)

    plt.plot(

        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()]

    )

    plt.xlabel("Actual SOH (%)")

    plt.ylabel("Predicted SOH (%)")

    plt.title("Predicted vs Actual SOH")

    prediction_plot = "../results/prediction_vs_actual.png"

    plt.savefig(prediction_plot, dpi=300)

    plt.close()

    print(f"Prediction Plot Saved : {prediction_plot}")

    # ----------------------------------------------------
    # Feature Importance Plot
    # ----------------------------------------------------

    importance = model.feature_importances_

    plt.figure(figsize=(8,5))

    plt.bar(feature_columns, importance)

    plt.xticks(rotation=45)

    plt.ylabel("Importance")

    plt.title("Feature Importance")

    plt.tight_layout()

    importance_plot = "../results/feature_importance.png"

    plt.savefig(importance_plot, dpi=300)

    plt.close()

    print(f"Feature Importance Plot Saved : {importance_plot}")

    print("\nTraining Completed Successfully!")

    print("=" * 60)


if __name__ == "__main__":

    main()

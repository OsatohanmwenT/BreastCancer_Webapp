import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model():
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = data.target  # 0: malignant, 1: benign

    # Rename columns to match prompt requirements (snake_case)
    # The prompt uses: radius_mean, texture_mean, etc.
    # Sklearn uses: 'mean radius', 'mean texture', etc.
    feature_mapping = {
        "mean radius": "radius_mean",
        "mean texture": "texture_mean",
        "mean perimeter": "perimeter_mean",
        "mean area": "area_mean",
        "mean concavity": "concavity_mean",
    }

    # We only need these 5, so let's check if they exist or rename all
    # Let's just select the sklearn names that correspond and then rename for clarity if needed
    # Or just use the sklearn names to extract data, and use those names.
    # But for the app input, we want consistent names.

    # Let's rename the columns in the dataframe for the ones we want
    df = df.rename(columns=feature_mapping)

    # Selected features (using the prompt's naming convention now)
    selected_features = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "concavity_mean",
    ]

    print(f"\nSelected features: {selected_features}")
    X = df[selected_features]
    y = df["diagnosis"]

    # Preprocessing
    # Check for missing values
    if X.isnull().sum().sum() > 0:
        print("Handling missing values...")
        X = X.fillna(X.mean())
    else:
        print("No missing values found.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # Save Model and Scaler
    print("\nSaving model and scaler...")
    # We need to save both because we need to scale new inputs
    model_data = {
        "model": model,
        "scaler": scaler,
        "features": selected_features,
        "target_names": data.target_names,
    }

    joblib.dump(model_data, "model/breast_cancer_model.pkl")
    print("Model saved to model/breast_cancer_model.pkl")

    # Demonstration of reloading
    print("\nDemonstrating model reload...")
    loaded_data = joblib.load("model/breast_cancer_model.pkl")
    loaded_model = loaded_data["model"]
    loaded_scaler = loaded_data["scaler"]

    # Test prediction with a sample
    sample_data = X_test.iloc[0].values.reshape(1, -1)
    sample_scaled = loaded_scaler.transform(sample_data)
    prediction = loaded_model.predict(sample_scaled)
    prediction_proba = loaded_model.predict_proba(sample_scaled)
    predicted_class = data.target_names[prediction[0]]

    print(f"Sample input: {sample_data}")
    print(f"Prediction: {predicted_class}")
    print(f"Probability: {prediction_proba}")


if __name__ == "__main__":
    train_model()

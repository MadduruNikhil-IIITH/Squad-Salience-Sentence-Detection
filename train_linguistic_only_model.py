import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
from pathlib import Path

RUNS = [100, 250, 500, 1000, 2000]

def train_linguistic_only_model(run_size: int):
    """Train linguistic-only model for a specific run size."""
    run_folder = f"results/run_{run_size}_passages"
    csv_path = Path(run_folder) / "sentences_with_features.csv"
    
    if not csv_path.exists():
        print(f"SKIP: {csv_path} not found")
        return False
    
    print(f"\n{'='*60}")
    print(f"Training linguistic-only model for {run_size} passages")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Define linguistic-only features (exclude gpt2_*, bert_*, and metadata)
    metadata_cols = {"para_id", "sent_id", "sentence", "label"}
    all_feature_cols = set(df.columns) - metadata_cols
    
    # Filter out gpt2_* and bert_* features
    linguistic_features = [
        col for col in all_feature_cols 
        if not col.startswith("gpt2_") and not col.startswith("bert_")
    ]
    linguistic_features = sorted(linguistic_features)
    
    print(f"Using {len(linguistic_features)} linguistic-only features:")
    for i, feat in enumerate(linguistic_features, 1):
        print(f"  {i}. {feat}")
    
    # Prepare data
    X = df[linguistic_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit scaler on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with same config as classifier.py
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    preds = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # Print results
    print(f"\nResults:")
    print(classification_report(y_test, preds, digits=4))
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Save model and scaler
    model_path = Path(run_folder) / "model_linguistic_only.pkl"
    scaler_path = Path(run_folder) / "scaler_linguistic_only.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    return True

def main():
    print(f"\n{'='*60}")
    print("Training Linguistic-Only Models for All Run Sizes")
    print(f"{'='*60}")
    
    results = {}
    for run_size in RUNS:
        success = train_linguistic_only_model(run_size)
        results[run_size] = success
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for run_size in RUNS:
        status = "✓ SUCCESS" if results[run_size] else "✗ SKIPPED"
        print(f"run_{run_size}_passages: {status}")

if __name__ == "__main__":
    main()

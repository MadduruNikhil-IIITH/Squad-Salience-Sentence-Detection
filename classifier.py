import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
import numpy as np

def train_model(df: pd.DataFrame, run_folder: str):
    """
    Train a single full model with all available features.
    
    DEPRECATED: Use training.py module for new work.
    Kept for backward compatibility only.
    """
    cols = [c for c in df.columns if c not in ["para_id","sent_id","sentence","label"]]
    X = df[cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    print("\nFINAL RESULTS")
    print(classification_report(y_test, preds, digits=4))

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"Accuracy: {accuracy:.4f} | F1(Answer): {f1:.4f}")

    # Save with run sub directory
    joblib.dump(model, f"{run_folder}/model.pkl")
    joblib.dump(scaler, f"{run_folder}/scaler.pkl")

    coef = pd.DataFrame({"feature": cols, "coefficient": model.coef_[0]})
    coef["abs"] = coef["coefficient"].abs()
    top20 = coef.sort_values("abs", ascending=False).head(20)
    top_features = top20["feature"].tolist()[:10]
    top20[["feature", "coefficient"]].to_csv(f"{run_folder}/top20_features.csv", index=False)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "top_features": top_features
    }
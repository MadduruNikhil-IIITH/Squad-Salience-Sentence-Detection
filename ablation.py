import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def run_ablation(run_folder: str = "results/run_2000_passages"):
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY — {os.path.basename(run_folder)}")
    print(f"{'='*70}")

    # Load data
    csv_path = f"{run_folder}/sentences_with_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)
    X_full = df.drop(columns=["para_id", "sent_id", "sentence", "label"])
    y = df["label"]

    # Load official full model results (from your original run)
    stats_file = "results/run_statistics.json"
    if not os.path.exists(stats_file):
        print("Error: run_statistics.json not found!")
        return

    with open(stats_file, "r") as f:
        stats = json.load(f)

    key = run_folder.split("_")[1]  # e.g., "2000"
    full_result = stats["runs"][key]
    full_acc = full_result["accuracy"]
    full_f1 = full_result["f1_answer"]
    full_recall = full_result["recall"]
    full_precision = full_result["precision"]

    print(f"Full Model : Accuracy = {full_acc:.4f} | F1 = {full_f1:.4f} | Precision = {full_precision:.4f} | Recall = {full_recall:.4f}")

    # Helper: train and evaluate
    def evaluate(X_subset, name):
        X_clean = X_subset.fillna(0).replace([np.inf, -np.inf], 0)
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        report = classification_report(y_test, pred, digits=4, output_dict=True)
        return {
            "name": name,
            "features": X_subset.shape[1],
            "accuracy": accuracy_score(y_test, pred),
            "precision_ans": report['1']['precision'],
            "recall_ans": report['1']['recall'],
            "f1_ans": report['1']['f1-score'],
            "macro_f1": report['macro avg']['f1-score']
        }

    results = []

    # 1. Top-10 features
    top20_path = f"{run_folder}/top20_features.csv"
    if os.path.exists(top20_path):
        top10_feats = pd.read_csv(top20_path)["feature"].head(10).tolist()
        res = evaluate(X_full[top10_feats], "Top-10 Features")
        results.append(res)
        print(f"{res['name']:<25}: Acc {res['accuracy']:.4f} | F1 {res['f1_ans']:.4f} | Precision {res['precision_ans']:.4f} | Recall {res['recall_ans']:.4f} ({res['features']} feats)")

    # 2. Without any surprisal (GPT-2 + BERT)
    surprisal_cols = [c for c in X_full.columns if c.startswith(("gpt2_", "bert_"))]
    linguistic_cols = [c for c in X_full.columns if c not in surprisal_cols]
    res = evaluate(X_full[linguistic_cols], "No Surprisal (Linguistic Only)")
    results.append(res)
    print(f"{res['name']:<25}: Acc {res['accuracy']:.4f} | F1 {res['f1_ans']:.4f} | Precision {res['precision_ans']:.4f} | Recall {res['recall_ans']:.4f} ({res['features']} feats)")

    # Save full ablation report
    ablation_report = {
        "full_model": {"accuracy": full_acc, "f1": full_f1, "precision": full_precision, "recall": full_recall, "features": X_full.shape[1]},
        "ablations": {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in results}
    }

    with open(f"{run_folder}/ablation_full_report.json", "w") as f:
        json.dump(ablation_report, f, indent=2)

    print(f"\nFull ablation report saved to {run_folder}/ablation_full_report.json")
    print(f"{'='*70}\n")

# Run for your main experiment
if __name__ == "__main__":
    run_ablation("results/run_100_passages")
    run_ablation("results/run_250_passages")
    run_ablation("results/run_500_passages")
    run_ablation("results/run_1000_passages")
    run_ablation("results/run_2000_passages")
import json
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def _extract_run_key(run_folder: str) -> str:
    match = re.search(r"run_(\d+)_passages", Path(run_folder).as_posix())
    if not match:
        raise ValueError(f"Could not extract run key from path: {run_folder}")
    return match.group(1)

def run_pca_analysis(run_folder):

    key = _extract_run_key(run_folder)
    pca_run_folder = f"results/pca/run_{key}"
    os.makedirs(pca_run_folder, exist_ok=True)
    
    
    STATS_FILE = "results/run_statistics.json"
    
    if not os.path.exists(STATS_FILE):
        print("No results found. Run main.py first.")
        return

    with open(STATS_FILE, "r") as f:
        data = json.load(f)

    run_data = data["runs"].get(key, None)
    if run_data is None:
        print(f"No run statistics found for key={key} in {STATS_FILE}.")
        return
    
    print(f"\nRunning PCA analysis on: {run_folder}")
    df = pd.read_csv(f"{run_folder}/sentences_with_features.csv")
    feature_cols = [c for c in df.columns if c not in ["para_id", "sent_id", "sentence", "label"]]
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 1. FULL PCA + 90% VARIANCE ===
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)

    # Find minimum components for 90% variance
    n_components_90 = np.argmax(cumsum >= 0.90) + 1
    variance_90 = cumsum[n_components_90 - 1]

    print(f"Minimum components for 90% variance: {n_components_90}")
    print(f"Explained variance at {n_components_90} components: {variance_90:.4f}")

    # === 2. ELBOW PLOT WITH 90% LINE ===
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'b-o', linewidth=2, markersize=6, label='Cumulative Variance')
    plt.axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='90% Variance Threshold')
    plt.axvline(x=n_components_90, color='green', linestyle=':', linewidth=3, label=f'{n_components_90} components = 90%')
    plt.title("PCA Elbow Plot — Cumulative Explained Variance\n"
              f"Only {n_components_90} components needed for 90% variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    elbow_path = f"{pca_run_folder}/elbow_90_variance.png"
    plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Elbow plot (90% variance) saved: {elbow_path}")

    # === 3. 2D PCA VISUALIZATION ===
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='RdYlBu', alpha=0.7, s=15)
    plt.colorbar(scatter, label='Contains Answer')
    plt.title("2D PCA of Feature Space\nAnswer vs Non-Answer Sentences")
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    viz_path = f"{pca_run_folder}/2d_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    # === 4. CLASSIFICATION ON 90% VARIANCE COMPONENTS ===
    pca_90 = PCA(n_components=n_components_90)
    X_pca_90 = pca_90.fit_transform(X_scaled)
    model_pca = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    model_pca.fit(X_pca_90, y)
    preds_pca_90 = model_pca.predict(X_pca_90)
    acc_pca = accuracy_score(y, preds_pca_90)
    f1_pca = f1_score(y, preds_pca_90)

    # === SAVE ALL RESULTS ===
    results_text = f"""PCA ANALYSIS RESULTS
    Minimum components for 90% variance: {n_components_90}
    Variance explained by {n_components_90} components: {variance_90:.4f}
    PC1 variance: {pca_2d.explained_variance_ratio_[0]:.4f}
    PC2 variance: {pca_2d.explained_variance_ratio_[1]:.4f}
    Classification using only PC1 & PC2:
    Accuracy: {acc_pca:.4f}
    F1 (Answer): {f1_pca:.4f}
    Original Model Accuracy: {run_data['accuracy']:.4f}
    Original Model F1 (Answer): {run_data['f1_answer']:.4f} 
    """

    with open(f"{pca_run_folder}/analysis_summary.txt", "w") as f:
        f.write(results_text)

    print(f"\nPCA ANALYSIS COMPLETE")
    print(f"Only {n_components_90} components needed for 90% variance!")
    print(f"Classification on 90% variance components: Acc {acc_pca:.4f} | F1 {f1_pca:.4f}")
    print(f"All results saved in: {run_folder}/")


if __name__ == "__main__":
    run_pca_analysis("results/run_100_passages")
    print("-" * 50)
    run_pca_analysis("results/run_250_passages")
    print("-" * 50)
    run_pca_analysis("results/run_500_passages")
    print("-" * 50)
    run_pca_analysis("results/run_1000_passages")
    print("-" * 50)
    run_pca_analysis("results/run_2000_passages")
"""
Consolidated training module for all model variants and ablation studies.
Streamlines training logic for linguistic, RST, surprisal, and combined models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)
import joblib
from pathlib import Path


# ============================================================================
# Feature Group Definitions
# ============================================================================

LINGUISTIC_FEATURES = [
    "avg_word_length", "sentence_length_words", "sentence_position",
    "sentence_position_norm", "type_token_ratio", "lexical_density",
    "noun_ratio", "verb_ratio", "adj_ratio", "pronoun_ratio",
    "noun_verb_ratio", "causal_marker_ratio", "contrast_marker_ratio",
    "named_entity_density"
]

RST_PASSAGE_FEATURES = [
    "rst_depth", "rst_is_nucleus", "rst_relation_type_encoded",
    "rst_centrality_score", "rst_is_root", "rst_relation_direction",
    "rst_span_length"
]

RST_INTRA_FEATURES = [
    "intra_clause_count", "intra_clause_density",
    "intra_has_nucleus_satellite_relation", "intra_avg_clause_depth",
    "intra_top_clause_is_nucleus"
]

RST_ALL_FEATURES = RST_PASSAGE_FEATURES + RST_INTRA_FEATURES

SURPRISAL_GPT2_FEATURES = [
    "gpt2_surprisal_mean", "gpt2_surprisal_max", "gpt2_surprisal_min",
    "gpt2_surprisal_var", "gpt2_surprisal_sum", "gpt2_surprisal_std"
]

SURPRISAL_BERT_FEATURES = [
    "bert_surprisal_mean", "bert_surprisal_max", "bert_surprisal_min",
    "bert_surprisal_var", "bert_surprisal_sum", "bert_surprisal_std"
]

SURPRISAL_ALL_FEATURES = SURPRISAL_GPT2_FEATURES + SURPRISAL_BERT_FEATURES


# ============================================================================
# Core Training Function
# ============================================================================

def train_model_variant(
    df: pd.DataFrame,
    feature_list: list,
    model_name: str,
    run_folder: str,
    random_state: int = 42,
    test_size: float = 0.2,
    verbose: bool = True
) -> dict:
    """
    Train a single model variant with specified feature set.
    
    Args:
        df: DataFrame with feature columns and 'label' column
        feature_list: List of feature column names to use
        model_name: Name for model (used in saved filenames)
        run_folder: Directory to save model, scaler, and metrics
        random_state: Random seed for reproducibility
        test_size: Test split ratio
        verbose: Print training details
        
    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, auc_roc, top_features
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"Features: {len(feature_list)}")
        print(f"{'='*70}")
    
    # Filter available features
    available_features = [f for f in feature_list if f in df.columns]
    
    if not available_features:
        raise ValueError(f"No valid features found for {model_name}")
    
    y = df["label"]
    X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    model.fit(X_train_s, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    if verbose:
        print(classification_report(y_test, y_pred, digits=4))
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
    
    # Save model and scaler
    model_path = Path(run_folder) / f"{model_name}.pkl"
    scaler_path = Path(run_folder) / f"scaler_{model_name}.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save feature importance
    coef_df = pd.DataFrame({
        "feature": available_features,
        "coefficient": model.coef_[0]
    })
    coef_df["abs"] = coef_df["coefficient"].abs()
    top20 = coef_df.sort_values("abs", ascending=False).head(20)
    top20[["feature", "coefficient"]].to_csv(
        Path(run_folder) / f"top20_features_{model_name}.csv", index=False
    )
    
    if verbose:
        print(f"✓ Saved to: {model_path}")
    
    return {
        "model_name": model_name,
        "num_features": len(available_features),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "top_5_features": ", ".join(top20.head(5)["feature"].tolist())
    }


# ============================================================================
# Predefined Model Variants
# ============================================================================

def train_linguistic_model(df: pd.DataFrame, run_folder: str, **kwargs) -> dict:
    """Train linguistic-only model (14 features)."""
    return train_model_variant(
        df, LINGUISTIC_FEATURES, "model_linguistic", run_folder, **kwargs
    )


def train_rst_only_model(df: pd.DataFrame, run_folder: str, **kwargs) -> dict:
    """Train RST-only model (12 features)."""
    return train_model_variant(
        df, RST_ALL_FEATURES, "model_rst_only", run_folder, **kwargs
    )


def train_linguistic_rst_model(df: pd.DataFrame, run_folder: str, **kwargs) -> dict:
    """Train linguistic + RST model (26 features, no surprisal)."""
    return train_model_variant(
        df,
        LINGUISTIC_FEATURES + RST_ALL_FEATURES,
        "model_ling_rst",
        run_folder,
        **kwargs
    )


def train_surprisal_only_model(df: pd.DataFrame, run_folder: str, **kwargs) -> dict:
    """Train surprisal-only model (24 features)."""
    return train_model_variant(
        df, SURPRISAL_ALL_FEATURES, "model_surprisal_only", run_folder, **kwargs
    )


def train_full_model(df: pd.DataFrame, run_folder: str, **kwargs) -> dict:
    """Train full model with all features (50 features)."""
    all_features = LINGUISTIC_FEATURES + SURPRISAL_ALL_FEATURES + RST_ALL_FEATURES
    return train_model_variant(
        df, all_features, "model_full", run_folder, **kwargs
    )


# ============================================================================
# Ablation Study: Multiple Models
# ============================================================================

def train_ablation_study(df: pd.DataFrame, run_folder: str) -> pd.DataFrame:
    """
    Train all model variants for ablation study:
    - Linguistic only (14 features)
    - RST only (12 features)
    - Linguistic + RST (26 features)
    - Surprisal only (24 features)
    - Full (50 features)
    
    Returns DataFrame with side-by-side comparison of all models.
    """
    print("\n" + "="*70)
    print("ABLATION STUDY: TRAINING ALL MODEL VARIANTS")
    print("="*70)
    
    results = []
    
    # Train all variants
    try:
        results.append(train_linguistic_model(df, run_folder))
    except ValueError as e:
        print(f"⚠️ Skipped linguistic model: {e}")
    
    try:
        results.append(train_rst_only_model(df, run_folder))
    except ValueError as e:
        print(f"⚠️ Skipped RST-only model: {e}")
    
    try:
        results.append(train_linguistic_rst_model(df, run_folder))
    except ValueError as e:
        print(f"⚠️ Skipped linguistic+RST model: {e}")
    
    try:
        results.append(train_surprisal_only_model(df, run_folder))
    except ValueError as e:
        print(f"⚠️ Skipped surprisal-only model: {e}")
    
    try:
        results.append(train_full_model(df, run_folder))
    except ValueError as e:
        print(f"⚠️ Skipped full model: {e}")
    
    # Create comparison table
    results_df = pd.DataFrame(results)
    comparison_path = Path(run_folder) / "ablation_study.csv"
    results_df.to_csv(comparison_path, index=False)
    
    print(f"\n✓ Ablation study results saved to: {comparison_path}")
    print(f"\nSummary:\n{results_df.to_string(index=False)}")
    
    return results_df


# ============================================================================
# Multi-Run Training (for multiple dataset sizes)
# ============================================================================

def train_all_runs(run_sizes: list) -> dict:
    """
    Train models for multiple dataset sizes.
    
    Args:
        run_sizes: List of integers (e.g., [100, 250, 500, 1000, 2000])
        
    Returns:
        Dictionary mapping run_size -> success status
    """
    print(f"\n{'='*70}")
    print("TRAINING MODELS FOR MULTIPLE RUN SIZES")
    print(f"{'='*70}")
    
    results = {}
    
    for run_size in run_sizes:
        run_folder = f"results/run_{run_size}_passages"
        csv_path = Path(run_folder) / "sentences_with_features.csv"
        
        if not csv_path.exists():
            print(f"\n✗ SKIP run_{run_size}: CSV not found ({csv_path})")
            results[run_size] = False
            continue
        
        try:
            df = pd.read_csv(csv_path)
            print(f"\n✓ Loaded {len(df)} rows from {run_size}-passage run")
            
            # Train linguistic-only model variant
            train_linguistic_model(df, run_folder, verbose=True)
            results[run_size] = True
            
        except Exception as e:
            print(f"✗ Error training run_{run_size}: {e}")
            results[run_size] = False
    
    return results


# ============================================================================
# Backward Compatibility: Original train_model function
# ============================================================================

def train_model(df: pd.DataFrame, run_folder: str) -> dict:
    """
    Legacy function for backward compatibility.
    Trains full model with all available features.
    """
    cols = [c for c in df.columns if c not in ["para_id", "sent_id", "sentence", "label"]]
    X = df[cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    
    print("\nFINAL RESULTS (Legacy Full Model)")
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

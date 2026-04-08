#!/usr/bin/env python
"""Configurable ablation runner."""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from qg_pipeline import (
    _load_examples_from_pipeline_output,
    _load_existing_metrics,
    _load_squad_dev_examples,
    _load_squad_train_examples,
    _write_generation_error_log,
    run_pipeline,
)


def _ensure_top10_model_artifacts(model_artifact_dir: str) -> Dict[str, str]:
    """Create reusable top-10 feature artifacts if missing, then return their paths."""
    artifact_dir = Path(model_artifact_dir)
    model_path = artifact_dir / "model_top10.pkl"
    scaler_path = artifact_dir / "scaler_top10.pkl"

    if model_path.exists() and scaler_path.exists():
        return {"model_path": str(model_path), "scaler_path": str(scaler_path)}

    features_csv = artifact_dir / "sentences_with_features.csv"
    top20_csv = artifact_dir / "top20_features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"Missing features CSV for top-10 ablation: {features_csv}")
    if not top20_csv.exists():
        raise FileNotFoundError(f"Missing top20 features CSV for top-10 ablation: {top20_csv}")

    df = pd.read_csv(features_csv)
    if "label" not in df.columns:
        raise ValueError(f"Expected 'label' column in {features_csv}")

    top_df = pd.read_csv(top20_csv)
    if "feature" not in top_df.columns:
        raise ValueError(f"Expected 'feature' column in {top20_csv}")

    top10_features = [str(x) for x in top_df["feature"].dropna().head(10).tolist()]
    selected_features = [name for name in top10_features if name in df.columns]
    if not selected_features:
        raise ValueError(
            f"No top-10 features from {top20_csv} were found in {features_csv}."
        )

    X = (
        df[selected_features]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    y = df["label"]
    if y.nunique() < 2:
        raise ValueError("Top-10 model training requires at least 2 label classes.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return {"model_path": str(model_path), "scaler_path": str(scaler_path)}


def run_ablation_study(
    num_examples: int = 2000,
    seed: int = 2026,
    source: str = "train",
    skip_first_passages: int = 2000,
    output_dir: str = "results/qg_ablation",
    model_artifact_dir: str = "results/run_2000_passages",
    overwrite: bool = False,
    batch_size: int = 4,
    threshold: float = 0.55,
    top_k: int = 3,
    num_questions: int = 2,
    qg_model: str = "microsoft/Phi-4-mini-instruct",
    eval_source_path: str = "",
    include_full_salience: bool = True,
    include_top10_features: bool = True,
) -> None:
    """Run ablation on the same seeded subset with configurable arms."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    run_root = output_dir_path.parent if output_dir_path.name == "ablation" else output_dir_path
    arms_root = output_dir_path
    arms_root.mkdir(parents=True, exist_ok=True)

    configs = []
    if include_full_salience:
        configs.append("full_salience")
    if include_top10_features:
        configs.append("top10_features")
    configs.append("no_surprisal")

    if source == "train":
        source_path = Path("data/train.json")
        all_examples = _load_squad_train_examples(source_path, num_examples * 2, skip_first_passages)
    else:
        source_path = Path("data/dev.json")
        all_examples = _load_squad_dev_examples(source_path, num_examples * 2)

    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(all_examples)), min(num_examples, len(all_examples))))
    sampled_examples = [all_examples[i] for i in sampled_indices]

    root_full_salience_path = None
    prefer_root_full_salience = False
    if include_full_salience and not overwrite:
        source_tag = f"train_after{skip_first_passages}" if source == "train" else "dev"
        search_roots = [run_root]
        canonical_qg_root = Path("results") / "qg" / f"run_{num_examples}_passages"
        if canonical_qg_root not in search_roots:
            search_roots.append(canonical_qg_root)

        for search_root in search_roots:
            preferred_candidates = sorted(search_root.glob(f"qg_pipeline_{source_tag}_{num_examples}*.json"))
            if preferred_candidates:
                root_full_salience_path = preferred_candidates[-1]
                break

        if root_full_salience_path is None:
            for search_root in search_roots:
                fallback_candidates = sorted(search_root.glob("qg_pipeline*.json"))
                if fallback_candidates:
                    root_full_salience_path = fallback_candidates[-1]
                    break

        if root_full_salience_path is not None:
            root_examples = _load_examples_from_pipeline_output(root_full_salience_path, num_examples)
            if len(root_examples) == len(sampled_examples):
                sampled_examples = root_examples
                sampled_indices = [
                    int(ex.get("original_passage_index", ex.get("example_index", i)))
                    for i, ex in enumerate(sampled_examples)
                ]
                prefer_root_full_salience = True

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY — {' vs '.join(configs)}")
    print(f"{'='*70}")
    print(f"Seed: {seed}")
    print(f"Source: {source}")
    print(f"Skip first passages: {skip_first_passages}")
    print(f"Sample size: {len(sampled_examples)}")
    print(f"Model artifact dir: {model_artifact_dir}")
    print(f"Batch size: {batch_size} | Threshold: {threshold} | Top-K: {top_k}")
    print(f"Questions per method: {num_questions} | QG model: {qg_model}")
    print(f"Sample indices: {sampled_indices[:10]}... (showing first 10)")
    if include_full_salience and prefer_root_full_salience and root_full_salience_path is not None:
        print(f"Full-salience root reuse candidate: {root_full_salience_path}")
    elif include_full_salience and not overwrite:
        print("No root full-salience pipeline candidate found; ablation will generate full_salience arm.")

    resolved_eval_source_path = eval_source_path.strip()
    if not resolved_eval_source_path:
        resolved_eval_source_path = "data/train.json" if source == "train" else "data/dev.json"

    results_by_config: Dict[str, Dict[str, Any]] = {}
    top10_artifacts: Dict[str, str] = {}
    if "top10_features" in configs:
        try:
            top10_artifacts = _ensure_top10_model_artifacts(model_artifact_dir)
            print(
                "Top-10 feature artifacts ready: "
                f"{top10_artifacts['model_path']}, {top10_artifacts['scaler_path']}"
            )
        except Exception as exc:
            print(f"ERROR: Could not prepare top-10 feature artifacts: {type(exc).__name__}: {exc}")
            return

    needs_generation = False
    for config in configs:
        config_output_dir = arms_root / config
        results_path = config_output_dir / "qg_pipeline.json"
        if overwrite or not results_path.exists():
            needs_generation = True
            break

    temp_subset_path = run_root / f"_ablation_subset_seed{seed}.json"
    if needs_generation or overwrite:
        subset_payload = {
            "metadata": {
                "seed": seed,
                "source": source,
                "skip_first_passages": skip_first_passages,
                "sample_indices": sampled_indices,
            },
            "data": [{"paragraphs": [{"context": ex["context"], "qas": [{"id": qid} for qid in ex["qa_ids"]]}]} for ex in sampled_examples],
        }
        with temp_subset_path.open("w", encoding="utf-8") as f:
            json.dump(subset_payload, f, indent=2)

    for config in configs:
        print(f"\n--- Running {config} ---")
        config_output_dir = arms_root / config
        config_output_dir.mkdir(parents=True, exist_ok=True)
        results_path = config_output_dir / "qg_pipeline.json"
        eval_path = config_output_dir / "evaluation.json"

        if config == "no_surprisal":
            model_path = str(Path(model_artifact_dir) / "model_linguistic_only.pkl")
            scaler_path = str(Path(model_artifact_dir) / "scaler_linguistic_only.pkl")
        elif config == "top10_features":
            model_path = top10_artifacts["model_path"]
            scaler_path = top10_artifacts["scaler_path"]
        else:
            model_path = str(Path(model_artifact_dir) / "model.pkl")
            scaler_path = str(Path(model_artifact_dir) / "scaler.pkl")

        if results_path.exists() and eval_path.exists() and not overwrite:
            print(f"Reusing existing outputs for {config} (set overwrite=True to regenerate).")
            results_by_config[config] = _load_existing_metrics(eval_path)
            continue

        if (
            config == "full_salience"
            and not overwrite
            and prefer_root_full_salience
            and root_full_salience_path is not None
            and root_full_salience_path.exists()
        ):
            if not results_path.exists():
                shutil.copy2(root_full_salience_path, results_path)
                print(f"Reused root qg_pipeline for full_salience: {root_full_salience_path}")

            root_error_log_path = root_full_salience_path.with_name("qa_generation_errors.json")
            target_error_log_path = results_path.with_name("qa_generation_errors.json")
            if root_error_log_path.exists() and not target_error_log_path.exists():
                shutil.copy2(root_error_log_path, target_error_log_path)
            elif not target_error_log_path.exists():
                with results_path.open("r", encoding="utf-8") as f:
                    reused_payload = json.load(f)
                _write_generation_error_log(results_path, reused_payload)

            if not eval_path.exists():
                eval_cmd = [
                    sys.executable,
                    "evaluation.py",
                    "--pipeline-json", str(results_path),
                    "--source-path", resolved_eval_source_path,
                    "--output", str(eval_path),
                ]
                print(f"Running evaluation: {' '.join(eval_cmd)}")
                subprocess.run(eval_cmd, check=False)

            if eval_path.exists():
                with eval_path.open("r", encoding="utf-8") as f:
                    eval_result = json.load(f)
                results_by_config[config] = eval_result.get("metrics", {})
            else:
                results_by_config[config] = {}
            continue

        if results_path.exists() and not overwrite:
            print(f"Reusing existing qg_pipeline for {config}; regenerating evaluation only.")
            eval_cmd = [
                sys.executable,
                "evaluation.py",
                "--pipeline-json", str(results_path),
                "--source-path", resolved_eval_source_path,
                "--output", str(eval_path),
            ]
            print(f"Running evaluation: {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd, check=False)
            if eval_path.exists():
                with eval_path.open("r", encoding="utf-8") as f:
                    eval_result = json.load(f)
                results_by_config[config] = eval_result.get("metrics", {})
            else:
                results_by_config[config] = {}
            continue

        class AblationArgs:
            pass

        args = AblationArgs()
        args.source = "dev"
        args.dev_path = str(temp_subset_path)
        args.train_path = str(temp_subset_path)
        args.skip_first_passages = 0
        args.model_path = model_path
        args.scaler_path = scaler_path
        args.max_examples = len(sampled_examples)
        args.batch_size = batch_size
        args.threshold = threshold
        args.top_k = top_k
        args.num_questions = num_questions
        args.qg_model = qg_model
        args.gpu_cache_every = 8
        args.clear_model_cache_every = 0
        args.output = str(results_path)
        args.linguistic_only = (config == "no_surprisal")

        if config == "no_surprisal" and (not Path(model_path).exists() or not Path(scaler_path).exists()):
            print("ERROR: Linguistic-only artifacts missing!")
            print(f"  Expected: {model_path}")
            print(f"  Expected: {scaler_path}")
            print("  Run train_linguistic_only_model.py first.")
            return

        payload = run_pipeline(args)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

        error_log_path = _write_generation_error_log(results_path, payload)
        print(f"Saved {config} results: {results_path}")
        print(f"Saved {config} QA error log: {error_log_path}")

        eval_cmd = [
            sys.executable,
            "evaluation.py",
            "--pipeline-json", str(results_path),
            "--source-path", resolved_eval_source_path,
            "--output", str(eval_path),
        ]
        print(f"Running evaluation: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=False)

        if eval_path.exists():
            with eval_path.open("r", encoding="utf-8") as f:
                eval_result = json.load(f)
            results_by_config[config] = eval_result.get("metrics", {})

    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")

    summary_rows = []
    for config in configs:
        metrics = results_by_config.get(config, {})
        if isinstance(metrics, dict) and "salience" in metrics:
            m = metrics["salience"]
            row = {
                "config": config,
                "count": m.get("count", 0),
                "rouge_l": m.get("rouge_l", 0.0),
                "bertscore_f1": m.get("bertscore_f1", 0.0),
                "qa_em": m.get("qa_em", 0.0),
                "qa_consistency_f1": m.get("qa_consistency_f1", 0.0),
            }
        else:
            row = {
                "config": config,
                "count": metrics.get("num_examples_processed", 0),
                "rouge_l": 0.0,
                "bertscore_f1": 0.0,
                "qa_em": 0.0,
                "qa_consistency_f1": 0.0,
            }
        summary_rows.append(row)

    if summary_rows:
        print(f"{'Config':<20} {'Count':<8} {'ROUGE-L':<12} {'BERTScore':<12} {'QA-EM':<12} {'QA-F1':<12}")
        print("-" * 76)
        for row in summary_rows:
            print(f"{row['config']:<20} {row['count']:<8} {row['rouge_l']:<12.4f} {row['bertscore_f1']:<12.4f} {row['qa_em']:<12.4f} {row['qa_consistency_f1']:<12.4f}")

    summary_path = run_root / "ablation_summary.json"
    summary_data = {
        "seed": seed,
        "sample_size": len(sampled_examples),
        "sample_indices": sampled_indices,
        "results": {row["config"]: {k: v for k, v in row.items() if k != "config"} for row in summary_rows},
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved ablation summary: {summary_path}")
    print(f"{'='*70}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configurable ablation test.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=2000,
        help="Number of passages to sample for the ablation run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for deterministic passage sampling.",
    )
    parser.add_argument(
        "--source",
        choices=["dev", "train"],
        default="train",
        help="Data source to sample from.",
    )
    parser.add_argument(
        "--skip-first-passages",
        type=int,
        default=2000,
        help="For train runs, skip the first N passages before sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for the ablation run. Defaults to results/qg_ablation/run_<N>_passages.",
    )
    parser.add_argument(
        "--model-run",
        type=int,
        default=2000,
        help="Use artifacts from results/run_<model_run>_passages for full/no-surprisal ablation arms.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for salience inference inside ablation runs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Salience threshold for ablation runs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K salient sentences to keep.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=2,
        help="Questions generated per method.",
    )
    parser.add_argument(
        "--qg-model",
        default="microsoft/Phi-4-mini-instruct",
        help="QG model id used in ablation runs.",
    )
    parser.add_argument(
        "--eval-source-path",
        default="",
        help="Optional source JSON path for evaluation. Defaults by source (train/dev).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing ablation outputs instead of reusing them.",
    )
    parser.add_argument(
        "--skip-full-salience",
        action="store_true",
        help="Do not run the full_salience arm; only run no_surprisal.",
    )
    parser.add_argument(
        "--skip-top10-features",
        action="store_true",
        help="Do not run the top10_features arm.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir
        else str(Path("results/qg_ablation") / f"run_{args.num_examples}_passages")
    )

    print(f"\n{'='*70}")
    print(f"Running ablation for {args.num_examples} examples")
    print(f"{'='*70}")
    run_ablation_study(
        num_examples=args.num_examples,
        seed=args.seed,
        source=args.source,
        skip_first_passages=args.skip_first_passages,
        output_dir=output_dir,
        model_artifact_dir=str(Path("results") / f"run_{args.model_run}_passages"),
        batch_size=args.batch_size,
        threshold=args.threshold,
        top_k=args.top_k,
        num_questions=args.num_questions,
        qg_model=args.qg_model,
        eval_source_path=args.eval_source_path,
        include_full_salience=not args.skip_full_salience,
        include_top10_features=not args.skip_top10_features,
        overwrite=args.overwrite,
    )
    print(f"Completed: {output_dir}")


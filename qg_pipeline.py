import argparse
import gc
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm.auto import tqdm

from llm_question_generator import (
    clear_generator_cache,
    generate_questions_baseline,
    generate_questions_with_salience,
)
from salience_inference import SalienceInferencer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run salience + QA generation pipeline on SQuAD dev/train examples."
    )
    parser.add_argument(
        "--source",
        choices=["dev", "train"],
        default="dev",
        help="Data source. Use train with --skip-first-passages to continue after training slice.",
    )
    parser.add_argument(
        "--dev-path",
        default="data/dev.json",
        help="Path to SQuAD dev JSON.",
    )
    parser.add_argument(
        "--train-path",
        default="data/train.json",
        help="Path to SQuAD train JSON.",
    )
    parser.add_argument(
        "--skip-first-passages",
        type=int,
        default=2000,
        help="For --source train, skip first N passages (default 2000).",
    )
    parser.add_argument(
        "--model-path",
        default="results/run_2000_passages/model.pkl",
        help="Path to salience model.pkl.",
    )
    parser.add_argument(
        "--scaler-path",
        default="results/run_2000_passages/scaler.pkl",
        help="Path to salience scaler.pkl.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=250,
        help="Number of dev examples to process (recommended 200-300 for RTX 4060).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Small batch size for salience inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Salience threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K salient sentences.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=2,
        help="Questions to generate for each method.",
    )
    parser.add_argument(
        "--qg-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model id used for question generation.",
    )
    parser.add_argument(
        "--gpu-cache-every",
        type=int,
        default=8,
        help="Call torch.cuda.empty_cache every N examples (0 disables).",
    )
    parser.add_argument(
        "--clear-model-cache-every",
        type=int,
        default=0,
        help="Call clear_generator_cache every N examples (0 disables).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. Defaults to results/qg_pipeline_<config>.json.",
    )
    parser.add_argument(
        "--linguistic-only",
        action="store_true",
        help="Use linguistic-only model (no surprisal features).",
    )
    parser.add_argument(
        "--eval-source-path",
        default="",
        help="Optional source JSON used for evaluation; defaults by --source.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step after pipeline generation.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_examples <= 0:
        raise ValueError("--max-examples must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_questions <= 0:
        raise ValueError("--num-questions must be > 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.gpu_cache_every < 0:
        raise ValueError("--gpu-cache-every must be >= 0")
    if args.clear_model_cache_every < 0:
        raise ValueError("--clear-model-cache-every must be >= 0")
    if args.skip_first_passages < 0:
        raise ValueError("--skip-first-passages must be >= 0")


def _load_squad_dev_examples(dev_path: Path, max_examples: int) -> List[Dict[str, Any]]:
    with dev_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Expected SQuAD format with top-level 'data'")

    examples: List[Dict[str, Any]] = []
    global_idx = 0
    for article in payload.get("data", []):
        title = article.get("title", "")
        for para_idx, paragraph in enumerate(article.get("paragraphs", [])):
            context = (paragraph.get("context") or "").strip()
            if not context:
                continue

            qa_ids = []
            for qa in paragraph.get("qas", []):
                qa_id = qa.get("id")
                if isinstance(qa_id, str):
                    qa_ids.append(qa_id)

            examples.append(
                {
                    "example_index": global_idx,
                    "title": title,
                    "paragraph_index": para_idx,
                    "context": context,
                    "qa_ids": qa_ids,
                }
            )
            global_idx += 1

            if len(examples) >= max_examples:
                return examples

    return examples


def _load_squad_train_examples(
    train_path: Path,
    max_examples: int,
    skip_first_passages: int,
) -> List[Dict[str, Any]]:
    with train_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Expected SQuAD format with top-level 'data'")

    all_examples: List[Dict[str, Any]] = []
    global_idx = 0
    for article in payload.get("data", []):
        title = article.get("title", "")
        for para_idx, paragraph in enumerate(article.get("paragraphs", [])):
            context = (paragraph.get("context") or "").strip()
            if not context:
                continue

            qa_ids = []
            for qa in paragraph.get("qas", []):
                qa_id = qa.get("id")
                if isinstance(qa_id, str):
                    qa_ids.append(qa_id)

            all_examples.append(
                {
                    "example_index": global_idx,
                    "original_passage_index": global_idx,
                    "title": title,
                    "paragraph_index": para_idx,
                    "context": context,
                    "qa_ids": qa_ids,
                }
            )
            global_idx += 1

    start = min(skip_first_passages, len(all_examples))
    filtered = all_examples[start : start + max_examples]

    for local_idx, row in enumerate(filtered):
        row["example_index"] = local_idx

    return filtered


def _cleanup_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _default_output_path(args: argparse.Namespace) -> Path:
    source_tag = (
        f"train_after{args.skip_first_passages}"
        if args.source == "train"
        else "dev"
    )
    run_dir = Path("results") / "qg" / f"run_{args.max_examples}_passages"
    return run_dir / (
        f"qg_pipeline_{source_tag}{args.max_examples}_b{args.batch_size}"
        f"_q{args.num_questions}_t{str(args.threshold).replace('.', 'p')}.json"
    )


def _collect_generation_errors(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    error_rows: List[Dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            continue

        error = row.get("error")
        if not error:
            continue

        salience_questions = row.get("salience_questions", [])
        baseline_questions = row.get("baseline_questions", [])

        error_rows.append(
            {
                "example_index": row.get("example_index"),
                "original_passage_index": row.get("original_passage_index"),
                "title": row.get("title", ""),
                "paragraph_index": row.get("paragraph_index"),
                "qa_ids": row.get("qa_ids", []),
                "error_type": error.split(":", 1)[0] if isinstance(error, str) else type(error).__name__,
                "error_message": error,
                "salience_question_count": len(salience_questions)
                if isinstance(salience_questions, list)
                else 0,
                "baseline_question_count": len(baseline_questions)
                if isinstance(baseline_questions, list)
                else 0,
                "passage_excerpt": (row.get("passage", "") or "")[:500],
            }
        )

    return error_rows


def _write_generation_error_log(output_path: Path, payload: Dict[str, Any]) -> Path:
    error_log_path = output_path.with_name("qa_generation_errors.json")
    error_rows = _collect_generation_errors(payload.get("results", []))
    error_log = {
        "meta": {
            "source_output": str(output_path),
            "num_examples_processed": payload.get("metadata", {}).get("num_examples_processed", 0),
            "num_generation_errors": len(error_rows),
            "created_utc": datetime.utcnow().isoformat() + "Z",
        },
        "errors": error_rows,
    }

    with error_log_path.open("w", encoding="utf-8") as f:
        json.dump(error_log, f, indent=2, ensure_ascii=True)

    return error_log_path


def _build_sentence_feature_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in payload.get("results", []):
        if not isinstance(record, dict):
            continue

        salience = record.get("salience", {})
        if not isinstance(salience, dict):
            continue

        sentences = salience.get("sentences", [])
        features = salience.get("features", [])
        scores = salience.get("scores", [])
        top_salient = salience.get("top_salient", [])
        if not isinstance(sentences, list) or not sentences:
            continue

        top_indices = {
            int(item.get("sentence_index"))
            for item in top_salient
            if isinstance(item, dict) and isinstance(item.get("sentence_index"), (int, float))
        }
        total_sentences = len(sentences)
        para_index = int(record.get("original_passage_index", record.get("example_index", 0)))
        para_id = f"P{para_index:06d}"

        for i, sentence in enumerate(sentences):
            if not isinstance(sentence, str) or not sentence.strip():
                continue

            feature_map = {}
            if isinstance(features, list) and i < len(features) and isinstance(features[i], dict):
                feature_map = dict(features[i])
            score = float(scores[i]) if i < len(scores) else 0.0

            rows.append(
                {
                    "para_id": para_id,
                    "sent_id": f"{para_id}_S{i:03d}",
                    "sentence": sentence,
                    "label": int(i in top_indices),
                    "salience_score": score,
                    **feature_map,
                }
            )

    return rows


def _write_sentence_feature_csvs(output_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = _build_sentence_feature_rows(payload)
    if not rows:
        return {"sentences": 0, "features_path": "", "raw_path": ""}

    df = pd.DataFrame(rows)
    features_path = output_path.with_name("sentences_with_features.csv")
    raw_path = output_path.with_name("sentences_raw.csv")

    df.to_csv(features_path, index=False)
    df[["para_id", "sent_id", "sentence", "label"]].to_csv(raw_path, index=False)

    return {
        "sentences": int(len(df)),
        "features_path": str(features_path),
        "raw_path": str(raw_path),
    }


def _load_existing_metrics(eval_path: Path) -> Dict[str, Any]:
    if not eval_path.exists():
        return {}
    with eval_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _load_examples_from_pipeline_output(
    pipeline_path: Path,
    max_examples: int,
) -> List[Dict[str, Any]]:
    with pipeline_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return []

    examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        context = row.get("passage", "")
        if not isinstance(context, str) or not context.strip():
            continue

        qa_ids = row.get("qa_ids", [])
        if not isinstance(qa_ids, list):
            qa_ids = []

        examples.append(
            {
                "example_index": idx,
                "original_passage_index": row.get("original_passage_index", idx),
                "title": row.get("title", ""),
                "paragraph_index": row.get("paragraph_index", 0),
                "context": context,
                "qa_ids": [qid for qid in qa_ids if isinstance(qid, str)],
            }
        )
        if len(examples) >= max_examples:
            break

    return examples


def _resolve_eval_source_path(args: argparse.Namespace) -> Path:
    if args.eval_source_path:
        return Path(args.eval_source_path)
    return Path(args.train_path) if args.source == "train" else Path(args.dev_path)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    source_path = Path(args.dev_path) if args.source == "dev" else Path(args.train_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if args.source == "dev":
        examples = _load_squad_dev_examples(
            dev_path=source_path,
            max_examples=args.max_examples,
        )
    else:
        examples = _load_squad_train_examples(
            train_path=source_path,
            max_examples=args.max_examples,
            skip_first_passages=args.skip_first_passages,
        )

    if not examples:
        raise ValueError("No valid examples found for selected source")

    inferencer = SalienceInferencer(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        threshold=args.threshold,
        top_k=args.top_k,
        linguistic_only=getattr(args, "linguistic_only", False),
    )

    started_utc = datetime.utcnow().isoformat() + "Z"
    outputs: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(examples), desc="QG pipeline", unit="example")
    processed = 0
    errors = 0

    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        contexts = [row["context"] for row in batch]

        salience_batch = inferencer.infer_many(contexts)

        for row, salience in zip(batch, salience_batch):
            record: Dict[str, Any] = {
                "example_index": row["example_index"],
                "title": row["title"],
                "paragraph_index": row["paragraph_index"],
                "qa_ids": row["qa_ids"],
                "passage": row["context"],
                "salience": salience,
                "salience_questions": [],
                "baseline_questions": [],
                "error": None,
            }
            if "original_passage_index" in row:
                record["original_passage_index"] = row["original_passage_index"]

            try:
                salient_sentences = [
                    item["sentence"]
                    for item in salience.get("top_salient", [])
                    if isinstance(item, dict) and isinstance(item.get("sentence"), str)
                ]
                salient_scores = [
                    float(item.get("score", 0.0))
                    for item in salience.get("top_salient", [])
                    if isinstance(item, dict) and isinstance(item.get("sentence"), str)
                ]

                record["salience_questions"] = generate_questions_with_salience(
                    full_passage=row["context"],
                    salient_sentences=salient_sentences,
                    salient_scores=salient_scores,
                    num_questions=args.num_questions,
                    preferred_model=args.qg_model,
                )
                record["baseline_questions"] = generate_questions_baseline(
                    full_passage=row["context"],
                    num_questions=args.num_questions,
                    preferred_model=args.qg_model,
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                errors += 1

            outputs.append(record)
            processed += 1
            pbar.update(1)

            if args.gpu_cache_every > 0 and processed % args.gpu_cache_every == 0:
                _cleanup_gpu_cache()

            if (
                args.clear_model_cache_every > 0
                and processed % args.clear_model_cache_every == 0
            ):
                clear_generator_cache()

    pbar.close()
    finished_utc = datetime.utcnow().isoformat() + "Z"

    return {
        "metadata": {
            "source": args.source,
            "source_path": str(source_path),
            "skip_first_passages": args.skip_first_passages if args.source == "train" else 0,
            "num_examples_requested": args.max_examples,
            "num_examples_processed": processed,
            "num_errors": errors,
            "batch_size": args.batch_size,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "num_questions": args.num_questions,
            "qg_model": args.qg_model,
            "linguistic_only": bool(getattr(args, "linguistic_only", False)),
            "gpu_cache_every": args.gpu_cache_every,
            "clear_model_cache_every": args.clear_model_cache_every,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "efficiency_notes": [
                "Single SalienceInferencer instance reused across all batches.",
                "Batched salience inference via infer_many to reduce overhead.",
                "Question generation is sequential to avoid loading multiple LLMs at once.",
                "Periodic torch.cuda.empty_cache and gc.collect to limit VRAM growth.",
                "Optional full model cache clear available for long runs.",
            ],
        },
        "results": outputs,
    }


def main() -> int:
    args = parse_args()
    try:
        _validate_args(args)
        payload = run_pipeline(args)
    except Exception as exc:
        print(f"Pipeline failed: {type(exc).__name__}: {exc}")
        return 1

    output_path = Path(args.output) if args.output else _default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    error_log_path = _write_generation_error_log(output_path, payload)
    csv_summary = _write_sentence_feature_csvs(output_path, payload)

    print(f"Saved pipeline output: {output_path}")
    print(f"Saved QA error log: {error_log_path}")
    if csv_summary["sentences"] > 0:
        print(f"Saved sentence features CSV: {csv_summary['features_path']}")
        print(f"Saved sentence raw CSV: {csv_summary['raw_path']}")

    if not args.skip_evaluation:
        eval_source_path = _resolve_eval_source_path(args)
        if eval_source_path.exists():
            evaluation_output_path = output_path.with_name("evaluation.json")
            eval_cmd = [
                sys.executable,
                "evaluation.py",
                "--pipeline-json",
                str(output_path),
                "--source-path",
                str(eval_source_path),
                "--output",
                str(evaluation_output_path),
            ]
            print(f"Running evaluation: {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd, check=False)
        else:
            print(f"Skipping evaluation: source file not found: {eval_source_path}")

    print(
        "Summary: "
        f"processed={payload['metadata']['num_examples_processed']} "
        f"errors={payload['metadata']['num_errors']}"
    )
    return 0


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
    qg_model: str = "Qwen/Qwen2.5-7B-Instruct",
    eval_source_path: str = "",
    include_full_salience: bool = True,
    include_top10_features: bool = True,
) -> None:
    """Compatibility wrapper. The implementation now lives in qg_ablations.py."""
    from qg_ablations import run_ablation_study as _run_ablation_study_impl

    return _run_ablation_study_impl(
        num_examples=num_examples,
        seed=seed,
        source=source,
        skip_first_passages=skip_first_passages,
        output_dir=output_dir,
        model_artifact_dir=model_artifact_dir,
        overwrite=overwrite,
        batch_size=batch_size,
        threshold=threshold,
        top_k=top_k,
        num_questions=num_questions,
        qg_model=qg_model,
        eval_source_path=eval_source_path,
        include_full_salience=include_full_salience,
        include_top10_features=include_top10_features,
    )


if __name__ == "__main__":
    raise SystemExit(main())
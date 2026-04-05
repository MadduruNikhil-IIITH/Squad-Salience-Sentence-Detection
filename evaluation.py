import argparse
import json
import re
import statistics
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from bert_score import score as bertscore_score
from tqdm.auto import tqdm
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated questions with ROUGE-L, BERTScore against gold questions, "
            "and QA-consistency (EM/F1) against generated answers using roberta-base-squad2."
        )
    )
    parser.add_argument(
        "--pipeline-json",
        required=True,
        help="Path to qg_pipeline output JSON.",
    )
    parser.add_argument(
        "--source-path",
        default="data/train.json",
        help="Path to SQuAD source JSON used to recover gold questions/answers.",
    )
    parser.add_argument(
        "--qa-model",
        default="deepset/roberta-base-squad2",
        help="Hugging Face QA model for consistency scoring.",
    )
    parser.add_argument(
        "--qa-batch-size",
        type=int,
        default=16,
        help="Batch size for QA model inference.",
    )
    parser.add_argument(
        "--bertscore-model",
        default="roberta-base",
        help="Model used for BERTScore computation.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path.",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    # SQuAD-style normalization with a few practical additions so equivalent
    # answers match despite formatting differences.
    text = text.lower().replace("’", "'")
    # Normalize possessives before punctuation stripping (e.g., "country's").
    text = re.sub(r"\b(\w+)'s\b", r"\1", text)
    text = re.sub(r"\b(\w+)s'\b", r"\1s", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(prediction: str, ground_truth: str) -> float:
    # Standard SQuAD token-overlap F1.
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(ground_truth))


def _max_over_ground_truths(metric_fn, prediction: str, ground_truths: Sequence[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def _question_issue_reasons(
    rouge_l: float,
    qa_em: float,
    qa_consistency_f1: float,
    answer_recovery_f1: float,
) -> List[str]:
    reasons: List[str] = []
    if rouge_l < 0.35:
        reasons.append("Low lexical overlap with the closest gold question")
    if qa_consistency_f1 < 0.55:
        reasons.append("QA model answer does not match the generated answer well")
    if answer_recovery_f1 < 0.45:
        reasons.append("QA model answer is weakly aligned with the gold answers")
    if qa_em == 0.0 and qa_consistency_f1 < 0.35:
        reasons.append("Generated question likely targets a different fact than the gold question")
    return reasons


def _build_question_issue_log(
    issue_rows: Dict[str, List[Dict[str, Any]]],
    pipeline_path: Path,
    evaluated_examples: int,
    pipeline_meta: Dict[str, Any],
) -> Dict[str, Any]:
    flagged_rows: List[Dict[str, Any]] = []
    for method, rows in issue_rows.items():
        for row in rows:
            reasons = _question_issue_reasons(
                rouge_l=float(row.get("rouge_l", 0.0)),
                qa_em=float(row.get("qa_em", 0.0)),
                qa_consistency_f1=float(row.get("qa_consistency_f1", 0.0)),
                answer_recovery_f1=float(row.get("answer_recovery_f1", 0.0)),
            )
            if not reasons:
                continue

            flagged_rows.append(
                {
                    **row,
                    "method": method,
                    "reasons": reasons,
                }
            )

    flagged_rows.sort(
        key=lambda row: (
            row.get("rouge_l", 0.0),
            row.get("qa_consistency_f1", 0.0),
            row.get("answer_recovery_f1", 0.0),
        )
    )

    return {
        "meta": {
            "pipeline_json": str(pipeline_path),
            "pipeline_metadata": pipeline_meta,
            "evaluated_examples": evaluated_examples,
            "issue_count": len(flagged_rows),
            "reason_thresholds": {
                "rouge_l": 0.35,
                "qa_consistency_f1": 0.55,
                "answer_recovery_f1": 0.45,
            },
        },
        "issues": flagged_rows,
    }


def _collect_generation_errors(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    error_rows: List[Dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        error = row.get("error")
        if not error:
            continue
        error_rows.append(
            {
                "example_index": row.get("example_index"),
                "original_passage_index": row.get("original_passage_index"),
                "title": row.get("title", ""),
                "paragraph_index": row.get("paragraph_index"),
                "qa_ids": row.get("qa_ids", []),
                "error_type": error.split(":", 1)[0] if isinstance(error, str) else type(error).__name__,
                "error_message": error,
            }
        )
    return error_rows


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class GoldQA:
    question: str
    answers: List[str]


def load_squad_qa_lookup(path: Path) -> Dict[str, GoldQA]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Expected SQuAD JSON with top-level 'data'")

    lookup: Dict[str, GoldQA] = {}
    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                qa_id = qa.get("id")
                question = qa.get("question", "")
                answers = [a.get("text", "") for a in qa.get("answers", []) if a.get("text")]
                if isinstance(qa_id, str) and isinstance(question, str):
                    lookup[qa_id] = GoldQA(question=question, answers=answers)
    return lookup


class QAEvaluatorRuntime:
    """Single-load QA runtime for batched answer extraction."""

    def __init__(self, model_name: str) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )

    def answer_batch(self, questions: List[str], contexts: List[str], batch_size: int) -> List[str]:
        items = [{"question": q, "context": c} for q, c in zip(questions, contexts)]
        outputs = self._pipe(items, batch_size=batch_size)
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [str(row.get("answer", "")) for row in outputs]


def _select_best_gold_question(candidate_q: str, gold_questions: List[str]) -> Tuple[str, float]:
    if not gold_questions:
        return "", 0.0
    best_q = gold_questions[0]
    best_score = rouge_l_f1(candidate_q, best_q)
    for gq in gold_questions[1:]:
        score = rouge_l_f1(candidate_q, gq)
        if score > best_score:
            best_score = score
            best_q = gq
    return best_q, best_score


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    pipeline_path = Path(args.pipeline_json)
    source_path = Path(args.source_path)

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing pipeline file: {pipeline_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    with pipeline_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Invalid pipeline JSON: 'results' must be a list")

    generation_errors = _collect_generation_errors(results)
    pipeline_meta = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}

    qa_lookup = load_squad_qa_lookup(source_path)
    qa_runtime = QAEvaluatorRuntime(model_name=args.qa_model)

    metrics = {
        "salience": defaultdict(float),
        "baseline": defaultdict(float),
    }

    bert_candidates: Dict[str, List[str]] = {"salience": [], "baseline": []}
    bert_references: Dict[str, List[str]] = {"salience": [], "baseline": []}

    qa_questions: Dict[str, List[str]] = {"salience": [], "baseline": []}
    qa_contexts: Dict[str, List[str]] = {"salience": [], "baseline": []}
    qa_target_answers: Dict[str, List[str]] = {"salience": [], "baseline": []}
    qa_gold_answers: Dict[str, List[List[str]]] = {"salience": [], "baseline": []}

    bert_f1_scores: Dict[str, List[float]] = {"salience": [], "baseline": []}
    issue_rows: Dict[str, List[Dict[str, Any]]] = {"salience": [], "baseline": []}

    evaluated_examples = 0

    for row in tqdm(results, desc="Preparing eval records", unit="example"):
        if not isinstance(row, dict):
            continue

        passage = row.get("passage", "")
        qa_ids = row.get("qa_ids", [])
        if not isinstance(passage, str) or not passage.strip():
            continue

        gold_items = [qa_lookup[qid] for qid in qa_ids if isinstance(qid, str) and qid in qa_lookup]
        gold_questions = [g.question for g in gold_items if g.question]
        gold_answers_union: List[str] = []
        for g in gold_items:
            gold_answers_union.extend(g.answers)
        gold_answers_union = [a for a in gold_answers_union if a]

        if not gold_questions:
            continue

        evaluated_examples += 1
        salient_sentences_for_row = [
            entry.get("sentence", "")
            for entry in (row.get("salience", {}) or {}).get("top_salient", [])
            if isinstance(entry, dict) and isinstance(entry.get("sentence"), str)
        ]

        for method, key in (("salience", "salience_questions"), ("baseline", "baseline_questions")):
            generated = row.get(key, [])
            if not isinstance(generated, list):
                continue

            for question_index, item in enumerate(generated):
                if not isinstance(item, dict):
                    continue
                candidate_q = item.get("question", "")
                if not isinstance(candidate_q, str) or not candidate_q.strip():
                    continue

                # Question quality metrics: compare generated question to the
                # closest gold SQuAD question for this passage.
                best_gold_q, rouge = _select_best_gold_question(candidate_q, gold_questions)
                metrics[method]["rouge_l_sum"] += rouge
                metrics[method]["count"] += 1

                bert_candidates[method].append(candidate_q)
                bert_references[method].append(best_gold_q)

                generated_answer = item.get("answer", "")
                if not isinstance(generated_answer, str):
                    generated_answer = str(generated_answer)

                issue_rows[method].append(
                    {
                        "example_index": row.get("example_index"),
                        "original_passage_index": row.get("original_passage_index"),
                        "title": row.get("title", ""),
                        "paragraph_index": row.get("paragraph_index"),
                        "qa_ids": row.get("qa_ids", []),
                        "method": method,
                        "question_index": question_index,
                        "generated_question_count": len(generated),
                        "candidate_question": candidate_q,
                        "gold_question": best_gold_q,
                        "rouge_l": round(rouge, 4),
                        "generated_answer": generated_answer,
                        "target_model_question": candidate_q,
                        "target_model_answer": generated_answer,
                        "all_generated_questions_for_example": generated,
                        "salient_sentences": salient_sentences_for_row,
                    }
                )

                # QA-consistency: ask the QA model the generated question on the source
                # passage, then compare the model's extracted answer to the generated
                # answer provided in the pipeline JSON (item["answer"]).
                qa_questions[method].append(candidate_q)
                qa_contexts[method].append(passage)
                qa_target_answers[method].append(generated_answer)
                qa_gold_answers[method].append(gold_answers_union)

    # BERTScore (question similarity) in one pass per method for speed.
    for method in ("salience", "baseline"):
        cands = bert_candidates[method]
        refs = bert_references[method]
        if cands:
            _, _, f1 = bertscore_score(
                cands,
                refs,
                model_type=args.bertscore_model,
                lang="en",
                batch_size=max(8, args.qa_batch_size),
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False,
            )
            bert_f1_scores[method].extend(float(x) for x in f1.tolist())

    # QA consistency in batched inference.
    for method in ("salience", "baseline"):
        questions = qa_questions[method]
        contexts = qa_contexts[method]
        target_answers = qa_target_answers[method]
        gold_answers = qa_gold_answers[method]
        if not questions:
            continue

        preds: List[str] = []
        for start in tqdm(
            range(0, len(questions), args.qa_batch_size),
            desc=f"QA consistency ({method})",
            unit="batch",
        ):
            q_batch = questions[start : start + args.qa_batch_size]
            c_batch = contexts[start : start + args.qa_batch_size]
            preds.extend(
                qa_runtime.answer_batch(
                    questions=q_batch,
                    contexts=c_batch,
                    batch_size=args.qa_batch_size,
                )
            )

        for idx, (pred, target, golds) in enumerate(zip(preds, target_answers, gold_answers)):
            em = _exact_match(pred, target)
            f1 = _token_f1(pred, target)
            metrics[method]["qa_em_sum"] += em
            metrics[method]["qa_f1_sum"] += f1

            recovery_f1 = _max_over_ground_truths(_token_f1, pred, golds)
            metrics[method]["answer_recovery_f1_sum"] += recovery_f1

            if idx < len(issue_rows[method]):
                issue_rows[method][idx]["qa_prediction"] = pred
                issue_rows[method][idx]["qa_em"] = round(em, 4)
                issue_rows[method][idx]["qa_consistency_f1"] = round(f1, 4)
                issue_rows[method][idx]["answer_recovery_f1"] = round(recovery_f1, 4)

    summary: Dict[str, Any] = {
        "meta": {
            "pipeline_json": str(pipeline_path),
            "source_path": str(source_path),
            "qa_model": args.qa_model,
            "qa_batch_size": args.qa_batch_size,
            "bertscore_model": args.bertscore_model,
            "evaluated_examples": evaluated_examples,
            "generation_error_count": len(generation_errors),
            "generation_error_log": str(pipeline_path.with_name("qa_generation_errors.json")),
            "question_issue_count": 0,
            "question_issue_log": str(pipeline_path.with_name("qa_generation_issues.json")),
            "efficiency_notes": [
                "QA model is loaded once and reused.",
                "QA consistency is computed in batches.",
                "BERTScore is computed in one pass per method.",
                "Gold lookup map is pre-built for O(1) QA-id access.",
            ],
        },
        "metrics": {},
    }

    for method in ("salience", "baseline"):
        count = int(metrics[method]["count"])
        if count == 0:
            summary["metrics"][method] = {
                "count": 0,
                "rouge_l": 0.0,
                "bertscore_f1": 0.0,
                "qa_em": 0.0,
                "qa_f1": 0.0,
                "qa_consistency_f1": 0.0,
                "answer_recovery_f1": 0.0,
            }
            continue

        summary["metrics"][method] = {
            "count": count,
            "rouge_l": round(metrics[method]["rouge_l_sum"] / count, 4),
            "bertscore_f1": round(statistics.mean(bert_f1_scores[method]), 4)
            if bert_f1_scores[method]
            else 0.0,
            "qa_em": round(metrics[method]["qa_em_sum"] / count, 4),
            # Backward-compatible alias for QA consistency F1.
            "qa_f1": round(metrics[method]["qa_f1_sum"] / count, 4),
            # Measures agreement between QA-model prediction and generated answer.
            "qa_consistency_f1": round(metrics[method]["qa_f1_sum"] / count, 4),
            # Measures agreement between QA-model prediction and SQuAD gold answers.
            "answer_recovery_f1": round(metrics[method]["answer_recovery_f1_sum"] / count, 4),
        }

    issue_log = _build_question_issue_log(
        issue_rows,
        pipeline_path=pipeline_path,
        evaluated_examples=evaluated_examples,
        pipeline_meta=pipeline_meta,
    )
    issue_log_path = pipeline_path.with_name("qa_generation_issues.json")
    with issue_log_path.open("w", encoding="utf-8") as f:
        json.dump(issue_log, f, indent=2, ensure_ascii=True)

    summary["meta"]["question_issue_count"] = issue_log["meta"]["issue_count"]
    summary["meta"]["question_issue_log"] = str(issue_log_path)

    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = evaluate(args)
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"evaluation_{Path(args.pipeline_json).stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"Saved evaluation: {output_path}")
    print("Average QA consistency metrics:")
    for method, vals in summary["metrics"].items():
        print(
            f"{method}: count={vals['count']} rouge_l={vals['rouge_l']:.4f} "
            f"bertscore_f1={vals['bertscore_f1']:.4f} qa_em={vals['qa_em']:.4f} "
            f"qa_f1={vals['qa_f1']:.4f} "
            f"qa_consistency_f1={vals['qa_consistency_f1']:.4f} "
            f"answer_recovery_f1={vals['answer_recovery_f1']:.4f}"
        )
        print(
            f"  {method} QA-EM={vals['qa_em']:.4f} "
            f"QA-F1={vals['qa_f1']:.4f} "
            f"QA-Consistency-F1={vals['qa_consistency_f1']:.4f} "
            f"Answer-Recovery-F1={vals['answer_recovery_f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
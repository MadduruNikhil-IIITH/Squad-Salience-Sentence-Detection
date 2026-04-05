import json
from pathlib import Path

import nltk

from salience_inference import SalienceInferencer

THRESHOLDS = [0.55, 0.60, 0.65]
TOP_K = 3
SKIP = 2000
NUM = 50


def main() -> None:
    payload = json.loads(Path("data/train.json").read_text(encoding="utf-8"))
    paragraphs = []
    for article in payload["data"]:
        for para in article["paragraphs"]:
            paragraphs.append(para)

    indices = list(range(SKIP, min(SKIP + NUM, len(paragraphs))))

    inferencer = SalienceInferencer(
        model_path="results/run_2000_passages/model.pkl",
        scaler_path="results/run_2000_passages/scaler.pkl",
        threshold=0.0,
        top_k=TOP_K,
    )

    summary = {
        t: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "total_salient": 0,
            "passages_with_any": 0,
        }
        for t in THRESHOLDS
    }

    rows = []
    for idx in indices:
        para = paragraphs[idx]
        context = para["context"]
        sents = nltk.sent_tokenize(context)

        spans = [
            (a["answer_start"], a["answer_start"] + len(a["text"]))
            for qa in para["qas"]
            for a in qa["answers"]
        ]

        gt = []
        pos = 0
        for i, sent in enumerate(sents):
            end = pos + len(sent)
            if any(not (end <= st or pos >= en) for st, en in spans):
                gt.append(i)
            pos = end + 1
        gt_set = set(gt)

        out = inferencer.infer(context)
        scores = out["scores"]

        row = {
            "original_passage_index": idx,
            "ground_truth": sorted(gt_set),
            "pred": {},
        }

        for thr in THRESHOLDS:
            candidates = [i for i, sc in enumerate(scores) if sc > thr]
            candidates_sorted = sorted(candidates, key=lambda i: scores[i], reverse=True)
            pred = candidates_sorted[:TOP_K]
            pred_set = set(pred)

            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            summary[thr]["TP"] += tp
            summary[thr]["FP"] += fp
            summary[thr]["FN"] += fn
            summary[thr]["total_salient"] += len(pred)
            if pred:
                summary[thr]["passages_with_any"] += 1

            row["pred"][str(thr)] = pred

        rows.append(row)

    report = {
        "settings": {
            "skip_first_passages": SKIP,
            "num_passages": len(indices),
            "top_k": TOP_K,
            "thresholds": THRESHOLDS,
        },
        "metrics": {},
        "sample_rows": rows[:10],
    }

    for thr in THRESHOLDS:
        tp = summary[thr]["TP"]
        fp = summary[thr]["FP"]
        fn = summary[thr]["FN"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        report["metrics"][str(thr)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_salient": summary[thr]["total_salient"],
            "passages_with_any": summary[thr]["passages_with_any"],
        }

    out_path = Path("results/inference/threshold_eval_50_skip2000_top3.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    for thr in THRESHOLDS:
        m = report["metrics"][str(thr)]
        print(
            f"thr={thr:.2f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} TP={m['tp']} FP={m['fp']} FN={m['fn']} "
            f"salient={m['total_salient']} passages_with_any={m['passages_with_any']}"
        )


if __name__ == "__main__":
    main()

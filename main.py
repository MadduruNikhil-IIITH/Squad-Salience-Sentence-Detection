import os
import json
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

from ablation import run_ablation
from data_stats_and_sampling import load_and_visualize
from feature_extractor import extract_linguistic_features
from surprisal import get_surprisal_features
from classifier import train_model

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)
os.makedirs("results", exist_ok=True)

print("CUDA Available:", torch.cuda.is_available())

# === CONFIG ===
MAX_PARAGRAPHS = 2000 # Set to None to use all passages

# Load + visualize
paragraphs, passage_count, total_available, stats = load_and_visualize(max_paragraphs=MAX_PARAGRAPHS)
run_folder = f"results/run_{passage_count}_passages"
os.makedirs(run_folder, exist_ok=True)

# === GLOBAL STATS ===
STATS_FILE = "results/run_statistics.json"
if os.path.exists(STATS_FILE):
    with open(STATS_FILE, "r") as f:
        all_runs = json.load(f)
else:
    all_runs = {
        "total_available_passages": total_available,
        "runs": {}
    }

print(f"\nRUN FOLDER: {run_folder}")

# === PROCESSING ===
all_rows = []
pid = 0

print("Processing sentences with GPU-accelerated surprisal...")
for para in tqdm(paragraphs, desc="Passages"):
    context = para["context"]
    sents = nltk.sent_tokenize(context)
    spans = [(a["answer_start"], a["answer_start"] + len(a["text"]))
             for qa in para["qas"] for a in qa["answers"]]
    
    char_pos = 0
    for i, sent in enumerate(sents):
        sent = sent.strip()
        if len(sent) < 5:
            char_pos += len(sent) + 1
            continue
            
        end = char_pos + len(sent)
        label = int(any(not(end <= s or char_pos >= e) for s, e in spans))
        
        ling_features = extract_linguistic_features(sent, i+1, len(sents))
        surp_features = get_surprisal_features(sent)  # Full token-level + n-grams
        
        all_rows.append({
            "para_id": f"P{pid:06d}",
            "sent_id": f"P{pid:06d}_S{i:03d}",
            "sentence": sent,
            "label": label,
            **ling_features,
            **surp_features
        })
        char_pos += len(sent) + 1
    pid += 1

# === SAVE DATA ===
df = pd.DataFrame(all_rows)
df.to_csv(f"{run_folder}/sentences_with_features.csv", index=False)
df[["para_id","sent_id","sentence","label"]].to_csv(f"{run_folder}/sentences_raw.csv", index=False)

# === CLASS SKEWNESS GRAPH ===
plt.figure(figsize=(8, 6))
counts = df["label"].value_counts().sort_index()
sns.barplot(x=counts.index, y=counts.values, palette="viridis")
plt.title(f"Class Distribution – {passage_count:,} Passages")
plt.xlabel("0 = Non-Answer | 1 = Answer")
plt.ylabel("Number of Sentences")
plt.xticks([0, 1], ["Non-Answer", "Answer"])
plt.tight_layout()
skew_path = f"{run_folder}/class_imbalance_skewness.png"
plt.savefig(skew_path, dpi=300, bbox_inches='tight')
plt.close()

# === TRAIN MODEL ===
results = train_model(df, run_folder)

# === UPDATE GLOBAL JSON ===
run_key = str(passage_count)
all_runs["runs"][run_key] = {
    "stats": {**stats},
    "run_folder": run_folder,
    "passages_used": passage_count,
    "total_sentences": len(df),
    "answer_sentences": int(df["label"].sum()),
    "accuracy": results["accuracy"],
    "f1_answer": results["f1"],
    "precision": results["precision"],
    "recall": results["recall"],
    "top_features": results["top_features"],
    "files": {
        "raw": "sentences_raw.csv",
        "features": "sentences_with_features.csv",
        "skewness": "class_imbalance_skewness.png",
        "model": "model.pkl"
    }
}

with open(STATS_FILE, "w") as f:
    json.dump(all_runs, f, indent=2)

print(f"Run folder: {run_folder}")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Results saved in: {STATS_FILE}")

run_ablation(run_folder=run_folder)
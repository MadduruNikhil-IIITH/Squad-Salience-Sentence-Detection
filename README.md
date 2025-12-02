# CL2 Project

This repository contains code to analyze passages from the SQuAD dataset and build sentence-level features (linguistic + surprisal) to classify whether a sentence contains the answer.

Overview
- The pipeline loads SQuAD-formatted JSON (data/train.json), extracts sentences, computes surprisal features using GPT-2/BERT (using HuggingFace Transformers), extracts hand-crafted linguistic features, trains a classifier (logistic regression), and runs ablation and PCA analyses.
- The project supports GPU acceleration for surprisal calculation if a CUDA-capable GPU is available.

Folder structure
- `ablation.py` — Run ablation experiments using top features and save metrics to `results`.
- `classifier.py` — Train a logistic regression model on features and save `model.pkl`, `scaler.pkl`, and `top20_features.csv`.
- `data_stats_and_sampling.py` — Load SQuAD `train.json`, generate dataset visualizations and simple dataset stats.
- `feature_extractor.py` — Cleanup text and compute linguistic features used by the classifier.
- `main.py` — Orchestrates the full pipeline (load, feature extraction, surprisal, training, results & graphs).
- `surprisal.py` — Compute token-level surprisal using GPT-2 and BERT. Uses transformers and PyTorch; can use GPU if available.
- `data/` — Place the `train.json` and `dev.json` (SQuAD) files here.
- `results/` — Output directory with `run_<N>_passages` subfolders (contains raw sentences, features, plots, model files, etc.).
- `report.pdf` — Project report.
- `README.md` — This file.
- `presentation_slides.pdf` — Slides for project presentation.
- `requirements.txt` — List of dependencies (optional).


## Final Results – Performance Across Training Data Size

| Passages Sampled | Total Sentences | Answer Ratio | Accuracy | F1 (Answer Class) |
|------------------|-----------------|--------------|----------|-------------------|
| 100              | 577             | 73.5%        | **70.69%**   | **0.7927** (best) |
| 250              | 1,236           | 66.6%        | 63.31%   | 0.7093            |
| 500              | 2,277           | 68.3%        | 66.01%   | 0.7404            |
| 1,000            | 4,485           | 63.8%        | 69.12%   | 0.7480            |
| **2,000**        | **8,328**       | **62.2%**    | **69.27%**   | **0.7377**        |

**Best model**: 2,000 passages → **69.27% accuracy**, **0.7377 F1** on the Answer class  
Ablation with only top-10 features: **68.29% accuracy / 0.7360 F1** → negligible drop!

## Feature List (26 Total)

| Category              | Features |
|-----------------------|--------|
| **Surface**           | `sentence_length_words`, `sentence_position`, `sentence_position_norm` |
| **Lexical**           | `type_token_ratio`, `lexical_density` |
| **POS Ratios**        | `noun_ratio`, `verb_ratio`, `adj_ratio`, `pronoun_ratio` |
| **Discourse**         | `named_entity_density`, causal/contrast marker ratios |
| **Surprisal – GPT-2** (CUDA) | `gpt2_surprisal_mean`, `sum`, `std`, `var`, `min`, `max` |
| **Surprisal – BERT**  (CUDA) | `bert_surprisal_mean`, `sum`, `std`, `var`, `min`, `max` |

**Top 3 most predictive features**:
1. `sentence_position` (coefficient: -0.796) → answers appear early
2. `gpt2_surprisal_sum` (+0.663)
3. `gpt2_surprisal_var` (-0.444)

# Getting started

1. Create a virtual environment (optional, recommended):

```powershell
python -m venv menv
./menv/Scripts/Activate.ps1
```

2. Install dependencies: copy the dependencies below into a `requirements.txt` or install manually:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # or pick the right CUDA version from pytorch.org
pip install transformers nltk pandas scikit-learn matplotlib seaborn joblib tqdm
```

Tip: visit https://pytorch.org/ to select the correct wheel for your CUDA version (e.g., `cu118`, `cu121`, or `cpu`).

3. Download the SQuAD dataset files into the `data/` directory (`train.json` and `dev.json`).

4. Run the full pipeline:

```powershell
python main.py
```

Notes
- `main.py` has a `MAX_PARAGRAPHS` variable near the top — adjust this to control how many SQuAD passages are used for a run (default is 2000). The script will create a `results/run_<N>_passages` subfolder with outputs.
- Surprisal computation uses Transformers and can be slow on CPU. If you have a CUDA GPU, PyTorch will use it automatically; otherwise CPU will be used.
- For reproducible results, consider creating a `requirements.txt` file with pinned versions.



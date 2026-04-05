# Squad Salience Sentence Detection Project

This repository contains code to analyze passages from the SQuAD dataset and build sentence-level features (linguistic + surprisal) to classify whether a sentence contains the answer.

## Python Version

- Recommended: **Python 3.11**
- Tested environment: `conda` env `squad-salience` with `python 3.11.15`

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
- `generate_plots.py` — Generate additional plots from results (feature importance, scaling metrics, prediction histograms).
- `presentation_slides.pdf` — Slides for project presentation.
- `requirements.txt` — List of pinned dependencies from the `squad-salience` environment.


## Final Results – Performance Across Training Data Size

| Passages Sampled | Total Sentences | Answer Ratio | Accuracy | F1     |
|------------------|-----------------|--------------|----------|-------------------|
| 100              | 577             | 73.5%        | **70.69%**   | **0.7927**  |
| 250              | 1,236           | 66.6%        | 63.31%   | 0.7093            |
| 500              | 2,277           | 68.3%        | 66.01%   | 0.7404            |
| 1,000            | 4,485           | 63.8%        | 69.12%   | 0.7480            |
| **2,000**        | **8,328**       | **62.2%**    | **69.27%**   | **0.7377**        |

- **Best model**: 2,000 passages → **69.27% accuracy**, **0.7377 F1** on the Answer class  
 - Ablation with only top-10 features: **68.29% accuracy / 0.7360 F1** → negligible drop!
 - ** No Suprisal features**: **69.21% accuracy / 0.7416 F1** → surprisal features help, but linguistic features are strong!

## Ablation Results – 750 Passages

The 750-passage ablation run compares the full feature set against reduced feature variants.

| Setting | Count | ROUGE-L | BERTScore F1 | QA EM | QA Consistency F1 |
|---------|-------|---------|--------------|-------|-------------------|
| Top-10 features | 1500 | 0.3626 | 0.9076 | 0.5160 | 0.7718 |
| No surprisal | 1500 | 0.3600 | 0.9077 | 0.5127 | 0.7644 |

- The gap between the two settings is small, which suggests the linguistic features carry most of the signal.
- Keeping surprisal features still gives a modest lift in QA consistency on this run.

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

# Getting Started

1. Create and activate a conda environment (recommended):

```powershell
conda create -n squad-salience python=3.11 -y
conda activate squad-salience
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. If you need a specific CUDA/CPU PyTorch build, install torch packages from https://pytorch.org/get-started/locally/ and then reinstall the remaining dependencies:

```powershell
pip install -r requirements.txt
```

4. Download SQuAD files into `data/`:
- Required files: `data/train.json` and `data/dev.json`
- Source: https://rajpurkar.github.io/SQuAD-explorer/

5. Run the full pipeline:

```powershell
python main.py
```

## Common Commands

```powershell
python main.py
python ablation.py
python pca.py
python salience_inference.py
python qg_pipeline.py --help
python evaluation.py --help
```

## Notes

- `main.py` has a `MAX_PARAGRAPHS` variable near the top. Adjust it to control the number of SQuAD passages per run (default: 2000).
- Surprisal computation can be slow on CPU. If CUDA is available and your torch build supports it, GPU will be used automatically.
- `requirements.txt` is pinned for reproducibility to the current `squad-salience` setup.
- `bitsandbytes` is included by default for the quantized LLM loading paths used in the repository.



"""
Train linguistic-only models for all dataset sizes.
Consolidated to use training.py module.
"""

import pandas as pd
from pathlib import Path
from training import train_linguistic_model

RUNS = [100, 250, 500, 1000, 2000]


def train_linguistic_only_for_size(run_size: int) -> bool:
    """Train linguistic-only model for a specific run size."""
    run_folder = f"results/run_{run_size}_passages"
    csv_path = Path(run_folder) / "sentences_with_features.csv"
    
    if not csv_path.exists():
        print(f"✗ SKIP run_{run_size}: {csv_path} not found")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(df)} rows from run_{run_size}")
        
        # Train linguistic-only model with verbose output
        train_linguistic_model(df, run_folder, verbose=True)
        return True
        
    except Exception as e:
        print(f"✗ Error training run_{run_size}: {e}")
        return False

def main():
    print(f"\n{'='*60}")
    print("TRAINING LINGUISTIC-ONLY MODELS FOR ALL RUN SIZES")
    print(f"{'='*60}")
    
    results = {}
    for run_size in RUNS:
        success = train_linguistic_only_for_size(run_size)
        results[run_size] = success
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for run_size in RUNS:
        status = "✓ SUCCESS" if results[run_size] else "✗ SKIPPED"
        print(f"run_{run_size}_passages: {status}")

if __name__ == "__main__":
    main()

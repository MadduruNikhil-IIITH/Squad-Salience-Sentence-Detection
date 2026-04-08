import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import nltk
import numpy as np
import pandas as pd

from feature_extractor import extract_linguistic_features
from surprisal import get_surprisal_features


_INFERENCER_CACHE: Dict[Tuple[str, str, float, int, bool], "SalienceInferencer"] = {}


class _LRUCache:
    """Small LRU cache to speed up repeated sentence feature extraction."""

    def __init__(self, max_size: int = 4096) -> None:
        self.max_size = max_size
        self._store: "OrderedDict[Tuple[str, int, int], Dict[str, float]]" = OrderedDict()

    def get(self, key: Tuple[str, int, int]) -> Optional[Dict[str, float]]:
        value = self._store.get(key)
        if value is None:
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: Tuple[str, int, int], value: Dict[str, float]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)


class SalienceInferencer:
    """
    Sentence salience inference utility.

    Loads model artifacts once, computes features per sentence, and returns:
    - sentence-wise salience probabilities
    - top-k salient sentences above a threshold
    - a passage with [SALIENT] sentence markers
    """

    _nltk_ready = False

    def __init__(
        self,
        model_path: str = "model.pkl",
        scaler_path: Optional[str] = None,
        threshold: float = 0.65,
        top_k: int = 3,
        cache_size: int = 4096,
        linguistic_only: bool = False,
    ) -> None:
        self.model_path = os.path.abspath(model_path)
        self.scaler_path = self._resolve_scaler_path(self.model_path, scaler_path)
        self.threshold = threshold
        self.top_k = top_k
        self.linguistic_only = linguistic_only
        self._feature_cache = _LRUCache(max_size=cache_size)

        self.model = self._load_artifact(self.model_path, name="model")
        self.scaler = self._load_optional_artifact(self.scaler_path, name="scaler")
        self._expected_features: Optional[List[str]] = None
        if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
            self._expected_features = list(self.scaler.feature_names_in_)

        self._ensure_nltk_resources()

    @staticmethod
    def _resolve_scaler_path(model_path: str, scaler_path: Optional[str]) -> str:
        if scaler_path:
            return os.path.abspath(scaler_path)
        model_dir = os.path.dirname(model_path) or "."
        return os.path.join(model_dir, "scaler.pkl")

    @staticmethod
    def _load_artifact(path: str, name: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} artifact not found at: {path}")
        return joblib.load(path)

    @staticmethod
    def _load_optional_artifact(path: str, name: str):
        if os.path.exists(path):
            return joblib.load(path)
        return None

    @classmethod
    def _ensure_nltk_resources(cls) -> None:
        if cls._nltk_ready:
            return
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("universal_tagset", quiet=True)
        cls._nltk_ready = True

    def _sent_tokenize(self, passage: str) -> List[str]:
        sentences = [s.strip() for s in nltk.sent_tokenize(passage or "")]
        return [s for s in sentences if s]

    def _sentence_features(self, sentence: str, idx_0_based: int, total_sentences: int, full_passage: str = "") -> Dict[str, float]:
        """Extract features for a sentence, with optional RST passage context."""
        # Update cache key to include passage hash for accuracy
        import hashlib
        passage_hash = hashlib.sha256(full_passage.encode()).hexdigest()[:8] if full_passage else ""
        key = (sentence, idx_0_based + 1, total_sentences, passage_hash)  # idx_1_based for compatibility
        
        cached = self._feature_cache.get(key)
        if cached is not None:
            return dict(cached)

        # Extract features with RST support
        features = extract_linguistic_features(
            sentence,
            idx_0_based + 1,  # 1-indexed position
            total_sentences,
            include_surprisal=not self.linguistic_only,
            include_rst=True,
            full_passage=full_passage,
            sent_index=idx_0_based
        )

        # Keep explicit reuse of surprisal module for compatibility if extractor output changes.
        if not self.linguistic_only and ("gpt2_surprisal_mean" not in features or "bert_surprisal_mean" not in features):
            surprisal_features = get_surprisal_features(sentence)
            features.update(surprisal_features)

        self._feature_cache.put(key, dict(features))
        return features

    def _build_feature_matrix(self, sentences: Sequence[str], passage: str = "") -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
        total = len(sentences)
        records: List[Dict[str, float]] = [
            self._sentence_features(sentence, i, total, full_passage=passage)
            for i, sentence in enumerate(sentences)
        ]

        X = pd.DataFrame.from_records(records)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        if self._expected_features is not None:
            for col in self._expected_features:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self._expected_features]

        return X, records

    def infer(self, passage: str) -> Dict[str, object]:
        sentences = self._sent_tokenize(passage)
        if not sentences:
            return {
                "sentences": [],
                "features": [],
                "scores": [],
                "top_salient": [],
                "augmented_passage": "",
                "threshold": self.threshold,
            }

        X, feature_records = self._build_feature_matrix(sentences, passage=passage)
        X_input = self.scaler.transform(X) if self.scaler is not None else X.values

        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(X_input)[:, 1]
        else:
            # Fallback for estimators without predict_proba.
            scores = self.model.decision_function(X_input)
            scores = 1.0 / (1.0 + np.exp(-scores))

        scores = np.asarray(scores, dtype=float)

        candidate_idx = np.where(scores > self.threshold)[0]
        ranked_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        top_idx = ranked_idx[: self.top_k]

        top_salient = [
            {
                "sentence_index": int(i),
                "sentence": sentences[i],
                "score": float(scores[i]),
            }
            for i in top_idx
        ]

        marked_sentences = [
            f"[SALIENT] {sent}" if scores[i] > self.threshold else sent
            for i, sent in enumerate(sentences)
        ]

        return {
            "sentences": sentences,
            "features": feature_records,
            "scores": [float(s) for s in scores],
            "top_salient": top_salient,
            "augmented_passage": " ".join(marked_sentences),
            "threshold": self.threshold,
            "linguistic_only": self.linguistic_only,
        }

    def infer_many(self, passages: Sequence[str]) -> List[Dict[str, object]]:
        return [self.infer(p) for p in passages]


def infer_salience(
    passage: str,
    model_path: str = "model.pkl",
    scaler_path: Optional[str] = None,
    threshold: float = 0.65,
    top_k: int = 3,
    linguistic_only: bool = False,
) -> Dict[str, object]:
    resolved_model = os.path.abspath(model_path)
    resolved_scaler = os.path.abspath(scaler_path) if scaler_path else ""
    cache_key = (resolved_model, resolved_scaler, float(threshold), int(top_k), bool(linguistic_only))

    inferencer = _INFERENCER_CACHE.get(cache_key)
    if inferencer is None:
        inferencer = SalienceInferencer(
            model_path=model_path,
            scaler_path=scaler_path,
            threshold=threshold,
            top_k=top_k,
            linguistic_only=linguistic_only,
        )
        _INFERENCER_CACHE[cache_key] = inferencer

    return inferencer.infer(passage)

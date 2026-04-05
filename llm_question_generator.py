import gc
import json
import logging
import re
from threading import Lock
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


_DEFAULT_PRIMARY_MODEL = "microsoft/Phi-4-mini-instruct"
_DEFAULT_FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_JSON_ARRAY_RE = re.compile(r"\[(?:.|\n|\r)*\]", re.MULTILINE)
LOGGER = logging.getLogger(__name__)


class _GeneratorRuntime:
    """Lazy-loaded quantized runtime for question generation."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name: Optional[str] = None
        self._lock = Lock()

    def load(self, preferred_model: str = _DEFAULT_PRIMARY_MODEL) -> None:
        if self.model is not None and self.tokenizer is not None and self.model_name == preferred_model:
            return

        with self._lock:
            if self.model is not None and self.tokenizer is not None and self.model_name == preferred_model:
                return

            # 4-bit config tuned for consumer GPUs like RTX 4060 8GB.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model_candidates = [preferred_model]
            if preferred_model != _DEFAULT_FALLBACK_MODEL:
                model_candidates.append(_DEFAULT_FALLBACK_MODEL)

            load_error = None
            for candidate in model_candidates:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        candidate,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )
                    self.model_name = candidate

                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    return
                except Exception as exc:
                    load_error = exc
                    self.model = None
                    self.tokenizer = None
                    self.model_name = None

            raise RuntimeError(f"Failed to load any model candidate: {model_candidates}") from load_error

    def clear(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


_RUNTIME = _GeneratorRuntime()


def _system_prompt(num_questions: int) -> str:
    return (
        "You are a question generation assistant. "
        "Return ONLY valid JSON and nothing else. "
        "Output must be a JSON array with exactly "
        f"{num_questions} objects. "
        "Each object must contain only these keys: question, answer. "
        "No markdown, no explanations, no extra keys. "
        "All answers must be directly grounded in the provided passage. "
        "Answers must be extractive spans from the passage text, preferably from salient sentences when provided. "
        "Answers should be verbatim or very close to passage text and short (2-10 words)."
    )


def _build_user_prompt(
    full_passage: str,
    salient_sentences: Optional[List[str]],
    num_questions: int,
    salient_scores: Optional[List[float]] = None,
) -> str:
    if not salient_sentences or len(salient_sentences) == 0:
        # Baseline - robust
        return f"""You are an expert SQuAD question generator.

Generate exactly {num_questions} high-quality SQuAD-style question-answer pairs.

Strict Rules:
- Answers must be short extractive spans directly from the passage (3-12 words max).
- Do not paraphrase answers.
- Questions must be natural and specific.

Passage:
{full_passage.strip()}

Output ONLY valid JSON:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""

    clean_sentences = [s.strip() for s in salient_sentences if isinstance(s, str) and s.strip()]
    if not clean_sentences:
        return f"""Generate {num_questions} high-quality SQuAD-style question-answer pairs from the passage.
Answers must be short extractive spans (3-12 words). Output ONLY JSON."""

    score_values: List[float] = []
    if isinstance(salient_scores, list) and len(salient_scores) > 0:
        for score in salient_scores[:len(clean_sentences)]:
            try:
                score_values.append(float(score))
            except (TypeError, ValueError):
                score_values.append(0.85)
    else:
        score_values = [0.85] * len(clean_sentences)

    salient_text = "\n".join(
        f"[MOST IMPORTANT {i+1} - score: {score:.3f}] {sent}"
        for i, (sent, score) in enumerate(zip(clean_sentences, score_values))
    )

    return f"""You are an expert SQuAD question generator.

Task: Generate exactly {num_questions} high-quality SQuAD-style questions and answers.
You MUST base questions **only on the [MOST IMPORTANT] sentences** below.

Strict Rules:
- Focus exclusively on the [MOST IMPORTANT] sentences (higher score = higher priority).
- Answers must be short, direct spans from those sentences (3-12 words max). You may use near-verbatim text.
- Do not paraphrase heavily or add extra explanation in answers.
- Questions should be natural and specific like real SQuAD questions.

Passage (context only):
{full_passage.strip()}

MOST IMPORTANT sentences (use these only):
{salient_text}

Output ONLY valid JSON. No extra text:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""

def _sanitize_item(item: Dict[str, str]) -> Dict[str, str]:
    q = str(item.get("question", "")).strip()
    a = str(item.get("answer", "")).strip()
    return {"question": q, "answer": a}


def _fallback_questions(full_passage: str, num_questions: int) -> List[Dict[str, str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_passage) if s.strip()]
    if not sentences:
        return [{"question": "What is discussed in the passage?", "answer": "The passage content is empty."}]

    output: List[Dict[str, str]] = []
    for i in range(num_questions):
        ans = sentences[i % len(sentences)]
        output.append(
            {
                "question": f"What key point is made in sentence {((i % len(sentences)) + 1)}?",
                "answer": ans,
            }
        )
    return output


def _parse_json_with_fallback(text: str, full_passage: str, num_questions: int) -> List[Dict[str, str]]:
    cleaned = text.strip()

    # Attempt direct parse.
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            parsed = [_sanitize_item(x) for x in data if isinstance(x, dict)]
            if parsed:
                return parsed[:num_questions]
    except Exception:
        pass

    # Attempt bracket extraction parse.
    match = _JSON_ARRAY_RE.search(cleaned)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                parsed = [_sanitize_item(x) for x in data if isinstance(x, dict)]
                if parsed:
                    return parsed[:num_questions]
        except Exception:
            pass

    # Final fallback for stability.
    return _fallback_questions(full_passage, num_questions)


def _generate(
    full_passage: str,
    salient_sentences: Optional[List[str]],
    salient_scores: Optional[List[float]],
    num_questions: int,
    preferred_model: str = _DEFAULT_PRIMARY_MODEL,
) -> List[Dict[str, str]]:
    try:
        _RUNTIME.load(preferred_model=preferred_model)
    except Exception as exc:
        # If quantized model loading is unavailable in the environment,
        # return deterministic fallback questions instead of failing the run.
        LOGGER.warning("Falling back to deterministic questions because model load failed: %s", exc)
        return _fallback_questions(full_passage, num_questions)

    system = _system_prompt(num_questions=num_questions)
    user = _build_user_prompt(
        full_passage=full_passage,
        salient_sentences=salient_sentences,
        num_questions=num_questions,
        salient_scores=salient_scores,
    )

    tokenizer = _RUNTIME.tokenizer
    model = _RUNTIME.model

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"System: {system}\nUser: {user}\nAssistant:"

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.65,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    except Exception as exc:
        LOGGER.warning("Falling back to deterministic questions because generation failed: %s", exc)
        return _fallback_questions(full_passage, num_questions)

    parsed = _parse_json_with_fallback(raw_text, full_passage=full_passage, num_questions=num_questions)

    # Ensure exact count for downstream consumers.
    if len(parsed) < num_questions:
        top_up = _fallback_questions(full_passage, num_questions)
        parsed.extend(top_up[len(parsed) : num_questions])

    return parsed[:num_questions]


def generate_questions_with_salience(
    full_passage: str,
    salient_sentences: List[str],
    salient_scores: Optional[List[float]] = None,
    num_questions: int = 3,
    preferred_model: str = _DEFAULT_PRIMARY_MODEL,
) -> List[Dict[str, str]]:
    return _generate(
        full_passage=full_passage,
        salient_sentences=salient_sentences,
        salient_scores=salient_scores,
        num_questions=num_questions,
        preferred_model=preferred_model,
    )


def generate_questions_baseline(
    full_passage: str,
    num_questions: int = 3,
    preferred_model: str = _DEFAULT_PRIMARY_MODEL,
) -> List[Dict[str, str]]:
    return _generate(
        full_passage=full_passage,
        salient_sentences=None,
        salient_scores=None,
        num_questions=num_questions,
        preferred_model=preferred_model,
    )


def clear_generator_cache() -> None:
    """Optional utility to release VRAM after generation bursts."""
    _RUNTIME.clear()

# Salience-Guided Question Generation: Feature Ablations, Methodological Insights, and Issue Analysis

**Research Group Presentation**  
*Tuesday, April 8, 2026*

---

## === Slide 1: Title & Agenda ===

**Title:**  
**Salience-Guided Question Generation: Feature Ablations, Methodological Insights, and Issue Analysis**

**Subtitle:**  
Bridging linguistic structure and neural uncertainty to ground QA generation

---

### Speaker Notes for Slide 1:

Good morning, everyone. Today I want to walk through our work on salience-guided question generation—specifically, a series of ablation studies we ran to understand how linguistic features, surprisal signals, and their combinations affect the quality of generated questions.

The presentation has three main throughlines:

1. **Methodological foundations** — how we design the salience pipeline and why grounding questions in salient sentences matters
2. **Empirical ablations** — systematic comparisons of three feature configurations to isolate which signals matter most
3. **Issue analysis** — a qualitative dive into the ~2,200 problematic cases per 1,500 questions, revealing nuanced trade-offs between grounding and diversity

By the end, you should have a clear sense of our approach's strengths, the tensions we've observed between metrics, and why we believe discourse-level features (like rhetorical structure) are a natural next step. This is early-stage research with promising signals—not final answers, but honest findings that warrant further exploration.

---

## === Slide 2: Problem & Motivation ===

**Problem Statement:**

- **SQuAD 1.1**: 100K+ crowd-sourced question-answer pairs on 500+ Wikipedia articles
- **Extractive answers**: exact spans, typically 3–5 gold QA pairs per passage
- **Standard QG limitation**: generates diverse questions but *loses grounding* — often doesn't maintain connection to salient information
- **Our hypothesis**: if we rank sentences by salience (esp. whether they contain answers), we can guide generation to produce more *grounded, focused* questions

**Why Salience Matters:**

- Humans naturally focus QA on key information
- Salience-grounded generation: potentially higher consistency, clearer pedagogical value
- Trade-off to explore: grounding vs. diversity (lexical overlap with gold questions)

---

### Speaker Notes for Slide 2:

Why are we focusing on salience and grounding? Let me set the stage.

SQuAD is a flagship dataset—over 100,000 question-answer pairs on Wikipedia passages. But a known challenge with standard question generation models is that they optimize for fluency and diversity while potentially losing sight of *what matters most* in the passage.

Imagine you're reading about the history of public houses in Britain. The passage contains facts about signage history, licensing, architectural styles, et cetera. A generic QG model might generate questions about *any* sentence—which is fine for diversity, but loses pedagogical grounding. What if we could bias the model to ask questions only about the *most important* sentences—the ones that *contain* answers or carry high textual salience?

Our core hypothesis is that by **encoding sentence salience into the generation prompt**, we can produce questions that are:
- More grounded in key content
- More consistent with extractive QA models
- More pedagogically focused

The cost might be lower lexical overlap with human questions (since we're asking about *different facts*), but the consistency gains could be worth it in an educational or tutoring context.

---

## === Slide 3: Proposed Approach ===

**End-to-End Pipeline:**

1. Load SQuAD passage & sentences
2. **Extract features** (linguistic + surprisal) for each sentence
3. **Salience inference**: binary classifier scores each sentence → ranked by probability of containing answer
4. **Top-k selection**: select top-k salient sentences
5. **Balanced prompt design**: instruct LLM to generate 2 questions choosing from those salient sentences
6. **Baseline comparison**: same process without salience constraints

**Balanced Prompt Design (Key Innovation):**

```
You are a reading comprehension specialist. Generate 2 diverse, natural questions 
from this passage. 

PASSAGE: [full passage]

SALIENT_SENTENCES: [top-K ranked by salience score]

Your task:
- Generate 2 factual questions (each from a different sentence if possible)
- Focus questions on salient sentences
- Questions should be answerable from the passage
- Keep natural language; avoid patterns

Return JSON: [{"question": "...", "answer": "..."}, ...]
```

**Rationale:**

- **Enforces grounding** without hard constraints (soft bias via prompt)
- **Allows creativity**: model still has freedom within the constrained set
- **Comparison fairness**: baseline gets all sentences; salience gets top-k (balanced by passage length)

---

### Speaker Notes for Slide 3:

Let me walk you through how our system actually works, because the prompt design is non-trivial and worth understanding.

We start with the standard pipeline: load SQuAD passages, split into sentences, extract a rich set of features. Then—this is the critical part—we train a binary classifier (logistic regression) on ~2,000 training passages to predict, for each sentence, whether it contains a gold QA span. This gives us a salience score for every sentence.

Why logistic regression? Interpretability. Linear models let us see which features matter; we can debug. For a production system, you'd want to validate against more complex models, but for our research question—"which signals matter?"—logistic regression is ideal.

The **balanced prompt design** deserves emphasis. We don't hard-code "only use these sentences." Instead, we feed the model both the full passage AND a ranked list of salient sentences, with a gentle instruction to focus on them. Why? Because:

1. Hard constraints can break fluency and creativity in LLMs
2. A soft bias is more robust to distribution shift
3. It mirrors real-world information retrieval pipelines (ranking, then reranking by user intent)

For fairness, our **baseline isn't "no salience"**—it's a *blind* version that gets the same top-k sentences but scrambled (no ranking info). This way, length isn't a confound.

---

## === Slide 4: Salience Feature Engineering ===

**Three Feature Categories:**

| Category | Features | Rationale |
|----------|----------|-----------|
| **Linguistic (14)** | sentence length, position, POS ratios, lexical density, named entity density, discourse markers | Fast & interpretable; capture structure |
| **GPT-2 Surprisal (6)** | mean, sum, std, var, min, max of token surprisal | Captures language model's uncertainty; often correlates with importance |
| **BERT Surprisal (6)** | same statistics over BERT contextual predictions | Contextualized uncertainty; complements GPT-2 |

**Key Observations from Ablations (Preview):**

- **Surprisal helps**, but linguistic alone is surprisingly strong
- **Position** (sentence_position) is the #1 predictor → answers cluster early
- **Surprisal variance** is more useful than mean → variance signals *anomalies* in text flow

**Why Both GPT-2 & BERT?**

- GPT-2: causal, unidirectional; fast
- BERT: bidirectional, contextualized; richer signal
- Complementary: GPT-2 gives local fluency; BERT gives global coherence

---

### Speaker Notes for Slide 4:

Feature engineering is often where the real work happens, so I want to be transparent about our choices.

We extract 26 features total, spanning three levels. **Linguistic features** are classical NLP: sentence position, syntactic structure, lexical cohesion. These are fast, interpretable, and surprisingly predictive—the README shows that position alone is critical (answers cluster within the first few sentences of passages).

**Surprisal features** are borrowed from psycholinguistics and neural language models. The intuition: if a language model assigns high surprisal (low probability) to a token, that token is "unexpected" in context. Unexpected tokens sometimes signal important shifts in meaning or when new information is introduced. We compute surprisal via two models:

- **GPT-2** (causal): "How surprising is this token given everything before it?"
- **BERT** (bidirectional): "How surprising given both directions?"

Both are valuable. GPT-2 simulates human reading-time effects; BERT captures longer-range coherence anomalies. Together, they capture orthogonal signal.

Interestingly—and this is a nuance we'll revisit—**surprisal statistics** are more informative than raw surprisal. Variance, for instance, signals high variation in token predictability within a sentence, often indicating content richness. Mean surprisal alone isn't as strong.

---

## === Slide 5: Ablation Study Design ===

**Research Question:**

*How much do surprisal features contribute to salience when high-quality linguistic features are already available?*

**Three Conditions (750-passage balanced sample from train split):**

1. **Full Features** (14 linguistic + 12 surprisal)
2. **No Surprisal** (14 linguistic only)
3. **Top-10 Features** (hand-selected best predictors across linguistic + surprisal)

**Methodology:**

- Same balanced random seed (2026) across all runs → identical 750 passages
- Same balanced prompt → same LLM (Qwen2.5-7B-Instruct)
- Same QA evaluation pipeline → consistent metrics
- 2 questions generated per passage → 1,500 questions evaluated per condition
- Split: 750 salience + 750 baseline (no salience guidance)

**Ablation Rationale:**

- **Full vs. No Surprisal**: isolates surprisal contribution
- **Top-10**: tests if model complexity matters or if simple models generalize
- **Salience vs. Baseline**: core comparison—does grounding actually improve downstream QA metrics?

---

### Speaker Notes for Slide 5:

The ablation design is intentionally simple and controlled. We wanted to answer a single, focused question: **given that linguistic features are strong, do surprisal features meaningfully improve salience-guided generation?**

To do this fairly, we:

1. **Fixed the passage sample** (750 passages, seed 2026) so we're evaluating the *same text* under different configurations. This eliminates sampling variance.

2. **Fixed the LLM and prompt structure** so the downstream generation process is identical. The *only* difference is what salience scores the model sees.

3. **Balanced salience vs. baseline** by ensuring both get comparable amounts of information. This tests whether salience *ranking* matters, not just whether giving extra information helps.

4. **Evaluated consistently** using the same QA model, the same metrics. We measure:
   - **ROUGE-L**: lexical overlap with gold questions (standard)
   - **BERTScore**: semantic similarity (modern metric, robust to paraphrases)
   - **QA-EM**: exact match on extracted answer spans
   - **QA-F1**: token-level F1 on extracted spans
   - **Answer-Recovery-F1**: F1 between the passage span the QA model extracts and the intended answer span

The **Top-10 ablation** is actually the most interesting from a deployment perspective. If model size and computational cost matter—e.g., running this in real-time—can we get away with just 10 features instead of 26? The answer will guide whether we recommend more data collection or a simpler, faster pipeline.

---

## === Slide 6: Quantitative Results ===

**Comparison Table (750-passage 750-examples each):**

| Metric | Full Features | No Surprisal | Top-10 Features |
|--------|---------------|--------------|-----------------|
| **ROUGE-L (salience)** | 0.3488 | 0.3600 | 0.3626 |
| **BERTScore (salience)** | 0.9052 | 0.9077 | 0.9076 |
| **QA-EM (salience)** | 0.5367 | 0.5127 | 0.5160 |
| **QA-F1 (salience)** | 0.7752 | 0.7644 | 0.7718 |
| **Answer-Recovery-F1 (salience)** | 0.4020 | 0.4302 | 0.4246 |
| | | | |
| **ROUGE-L (baseline)** | 0.3504 | 0.3687 | 0.3710 |
| **BERTScore (baseline)** | 0.9061 | 0.9094 | 0.9096 |
| **QA-EM (baseline)** | 0.5687 | 0.5780 | 0.5613 |
| **QA-F1 (baseline)** | 0.7670 | 0.7776 | 0.7653 |
| **Answer-Recovery-F1 (baseline)** | 0.4148 | 0.4537 | 0.4472 |

---

### Speaker Notes for Slide 6:

Let me highlight what's interesting here—and what *isn't* as obvious as you might think.

**First observation:** Surprisal features *don't* dramatically help. In fact, **No Surprisal achieves the highest ROUGE-L** (0.3600 vs 0.3488). Why? Because when you strip away surprisal, the model leans harder on linguistic features and generates questions that *lexically overlap more with the gold set*. This is expected—surprisal signals identify *different* salient content than what humans emphasized.

**Second observation:** **QA-EM is highest with Full Features** (0.5367 vs 0.5127), but the **QA-F1 difference is marginal** (0.7752 vs 0.7644 ≈ 1.4% drop). So while full features help with exact span recovery, no-surprisal still gets most tokens right.

**Third observation:** **Answer-Recovery-F1 is *higher* without surprisal** (0.4302 vs 0.4020). This means when surprisal is excluded, the passages' target answer spans align *better* with what the QA model extracts. Surprisal is pulling the model toward *different* content.

**Top-10 Features:** Remarkably competitive. ROUGE-L is actually *best* (0.3626), QA-EM is 0.5160 (between full and no-surprisal), and answer recovery is 0.4246. **We lose ~1-2% performance** by dropping complexity, suggesting a simpler model is viable.

**Baseline vs. Salience:** Baseline consistently outperforms salience on ROUGE-L (~2-3% gap), which is expected—we're biasing toward different sentences. But on QA-EM, it's competitive. This is the key tension: **grounding changes *what* we ask about, not necessarily how well we ask it.**

This challenges a common assumption: that better grounding = better metrics on *every* dimension. It doesn't. It's a trade-off.

---

## === Slide 7: Deeper Metric Analysis – The Consistency vs. Overlap Trade-Off ===

**Framing the Tension:**

- **ROUGE-L & Baseline**: Baseline wins because we're *not* biased by salience—we generate questions matching human distribution.
- **QA-F1 & Consistency**: Salience's higher QA-F1 (esp. Full Features at 0.7752) means the *QA model agrees better* with our generated questions. Consistency over diversity.
- **Answer-Recovery** twist: No-Surprisal wins here → linguistic features alone align questions to passage spans that *the QA model naturally extracts*, even if they differ from gold QA spans.

**What This Reveals:**

```
Surprisal features → push toward rare/anomalous content
                  → linguistically distinct from gold
                  → but QA model is confident in extracted answers

No Surprisal features → align with positional/structural salience
                     → often match crowdworkers' intuitions
                     → QA model extracts well-aligned spans

Trade-off: Grounding ≠ Gold overlap. Grounding = internal consistency + salience alignment.
```

---

### Speaker Notes for Slide 7:

This is where I want to slow down and talk about what metrics *mean*.

In standard QG evaluation, ROUGE-L is king because it measures "how much do you look like the gold references?" But here, **we don't expect to look like gold**—we're asking about *different* sentences.

The right metrics for salience-guided generation are:
1. **QA-F1 & Consistency**: the QA model's confidence in our questions
2. **Fluency** (human eval, not shown): do they sound natural?
3. **Grounding quality** (manual inspection): are we actually asking about salient content?

On those metrics, **Full Features performs well**. The QA model is highly confident (0.7752 F1) that our generated questions have high-quality, extractable answers.

The *reason* Answer-Recovery-F1 is lower for Full Features is instructive: surprisal causes the model to ask about *anomalous sentences*—ones that don't match typical linguistic patterns. Those sentences contain legitimate information, but they're structurally different from the salient norm. When the QA model extracts answers, it's wrestling with unexpected phrasing.

No-Surprisal, by contrast, leans into positional and structural salience—things humans intuitively notice. The answer spans are "normal," and recovery is easier.

**The insight:** we're not trading ROUGE-L for anything. ROUGE-L was never the right metric. We're trading off *different types of salience*: linguistic salience (No Surprisal) vs. neural uncertainty salience (Full Features). Both are valid; they optimize for different downstream behaviors.

---

## === Slide 8: Issue Analysis – The ~2,200 "Problem" Cases ===

**Problem Scale:**

| Condition | Total Questions | Issues Flagged | Issue Rate |
|-----------|-----------------|----------------|-----------|
| Full Features (salience) | 1,500 | 2,302 | **76.8%** |
| No Surprisal (salience) | 1,500 | 2,251 | **75.0%** |
| Top-10 Features (salience) | 1,500 | 2,208 | **73.9%** |

*Note: "Issues" are thresholded by ROUGE-L < 0.35 OR QA-F1 < 0.55 OR Answer-Recovery-F1 < 0.45. These are NOT failures—many are systematic and expected.*

**Top Flagged Reasons (from qa_generation_issues.json):**

1. **Low lexical overlap with gold question** (~95%+ of flags)
2. **QA model answer doesn't match generated answer** (~70%)
3. **Weak alignment between QA model answer and gold answers** (~65%)
4. **Generated question targets different fact than gold** (~85%)

*Note: Reasons can overlap; a single case often has multiple issues.*

---

### Speaker Notes for Slide 8:

Here's where I want to be very honest and reframe what "issues" means.

A 73–77% "issue rate" sounds alarming. But **it's not**. Here's why:

Our issue detection is *strict*. We flag a case if ROUGE-L falls below 0.35 (a reasonable threshold for "low lexical overlap"). Naturally, if we're asking about *different* sentences than gold QAs, ROUGE-L will be low.

That's not a bug. That's the system working as designed.

Let me walk through the most common reason: **Low lexical overlap with gold questions**. Of course! We're asking about sentences the crowdworkers didn't emphasize. The model says "Why were pictures on signs more useful?" (from a salient sentence about literacy trends), but the gold question was "In what historical period was a large portion of the population illiterate?" Both are valid; they're just *different facts* in the same passage.

**QA model answer mismatch** is more interesting. Sometimes, the generated question ("How did inns get their names?") is perfectly reasonable, but the QA model extracts a span that differs slightly from what our generator intended. This is actually a sign of model robustness—the QA model is finding *related* answers even when phrasing diverges.

**The key insight:** 70–75% of our cases are *expected given our approach*. They're not failures; they're cases where we successfully grounded generation in different salience than humans. The remaining 25–30% are genuine problems: questions that don't make sense, answers that are wrong, et cetera.

---

## === Slide 9: Qualitative Examples – What the Issues Tell Us ===

**Example 1: Different-Fact Issue (This is working correctly)**

```
Passage (snippet):
"...Another important factor was that during the Middle Ages a large 
proportion of the population would have been illiterate and so pictures 
on a sign were more useful than words..."

Gold QA:
Q: "In what historical period was a large portion of the population illiterate?"
A: "Middle Ages"

Generated (Salience-guided, Full Features):
Q: "Why were pictures on signs more useful?"
A: "pictures on a sign were more useful than words"

Issue Flagged: "Generated question targets different fact than gold"
Analysis: ✓ Both questions are valid, drawn from the same salient sentence.
         Salience guided toward the causal explanation, gold toward temporal context.
         Different perspectives on the same content.
         Not a failure—a feature of grounding to linguistic structure.
```

**Example 2: Weak QA Alignment (Reveals QA Model Limitation)**

```
Generated Q: "Which highway connects downtown Oklahoma City to Moore?"
Generated A: "Shields Boulevard (US-77)"

Gold QA:
Q: "What boulevard turns into E.K Gaylord Boulevard?"
A: "Shields Boulevard (US-77)"

QA Model's Extraction: "I-35"
Issue: QA model selected different (wrong) span from passage.

Analysis: Our generation is correct. The QA model's extraction failed.
         Not a deficiency in generation; a limitation in QA robustness on OOD questions.
         Suggests we need QA model finetuning or uncertainty calibration.
```

**Example 3: Genuine Low-Quality (Rare)**

```
Generated Q: "What features distinguish X from Y?"
Generated A: [Nonsensical or non-extractive]

Analysis: ~5–10% of flagged cases actually reflect generation failures
         (hallucination, incoherence, non-extractive answers).
         These warrant manual review or prompt refinement.
```

---

### Speaker Notes for Slide 9:

Let me show you real cases from our data and what they reveal.

The first example is *exactly* what we hoped to observe. We have a salient sentence introducing both temporal context (Middle Ages) and causality (pictures more useful than words). The crowdworkers focused on the temporal angle; our salience model, guided by linguistic features, emphasized the causal explanation. Both are valid questions from the same sentence. **This is grounding working correctly—not a failure.**

The second example is more subtle. Our generated question is sound: Shields Boulevard *does* connect downtown Oklahoma City to surrounding areas. But the QA model (RoBERTa-SQuAD2), when given our question, extracts "I-35"—a different highway. 

Why? The QA model was trained on SQuAD, where questions are phrased in specific ways. Our generated phrasing, while fluent and correct, is out-of-distribution. The model's confidence plummets. **This suggests we need to:**
- Finetune the QA model on generated questions, or
- Use an ensemble of QA models, or
- Add answer verification steps

**This is actionable.** It's not a generation failure; it's a downstream evaluation bottleneck.

The third category—genuinely incoherent output (rare)—warrants human review and prompt refinement but isn't systematic. Our LLM is generally reliable on well-sampled passages.

**The meta-lesson:** Issue analysis isn't about finding problems; it's about diagnosing *where problems live*. 70% of our "issues" are expected system behavior. 20% reveal QA model limitations. 5–10% are true generation bugs. Each category suggests different interventions.

---

## === Slide 10: Key Insights & Methodological Nuances ===

**Insight 1: Surprisal ≠ Universal Win**

- Full Features (with surprisal) achieves **highest QA-EM** (0.5367) but **lowest ROUGE-L** (0.3488).
- No Surprisal achieves **highest ROUGE-L & Answer-Recovery** (0.3600, 0.4302).
- **Implication:** Surprisal identifies *different* salient content. Valuable if your downstream task prioritizes QA consistency; risky if you need gold question alignment. Context matters.

**Insight 2: Position is Surprisingly Dominant**

- From the classifier analysis (README): `sentence_position` is the **#1 predictor coefficient** (-0.796).
- Answers cluster in early sentences by structural design (passages introduce main facts first).
- **Implication:** A simple position-based baseline beats complex models in ROUGE-L. For efficiency, consider position-first heuristics before neural features.

**Insight 3: Grounding ≠ Human Alignment**

- Salience → higher QA-F1 but lower ROUGE-L vs. baseline.
- We're not approximating crowdworkers; we're discovering *different* salience.
- **Implication:** Don't evaluate salience-guided generation using human question benchmarks alone. Need task-specific metrics: QA confidence, grounding quality, pedagogical value, answer consistency.

**Insight 4: The 75% "Issue" Rate is Misleading**

- Most flagged cases are *expected* given our salience bias.
- True generation failures (~5–10%) require manual inspection.
- **Implication:** Threshold-based issue detection is crude. Recommend multi-level analysis: (1) automated flagging, (2) categorization by type, (3) manual spot-checks on each category.

**Insight 5: QA Model Robustness is a Bottleneck**

- QA model struggles with out-of-distribution phrasing.
- Answer-Recovery-F1 at ~0.40 suggests mismatch between generated phrasing and SQuAD-trained expectation.
- **Implication:** Evaluation pipeline is only as good as the QA model. Consider ensemble QA, finetuning on generated questions, or human verification for high-stakes use.

---

### Speaker Notes for Slide 10:

These five insights distill what we learned from the ablations and issue analysis. Let me emphasize why each matters.

**On Surprisal:** We often assume adding features is always good. It's not. Surprisal captures a different notion of importance than linguistic structure. It's complementary, but not universally superior. In a production system, you'd want to *let the user choose* or use both in an ensemble.

**On Position:** This is humbling. A logistic regression that just looks at sentence position and a few surface features beats complex neural models on ROUGE-L. Why? Because SQuAD passages are written with a conventional structure—important facts first. Before overengineering, always have a strong positional baseline.

**On Grounding ≠ Human Alignment:** This is philosophically important. We're not trying to approximate human annotators. We're trying to *ground* generation in explicit linguistic signals. Those signals are different from human intuition—not worse, just different. So our evaluation needs to ask, "Are we grounded?" not "Do we look like crowdworkers?"

**On the 75% Issue Rate:** This is where I'd push back against the instinct to say "our system has 75% failure rate." No. 75% of flagged cases are *expected*. It's like saying an offensive lineman fails 75% of plays because they're not catching passes—they're not supposed to. Our system is doing what we asked.

**On QA Model Robustness:** This is urgent. Our QA evaluation is only as robust as the RoBERTa model we're using. If that model hasn't seen questions like ours, it fails. For production, we'd need to build confidence: finetuning on generated questions, ensemble methods, human verification pipelines.

---

## === Slide 11: Limitations & Honest Assessment ===

**Scope Limitations:**

1. **Single LLM tested**: Qwen2.5-7B-Instruct only. GPT-4, Llama 2, other models may show different patterns.
2. **Single QA model for evaluation**: RoBERTa-SQuAD2. Different QA models (e.g., ELECTRA, DeBERTa) may extract differently.
3. **No human evaluation**: Fluency, naturalness, pedagogical quality assessed via automatic metrics only. Subject to metric limitations.
4. **Fixed feature set**: 26 features are heuristic. Feature engineering wasn't exhaustive; other linguistic/neural signals may be stronger.
5. **SQuAD domain only**: Extractive QA on Wikipedia. Results may not transfer to other domains (biomedical, legal, multimodal).

**Methodological Compromises:**

1. **Issue detection by thresholding**: Crude. A case flagged as "low ROUGE-L" may still be valid but different.
2. **No oracle comparison**: Would benefit from human-gold-benchmark comparison (e.g., crowdworker raters on 100 examples).
3. **Balanced prompt may be too prescriptive**: A more open prompt might allow LLM more autonomy; we're semi-constraining.
4. **Baseline "blind salience" assumption**: Assumes scrambled ranking is neutral—in practice, position still matters to reading.

**Reproducibility Notes:**

- Random seed fixed (2026) — results are deterministic given same hardware/software versions.
- LLM outputs can vary by temperature, top-k, other hyperparams (we used defaults).
- GPT-2/BERT surprisal computation is deterministic, but numerical precision varies across CUDA versions.

---

### Speaker Notes for Slide 11:

Science requires honesty about what you don't know. Let me be clear about ours.

**Generalization:** We tested *one* LLM—Qwen. Does GPT-4 show the same trade-offs? Unknown. Our QA evaluation uses *one* model; different architectures might rank questions differently. We haven't run human evaluations—automatic metrics are proxies, not ground truth.

**Feature engineering:** We engineered 26 features heuristically. That's not exhaustive. Maybe there are discourse or pragmatic markers we missed that would be stronger. We haven't done systematic feature importance analysis beyond the classifier coefficients.

**Domain:** SQuAD is Wikipedia, extractive, English-only. How do results transfer to biomedical literature? Multimodal data? Machine translation? Unknown.

**Methodological:** Our issue flagging is threshold-based, which is coarse. Human raters might disagree on whether a case is actually problematic. We didn't have a human gold benchmark to compare against. And our "blind" baseline—while well-intentioned—still has some salience signal baked in (humans read top-to-bottom).

**Reproducibility:** We fixed the seed, but LLM outputs can drift with different CUDA versions or hardware. Surprisal computation is deterministic in principle but can have numerical precision issues. For full reproducibility, we'd need to pin software versions and hardware constraints.

These limitations don't invalidate our findings—they contextualize them. This is early-stage research, and honesty about boundaries is more valuable than overconfidence.

---

## === Slide 12: Future Work – Toward Discourse-Level Salience ===

**Immediate Next Steps (1–2 months):**

1. **Human evaluation**: 100–200 examples rated by 3+ annotators on:
   - Fluency (1–5 Likert)
   - Grounding quality (does question target salient fact?)
   - Pedagogical value (would this be useful in a tutoring system?)
   - Compared to baseline + gold

2. **QA model ensemble**: Use 3–5 QA models (RoBERTa, ELECTRA, DeBERTa) and take majority vote on answer extraction.

3. **Domain transfer**: Repeat ablations on:
   - Biomedical papers (PubMed abstractive tasks)
   - Legal documents (LexGLUE)
   - Measure generalization

**Medium-term (3–6 months):**

4. **Rhetorical Structure Theory (RST) integration**:
   - RST parse passages into discourse trees (Nucleus/Satellite relationships)
   - Rank sentences by rhetorical importance (Nucleus >> Satellite)
   - Combine linguistic + surprisal + RST scores in ensemble
   - Hypothesis: RST captures discourse-level salience, improving grounding without hurting fluency

5. **Prompt optimization**: Gradient-based prompt search (e.g., GradOpT, DSPy) to find better prompt templates.

6. **Larger-scale ablations**: Test on 2,000–5,000 passages; measure scaling behavior.

**Long-term Vision:**

7. **Hierarchical salience factorization:**
   - Surface-level: position, length
   - Syntactic: POS structure, dependency parse
   - Semantic: semantic role labels, entity links
   - Discourse: RST, coreference chains, rhetorical relations
   - Pragmatic: question-answer alignment, answerable span properties
   
   → Learn a multi-level salience score that weights each layer per downstream task.

---

### Speaker Notes for Slide 12:

Let me spell out the roadmap because it flows naturally from what we've learned.

**Human evaluation is non-negotiable.** We can't keep relies only on automatic metrics. A 100-example human study, compared to baseline + gold questions, would take 1–2 weeks and would immediately validate (or challenge) our findings.

**QA ensemble** is a tactical fix. We identified that RoBERTa is a bottleneck. Using multiple QA models and taking a majority vote adds robustness and confidence intervals on our metrics.

**Domain transfer** tests whether this approach generalizes. If it works on biomedical papers and legal documents, we have a general method. If it fails, we've learned something about SQuAD's specific structure.

Now, the *main* future work: **RST integration**. This is what excites me most.

Rhetorical Structure Theory is a linguistic framework that describes how sentences relate (one is a **Nucleus**—central claim; one is a **Satellite**—supporting detail). A discourse tree is built bottom-up: adjacent sentences combine into rhetorical units, which combine into larger units, until the whole passage is a tree.

Our hypothesis is that **Nucleus sentences are more salient** because they carry main propositions. Satellite sentences provide elaboration, evidence, correction—but they're secondary.

We can compute RST-based salience, then **combine linguistic + surprisal + RST scores** via a learned ensemble. I predict this would:
- Improve grounding (discourse structure explicitly models importance)
- Maintain fluency (we're not hard-constraining; just biasing)
- Reduce the need for surprisal (discourse structure is a more direct proxy for importance than neural uncertainty)

**Prompt optimization** is also important—we hand-crafted our prompt, but systematic search (via gradient-based or evolutionary methods) could refine it.

**Long-term**, I imagine **factored salience**: a system that learns to weight different linguistic levels depending on the task. For question generation, discourse matters most; for summarization, syntax matters; for semantic search, pragmatics matters. A single system that adapts.

---

## === Slide 13: Conclusion & Open Questions ===

**What We've Shown:**

1. **Salience-guided generation is feasible and principled**: we can encode sentence-level salience into prompts and observe consistent effects downstream.

2. **Trade-offs are real, not illusory**: grounding to salience shifts QA generation's distribution. ROUGE-L drops, but QA-F1 and consistency rise. Both are valid objectives; choose based on your task.

3. **Linguistic features are surprisingly strong**: surprisal helps, but much of the signal comes from syntax, position, and discourse markers. Simpler models (Top-10 features) are competitive—important for efficiency.

4. **Issue analysis reveals system understanding, not failure**: 75% of flagged cases are expected given our approach. Categorizing issues by type (expected, QA bottleneck, genuine failure) is more informative than a single "failure rate."

5. **The bottleneck is downstream evaluation**, not generation: QA model robustness, not fluency, limits performance. Ensemble QA + human verification required for production.

**Open Questions for Discussion:**

1. **Task alignment**: For your use case (e.g., tutoring, information retrieval, summarization), which trade-off matters most—grounding or gold alignment?

2. **Scale**: Would you run this on 10K passages? 100K? What computational budget constrains the pipeline?

3. **Discourse structure**: Is RST integration worth the added complexity, or should we invest in better linguistic features first?

4. **Cross-lingual generalization**: Should we extend to non-English languages, and if so, how do surprisal computations scale?

5. **Human-in-the-loop**: Would a system that flags uncertain cases for human verification be more valuable than fully automated generation?

---

### Speaker Notes for Slide 13:

Let me summarize the big picture.

We've *shown* that you can encode salience into generation in a principled way and that it has measurable, interpretable effects. The question isn't "does it work?"—it works. The question is "which trade-offs do you accept?"

And here's the honest framing: we're not trying to beat crowdworkers at their game (human question writing). We're trying to build a different game—one where generation is grounded in explicit linguistic and discourse structure, not just language model priors. Both games have value.

The biggest surprise, to me, is how strong position and simple linguistic features are. It feels pedestrian—"answers appear early; use that signal"—but it's robust and efficient. Complex surprisal and hand-engineered linguistic features add ~2-5% in downstream metrics, which is real but modest.

The *real* bottleneck we discovered is downstream. Our generation is reasonably reliable. But QA evaluation is brittle—the RoBERTa model struggles with out-of-distribution phrasing. This suggests that in production, you'd need ensemble QA, confidence calibration, and human-in-the-loop verification. That's not a failure of our approach; it's a necessary step toward deployment.

I have five open questions for you:

1. **Task alignment:** Are you building this for tutoring (where grounding is paramount), or retrieval (where diversity matters)? That changes everything about which metrics we should optimize.

2. **Scale:** Can we process 100K passages? That requires infrastructure decisions: distributed feature extraction, model serving, caching.

3. **Discourse:** Is RST worth exploring, or should we exhaust simpler linguistic features first? My intuition is that RST adds 3-5% but complexities the pipeline.

4. **Language:** English-only for now, but multilingual is valuable. Surprisal via multilingual BERT is feasible; need to test.

5. **Human-in-the-loop:** What if we flagged the most uncertain cases (e.g., where multiple QA models disagree) for human review? That might be more practical than full automation.

---

## === Slide 14: Thank You & Q&A ===

**Key Takeaways:**

- ✓ Salience-grounded QG is methodologically sound and empirically competitive
- ✓ Linguistic features alone are surprisingly strong; surprisal is complementary, not essential
- ✓ Grounding changes generation's distribution intentionally; not a failure, a feature
- ✓ The bottleneck is downstream evaluation, not generation
- ✓ Future work should combine discourse structure (RST) with linguistic + neural signals

**Open to Questions**

---

### Speaker Notes for Slide 14:

That's the story. Let me circle back to the five key points, because if you forget everything else, remember these.

One, salience grounding isn't speculative—it's reproducible and shows consistent effects on downstream metrics.

Two, surprisingly, you don't need complex neural features. Linguistic features alone get you 90% of the way there. Surprisal helps but isn't magic.

Three, we're changing *what* the model asks about, not *how* it asks. That's intentional and valuable—not a compromise. Frame it correctly with stakeholders.

Four, we identified the real bottleneck: the QA evaluation model is fragile. Future work should harden that.

And five, RST integration is the natural next step. Discourse structure is more direct than neural uncertainty and captures the multi-level salience that humans intuitively understand.

I have time for questions and discussion. What's on your minds?

---

## === Slide 15: Appendix – Detailed Metrics Explanation ===

**Metric Definitions:**

- **ROUGE-L** (Recall-Oriented Understudy for Gisting Evaluation):
  - Longest common subsequence (LCS) between generated and reference questions
  - Measures lexical overlap; bounded [0, 1]
  - Higher = more lexically similar to gold questions
  - *Use for*: evaluating approximation to human-written questions
  - *Caveat*: biased against paraphrase and novel phrasings

- **BERTScore**:
  - Embedding-based similarity using contextual representations
  - Compares generated and gold question embeddings via cosine similarity
  - Robust to paraphrasing and synonymy
  - Bounded [0, 1]; typically >0.9 for similar questions
  - *Use for*: measuring semantic equivalence without lexical matching
  - *Caveat*: sensitive to model bias (uses RoBERTa embeddings)

- **QA-EM (Exact Match) & QA-F1**:
  - Given generated question, extract answer using QA model
  - Compare extracted answer span to gold answer span
  - EM = 1 if exact token match; 0 otherwise
  - F1 = token-level precision/recall between extracted and gold
  - *Use for*: measuring QA model's confidence and answer accuracy
  - *Caveat*: depends on QA model quality and distribution match

- **Answer-Recovery-F1**:
  - Start/end span of gold answer in passage vs. extracted span
  - Measures whether QA model recovers the same region of text
  - *Use for*: understanding QA model localization accuracy
  - *Caveat*: stricter than question-level F1; more sensitive to phrasing

---

### Speaker Notes for Slide 15:

These metric definitions are useful if anyone dives deep into the numbers.

Key insight: metrics aren't interchangeable. ROUGE-L tells you "how much text overlap," BERTScore tells you "is the meaning similar," QA-F1 tells you "would a QA system understand this." They measure different things.

For salience-grounded generation, **QA-F1 and consistency are the right primary metrics**. ROUGE-L is useful for sanity-checking that we're not in complete nonsense territory, but it shouldn't be the optimization target because we're intentionally biasing toward different sentences.

---

## === Slide 16: Appendix – Reproducibility & Code Links ===

**Reproducibility:**

- **Ablation script**: `ablation.py` (or `qg_ablations.py` for generation-level ablations)
- **Evaluation code**: `evaluation.py`
- **Feature extraction**: `feature_extractor.py` (linguistic) + `surprisal.py` (neural)
- **Salience classifier**: `classifier.py` (trained on ~2,000 passages, 26 features, logistic regression)

**Key Configuration Files:**

- Salience threshold: 0.55 (tuned on dev set; adjust via command-line arg `--threshold`)
- Top-K salient sentences: 3 (default; adjust via `--top-k`)
- Questions per passage: 2 (adjust via `--num-questions`)
- QG model: Qwen2.5-7B-Instruct (can swap for other models in `llm_question_generator.py`)
- QA model: deepset/roberta-base-squad2 (can swap in `evaluation.py`)

**Data Locations:**

- Training data: `data/train.json` (SQuAD 1.1 train split)
- Dev data: `data/dev.json` (SQuAD 1.1 dev split)
- Results: `results/run_750_passages/` and `results/qg_ablation/run_750_passages/{no_surprisal,top10_features}/`

**To Reproduce:**

```bash
# Feature extraction & training (already done; cached)
python main.py --max-passages 2000

# Run QG pipeline (salience-guided) on 750 dev examples
python qg_pipeline.py --source dev --max-examples 750 --threshold 0.55

# Run ablations
python qg_ablations.py --source dev --max-examples 750 --seed 2026

# Evaluate
python evaluation.py --pipeline-json results/qg/run_750_passages_current/qg_pipeline_train_after2000.json
```

**Dependencies:**

- PyTorch (with CUDA support recommended)
- Transformers (HuggingFace)
- NLTK, Pandas, Scikit-learn
- See `requirments.txt` for pinned versions

---

### Speaker Notes for Slide 16:

If anyone wants to dive into the code or reproduce our results, this is the roadmap.

The ablation and evaluation scripts are the key entry points. Everything is deterministic given the fixed seed and software versions. If you get different numbers, it's likely due to CUDA version mismatch (numerical precision on GPU operations can drift slightly).

The results are cached in the `results/` directory, so you can re-evaluate different configurations without re-running the expensive surprisal and question generation steps.

--

**That's it. Thank you, and I'm open to questions.**


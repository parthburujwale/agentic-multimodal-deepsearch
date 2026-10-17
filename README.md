# Agentic Multimodal DeepSearch

An agentic multimodal retrieval and reasoning system for complex image search.

The system combines:

- a trained CLIP-based multimodal retriever
- LLM-based query decomposition
- weighted reciprocal rank fusion
- vision-language-model verification
- evidence sufficiency checking
- iterative evidence-seeking retrieval
- bounded cross-round promotion
- grounded multimodal answer generation

The goal is to answer complex compositional image-search queries while remaining grounded in retrieved visual evidence.

---

# Motivation

Traditional text-to-image retrieval systems often perform well when a query contains a single dominant visual concept.

For example:

> Find an image of a person riding a bicycle.

However, retrieval becomes significantly harder when the request contains multiple simultaneous requirements:

> Find an image of a dog standing beside water with several people visible in the background and a red bicycle nearby, but the dog should not be swimming and no boat should be visible.

Such queries require more than similarity matching.

The system must:

1. identify multiple positive requirements
2. preserve negative constraints
3. retrieve candidates for different semantic components
4. verify whether each candidate actually satisfies the request
5. determine whether the evidence is sufficient
6. search again when evidence is incomplete
7. avoid degrading a strong first-round result
8. generate an answer that does not claim more than the retrieved evidence supports

Agentic Multimodal DeepSearch is designed around this problem.

---

# System Architecture

```text
User Query
    ↓
Query Planner
    ↓
DeepSearch Retriever V2
    ↓
Weighted Reciprocal Rank Fusion
    ↓
Vision-Language Evidence Verification
    ↓
Evidence Sufficiency Check
    ↓
Baseline-Aware Iteration Router
    ↓
Optional Evidence-Seeking Round 2
    ↓
Bounded Cross-Round Promotion
    ↓
Structured Evidence Builder
    ↓
Grounded Multimodal Answer Generator
    ↓
Final Answer + Evidence + Confidence
```

The retrieval architecture is frozen before final evaluation.

The grounded answer-generation layer operates on top of the frozen retrieval system and does not modify retrieval rankings.

---

# 1. DeepSearch Retriever V2

The retrieval backbone is based on pretrained CLIP with lightweight trainable residual adapters.

The pretrained CLIP image encoder, text encoder, and projection layers remain frozen.

Separate trainable adapters are applied to normalized image and text embeddings.

The adapter architecture is:

```text
LayerNorm(512)
→ Linear(512, 1024)
→ GELU
→ Dropout
→ Linear(1024, 512)
```

The final adapted embedding is:

```text
normalize(
    base_embedding
    +
    alpha * adapter(base_embedding)
)
```

Separate trainable scaling parameters are used for image and text representations.

Only approximately **1.37% of the total model parameters are trainable**.

The retriever was trained using symmetric contrastive learning on 50,000 COCO training images.

## Retriever Performance

On the standard COCO validation retrieval task:

| System | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Pretrained CLIP | 28.68% | 53.20% | 64.66% |
| DeepSearch Retriever V2 | **35.02%** | **61.76%** | **73.48%** |

The learned residual adapters substantially improve the pretrained CLIP retrieval backbone while modifying only a small fraction of the model parameters.

---

# 2. Query Planning

A Qwen2.5-1.5B-Instruct model converts each natural-language request into a structured search objective containing:

- retrieval subqueries
- positive constraints
- negative constraints
- reasoning goal

Example:

```json
{
  "search_queries": [
    "dog standing beside water",
    "several people visible in background",
    "red bicycle nearby"
  ],
  "positive_constraints": [
    "dog standing beside water",
    "several people visible in background",
    "red bicycle nearby"
  ],
  "negative_constraints": [
    "dog swimming",
    "boat visible"
  ]
}
```

The original user request remains the fixed semantic verification objective even when later retrieval queries are refined.

This separates:

```text
What should I search for?
```

from:

```text
What does a correct result need to satisfy?
```

---

# 3. Weighted Multi-Query Retrieval

The original user query and planner-generated subqueries are independently retrieved against the image corpus.

Their rankings are combined using weighted Reciprocal Rank Fusion.

The original user query receives a larger weight than the decomposed subqueries.

Conceptually:

```text
Original query                weight = 2
Planner subquery 1            weight = 1
Planner subquery 2            weight = 1
Planner subquery 3            weight = 1
```

This allows the system to preserve the global semantic meaning of the user request while also searching for individual visual constraints.

---

# 4. Multimodal Evidence Verification

Top retrieval candidates are verified using Qwen2.5-VL-3B-Instruct.

For each candidate image, the verifier evaluates:

- each positive constraint
- each negative constraint
- positive evidence score
- negative-violation score
- overall semantic score
- overall semantic match
- natural-language explanation

Example:

```text
Positive constraints:
✓ dog standing beside water
✓ several people visible in background
✗ red bicycle nearby

Negative constraints:
✓ dog is not swimming
✓ no boat is visible
```

Retrieval similarity determines whether a candidate is relevant enough to inspect.

The verifier determines whether the candidate actually satisfies the semantic request.

---

# 5. Evidence Sufficiency

After verification, the system determines whether the best candidate provides sufficient evidence.

Evidence is considered sufficient only when:

- the semantic match is complete
- required positive evidence is present
- negative constraints are respected
- verifier confidence exceeds the frozen threshold

If evidence is insufficient, the system may initiate another retrieval round.

The final system uses a verification threshold of:

```text
0.75
```

---

# 6. Baseline-Aware Iteration Router

A second retrieval round is not always necessary.

The router considers signals such as:

- whether the current result is a complete semantic match
- verification confidence
- disagreement with the DeepSearch Retriever V2 baseline
- verifier ties
- missing semantic constraints

If the evidence is already sufficient and stable, the system stops after Round 1.

If important evidence is missing, the router may trigger Round 2.

---

# 7. Controlled Evidence-Seeking Query Refinement

Round 2 is not a simple repeat of the original search.

Instead, the query refiner focuses on evidence that was missing from the current best candidate.

Example:

```text
Original query:
dog beside water + several people + red bicycle

Round 1 missing:
several people
red bicycle

Round 2 retrieval query:
several people visible in background, red bicycle nearby
```

The original semantic verification objective remains fixed.

Only the retrieval query changes.

The refiner uses:

- missing positive constraints
- candidate ambiguity
- verifier disagreement
- controlled query length
- similarity constraints relative to previous queries
- deterministic fallback behavior

---

# 8. Bounded Cross-Round Promotion

Experiments showed that unrestricted Round-2 replacement often degraded strong Round-1 results.

The final architecture therefore uses conservative bounded promotion.

Round 2 may only promote the original Round-1 rank-2 candidate when all of the following conditions hold:

1. the Round-1 rank-2 candidate becomes Round-2 rank 1
2. the original Round-1 winner is absent from the Round-2 top candidates
3. the verifier-score difference is no greater than 0.20
4. the challenger contains no explicit negative-constraint violation

Otherwise, the Round-1 ranking is preserved.

This makes Round 2 an evidence-seeking exploration mechanism rather than an unrestricted replacement policy.

---

# 9. Structured Evidence Builder

The final retrieval result is converted into a compact structured evidence package.

For each retrieved image, the package contains:

- final rank
- image ID
- image path
- retrieval support
- RRF score
- verifier score
- positive constraint checks
- negative constraint checks
- missing requirements
- negative violations
- verifier explanation

This structured representation provides the grounding context for answer generation.

---

# 10. Grounded Multimodal Answer Generation

Qwen2.5-VL-3B-Instruct is also used as the final multimodal answer generator.

The answer generator receives:

```text
original query
+
retrieved images
+
structured verifier evidence
```

The verifier remains authoritative.

The generative model is responsible for natural-language synthesis, but it does not redefine:

- which positive constraints are satisfied
- which positive constraints are missing
- which negative constraints are violated
- whether the retrieval system considered the evidence sufficient
- which image is the primary final evidence

This prevents the answer-generation model from overriding the structured retrieval and verification system.

---

# Grounded Answer Behavior

The final system supports both confident answers and explicit abstention.

## Complete Match Example

Query:

```text
Find an image of a person riding a bicycle.
```

Example result:

```text
Complete match: True
Confidence: high

Satisfied requirements:
+ person riding bicycle

Missing requirements:
None

Violated requirements:
None
```

The system found a complete semantic match in Round 1 and stopped without unnecessary iteration.

## Insufficient Evidence Example

Query:

```text
Find an image of a dog standing beside water with several people
visible in the background and a red bicycle nearby, but the dog
should not be swimming and no boat should be visible.
```

Example result:

```text
Complete match: False
Confidence: low

Satisfied requirements:
+ dog standing beside water

Missing requirements:
- several people visible in background
- red bicycle nearby

Violated requirements:
None

Forbidden concepts not detected:
+ dog swimming
+ boat visible
```

The system explicitly abstains instead of hallucinating a complete match.

---

# Results and Ablations

## Final Independent Test Set

The final retrieval architecture was frozen before evaluation on an independently generated 200-query COCO benchmark.

The benchmark contains:

- 200 unique target images
- 0 target overlap with the 50-query development set
- 0 target overlap with the previous 100-query holdout set
- deterministic construction with a fixed random seed

Benchmark SHA256:

```text
723380c31e6842c3388f9c73f1bf251bbf4617bf7bd6f7099f822a25686e720c
```

## Main Results

| System | R@1 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|
| DeepSearch Retriever V2 | 41.50% | 69.50% | 79.50% | 0.5357 |
| + One-shot Agentic Verification | 46.00% | 73.50% | 79.50% | 0.5712 |
| + Iterative Retrieval with Bounded Promotion | **46.50%** | **73.50%** | **79.50%** | **0.5737** |

Compared with the trained DeepSearch Retriever V2, the complete agentic retrieval pipeline improved:

- **R@1:** 41.5% → 46.5% (**+5.0 percentage points**)
- **R@5:** 69.5% → 73.5% (**+4.0 percentage points**)
- **MRR@10:** 0.5357 → 0.5737

The majority of the improvement comes from multimodal semantic verification.

Controlled iteration provides a smaller additional improvement while preserving one-shot behavior for almost all final test queries.

---

# What Each Stage Contributes

## 1. Learned Multimodal Retriever

The first stage improves the underlying CLIP representation using lightweight trainable residual adapters.

| System | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Pretrained CLIP | 28.68% | 53.20% | 64.66% |
| DeepSearch Retriever V2 | **35.02%** | **61.76%** | **73.48%** |

This provides a stronger retrieval backbone before agentic reasoning is applied.

## 2. Multimodal Verification

Adding structured query decomposition and vision-language-model candidate verification produces the largest agentic improvement.

| System | R@1 | R@5 | MRR@10 |
|---|---:|---:|---:|
| Retriever V2 | 41.50% | 69.50% | 0.5357 |
| One-shot Verified Agentic Search | **46.00%** | **73.50%** | **0.5712** |
| Absolute improvement | **+4.50 pp** | **+4.00 pp** | **+0.0355** |

This suggests that retrieval similarity alone is insufficient for many complex compositional queries.

Explicit semantic verification contributes substantially to the final ranking quality.

---

# Iterative Evidence-Seeking Analysis

The iteration router triggered on:

```text
91 / 200 queries
```

A second retrieval round therefore ran for:

```text
45.5% of final test queries
```

Raw Round 2 compared with Round 1:

| Outcome | Queries |
|---|---:|
| Improved | 15 |
| Unchanged | 42 |
| Worsened | 34 |
| Round 2 not executed | 109 |

Among the 91 cases where Round 2 executed:

```text
Improved: 15
Unchanged: 42
Worsened: 34
```

This is an important result.

Iterative search can discover stronger candidates, but simply replacing Round 1 with Round 2 is not reliable.

---

# Ablation: Unrestricted Cross-Round Fusion

During development, unrestricted cross-round fusion was tested.

On the targeted development cases:

```text
Improved: 0
Unchanged: 6
Worsened: 2
```

The experiment showed that unrestricted fusion could degrade strong Round-1 results.

As a result, unrestricted cross-round fusion was removed from the final decision policy.

Cross-round candidate memory is retained only for diagnostics.

---

# Ablation: Bounded Promotion

The final system uses conservative bounded promotion instead of unrestricted Round-2 replacement.

Bounded promotions were applied in:

```text
5 / 200 queries
```

Final iterative ranking compared with one-shot agentic search:

| Outcome | Queries |
|---|---:|
| Improved | 1 |
| Unchanged | 199 |
| Worsened | 0 |

The iterative controller therefore added a small improvement while introducing **zero target-rank regressions relative to the one-shot system on this frozen 200-query test set**.

Final R@1 increased from:

```text
46.0% → 46.5%
```

while R@5 and R@10 remained unchanged.

---

# Final System vs DeepSearch Retriever V2

At the individual-query level:

| Outcome | Queries |
|---|---:|
| Improved target rank | 55 |
| Unchanged target rank | 110 |
| Worsened target rank | 35 |

Aggregate performance still improved:

```text
R@1:    41.5% → 46.5%
R@5:    69.5% → 73.5%
R@10:   79.5% → 79.5%
MRR@10: 0.5357 → 0.5737
```

This distinction is important.

The final agentic system improves aggregate retrieval quality, but it does not improve every individual query.

---

# Development Benchmark

A separate 50-query development benchmark was used while designing the agentic architecture.

| System | R@1 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|
| DeepSearch Retriever V2 | 58.00% | 82.00% | 90.00% | 0.6927 |
| One-shot Verified Agentic Search | 58.00% | 88.00% | 90.00% | 0.7225 |
| Iterative + Bounded Promotion | **60.00%** | **88.00%** | **90.00%** | **0.7325** |

Development-set iterative behavior:

```text
Router triggered: 19 / 50
Round 2 executed: 19 / 50
Bounded promotions: 1 / 50
```

Raw Round 2 behavior:

```text
Improved: 2
Unchanged: 7
Worsened: 10
Not run: 31
```

Final iterative result compared with one-shot:

```text
Improved: 1
Unchanged: 49
Worsened: 0
```

The development results motivated the conservative bounded-promotion policy before final independent evaluation.

---

# Ablation Summary

| Component | Observation |
|---|---|
| Pretrained CLIP | Strong starting point but weaker retrieval recall |
| Residual-adapter training | Improves the retrieval backbone substantially |
| Query decomposition | Enables constraint-aware multi-query retrieval |
| Weighted RRF | Combines global-query and subquery evidence |
| VLM verification | Produces the largest agentic performance gain |
| Evidence sufficiency | Prevents premature acceptance of incomplete results |
| Raw Round 2 | Sometimes improves retrieval but frequently introduces regressions |
| Unrestricted cross-round fusion | Rejected because it produced regressions |
| Bounded promotion | Preserves Round-1 stability while allowing rare safe improvements |
| Grounded answer generation | Adds natural-language responses without changing retrieval decisions |

---

# Main Experimental Takeaway

The strongest performance does not come from simply performing more retrieval rounds.

Instead, the results support combining:

```text
strong learned retrieval
+
structured query decomposition
+
multimodal semantic verification
+
controlled evidence-seeking
+
conservative final arbitration
```

The experiments also show that **exploration and final ranking should be treated as separate problems**.

Round 2 is useful for searching for missing evidence, but its candidates should not automatically replace a stronger existing result.

---

# Reproducibility

The final system uses deterministic inference settings:

```text
Seed: 42
cuDNN deterministic: enabled
cuDNN benchmark: disabled
TF32: disabled
PyTorch deterministic algorithms: enabled
LLM/VLM generation: do_sample=False
Attention implementation: eager
```

A four-case reproducibility test produced:

```text
Exact matches: 4 / 4
All cases reproducible: True
```

The final test benchmark was frozen and committed before final evaluation.

---

# Repository Structure

```text
src/
├── models/
│   ├── deepsearch_retriever.py
│   ├── deepsearch_retriever_v2.py
│   ├── test_deepsearch_retriever.py
│   └── test_deepsearch_retriever_v2.py
│
├── agents/
│   ├── query_planner.py
│   ├── agentic_retriever.py
│   ├── evidence_verifier.py
│   ├── verified_agentic_search.py
│   ├── evidence_sufficiency.py
│   ├── query_refiner.py
│   ├── iterative_agentic_search.py
│   └── reproducibility.py
│
├── rag/
│   ├── evidence_builder.py
│   ├── multimodal_answer_generator.py
│   └── grounded_deepsearch.py
│
├── evaluation/
│   ├── evaluate_agentic_pipeline.py
│   ├── evaluate_iterative_agentic_pipeline.py
│   ├── build_complex_coco_benchmark.py
│   ├── build_complex_coco_holdout.py
│   ├── build_final_coco_test_200.py
│   ├── evaluate_final_test_200.py
│   ├── analyze_retrieval_confidence.py
│   ├── analyze_verifier_regressions.py
│   ├── analyze_round2_behavior.py
│   ├── analyze_cross_round_fusion.py
│   ├── analyze_iterative_router_candidates.py
│   ├── test_iterative_reproducibility.py
│   └── ...
│
└── training/
```

---

# Models

The current system uses:

- **CLIP** as the multimodal retrieval backbone
- **custom residual adapters** for retriever training
- **Qwen2.5-1.5B-Instruct** for query planning
- **Qwen2.5-1.5B-Instruct** for controlled evidence-seeking query refinement
- **Qwen2.5-VL-3B-Instruct** for multimodal evidence verification
- **Qwen2.5-VL-3B-Instruct** for grounded answer generation

---

# Dataset

Experiments currently use MS COCO.

Retriever training:

```text
50,000 COCO train2017 images
```

Retrieval corpus:

```text
5,000 COCO val2017 images
```

Final independent benchmark:

```text
200 compositional queries
```

Development benchmark:

```text
50 compositional queries
```

The final test benchmark was frozen before final evaluation.

---

# Running Grounded DeepSearch

The complete system can be run from a single natural-language query.

Example:

```bash
python src/rag/grounded_deepsearch.py   --query "Find an image of a person riding a bicycle."   --output experiments/grounded_deepsearch_result.json
```

The system automatically performs:

```text
query planning
→ multimodal retrieval
→ weighted RRF
→ semantic verification
→ evidence sufficiency analysis
→ optional iterative retrieval
→ bounded promotion
→ evidence construction
→ grounded answer generation
```

---

# Example End-to-End Output

Example query:

```text
Find an image of a person riding a bicycle.
```

Output:

```text
Complete match:
True

Confidence:
high

Satisfied requirements:
+ person riding bicycle

Missing requirements:
None

Violated requirements:
None
```

Example difficult query:

```text
Find an image of a dog standing beside water with several people
visible in the background and a red bicycle nearby, but the dog
should not be swimming and no boat should be visible.
```

Output:

```text
Complete match:
False

Confidence:
low

Satisfied requirements:
+ dog standing beside water

Missing requirements:
- several people visible in background
- red bicycle nearby

Violated requirements:
None
```

---

# Key Findings

1. Lightweight residual-adapter training substantially improves the pretrained CLIP retrieval backbone.
2. LLM query decomposition enables structured retrieval over complex compositional requests.
3. Multimodal candidate verification provides the largest agentic improvement.
4. Separating retrieval queries from the fixed semantic verification objective is important for iterative search.
5. Iterative evidence seeking can discover stronger candidates, but unrestricted replacement is unreliable.
6. Conservative bounded promotion makes iteration safer.
7. Structured verifier outputs should remain authoritative during answer generation.
8. Explicit abstention is essential when visual evidence does not fully satisfy the request.
9. Agentic retrieval improves aggregate retrieval quality, but individual-query regressions still exist relative to the base retriever.
10. Exploration and final ranking should be treated as separate components.

---

# Research Question

The central research question is:

> Can agentic query decomposition, multimodal verification, and controlled iterative evidence seeking improve retrieval for complex compositional visual queries?

The final independent evaluation supports the usefulness of multimodal verification and shows that controlled iteration can add additional improvement while avoiding the regressions observed with unrestricted second-round replacement.

---

# Future Work

Potential extensions include:

- scaling Retriever V2 training to the full COCO training set
- evaluation on larger and more diverse multimodal datasets
- learned iteration routing
- learned cross-round arbitration
- confidence calibration
- larger vision-language models
- more complex relational queries
- spatial reasoning queries
- text + image input queries
- video retrieval
- document retrieval
- interactive multimodal search
- learned evidence aggregation
- human evaluation of grounded responses
- evaluation on real-world multimodal search tasks

---

# Project Status

Current system:

```text
✓ trained multimodal retriever
✓ lightweight residual adapters
✓ agentic query planning
✓ weighted multi-query retrieval
✓ multimodal semantic verification
✓ evidence sufficiency reasoning
✓ controlled iterative evidence-seeking retrieval
✓ deterministic inference
✓ bounded cross-round promotion
✓ independent 200-query evaluation
✓ ablation studies
✓ structured evidence builder
✓ grounded multimodal answer generation
✓ explicit abstention
✓ end-to-end DeepSearch pipeline
```

The core retrieval, reasoning, evaluation, and grounded RAG architecture are complete.
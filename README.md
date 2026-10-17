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

## Motivation

Traditional text-to-image retrieval systems often perform well when a query contains a single dominant visual concept.

For example:

> Find an image of a person riding a bicycle.

However, retrieval becomes significantly harder when the request contains multiple simultaneous requirements:

> Find an image of a dog standing beside water with several people visible in the background and a red bicycle nearby, but the dog should not be swimming and no boat should be visible.

Such queries require more than similarity matching.

The system must:

1. identify multiple positive requirements,
2. preserve negative constraints,
3. retrieve candidates for different semantic components,
4. verify whether each candidate actually satisfies the request,
5. determine whether the evidence is sufficient,
6. search again when evidence is incomplete,
7. avoid degrading a strong first-round result,
8. generate an answer that does not claim more than the retrieved evidence supports.

Agentic Multimodal DeepSearch is designed around this problem.

---

# System Architecture

```text
User Query
    ↓
Query Planner
    ↓
Retriever V2
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
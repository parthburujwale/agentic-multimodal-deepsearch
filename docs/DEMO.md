# Agentic Multimodal DeepSearch — Demo Cases

This page shows representative end-to-end behaviors of the system.

The examples demonstrate three important cases:

1. a successful complete match
2. explicit abstention when evidence is incomplete
3. iterative evidence seeking where Round 2 explores new candidates but conservative arbitration prevents an unsafe replacement

---

# Demo 1 — Complete Match

## Query

> Find an image of a person riding a bicycle.

## Query Plan

The planner decomposes the request into a simple visual search objective:

```text
Positive requirement:
+ person riding bicycle

Negative requirements:
None
```

## Retrieval

DeepSearch Retriever V2 retrieves candidate images from the 5,000-image COCO validation corpus.

Weighted Reciprocal Rank Fusion combines:

```text
Original query:
Find an image of a person riding a bicycle.

Planner subquery:
person riding bicycle
```

The top candidate is:

```text
Image ID: 549390
Verifier score: 0.90
Overall match: True
```

Several other top-ranked candidates also receive strong semantic-verification scores.

## Evidence Sufficiency

The top result satisfies all required evidence:

```text
Positive constraints satisfied: 1 / 1
Negative violations: 0 / 0
Verifier score: 0.90
Required threshold: 0.75
```

The evidence-sufficiency controller returns:

```text
Sufficient: True
```

## Router Decision

Because strong and complete evidence has already been found:

```text
Trigger Round 2: False
```

The system stops after one retrieval round.

## Grounded Answer

```text
Evidence 1 satisfies the positive requirement of a person riding a bicycle.
```

Final structured result:

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

## Why This Example Matters

This demonstrates the fast path through the architecture:

```text
Query
→ planning
→ retrieval
→ verification
→ sufficient evidence
→ grounded answer
```

The iterative controller does not perform unnecessary additional search when the evidence is already strong.

---

# Demo 2 — Explicit Abstention

## Query

> Find an image of a dog standing beside water with several people visible in the background and a red bicycle nearby, but the dog should not be swimming and no boat should be visible.

This query is significantly more difficult because it combines:

```text
Positive requirements:
+ dog standing beside water
+ several people visible in background
+ red bicycle nearby

Negative requirements:
- dog swimming
- boat visible
```

The system must find a single image that satisfies the positive requirements while avoiding both forbidden concepts.

---

## Round 1

The planner generates the following retrieval components:

```text
dog standing beside water
several people visible in background
red bicycle nearby
```

The original query remains the fixed semantic verification objective.

The Round-1 winner is:

```text
Image ID: 190140
Verifier score: 0.30
Overall match: False
```

Verifier analysis:

```text
Satisfied:
+ dog standing beside water

Missing:
- several people visible in background
- red bicycle nearby

Negative violations:
None
```

## Evidence Sufficiency

The evidence is insufficient:

```text
Positive constraints: 1 / 3
Negative violations: 0 / 2
Verifier score: 0.30
Required threshold: 0.75
```

Therefore:

```text
Sufficient: False
```

---

# Demo 3 — Iterative Evidence Seeking

The same difficult query illustrates why the system contains a second search round.

## Why Round 2 Is Triggered

The baseline-aware router identifies:

```text
incomplete semantic match
low verification score
```

and decides:

```text
Trigger another round: True
```

The system does not simply rerun the original query.

Instead, it asks:

> What evidence is missing from the best result?

The missing evidence after Round 1 is:

```text
several people visible in background
red bicycle nearby
```

---

## Controlled Query Refinement

The query refiner initially generates:

```text
people bike red background dogs near water
```

The controller rejects this query because it does not preserve enough of the required constraints:

```text
Validation: False
Reason: insufficient_constraint_overlap
```

The deterministic fallback is therefore used:

```text
several people visible in background, red bicycle nearby
```

This becomes the Round-2 retrieval query.

Importantly, the verification objective remains the original request.

---

# Round 2

Round 2 retrieves a new candidate:

```text
Image ID: 184611
Verifier score: 0.60
Overall match: False
```

This candidate improves the semantic evidence relative to the raw Round-1 winner.

Its verifier state is:

```text
Positive constraints: 2 / 3
Negative violations: 0 / 2

Missing:
- dog standing beside water
```

The score improves:

```text
Round 1 verifier score: 0.30
Round 2 verifier score: 0.60
```

However, the candidate still does not satisfy the complete original request.

---

# Why Round 2 Does Not Automatically Replace Round 1

Earlier experiments showed that unrestricted Round-2 replacement frequently degraded results.

The final system therefore uses bounded cross-round promotion.

The final policy checks whether the original Round-1 rank-2 challenger satisfies a strict set of promotion conditions.

For this example:

```text
Promote challenger: False

Reason:
challenger_did_not_become_round_2_rank_1
```

Therefore the system preserves the Round-1 final ranking.

This is intentional.

Round 2 is treated as exploratory evidence search, not as an automatic replacement mechanism.

---

# Final Evidence State

After bounded arbitration, the final primary evidence is:

```text
Image ID: 190140
Verifier score: 0.30
Overall match: False
```

Final structured interpretation:

```text
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

---

# Grounded Answer

Because the evidence remains incomplete, the answer generator abstains:

```text
No complete match was verified.
```

Final system output:

```text
Complete match: False
Confidence: low
```

The system does not claim that an image fully satisfies the request.

---

# What These Examples Demonstrate

These examples illustrate the main behaviors of Agentic Multimodal DeepSearch.

## 1. Fast Success Path

When evidence is strong:

```text
retrieval
→ semantic verification
→ sufficient evidence
→ stop after Round 1
```

The system avoids unnecessary additional reasoning.

## 2. Explicit Abstention

When retrieved evidence is incomplete:

```text
incomplete evidence
→ low confidence
→ grounded abstention
```

The answer generator does not hallucinate missing visual details.

## 3. Iterative Evidence Seeking

When specific requirements are missing:

```text
identify missing evidence
→ generate focused retrieval query
→ search again
```

Round 2 searches for information rather than blindly repeating the original request.

## 4. Separation of Retrieval and Verification

The Round-2 retrieval objective may change:

```text
several people visible in background, red bicycle nearby
```

while the semantic verification objective remains:

```text
dog beside water
+
several people
+
red bicycle
+
dog not swimming
+
no boat
```

This preserves the original meaning of the user request.

## 5. Conservative Final Arbitration

The system distinguishes:

```text
exploration
```

from:

```text
final ranking
```

Round 2 can discover useful evidence without automatically replacing a stronger previously selected candidate.

---

# End-to-End Behavior Summary

| Scenario | Rounds | Final Match | Confidence | Behavior |
|---|---:|---|---|---|
| Person riding bicycle | 1 | True | High | Answer confidently |
| Complex dog / water / bicycle query | 2 | False | Low | Abstain |
| Round-2 evidence search | 2 | Incomplete | Low | Explore but preserve conservative final ranking |

---

# Pipeline Demonstrated by These Examples

```text
User Query
    ↓
Query Planning
    ↓
DeepSearch Retriever V2
    ↓
Weighted Reciprocal Rank Fusion
    ↓
Multimodal Semantic Verification
    ↓
Evidence Sufficiency
    ↓
If needed:
Evidence-Seeking Round 2
    ↓
Bounded Cross-Round Promotion
    ↓
Structured Evidence Builder
    ↓
Grounded Multimodal Answer Generator
    ↓
Answer or Explicit Abstention
```

These cases demonstrate that the system is designed not only to retrieve relevant images, but also to reason about whether the retrieved visual evidence is sufficient to support the final answer.

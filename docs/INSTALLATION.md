# Installation and Reproducibility

This document describes how to set up **Agentic Multimodal DeepSearch** and reproduce the working environment used for the current implementation.

The project was validated on:

```text
Python:       3.11.17
PyTorch:      2.14.1+cu130
torchvision:  0.29.1+cu130
Transformers: 5.19.0
NumPy:        2.4.6
Pillow:       12.3.0
tqdm:         4.70.1
safetensors:  0.8.0
HF Hub:       2.2.0
```

The end-to-end pipeline uses GPU inference for the query planner, multimodal verifier, query refiner, retriever, and grounded answer generator.

---

# 1. Clone the Repository

```bash
git clone git@github.com:parthburujwale/agentic-multimodal-deepsearch.git
cd agentic-multimodal-deepsearch
```

HTTPS may also be used if preferred.

---

# 2. Create a Python Environment

A Python 3.11 environment is recommended because the validated environment uses Python 3.11.17.

Using Conda:

```bash
conda create -n deepsearch python=3.11 -y
conda activate deepsearch
```

---

# 3. Install PyTorch

PyTorch should be installed separately because the correct package depends on the machine's CUDA, ROCm, Apple Silicon, or CPU configuration.

The validated PSC environment uses:

```text
torch==2.14.1+cu130
torchvision==0.29.1+cu130
```

Install a PyTorch build that is compatible with your hardware and CUDA runtime before installing the remaining dependencies.

Verify:

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

# 4. Install Project Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

The minimal validated dependency set is:

```text
transformers==5.19.0
numpy==2.4.6
Pillow==12.3.0
tqdm==4.70.1
safetensors==0.8.0
huggingface_hub==2.2.0
requests
```

The current core pipeline does not require pandas, matplotlib, scikit-learn, or accelerate.

---

# 5. COCO Dataset Layout

The current implementation expects the COCO 2017 validation images and captions.

Expected layout:

```text
data/
└── coco/
    ├── val2017/
    │   ├── 000000000139.jpg
    │   ├── ...
    │   └── ...
    └── annotations/
        └── captions_val2017.json
```

The trained retriever was developed using COCO train2017 data, while the active retrieval corpus contains 5,000 COCO val2017 images.

Dataset files are not intended to be stored directly in Git.

---

# 6. Retriever Checkpoint Layout

The current grounded orchestrator expects the trained DeepSearch Retriever V2 checkpoint under:

```text
experiments/
└── deepsearch_retriever_v2/
    ├── deepsearch_retriever_v2.pt
    └── ...
```

The current orchestrator also uses this directory as the retriever processor/configuration location.

If your checkpoint is stored elsewhere, update the corresponding paths in:

```text
src/rag/grounded_deepsearch.py
```

---

# 7. Hugging Face Models

The current system uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-VL-3B-Instruct
```

These models are downloaded automatically by Hugging Face Transformers when first used.

For systems with limited home-directory quota, define a dedicated Hugging Face cache:

```bash
export HF_HOME=/path/to/large/storage/hf_cache
export HF_HUB_CACHE=/path/to/large/storage/hf_cache/hub
export TRANSFORMERS_CACHE=/path/to/large/storage/hf_cache/transformers
```

An authenticated Hugging Face token is optional for the models currently used, but authentication can provide higher Hub rate limits.

---

# 8. Run Grounded DeepSearch

From the repository root:

```bash
python src/rag/grounded_deepsearch.py \
  --query "Find an image of a person riding a bicycle." \
  --output experiments/grounded_deepsearch_result.json
```

The pipeline performs:

```text
Query
→ Query Planning
→ DeepSearch Retriever V2
→ Weighted Reciprocal Rank Fusion
→ Multimodal Semantic Verification
→ Evidence Sufficiency
→ Optional Round-2 Evidence Search
→ Bounded Cross-Round Promotion
→ Structured Evidence Building
→ Grounded Multimodal Answer Generation
```

---

# 9. Reproducibility Settings

The final retrieval pipeline uses deterministic inference settings:

```text
Seed: 42
cuDNN benchmark: False
cuDNN deterministic: True
TF32: disabled
PyTorch deterministic algorithms: enabled
Attention implementation: eager
Generation: do_sample=False
```

The four-case reproducibility test produced:

```text
Exact matches: 4 / 4
All cases reproducible: True
```

The relevant implementation is located in:

```text
src/agents/reproducibility.py
src/evaluation/test_iterative_reproducibility.py
```

---

# 10. PSC Bridges-2 Setup

The project was also validated on PSC Bridges-2.

Project root:

```text
/ocean/projects/bio260013p/pburujwale/multimodal-deepsearch
```

Validated Conda environment:

```text
/ocean/projects/bio260013p/pburujwale/multimodal-deepsearch/envs/deepsearch-psc
```

## Activate the environment

```bash
cd /ocean/projects/bio260013p/pburujwale/multimodal-deepsearch

conda activate /ocean/projects/bio260013p/pburujwale/multimodal-deepsearch/envs/deepsearch-psc
```

## Configure Hugging Face cache

```bash
cd /ocean/projects/bio260013p/pburujwale/multimodal-deepsearch

export HF_HOME=/ocean/projects/bio260013p/pburujwale/multimodal-deepsearch/hf_cache
export HF_HUB_CACHE=/ocean/projects/bio260013p/pburujwale/multimodal-deepsearch/hf_cache/hub
export TRANSFORMERS_CACHE=/ocean/projects/bio260013p/pburujwale/multimodal-deepsearch/hf_cache/transformers
```

## Request an interactive H100

Do not run model inference or training jobs on a login node.

From the project root:

```bash
cd /ocean/projects/bio260013p/pburujwale/multimodal-deepsearch

interact -A bio260013p -p GPU-shared --gres=gpu:h100-80:1 -t 01:00:00
```

After the allocation begins, activate the environment again if needed:

```bash
cd /ocean/projects/bio260013p/pburujwale/multimodal-deepsearch

conda activate /ocean/projects/bio260013p/pburujwale/multimodal-deepsearch/envs/deepsearch-psc
```

Then set the Hugging Face cache variables and run the pipeline.

---

# 11. Validate the Installation

Check syntax without loading models:

```bash
python -m py_compile src/rag/evidence_builder.py
python -m py_compile src/rag/multimodal_answer_generator.py
python -m py_compile src/rag/grounded_deepsearch.py
```

On a GPU node, run the end-to-end smoke test:

```bash
python src/rag/grounded_deepsearch.py \
  --query "Find an image of a person riding a bicycle." \
  --output experiments/grounded_deepsearch_result.json
```

A successful run should produce a structured grounded response and write the final JSON file to the requested output path.

---

# 12. Important Notes

- The current public environment specification is intentionally minimal.
- `requirements.txt` contains direct runtime dependencies used by the project rather than the full Conda environment.
- PyTorch is installed separately because hardware-specific builds differ across systems.
- COCO data and large model checkpoints should not be committed to Git.
- The frozen 200-query final benchmark should not be used for architecture tuning.
- The retrieval architecture used for the reported final evaluation should remain unchanged when reproducing the published metrics.

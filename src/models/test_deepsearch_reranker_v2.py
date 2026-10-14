import sys

import torch


sys.path.append("..")

from models.deepsearch_reranker_v2 import (
    DeepSearchRerankerV2
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Using device: {device}"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = DeepSearchRerankerV2(
    embedding_dim=512,
    hidden_dim=1024,
    dropout=0.1,
    initial_alpha=0.01
)


model = model.to(
    device
)


# --------------------------------------------------
# Parameter counts
# --------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Initial alpha: "
    f"{model.alpha.item():.6f}"
)


# --------------------------------------------------
# Dummy batch
#
# 4 query-image candidate pairs
# --------------------------------------------------

batch_size = 4


text_embeddings = torch.randn(
    batch_size,
    512,
    device=device
)


image_embeddings = torch.randn(
    batch_size,
    512,
    device=device
)


# Normalize just like Retriever V2 embeddings
text_embeddings = torch.nn.functional.normalize(
    text_embeddings,
    dim=-1
)

image_embeddings = torch.nn.functional.normalize(
    image_embeddings,
    dim=-1
)


retriever_scores = torch.tensor(
    [
        0.42,
        0.39,
        0.36,
        0.31
    ],
    dtype=torch.float32,
    device=device
)


# --------------------------------------------------
# Forward
# --------------------------------------------------

model.eval()


with torch.inference_mode():

    outputs = model(
        text_embeddings=
            text_embeddings,

        image_embeddings=
            image_embeddings,

        retriever_scores=
            retriever_scores
    )


print(
    "Retriever scores:"
)

print(
    retriever_scores
)


print(
    "\nCorrection scores:"
)

print(
    outputs[
        "correction_scores"
    ]
)


print(
    "\nFinal scores:"
)

print(
    outputs[
        "final_scores"
    ]
)


print(
    "\nAlpha:"
)

print(
    outputs[
        "alpha"
    ]
)
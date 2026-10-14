import sys

import torch

from PIL import Image
from transformers import CLIPProcessor


sys.path.append("..")

from models.deepsearch_cross_attention_reranker import (
    DeepSearchCrossAttentionReranker
)


# --------------------------------------------------
# 1. Device
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
# 2. Model configuration
# --------------------------------------------------

model_name = (
    "openai/"
    "clip-vit-base-patch32"
)


processor = CLIPProcessor.from_pretrained(
    model_name
)


model = DeepSearchCrossAttentionReranker(
    backbone_name=model_name,
    fusion_dim=512,
    num_heads=8,
    ff_hidden_dim=1024,
    scorer_hidden_dim=512,
    dropout=0.1,
    initial_alpha=0.01,
    freeze_backbone=True
)


model = model.to(
    device
)


# --------------------------------------------------
# 3. Parameter counts
# --------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
    if parameter.requires_grad
)


print(
    f"\nTotal parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Trainable percentage: "
    f"{100 * trainable_parameters / total_parameters:.4f}%"
)

print(
    f"Initial alpha: "
    f"{model.alpha.item():.6f}"
)


# --------------------------------------------------
# 4. Dummy text-image pairs
# --------------------------------------------------

images = [

    Image.new(
        "RGB",
        (224, 224),
        color="brown"
    ),

    Image.new(
        "RGB",
        (224, 224),
        color="green"
    )
]


captions = [

    "a brown dog running through grass",

    "a green field with trees"
]


inputs = processor(
    text=captions,
    images=images,
    return_tensors="pt",
    padding=True,
    truncation=True
)


inputs = {
    key: value.to(device)

    for key, value
    in inputs.items()
}


# --------------------------------------------------
# 5. Pretend these came from Retriever V2
# --------------------------------------------------

retriever_scores = torch.tensor(
    [
        0.42,
        0.38
    ],
    dtype=torch.float32,
    device=device
)


# --------------------------------------------------
# 6. Forward
# --------------------------------------------------

model.eval()


with torch.inference_mode():

    outputs = model(

        pixel_values=(
            inputs[
                "pixel_values"
            ]
        ),

        input_ids=(
            inputs[
                "input_ids"
            ]
        ),

        attention_mask=(
            inputs[
                "attention_mask"
            ]
        ),

        retriever_scores=
            retriever_scores
    )


# --------------------------------------------------
# 7. Inspect outputs
# --------------------------------------------------

print(
    "\nRetriever scores:"
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


print(
    "\nAttention weight shape:"
)

print(
    outputs[
        "attention_weights"
    ].shape
)
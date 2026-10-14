import sys

import torch

from PIL import Image
from transformers import CLIPProcessor


sys.path.append("..")

from models.deepsearch_cross_attention_reranker_v4 import (
    DeepSearchCrossAttentionRerankerV4
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Using device: {device}"
)


model_name = (
    "openai/"
    "clip-vit-base-patch32"
)


processor = CLIPProcessor.from_pretrained(
    model_name
)


model = (
    DeepSearchCrossAttentionRerankerV4(

        backbone_name=model_name,

        fusion_dim=512,

        num_heads=8,

        ff_hidden_dim=1024,

        scorer_hidden_dim=512,

        dropout=0.1,

        max_correction=0.02,

        freeze_backbone=True
    )
)


model = model.to(
    device
)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    f"\nTotal parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


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


retriever_scores = torch.tensor(
    [
        0.4200,
        0.3800
    ],
    dtype=torch.float32,
    device=device
)


model.eval()


with torch.inference_mode():

    outputs = model(

        pixel_values=
            inputs[
                "pixel_values"
            ],

        input_ids=
            inputs[
                "input_ids"
            ],

        attention_mask=
            inputs[
                "attention_mask"
            ],

        retriever_scores=
            retriever_scores
    )


print(
    "\nRetriever scores:"
)

print(
    retriever_scores
)


print(
    "\nRaw corrections:"
)

print(
    outputs[
        "raw_corrections"
    ]
)


print(
    "\nBounded corrections:"
)

print(
    outputs[
        "bounded_corrections"
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
    "\nMaximum absolute correction:"
)

print(
    outputs[
        "bounded_corrections"
    ]
    .abs()
    .max()
    .item()
)
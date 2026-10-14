import sys

import torch

from PIL import Image
from transformers import CLIPProcessor


sys.path.append("..")

from models.deepsearch_retriever import (
    DeepSearchRetriever
)


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print(
    f"Using device: {device}"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model_name = (
    "openai/"
    "clip-vit-base-patch32"
)


processor = CLIPProcessor.from_pretrained(
    model_name
)


model = DeepSearchRetriever(
    backbone_name=model_name,
    embedding_dim=512,
    hidden_dim=1024,
    dropout=0.1,
    freeze_backbones=True
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
    f"Trainable percentage: "
    f"{100 * trainable_parameters / total_parameters:.4f}%"
)


# --------------------------------------------------
# Dummy image
# --------------------------------------------------

image = Image.new(
    "RGB",
    (224, 224),
    color="white"
)


caption = (
    "a white image"
)


inputs = processor(
    text=[caption],
    images=[image],
    return_tensors="pt",
    padding=True
)


inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

model.eval()


with torch.inference_mode():

    outputs = model(
        pixel_values=(
            inputs["pixel_values"]
        ),

        input_ids=(
            inputs["input_ids"]
        ),

        attention_mask=(
            inputs["attention_mask"]
        )
    )


print(
    "Image embedding shape:",
    outputs[
        "image_embeddings"
    ].shape
)


print(
    "Text embedding shape:",
    outputs[
        "text_embeddings"
    ].shape
)


print(
    "Similarity matrix shape:",
    outputs[
        "logits_per_text"
    ].shape
)
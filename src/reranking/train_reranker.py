import json
import os
import random
import sys

import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


# --------------------------------------------------
# Import our reranker
# --------------------------------------------------

sys.path.append("..")

from models.deepsearch_reranker import (
    DeepSearchReranker
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

image_folder = (
    "../../data/coco/train2017"
)

reranker_dataset_file = (
    "../../data/coco/"
    "reranker_dataset_50k.json"
)

backbone_name = (
    "openai/"
    "clip-vit-base-patch32"
)

output_dir = (
    "../../experiments/"
    "deepsearch_reranker_v1"
)


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

batch_size = 16

num_epochs = 3

learning_rate = 1e-4

weight_decay = 0.01

# Positive + 4 negatives = 5 candidates/query
num_negatives = 4

hidden_dim = 1024
embedding_dim = 512
dropout = 0.1

random_seed = 42


# --------------------------------------------------
# 2. Random seed
# --------------------------------------------------

random.seed(
    random_seed
)

torch.manual_seed(
    random_seed
)


# --------------------------------------------------
# 3. Select device
# --------------------------------------------------

if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

elif torch.backends.mps.is_available():

    device = torch.device(
        "mps"
    )

else:

    device = torch.device(
        "cpu"
    )


print(
    f"Using device: {device}"
)


# --------------------------------------------------
# 4. Load reranker dataset
# --------------------------------------------------

with open(
    reranker_dataset_file,
    "r"
) as file:

    reranker_data = json.load(
        file
    )


training_examples = (
    reranker_data[
        "examples"
    ]
)


print(
    f"Loaded "
    f"{len(training_examples)} "
    f"reranker queries"
)


# --------------------------------------------------
# 5. Dataset
# --------------------------------------------------

class RerankerDataset(Dataset):

    def __init__(
        self,
        examples,
        image_folder,
        num_negatives
    ):

        self.examples = examples

        self.image_folder = (
            image_folder
        )

        self.num_negatives = (
            num_negatives
        )


    def __len__(self):

        return len(
            self.examples
        )


    def __getitem__(
        self,
        index
    ):

        example = self.examples[
            index
        ]


        caption = example[
            "caption"
        ]


        # --------------------------------------------------
        # Positive image
        # --------------------------------------------------

        positive_file = (
            example[
                "positive"
            ][
                "file_name"
            ]
        )


        positive_path = os.path.join(
            self.image_folder,
            positive_file
        )


        positive_image = Image.open(
            positive_path
        ).convert(
            "RGB"
        )


        # --------------------------------------------------
        # Candidate negatives
        # --------------------------------------------------

        negatives = example[
            "negative_candidates"
        ]


        # We sample from the top retrieved candidates.
        #
        # Using several retrieved negatives gives us
        # difficult examples without always training
        # against exactly the same candidates.
        if len(negatives) >= self.num_negatives:

            selected_negatives = random.sample(
                negatives,
                self.num_negatives
            )

        else:

            selected_negatives = negatives


        negative_images = []


        for negative in selected_negatives:

            negative_path = os.path.join(
                self.image_folder,
                negative[
                    "file_name"
                ]
            )


            negative_image = Image.open(
                negative_path
            ).convert(
                "RGB"
            )


            negative_images.append(
                negative_image
            )


        return (
            caption,
            positive_image,
            negative_images
        )


# --------------------------------------------------
# 6. Processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    backbone_name
)


# --------------------------------------------------
# 7. Collate function
# --------------------------------------------------

def collate_fn(batch):

    captions = []

    candidate_images = []

    candidate_counts = []


    for (
        caption,
        positive_image,
        negative_images
    ) in batch:

        images = [
            positive_image
        ] + negative_images


        candidate_counts.append(
            len(images)
        )


        candidate_images.extend(
            images
        )


        # Repeat caption for every candidate.
        captions.extend(
            [
                caption
                for _ in images
            ]
        )


    inputs = processor(
        text=captions,
        images=candidate_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )


    return (
        inputs,
        candidate_counts
    )


# --------------------------------------------------
# 8. Dataset + DataLoader
# --------------------------------------------------

dataset = RerankerDataset(
    examples=training_examples,
    image_folder=image_folder,
    num_negatives=num_negatives
)


dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)


print(
    f"Number of training batches: "
    f"{len(dataloader)}"
)


print(
    f"Candidates per query: "
    f"{num_negatives + 1}"
)


# --------------------------------------------------
# 9. Load reranker
# --------------------------------------------------

model = DeepSearchReranker(
    backbone_name=backbone_name,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    dropout=dropout,
    freeze_backbone=True
)


model = model.to(
    device
)


# --------------------------------------------------
# 10. Count parameters
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
# 11. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ],
    lr=learning_rate,
    weight_decay=weight_decay
)


# --------------------------------------------------
# 12. Training mode
# --------------------------------------------------

model.train()


# --------------------------------------------------
# 13. Training loop
# --------------------------------------------------

for epoch in range(
    num_epochs
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/"
        f"{num_epochs}"
    )


    running_loss = 0.0

    running_accuracy = 0.0


    for batch_index, (
        inputs,
        candidate_counts
    ) in enumerate(
        dataloader
    ):

        inputs = {
            key: value.to(
                device,
                non_blocking=True
            )

            for key, value
            in inputs.items()
        }


        optimizer.zero_grad()


        # --------------------------------------------------
        # Score every text-image candidate pair
        # --------------------------------------------------

        scores = model(
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
            )
        )


        # --------------------------------------------------
        # Reconstruct one candidate list per query
        #
        # Candidate 0 is always the positive image.
        # --------------------------------------------------

        query_losses = []

        query_correct = []


        offset = 0


        for candidate_count in candidate_counts:

            query_scores = scores[
                offset:
                offset + candidate_count
            ]


            # Shape:
            # [1, number_of_candidates]
            query_scores = (
                query_scores.unsqueeze(
                    0
                )
            )


            # Candidate 0 is positive.
            target = torch.zeros(
                1,
                dtype=torch.long,
                device=device
            )


            query_loss = (
                F.cross_entropy(
                    query_scores,
                    target
                )
            )


            query_losses.append(
                query_loss
            )


            predicted_index = int(
                torch.argmax(
                    query_scores,
                    dim=1
                ).item()
            )


            query_correct.append(
                1.0
                if predicted_index == 0
                else 0.0
            )


            offset += (
                candidate_count
            )


        # --------------------------------------------------
        # Average listwise ranking loss
        # --------------------------------------------------

        loss = torch.stack(
            query_losses
        ).mean()


        loss.backward()


        optimizer.step()


        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        batch_accuracy = (
            sum(query_correct)
            / len(query_correct)
        )


        running_loss += (
            loss.item()
        )


        running_accuracy += (
            batch_accuracy
        )


        average_loss = (
            running_loss
            / (batch_index + 1)
        )


        average_accuracy = (
            running_accuracy
            / (batch_index + 1)
        )


        if (
            batch_index + 1
        ) % 25 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(dataloader)} "
                f"| Loss: "
                f"{average_loss:.4f} "
                f"| Query Accuracy: "
                f"{average_accuracy:.4f}"
            )


    print(
        f"\nEpoch {epoch + 1} complete"
    )


    print(
        f"Average loss: "
        f"{average_loss:.4f}"
    )


    print(
        f"Average query accuracy: "
        f"{average_accuracy:.4f}"
    )


# --------------------------------------------------
# 14. Save reranker
# --------------------------------------------------

os.makedirs(
    output_dir,
    exist_ok=True
)


checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "backbone_name":
        backbone_name,

    "embedding_dim":
        embedding_dim,

    "hidden_dim":
        hidden_dim,

    "dropout":
        dropout,

    "num_negatives":
        num_negatives,

    "batch_size":
        batch_size,

    "num_epochs":
        num_epochs,

    "learning_rate":
        learning_rate,

    "weight_decay":
        weight_decay
}


checkpoint_path = os.path.join(
    output_dir,
    "deepsearch_reranker_v1.pt"
)


torch.save(
    checkpoint,
    checkpoint_path
)


processor.save_pretrained(
    output_dir
)


print(
    "\nSaved DeepSearch reranker to:"
)

print(
    checkpoint_path
)
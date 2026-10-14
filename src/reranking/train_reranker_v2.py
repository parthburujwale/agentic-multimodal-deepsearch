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
# Import DeepSearch models
# --------------------------------------------------

sys.path.append("..")

from models.deepsearch_retriever_v2 import (
    DeepSearchRetrieverV2
)

from models.deepsearch_reranker_v2 import (
    DeepSearchRerankerV2
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

retriever_checkpoint_path = (
    "../../experiments/"
    "deepsearch_retriever_v2/"
    "deepsearch_retriever_v2.pt"
)

processor_path = (
    "../../experiments/"
    "deepsearch_retriever_v2"
)

output_dir = (
    "../../experiments/"
    "deepsearch_reranker_v2"
)


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

train_size = 45000
validation_size = 5000

batch_size = 16

num_epochs = 3

learning_rate = 1e-4
weight_decay = 0.01

num_negatives = 4

embedding_dim = 512
hidden_dim = 1024
dropout = 0.1

initial_alpha = 0.01

random_seed = 42


# --------------------------------------------------
# 2. Reproducibility
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

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


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

    reranker_data = json.load(file)


all_examples = (
    reranker_data["examples"]
)


print(
    f"Loaded {len(all_examples)} "
    f"reranker examples"
)


# --------------------------------------------------
# 5. Train / validation split
# --------------------------------------------------

indices = list(
    range(
        len(all_examples)
    )
)


random.shuffle(
    indices
)


train_indices = (
    indices[:train_size]
)

validation_indices = (
    indices[
        train_size:
        train_size + validation_size
    ]
)


train_examples = [
    all_examples[index]
    for index in train_indices
]


validation_examples = [
    all_examples[index]
    for index in validation_indices
]


print(
    f"Training queries: "
    f"{len(train_examples)}"
)

print(
    f"Validation queries: "
    f"{len(validation_examples)}"
)


# --------------------------------------------------
# 6. Dataset
# --------------------------------------------------

class ResidualRerankerDataset(Dataset):

    def __init__(
        self,
        examples,
        image_folder,
        num_negatives,
        training=True
    ):

        self.examples = examples
        self.image_folder = image_folder

        self.num_negatives = (
            num_negatives
        )

        self.training = training


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
        # Positive
        # --------------------------------------------------

        positive = example[
            "positive"
        ]


        positive_path = os.path.join(
            self.image_folder,
            positive[
                "file_name"
            ]
        )


        positive_image = Image.open(
            positive_path
        ).convert(
            "RGB"
        )


        positive_score = float(
            positive[
                "retrieval_score"
            ]
        )


        # --------------------------------------------------
        # Negatives
        # --------------------------------------------------

        negatives = example[
            "negative_candidates"
        ]


        if self.training:

            selected_negatives = (
                random.sample(
                    negatives,
                    self.num_negatives
                )
            )

        else:

            # Fixed validation candidates
            # for deterministic evaluation.
            selected_negatives = (
                negatives[
                    :self.num_negatives
                ]
            )


        negative_images = []

        negative_scores = []


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


            negative_scores.append(
                float(
                    negative[
                        "retrieval_score"
                    ]
                )
            )


        images = (
            [positive_image]
            + negative_images
        )


        retriever_scores = (
            [positive_score]
            + negative_scores
        )


        return (
            caption,
            images,
            retriever_scores
        )


# --------------------------------------------------
# 7. Processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    processor_path
)


# --------------------------------------------------
# 8. Collate function
# --------------------------------------------------

def collate_fn(batch):

    captions = []

    all_images = []

    all_retriever_scores = []


    for (
        caption,
        images,
        retriever_scores
    ) in batch:

        captions.append(
            caption
        )

        all_images.extend(
            images
        )

        all_retriever_scores.append(
            retriever_scores
        )


    # --------------------------------------------------
    # Process captions once per query
    # --------------------------------------------------

    text_inputs = processor(
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )


    # --------------------------------------------------
    # Process all candidate images
    # --------------------------------------------------

    image_inputs = processor(
        images=all_images,
        return_tensors="pt"
    )


    retriever_scores = torch.tensor(
        all_retriever_scores,
        dtype=torch.float32
    )


    return (
        text_inputs,
        image_inputs,
        retriever_scores
    )


# --------------------------------------------------
# 9. DataLoaders
# --------------------------------------------------

train_dataset = ResidualRerankerDataset(
    examples=train_examples,
    image_folder=image_folder,
    num_negatives=num_negatives,
    training=True
)


validation_dataset = ResidualRerankerDataset(
    examples=validation_examples,
    image_folder=image_folder,
    num_negatives=num_negatives,
    training=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)


print(
    f"Training batches: "
    f"{len(train_loader)}"
)

print(
    f"Validation batches: "
    f"{len(validation_loader)}"
)

print(
    f"Candidates per query: "
    f"{num_negatives + 1}"
)


# --------------------------------------------------
# 10. Load frozen DeepSearch Retriever V2
# --------------------------------------------------

print(
    "\nLoading DeepSearch Retriever V2..."
)


retriever_checkpoint = torch.load(
    retriever_checkpoint_path,
    map_location=device,
    weights_only=False
)


retriever = DeepSearchRetrieverV2(
    backbone_name=(
        retriever_checkpoint[
            "backbone_name"
        ]
    ),

    embedding_dim=(
        retriever_checkpoint[
            "embedding_dim"
        ]
    ),

    hidden_dim=(
        retriever_checkpoint[
            "hidden_dim"
        ]
    ),

    dropout=(
        retriever_checkpoint[
            "dropout"
        ]
    ),

    freeze_backbone=True
)


retriever.load_state_dict(
    retriever_checkpoint[
        "model_state_dict"
    ]
)


retriever = retriever.to(
    device
)

retriever.eval()


for parameter in retriever.parameters():

    parameter.requires_grad = False


print(
    "Retriever V2 loaded and frozen"
)


# --------------------------------------------------
# 11. Load Residual Reranker V2
# --------------------------------------------------

reranker = DeepSearchRerankerV2(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    dropout=dropout,
    initial_alpha=initial_alpha
)


reranker = reranker.to(
    device
)


total_parameters = sum(
    parameter.numel()
    for parameter
    in reranker.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter
    in reranker.parameters()
    if parameter.requires_grad
)


print(
    f"Reranker parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Initial alpha: "
    f"{reranker.alpha.item():.6f}"
)


# --------------------------------------------------
# 12. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    reranker.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)


# --------------------------------------------------
# 13. Encode one batch using frozen Retriever V2
# --------------------------------------------------

def build_reranker_inputs(
    text_inputs,
    image_inputs,
    retriever_scores
):

    text_inputs = {
        key: value.to(
            device,
            non_blocking=True
        )

        for key, value
        in text_inputs.items()
    }


    image_inputs = {
        key: value.to(
            device,
            non_blocking=True
        )

        for key, value
        in image_inputs.items()
    }


    retriever_scores = (
        retriever_scores.to(
            device,
            non_blocking=True
        )
    )


    with torch.inference_mode():

        # ------------------------------------------
        # One text embedding per query
        # [B, 512]
        # ------------------------------------------

        text_embeddings = (
            retriever.encode_text(
                text_inputs[
                    "input_ids"
                ],

                text_inputs[
                    "attention_mask"
                ]
            )
        )


        # ------------------------------------------
        # One image embedding per candidate
        # [B * candidates, 512]
        # ------------------------------------------

        image_embeddings = (
            retriever.encode_image(
                image_inputs[
                    "pixel_values"
                ]
            )
        )


    current_batch_size = (
        text_embeddings.shape[0]
    )


    candidate_count = (
        retriever_scores.shape[1]
    )


    # --------------------------------------------------
    # Expand text embedding across candidates
    # --------------------------------------------------

    expanded_text_embeddings = (
        text_embeddings
        .unsqueeze(1)
        .expand(
            -1,
            candidate_count,
            -1
        )
    )


    # --------------------------------------------------
    # Restore image candidate grouping
    # --------------------------------------------------

    image_embeddings = (
        image_embeddings.view(
            current_batch_size,
            candidate_count,
            -1
        )
    )


    return (
        expanded_text_embeddings,
        image_embeddings,
        retriever_scores
    )


# --------------------------------------------------
# 14. Run one epoch
# --------------------------------------------------

def run_epoch(
    dataloader,
    training
):

    if training:

        reranker.train()

    else:

        reranker.eval()


    running_loss = 0.0

    correct_final = 0

    correct_retriever = 0

    total_queries = 0


    for batch_index, batch in enumerate(
        dataloader
    ):

        (
            text_inputs,
            image_inputs,
            retriever_scores
        ) = batch


        (
            text_embeddings,
            image_embeddings,
            retriever_scores
        ) = build_reranker_inputs(
            text_inputs,
            image_inputs,
            retriever_scores
        )


        current_batch_size = (
            text_embeddings.shape[0]
        )


        candidate_count = (
            text_embeddings.shape[1]
        )


        # --------------------------------------------------
        # Flatten pairs for reranker
        # --------------------------------------------------

        text_flat = (
            text_embeddings.reshape(
                -1,
                text_embeddings.shape[-1]
            )
        )


        image_flat = (
            image_embeddings.reshape(
                -1,
                image_embeddings.shape[-1]
            )
        )


        score_flat = (
            retriever_scores.reshape(
                -1
            )
        )


        if training:

            optimizer.zero_grad()


        with torch.set_grad_enabled(
            training
        ):

            outputs = reranker(
                text_embeddings=
                    text_flat,

                image_embeddings=
                    image_flat,

                retriever_scores=
                    score_flat
            )


            final_scores = (
                outputs[
                    "final_scores"
                ].view(
                    current_batch_size,
                    candidate_count
                )
            )


            # Positive is candidate 0
            targets = torch.zeros(
                current_batch_size,
                dtype=torch.long,
                device=device
            )


            loss = F.cross_entropy(
                final_scores,
                targets
            )


            if training:

                loss.backward()

                optimizer.step()


        # --------------------------------------------------
        # Accuracy before reranking
        # --------------------------------------------------

        retriever_predictions = (
            torch.argmax(
                retriever_scores,
                dim=1
            )
        )


        correct_retriever += (
            retriever_predictions
            == 0
        ).sum().item()


        # --------------------------------------------------
        # Accuracy after reranking
        # --------------------------------------------------

        final_predictions = (
            torch.argmax(
                final_scores,
                dim=1
            )
        )


        correct_final += (
            final_predictions
            == 0
        ).sum().item()


        total_queries += (
            current_batch_size
        )


        running_loss += (
            loss.item()
            * current_batch_size
        )


        if (
            training
            and (
                batch_index + 1
            ) % 50 == 0
        ):

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(dataloader)} "
                f"| Loss: "
                f"{running_loss / total_queries:.4f} "
                f"| Retriever Acc: "
                f"{correct_retriever / total_queries:.4f} "
                f"| Reranked Acc: "
                f"{correct_final / total_queries:.4f} "
                f"| Alpha: "
                f"{reranker.alpha.item():.5f}"
            )


    average_loss = (
        running_loss
        / total_queries
    )


    retriever_accuracy = (
        correct_retriever
        / total_queries
    )


    reranked_accuracy = (
        correct_final
        / total_queries
    )


    return (
        average_loss,
        retriever_accuracy,
        reranked_accuracy
    )


# --------------------------------------------------
# 15. Training
# --------------------------------------------------

os.makedirs(
    output_dir,
    exist_ok=True
)


best_validation_accuracy = -1.0


for epoch in range(
    num_epochs
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Epoch "
        f"{epoch + 1}/"
        f"{num_epochs}"
    )

    print(
        f"{'=' * 60}"
    )


    (
        train_loss,
        train_retriever_accuracy,
        train_reranked_accuracy
    ) = run_epoch(
        train_loader,
        training=True
    )


    print(
        "\nTraining Results"
    )

    print(
        "-" * 50
    )

    print(
        f"Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Retriever accuracy: "
        f"{train_retriever_accuracy:.4f}"
    )

    print(
        f"Reranked accuracy: "
        f"{train_reranked_accuracy:.4f}"
    )

    print(
        f"Alpha: "
        f"{reranker.alpha.item():.6f}"
    )


    # --------------------------------------------------
    # Held-out validation
    # --------------------------------------------------

    with torch.inference_mode():

        (
            validation_loss,
            validation_retriever_accuracy,
            validation_reranked_accuracy
        ) = run_epoch(
            validation_loader,
            training=False
        )


    print(
        "\nHeld-Out Reranker Validation"
    )

    print(
        "-" * 50
    )

    print(
        f"Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Retriever accuracy: "
        f"{validation_retriever_accuracy:.4f}"
    )

    print(
        f"Reranked accuracy: "
        f"{validation_reranked_accuracy:.4f}"
    )

    print(
        f"Improvement: "
        f"{validation_reranked_accuracy - validation_retriever_accuracy:+.4f}"
    )


    # --------------------------------------------------
    # Save best validation checkpoint
    # --------------------------------------------------

    if (
        validation_reranked_accuracy
        > best_validation_accuracy
    ):

        best_validation_accuracy = (
            validation_reranked_accuracy
        )


        checkpoint = {

            "model_state_dict":
                reranker.state_dict(),

            "embedding_dim":
                embedding_dim,

            "hidden_dim":
                hidden_dim,

            "dropout":
                dropout,

            "alpha":
                reranker.alpha.item(),

            "num_negatives":
                num_negatives,

            "train_size":
                train_size,

            "validation_size":
                validation_size,

            "batch_size":
                batch_size,

            "learning_rate":
                learning_rate,

            "epoch":
                epoch + 1,

            "validation_accuracy":
                validation_reranked_accuracy,

            "retriever_validation_accuracy":
                validation_retriever_accuracy
        }


        checkpoint_path = os.path.join(
            output_dir,
            "deepsearch_reranker_v2_best.pt"
        )


        torch.save(
            checkpoint,
            checkpoint_path
        )


        print(
            "\nSaved new best checkpoint:"
        )

        print(
            checkpoint_path
        )


# --------------------------------------------------
# 16. Final summary
# --------------------------------------------------

print(
    "\nTraining complete"
)

print(
    f"Best held-out reranker accuracy: "
    f"{best_validation_accuracy:.4f}"
)

print(
    f"Final alpha: "
    f"{reranker.alpha.item():.6f}"
)
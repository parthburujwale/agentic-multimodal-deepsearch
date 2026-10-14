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
# Import our cross-attention reranker
# --------------------------------------------------

sys.path.append("..")

from models.deepsearch_cross_attention_reranker import (
    DeepSearchCrossAttentionReranker
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

image_folder = (
    "../../data/coco/train2017"
)

train_dataset_file = (
    "../../data/coco/"
    "reranker_v3_train_45k.json"
)

dev_dataset_file = (
    "../../data/coco/"
    "reranker_v3_dev_5k.json"
)

backbone_name = (
    "openai/"
    "clip-vit-base-patch32"
)

output_dir = (
    "../../experiments/"
    "deepsearch_cross_attention_reranker_v3"
)


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

batch_size = 12

num_epochs = 3

learning_rate = 5e-5

weight_decay = 0.01

num_training_negatives = 4

candidate_pool_size = 20

fusion_dim = 512

num_heads = 8

ff_hidden_dim = 1024

scorer_hidden_dim = 512

dropout = 0.1

initial_alpha = 0.01

max_alpha = 0.10

residual_regularization_weight = 1.0

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
# 3. Device
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
# 4. Load datasets
# --------------------------------------------------

with open(
    train_dataset_file,
    "r"
) as file:

    train_data = json.load(
        file
    )


with open(
    dev_dataset_file,
    "r"
) as file:

    dev_data = json.load(
        file
    )


all_train_examples = (
    train_data[
        "examples"
    ]
)

all_dev_examples = (
    dev_data[
        "examples"
    ]
)


print(
    f"Original training queries: "
    f"{len(all_train_examples)}"
)

print(
    f"DEV queries: "
    f"{len(all_dev_examples)}"
)


# --------------------------------------------------
# 5. Keep only training examples where positive
#    is inside Retriever V2 top-20
#
# A top-20 reranker cannot help examples whose
# positive image was never retrieved.
# --------------------------------------------------

train_examples = [

    example

    for example in all_train_examples

    if example[
        "positive"
    ][
        "retrieval_rank"
    ] <= candidate_pool_size
]


print(
    f"Train queries with positive "
    f"in top-{candidate_pool_size}: "
    f"{len(train_examples)}"
)


# --------------------------------------------------
# 6. Training Dataset
# --------------------------------------------------

class CrossAttentionTrainingDataset(
    Dataset
):

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


        # ------------------------------------------
        # Positive image
        # ------------------------------------------

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


        # ------------------------------------------
        # Negative candidates
        #
        # Mix:
        # 2 very difficult negatives from top 5
        # 2 semi-hard negatives from ranks 6-20
        #
        # This avoids relying entirely on the
        # absolute hardest candidates.
        # ------------------------------------------

        negatives = example[
            "negative_candidates"
        ]


        hard_pool = (
            negatives[:5]
        )

        semi_hard_pool = (
            negatives[5:20]
        )


        selected_hard = random.sample(
            hard_pool,
            k=min(
                2,
                len(hard_pool)
            )
        )


        remaining_count = (
            self.num_negatives
            - len(selected_hard)
        )


        selected_semi_hard = (
            random.sample(
                semi_hard_pool,
                k=min(
                    remaining_count,
                    len(semi_hard_pool)
                )
            )
        )


        selected_negatives = (
            selected_hard
            + selected_semi_hard
        )


        # Safety fallback
        if len(selected_negatives) < (
            self.num_negatives
        ):

            unused = [

                negative

                for negative in negatives

                if negative
                not in selected_negatives
            ]


            needed = (
                self.num_negatives
                - len(selected_negatives)
            )


            selected_negatives.extend(
                unused[:needed]
            )


        images = [
            positive_image
        ]


        retriever_scores = [
            positive_score
        ]


        for negative in (
            selected_negatives
        ):

            negative_path = os.path.join(
                self.image_folder,
                negative[
                    "file_name"
                ]
            )


            image = Image.open(
                negative_path
            ).convert(
                "RGB"
            )


            images.append(
                image
            )


            retriever_scores.append(
                float(
                    negative[
                        "retrieval_score"
                    ]
                )
            )


        return (
            caption,
            images,
            retriever_scores
        )


# --------------------------------------------------
# 7. DEV Dataset
#
# For actual reranking evaluation we only need
# queries whose positive is in top-20.
#
# The final Recall denominator remains ALL 5000
# DEV queries.
# --------------------------------------------------

class CrossAttentionDevDataset(
    Dataset
):

    def __init__(
        self,
        examples,
        image_folder,
        candidate_pool_size
    ):

        self.total_queries = len(
            examples
        )


        self.examples = [

            example

            for example in examples

            if example[
                "positive"
            ][
                "retrieval_rank"
            ] <= candidate_pool_size
        ]


        self.image_folder = (
            image_folder
        )

        self.candidate_pool_size = (
            candidate_pool_size
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


        # ------------------------------------------
        # Exact top-20 reconstruction
        #
        # Because positive was removed before
        # negative mining:
        #
        # positive + first 19 wrong candidates
        #
        # gives 20 total candidates.
        # ------------------------------------------

        negative_candidates = (
            example[
                "negative_candidates"
            ][
                :self.candidate_pool_size - 1
            ]
        )


        images = [
            positive_image
        ]


        retriever_scores = [
            positive_score
        ]


        for negative in (
            negative_candidates
        ):

            negative_path = os.path.join(
                self.image_folder,
                negative[
                    "file_name"
                ]
            )


            image = Image.open(
                negative_path
            ).convert(
                "RGB"
            )


            images.append(
                image
            )


            retriever_scores.append(
                float(
                    negative[
                        "retrieval_score"
                    ]
                )
            )


        return (
            caption,
            images,
            retriever_scores
        )


# --------------------------------------------------
# 8. Processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    backbone_name
)


# --------------------------------------------------
# 9. Generic collate function
# --------------------------------------------------

def collate_fn(batch):

    repeated_captions = []

    all_images = []

    all_scores = []

    candidate_count = len(
        batch[0][1]
    )


    for (
        caption,
        images,
        retriever_scores
    ) in batch:

        # Repeat caption once for every candidate
        repeated_captions.extend(
            [
                caption
                for _ in images
            ]
        )


        all_images.extend(
            images
        )


        all_scores.extend(
            retriever_scores
        )


    inputs = processor(
        text=repeated_captions,
        images=all_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )


    retriever_scores = torch.tensor(
        all_scores,
        dtype=torch.float32
    )


    return (
        inputs,
        retriever_scores,
        candidate_count
    )


# --------------------------------------------------
# 10. Datasets
# --------------------------------------------------

train_dataset = (
    CrossAttentionTrainingDataset(
        examples=train_examples,
        image_folder=image_folder,
        num_negatives=(
            num_training_negatives
        )
    )
)


dev_dataset = (
    CrossAttentionDevDataset(
        examples=all_dev_examples,
        image_folder=image_folder,
        candidate_pool_size=(
            candidate_pool_size
        )
    )
)


print(
    f"Eligible DEV queries: "
    f"{len(dev_dataset)}"
)

print(
    f"Total DEV queries: "
    f"{dev_dataset.total_queries}"
)


# --------------------------------------------------
# 11. DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)


# DEV has 20 candidate images/query,
# so use a smaller query batch.
dev_loader = DataLoader(
    dev_dataset,
    batch_size=4,
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
    f"DEV batches: "
    f"{len(dev_loader)}"
)


# --------------------------------------------------
# 12. Model
# --------------------------------------------------

model = (
    DeepSearchCrossAttentionReranker(

        backbone_name=
            backbone_name,

        fusion_dim=
            fusion_dim,

        num_heads=
            num_heads,

        ff_hidden_dim=
            ff_hidden_dim,

        scorer_hidden_dim=
            scorer_hidden_dim,

        dropout=
            dropout,

        initial_alpha=
            initial_alpha,

        freeze_backbone=True
    )
)


model = model.to(
    device
)


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
    f"Initial alpha: "
    f"{model.alpha.item():.6f}"
)


# --------------------------------------------------
# 13. Optimizer
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
# 14. Run model on one batch
# --------------------------------------------------

def forward_batch(
    inputs,
    retriever_scores,
    candidate_count
):

    inputs = {

        key:
            value.to(
                device,
                non_blocking=True
            )

        for key, value
        in inputs.items()
    }


    retriever_scores = (
        retriever_scores.to(
            device,
            non_blocking=True
        )
    )


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


    total_pairs = (
        retriever_scores.shape[0]
    )


    query_count = (
        total_pairs
        // candidate_count
    )


    final_scores = (
        outputs[
            "final_scores"
        ].view(
            query_count,
            candidate_count
        )
    )


    original_scores = (
        retriever_scores.view(
            query_count,
            candidate_count
        )
    )


    return (
        outputs,
        final_scores,
        original_scores
    )


# --------------------------------------------------
# 15. Training epoch
# --------------------------------------------------

def train_one_epoch():

    model.train()


    total_loss = 0.0

    total_ranking_loss = 0.0

    total_regularization = 0.0

    total_queries = 0

    retriever_correct = 0

    reranker_correct = 0


    for batch_index, (
        inputs,
        retriever_scores,
        candidate_count
    ) in enumerate(
        train_loader
    ):

        optimizer.zero_grad()


        (
            outputs,
            final_scores,
            original_scores
        ) = forward_batch(
            inputs,
            retriever_scores,
            candidate_count
        )


        query_count = (
            final_scores.shape[0]
        )


        targets = torch.zeros(
            query_count,
            dtype=torch.long,
            device=device
        )


        # ------------------------------------------
        # Main listwise ranking loss
        # ------------------------------------------

        ranking_loss = (
            F.cross_entropy(
                final_scores,
                targets
            )
        )


        # ------------------------------------------
        # Keep residual corrections controlled.
        #
        # Penalize large deviations from V2 scores.
        # ------------------------------------------

        score_delta = (
            final_scores
            - original_scores
        )


        residual_regularization = (
            score_delta
            .pow(2)
            .mean()
        )


        loss = (
            ranking_loss

            + residual_regularization_weight
            * residual_regularization
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        # ------------------------------------------
        # Prevent alpha from exploding.
        # ------------------------------------------

        with torch.no_grad():

            model.alpha.clamp_(
                min=0.0,
                max=max_alpha
            )


        # ------------------------------------------
        # Diagnostics
        # ------------------------------------------

        retriever_predictions = (
            original_scores.argmax(
                dim=1
            )
        )


        reranker_predictions = (
            final_scores.argmax(
                dim=1
            )
        )


        retriever_correct += (
            retriever_predictions
            == 0
        ).sum().item()


        reranker_correct += (
            reranker_predictions
            == 0
        ).sum().item()


        total_queries += (
            query_count
        )


        total_loss += (
            loss.item()
            * query_count
        )


        total_ranking_loss += (
            ranking_loss.item()
            * query_count
        )


        total_regularization += (
            residual_regularization.item()
            * query_count
        )


        if (
            batch_index + 1
        ) % 50 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss: "
                f"{total_loss / total_queries:.4f} "
                f"| Rank Loss: "
                f"{total_ranking_loss / total_queries:.4f} "
                f"| Retriever Acc: "
                f"{retriever_correct / total_queries:.4f} "
                f"| Reranker Acc: "
                f"{reranker_correct / total_queries:.4f} "
                f"| Alpha: "
                f"{model.alpha.item():.5f}"
            )


    return {

        "loss":
            total_loss
            / total_queries,

        "ranking_loss":
            total_ranking_loss
            / total_queries,

        "regularization":
            total_regularization
            / total_queries,

        "retriever_accuracy":
            retriever_correct
            / total_queries,

        "reranker_accuracy":
            reranker_correct
            / total_queries
    }


# --------------------------------------------------
# 16. Actual top-20 DEV evaluation
# --------------------------------------------------

def evaluate_dev():

    model.eval()


    # Numerators are over ALL 5000 DEV queries.
    correct_at_1 = 0

    correct_at_5 = 0

    correct_at_10 = 0

    correct_at_20 = 0


    retriever_at_1 = 0

    retriever_at_5 = 0

    retriever_at_10 = 0

    retriever_at_20 = 0


    total_dev_queries = (
        dev_dataset.total_queries
    )


    processed_eligible = 0


    with torch.inference_mode():

        for batch_index, (
            inputs,
            retriever_scores,
            candidate_count
        ) in enumerate(
            dev_loader
        ):

            (
                outputs,
                final_scores,
                original_scores
            ) = forward_batch(
                inputs,
                retriever_scores,
                candidate_count
            )


            query_count = (
                final_scores.shape[0]
            )


            # --------------------------------------
            # Position of positive candidate
            #
            # Positive was inserted at index 0.
            # --------------------------------------

            retriever_order = (
                torch.argsort(
                    original_scores,
                    dim=1,
                    descending=True
                )
            )


            reranker_order = (
                torch.argsort(
                    final_scores,
                    dim=1,
                    descending=True
                )
            )


            # --------------------------------------
            # Rank of candidate 0
            # --------------------------------------

            retriever_positive_rank = (
                (
                    retriever_order
                    == 0
                )
                .nonzero(
                    as_tuple=False
                )[
                    :,
                    1
                ]
                + 1
            )


            reranker_positive_rank = (
                (
                    reranker_order
                    == 0
                )
                .nonzero(
                    as_tuple=False
                )[
                    :,
                    1
                ]
                + 1
            )


            # --------------------------------------
            # Retriever metrics
            # --------------------------------------

            retriever_at_1 += (
                retriever_positive_rank
                <= 1
            ).sum().item()


            retriever_at_5 += (
                retriever_positive_rank
                <= 5
            ).sum().item()


            retriever_at_10 += (
                retriever_positive_rank
                <= 10
            ).sum().item()


            retriever_at_20 += (
                retriever_positive_rank
                <= 20
            ).sum().item()


            # --------------------------------------
            # Reranker metrics
            # --------------------------------------

            correct_at_1 += (
                reranker_positive_rank
                <= 1
            ).sum().item()


            correct_at_5 += (
                reranker_positive_rank
                <= 5
            ).sum().item()


            correct_at_10 += (
                reranker_positive_rank
                <= 10
            ).sum().item()


            correct_at_20 += (
                reranker_positive_rank
                <= 20
            ).sum().item()


            processed_eligible += (
                query_count
            )


            if (
                batch_index + 1
            ) % 100 == 0:

                print(
                    f"DEV reranked: "
                    f"{processed_eligible}/"
                    f"{len(dev_dataset)} "
                    f"eligible queries"
                )


    # --------------------------------------------------
    # IMPORTANT:
    #
    # Divide by ALL 5000 queries.
    #
    # Queries whose positive was not retrieved in
    # top-20 automatically contribute zero.
    # --------------------------------------------------

    results = {

        "retriever_r1":
            retriever_at_1
            / total_dev_queries,

        "retriever_r5":
            retriever_at_5
            / total_dev_queries,

        "retriever_r10":
            retriever_at_10
            / total_dev_queries,

        "retriever_r20":
            retriever_at_20
            / total_dev_queries,

        "reranker_r1":
            correct_at_1
            / total_dev_queries,

        "reranker_r5":
            correct_at_5
            / total_dev_queries,

        "reranker_r10":
            correct_at_10
            / total_dev_queries,

        "reranker_r20":
            correct_at_20
            / total_dev_queries
    }


    return results


# --------------------------------------------------
# 17. Train
# --------------------------------------------------

os.makedirs(
    output_dir,
    exist_ok=True
)


best_dev_r1 = -1.0


for epoch in range(
    num_epochs
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"Epoch "
        f"{epoch + 1}/"
        f"{num_epochs}"
    )

    print(
        "=" * 60
    )


    train_results = (
        train_one_epoch()
    )


    print(
        "\nTraining Results"
    )

    print(
        "-" * 50
    )

    print(
        f"Loss: "
        f"{train_results['loss']:.4f}"
    )

    print(
        f"Ranking loss: "
        f"{train_results['ranking_loss']:.4f}"
    )

    print(
        f"Residual regularization: "
        f"{train_results['regularization']:.6f}"
    )

    print(
        f"Retriever accuracy: "
        f"{train_results['retriever_accuracy']:.4f}"
    )

    print(
        f"Reranker accuracy: "
        f"{train_results['reranker_accuracy']:.4f}"
    )

    print(
        f"Alpha: "
        f"{model.alpha.item():.6f}"
    )


    # --------------------------------------------------
    # True top-20 DEV evaluation
    # --------------------------------------------------

    print(
        "\nEvaluating image-disjoint DEV corpus..."
    )


    dev_results = (
        evaluate_dev()
    )


    print(
        "\nDEV Retriever V2"
    )

    print(
        "-" * 50
    )

    print(
        f"Recall@1:  "
        f"{dev_results['retriever_r1']:.4f}"
    )

    print(
        f"Recall@5:  "
        f"{dev_results['retriever_r5']:.4f}"
    )

    print(
        f"Recall@10: "
        f"{dev_results['retriever_r10']:.4f}"
    )

    print(
        f"Recall@20: "
        f"{dev_results['retriever_r20']:.4f}"
    )


    print(
        "\nDEV Retriever V2 + "
        "Cross-Attention Reranker V3"
    )

    print(
        "-" * 50
    )

    print(
        f"Recall@1:  "
        f"{dev_results['reranker_r1']:.4f}"
    )

    print(
        f"Recall@5:  "
        f"{dev_results['reranker_r5']:.4f}"
    )

    print(
        f"Recall@10: "
        f"{dev_results['reranker_r10']:.4f}"
    )

    print(
        f"Recall@20: "
        f"{dev_results['reranker_r20']:.4f}"
    )


    print(
        "\nDEV Improvement"
    )

    print(
        "-" * 50
    )

    print(
        f"Recall@1:  "
        f"{dev_results['reranker_r1'] - dev_results['retriever_r1']:+.4f}"
    )

    print(
        f"Recall@5:  "
        f"{dev_results['reranker_r5'] - dev_results['retriever_r5']:+.4f}"
    )

    print(
        f"Recall@10: "
        f"{dev_results['reranker_r10'] - dev_results['retriever_r10']:+.4f}"
    )


    # --------------------------------------------------
    # Save best based on real DEV Recall@1
    # --------------------------------------------------

    if (
        dev_results[
            "reranker_r1"
        ]
        > best_dev_r1
    ):

        best_dev_r1 = (
            dev_results[
                "reranker_r1"
            ]
        )


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "backbone_name":
                backbone_name,

            "fusion_dim":
                fusion_dim,

            "num_heads":
                num_heads,

            "ff_hidden_dim":
                ff_hidden_dim,

            "scorer_hidden_dim":
                scorer_hidden_dim,

            "dropout":
                dropout,

            "alpha":
                model.alpha.item(),

            "epoch":
                epoch + 1,

            "learning_rate":
                learning_rate,

            "candidate_pool_size":
                candidate_pool_size,

            "dev_results":
                dev_results
        }


        checkpoint_path = os.path.join(
            output_dir,
            "deepsearch_cross_attention_reranker_v3_best.pt"
        )


        torch.save(
            checkpoint,
            checkpoint_path
        )


        print(
            "\nSaved new best V3 checkpoint:"
        )

        print(
            checkpoint_path
        )


# --------------------------------------------------
# 18. Final
# --------------------------------------------------

print(
    "\nTraining complete."
)

print(
    f"Best DEV Recall@1: "
    f"{best_dev_r1:.4f}"
)

print(
    f"Final alpha: "
    f"{model.alpha.item():.6f}"
)
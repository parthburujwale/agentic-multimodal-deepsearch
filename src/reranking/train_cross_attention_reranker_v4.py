import json
import os
import random
import sys

import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


sys.path.append("..")

from models.deepsearch_cross_attention_reranker_v4 import (
    DeepSearchCrossAttentionRerankerV4
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

output_dir = (
    "../../experiments/"
    "deepsearch_cross_attention_reranker_v4"
)

backbone_name = (
    "openai/"
    "clip-vit-base-patch32"
)


candidate_pool_size = 20

batch_size = 2

dev_batch_size = 4

num_epochs = 3

learning_rate = 2e-5

weight_decay = 0.01

fusion_dim = 512

num_heads = 8

ff_hidden_dim = 1024

scorer_hidden_dim = 512

dropout = 0.1

max_correction = 0.02

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

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Using device: {device}"
)


# --------------------------------------------------
# 4. Load data
# --------------------------------------------------

with open(
    train_dataset_file,
    "r"
) as file:

    train_data = json.load(file)


with open(
    dev_dataset_file,
    "r"
) as file:

    dev_data = json.load(file)


all_train_examples = (
    train_data["examples"]
)

all_dev_examples = (
    dev_data["examples"]
)


# --------------------------------------------------
# Only train where positive is recoverable
# --------------------------------------------------

train_examples = [

    example

    for example in all_train_examples

    if (
        example[
            "positive"
        ][
            "retrieval_rank"
        ]
        <= candidate_pool_size
    )
]


eligible_dev_examples = [

    example

    for example in all_dev_examples

    if (
        example[
            "positive"
        ][
            "retrieval_rank"
        ]
        <= candidate_pool_size
    )
]


print(
    f"Training queries: "
    f"{len(train_examples)}"
)

print(
    f"Eligible DEV queries: "
    f"{len(eligible_dev_examples)}"
)

print(
    f"Total DEV queries: "
    f"{len(all_dev_examples)}"
)


# --------------------------------------------------
# 5. Dataset
# --------------------------------------------------

class Top20Dataset(Dataset):

    def __init__(
        self,
        examples,
        image_folder
    ):

        self.examples = examples

        self.image_folder = (
            image_folder
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


        images = [
            positive_image
        ]


        scores = [
            float(
                positive[
                    "retrieval_score"
                ]
            )
        ]


        # Positive + first 19 negatives
        # = complete 20-candidate set
        negatives = (
            example[
                "negative_candidates"
            ][:19]
        )


        for negative in negatives:

            image_path = os.path.join(
                self.image_folder,
                negative[
                    "file_name"
                ]
            )


            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )


            images.append(
                image
            )


            scores.append(
                float(
                    negative[
                        "retrieval_score"
                    ]
                )
            )


        return (
            caption,
            images,
            scores
        )


# --------------------------------------------------
# 6. Processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    backbone_name
)


# --------------------------------------------------
# 7. Collate
# --------------------------------------------------

def collate_fn(batch):

    captions = []

    images = []

    scores = []


    for (
        caption,
        candidate_images,
        candidate_scores
    ) in batch:

        captions.extend(
            [
                caption
                for _ in candidate_images
            ]
        )


        images.extend(
            candidate_images
        )


        scores.extend(
            candidate_scores
        )


    inputs = processor(

        text=captions,

        images=images,

        return_tensors="pt",

        padding=True,

        truncation=True
    )


    scores = torch.tensor(
        scores,
        dtype=torch.float32
    )


    return (
        inputs,
        scores
    )


# --------------------------------------------------
# 8. DataLoaders
# --------------------------------------------------

train_dataset = Top20Dataset(
    train_examples,
    image_folder
)


dev_dataset = Top20Dataset(
    eligible_dev_examples,
    image_folder
)


train_loader = DataLoader(

    train_dataset,

    batch_size=batch_size,

    shuffle=True,

    collate_fn=collate_fn,

    num_workers=4,

    pin_memory=True
)


dev_loader = DataLoader(

    dev_dataset,

    batch_size=dev_batch_size,

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
# 9. Model
# --------------------------------------------------

model = (
    DeepSearchCrossAttentionRerankerV4(

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

        max_correction=
            max_correction,

        freeze_backbone=True
    )
)


model = model.to(
    device
)


print(
    f"Trainable parameters: "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

print(
    f"Maximum score correction: "
    f"±{max_correction:.4f}"
)


# --------------------------------------------------
# 10. Optimizer
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
# 11. Forward helper
# --------------------------------------------------

def forward_batch(
    inputs,
    retriever_scores
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


    query_count = (
        retriever_scores.shape[0]
        // candidate_pool_size
    )


    original_scores = (
        retriever_scores.view(
            query_count,
            candidate_pool_size
        )
    )


    final_scores = (
        outputs[
            "final_scores"
        ].view(
            query_count,
            candidate_pool_size
        )
    )


    bounded_corrections = (
        outputs[
            "bounded_corrections"
        ].view(
            query_count,
            candidate_pool_size
        )
    )


    return (
        original_scores,
        final_scores,
        bounded_corrections
    )


# --------------------------------------------------
# 12. Training epoch
# --------------------------------------------------

def train_epoch():

    model.train()


    total_queries = 0

    total_loss = 0.0

    retriever_correct = 0

    reranker_correct = 0

    correction_sum = 0.0

    max_seen_correction = 0.0


    for batch_index, (
        inputs,
        scores
    ) in enumerate(
        train_loader
    ):

        optimizer.zero_grad()


        (
            original_scores,
            final_scores,
            bounded_corrections
        ) = forward_batch(
            inputs,
            scores
        )


        query_count = (
            final_scores.shape[0]
        )


        targets = torch.zeros(
            query_count,
            dtype=torch.long,
            device=device
        )


        loss = F.cross_entropy(
            final_scores,
            targets
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


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


        total_loss += (
            loss.item()
            * query_count
        )


        total_queries += (
            query_count
        )


        correction_sum += (
            bounded_corrections
            .abs()
            .sum()
            .item()
        )


        max_seen_correction = max(

            max_seen_correction,

            bounded_corrections
            .abs()
            .max()
            .item()
        )


        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss "
                f"{total_loss / total_queries:.4f} "
                f"| Retriever Acc "
                f"{retriever_correct / total_queries:.4f} "
                f"| Reranker Acc "
                f"{reranker_correct / total_queries:.4f} "
                f"| Max Corr "
                f"{max_seen_correction:.5f}"
            )


    average_abs_correction = (

        correction_sum

        / (
            total_queries
            * candidate_pool_size
        )
    )


    return {

        "loss":
            total_loss
            / total_queries,

        "retriever_accuracy":
            retriever_correct
            / total_queries,

        "reranker_accuracy":
            reranker_correct
            / total_queries,

        "average_abs_correction":
            average_abs_correction,

        "max_correction":
            max_seen_correction
    }


# --------------------------------------------------
# 13. DEV evaluation
# --------------------------------------------------

def evaluate_dev():

    model.eval()


    total_queries = len(
        all_dev_examples
    )


    retriever_counts = {
        1: 0,
        5: 0,
        10: 0,
        20: 0
    }


    reranker_counts = {
        1: 0,
        5: 0,
        10: 0,
        20: 0
    }


    processed = 0


    with torch.inference_mode():

        for batch_index, (
            inputs,
            scores
        ) in enumerate(
            dev_loader
        ):

            (
                original_scores,
                final_scores,
                bounded_corrections
            ) = forward_batch(
                inputs,
                scores
            )


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


            retriever_rank = (
                (
                    retriever_order
                    == 0
                )
                .nonzero(
                    as_tuple=False
                )[:, 1]
                + 1
            )


            reranker_rank = (
                (
                    reranker_order
                    == 0
                )
                .nonzero(
                    as_tuple=False
                )[:, 1]
                + 1
            )


            for k in [
                1,
                5,
                10,
                20
            ]:

                retriever_counts[k] += (
                    retriever_rank <= k
                ).sum().item()


                reranker_counts[k] += (
                    reranker_rank <= k
                ).sum().item()


            processed += (
                original_scores.shape[0]
            )


            if (
                batch_index + 1
            ) % 100 == 0:

                print(
                    f"DEV reranked: "
                    f"{processed}/"
                    f"{len(dev_dataset)} "
                    f"eligible"
                )


    return {

        "retriever_r1":
            retriever_counts[1]
            / total_queries,

        "retriever_r5":
            retriever_counts[5]
            / total_queries,

        "retriever_r10":
            retriever_counts[10]
            / total_queries,

        "retriever_r20":
            retriever_counts[20]
            / total_queries,

        "reranker_r1":
            reranker_counts[1]
            / total_queries,

        "reranker_r5":
            reranker_counts[5]
            / total_queries,

        "reranker_r10":
            reranker_counts[10]
            / total_queries,

        "reranker_r20":
            reranker_counts[20]
            / total_queries
    }


# --------------------------------------------------
# 14. Training
# --------------------------------------------------

os.makedirs(
    output_dir,
    exist_ok=True
)


# Important:
# we only save V4 if it actually beats Retriever V2.
best_dev_r1 = float(
    dev_data[
        "retriever_diagnostics"
    ][
        "recall_at_1"
    ]
)


print(
    f"\nV2 DEV R@1 to beat: "
    f"{best_dev_r1:.4f}"
)


saved_improved_checkpoint = False


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
        train_epoch()
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
        f"Retriever accuracy: "
        f"{train_results['retriever_accuracy']:.4f}"
    )

    print(
        f"Reranker accuracy: "
        f"{train_results['reranker_accuracy']:.4f}"
    )

    print(
        f"Average abs correction: "
        f"{train_results['average_abs_correction']:.6f}"
    )

    print(
        f"Maximum correction seen: "
        f"{train_results['max_correction']:.6f}"
    )


    print(
        "\nEvaluating full top-20 DEV..."
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
        "Bounded Reranker V4"
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
    # ONLY save if V4 beats V2 R@1
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

            "max_correction":
                max_correction,

            "candidate_pool_size":
                candidate_pool_size,

            "epoch":
                epoch + 1,

            "dev_results":
                dev_results
        }


        checkpoint_path = os.path.join(
            output_dir,
            "deepsearch_cross_attention_reranker_v4_best.pt"
        )


        torch.save(
            checkpoint,
            checkpoint_path
        )


        saved_improved_checkpoint = True


        print(
            "\nNEW BEST:"
        )

        print(
            "V4 beat Retriever V2 R@1."
        )

        print(
            "Saved checkpoint:"
        )

        print(
            checkpoint_path
        )


# --------------------------------------------------
# 15. Summary
# --------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "V4 experiment complete"
)

print(
    "=" * 60
)


if saved_improved_checkpoint:

    print(
        f"Best DEV R@1: "
        f"{best_dev_r1:.4f}"
    )

    print(
        "V4 improved over Retriever V2."
    )

else:

    print(
        "V4 never beat Retriever V2."
    )

    print(
        "No improved V4 checkpoint was saved."
    )
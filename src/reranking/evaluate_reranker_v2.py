import json
import os
import sys

import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


# --------------------------------------------------
# Import models
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
    "../../data/coco/val2017"
)

captions_file = (
    "../../data/coco/annotations/"
    "captions_val2017.json"
)

retriever_checkpoint_path = (
    "../../experiments/"
    "deepsearch_retriever_v2/"
    "deepsearch_retriever_v2.pt"
)

reranker_checkpoint_path = (
    "../../experiments/"
    "deepsearch_reranker_v2/"
    "deepsearch_reranker_v2_best.pt"
)

processor_path = (
    "../../experiments/"
    "deepsearch_retriever_v2"
)

output_file = (
    "../../experiments/"
    "deepsearch_reranker_v2/"
    "evaluation_results.json"
)

batch_size = 128

candidate_pool_size = 20

reranker_query_batch_size = 512


# --------------------------------------------------
# 2. Device
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
# 3. Processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    processor_path
)


# --------------------------------------------------
# 4. Load Retriever V2
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


retriever = retriever.to(device)

retriever.eval()


print(
    "Retriever V2 loaded"
)


# --------------------------------------------------
# 5. Load Reranker V2
# --------------------------------------------------

print(
    "\nLoading Residual Reranker V2..."
)


reranker_checkpoint = torch.load(
    reranker_checkpoint_path,
    map_location=device,
    weights_only=False
)


reranker = DeepSearchRerankerV2(
    embedding_dim=(
        reranker_checkpoint[
            "embedding_dim"
        ]
    ),

    hidden_dim=(
        reranker_checkpoint[
            "hidden_dim"
        ]
    ),

    dropout=(
        reranker_checkpoint[
            "dropout"
        ]
    ),

    initial_alpha=(
        reranker_checkpoint[
            "alpha"
        ]
    )
)


reranker.load_state_dict(
    reranker_checkpoint[
        "model_state_dict"
    ]
)


reranker = reranker.to(device)

reranker.eval()


print(
    "Reranker V2 loaded"
)

print(
    f"Learned alpha: "
    f"{reranker.alpha.item():.6f}"
)

print(
    f"Best held-out accuracy: "
    f"{reranker_checkpoint['validation_accuracy']:.4f}"
)

print(
    f"Checkpoint epoch: "
    f"{reranker_checkpoint['epoch']}"
)


# --------------------------------------------------
# 6. COCO annotations
# --------------------------------------------------

with open(
    captions_file,
    "r"
) as file:

    coco_data = json.load(file)


# --------------------------------------------------
# First caption per image
# --------------------------------------------------

image_id_to_caption = {}


for annotation in coco_data[
    "annotations"
]:

    image_id = annotation[
        "image_id"
    ]

    if image_id not in image_id_to_caption:

        image_id_to_caption[
            image_id
        ] = annotation[
            "caption"
        ]


# --------------------------------------------------
# 7. Build aligned validation set
# --------------------------------------------------

validation_pairs = []


for image_info in coco_data[
    "images"
]:

    image_id = image_info[
        "id"
    ]


    if image_id not in image_id_to_caption:

        continue


    validation_pairs.append(
        {
            "image_id":
                image_id,

            "file_name":
                image_info[
                    "file_name"
                ],

            "caption":
                image_id_to_caption[
                    image_id
                ]
        }
    )


print(
    f"\nValidation pairs: "
    f"{len(validation_pairs)}"
)


# --------------------------------------------------
# 8. Image Dataset
# --------------------------------------------------

class COCOImageDataset(Dataset):

    def __init__(
        self,
        pairs,
        image_folder
    ):

        self.pairs = pairs
        self.image_folder = image_folder


    def __len__(self):

        return len(
            self.pairs
        )


    def __getitem__(
        self,
        index
    ):

        pair = self.pairs[
            index
        ]


        image_path = os.path.join(
            self.image_folder,
            pair[
                "file_name"
            ]
        )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        return image


def image_collate_fn(batch):

    return processor(
        images=batch,
        return_tensors="pt"
    )


image_dataset = COCOImageDataset(
    validation_pairs,
    image_folder
)


image_loader = DataLoader(
    image_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=image_collate_fn,
    num_workers=4,
    pin_memory=True
)


# --------------------------------------------------
# 9. Encode all validation images
# --------------------------------------------------

image_embeddings = []


print(
    "\nEncoding validation images..."
)


with torch.inference_mode():

    for batch_index, batch in enumerate(
        image_loader
    ):

        pixel_values = (
            batch[
                "pixel_values"
            ].to(
                device,
                non_blocking=True
            )
        )


        embeddings = (
            retriever.encode_image(
                pixel_values
            )
        )


        image_embeddings.append(
            embeddings.cpu()
        )


        count = min(
            (
                batch_index + 1
            )
            * batch_size,

            len(validation_pairs)
        )


        print(
            f"Encoded images: "
            f"{count}/"
            f"{len(validation_pairs)}"
        )


image_embeddings = torch.cat(
    image_embeddings,
    dim=0
).to(device)


print(
    "Image embeddings:",
    image_embeddings.shape
)


# --------------------------------------------------
# 10. Encode captions
# --------------------------------------------------

captions = [
    pair[
        "caption"
    ]
    for pair
    in validation_pairs
]


text_embeddings = []


print(
    "\nEncoding captions..."
)


with torch.inference_mode():

    for start_index in range(
        0,
        len(captions),
        batch_size
    ):

        end_index = min(
            start_index
            + batch_size,

            len(captions)
        )


        batch_captions = captions[
            start_index:
            end_index
        ]


        inputs = processor(
            text=batch_captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )


        input_ids = (
            inputs[
                "input_ids"
            ].to(device)
        )


        attention_mask = (
            inputs[
                "attention_mask"
            ].to(device)
        )


        embeddings = (
            retriever.encode_text(
                input_ids,
                attention_mask
            )
        )


        text_embeddings.append(
            embeddings.cpu()
        )


        print(
            f"Encoded captions: "
            f"{end_index}/"
            f"{len(captions)}"
        )


text_embeddings = torch.cat(
    text_embeddings,
    dim=0
).to(device)


print(
    "Text embeddings:",
    text_embeddings.shape
)


# --------------------------------------------------
# 11. Full Retriever V2 search
# --------------------------------------------------

print(
    "\nRunning Retriever V2 search..."
)


similarity = (
    text_embeddings
    @ image_embeddings.T
)


retriever_scores, candidate_indices = (
    torch.topk(
        similarity,
        k=candidate_pool_size,
        dim=1
    )
)


# --------------------------------------------------
# 12. Ground truth
# --------------------------------------------------

ground_truth = torch.arange(
    len(validation_pairs),
    device=device
)


# --------------------------------------------------
# Recall utility
# --------------------------------------------------

def recall_at_k(
    ranked_indices,
    k
):

    correct = (
        ranked_indices[
            :,
            :k
        ]
        == ground_truth.unsqueeze(1)
    ).any(
        dim=1
    )


    return (
        correct.float()
        .mean()
        .item()
    )


# --------------------------------------------------
# 13. Retriever metrics
# --------------------------------------------------

retriever_r1 = recall_at_k(
    candidate_indices,
    1
)

retriever_r5 = recall_at_k(
    candidate_indices,
    5
)

retriever_r10 = recall_at_k(
    candidate_indices,
    10
)

retriever_r20 = recall_at_k(
    candidate_indices,
    20
)


print(
    "\nDeepSearch Retriever V2"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{retriever_r1:.4f}"
)

print(
    f"Recall@5:  "
    f"{retriever_r5:.4f}"
)

print(
    f"Recall@10: "
    f"{retriever_r10:.4f}"
)

print(
    f"Recall@20: "
    f"{retriever_r20:.4f}"
)


# --------------------------------------------------
# 14. Residual reranking
# --------------------------------------------------

print(
    "\nRunning Residual Reranker V2..."
)


reranked_batches = []


with torch.inference_mode():

    for start_index in range(
        0,
        len(validation_pairs),
        reranker_query_batch_size
    ):

        end_index = min(
            start_index
            + reranker_query_batch_size,

            len(validation_pairs)
        )


        batch_candidate_indices = (
            candidate_indices[
                start_index:
                end_index
            ]
        )


        batch_retriever_scores = (
            retriever_scores[
                start_index:
                end_index
            ]
        )


        current_batch_size = (
            end_index
            - start_index
        )


        # --------------------------------------------------
        # Text:
        # [B, 512] -> [B, 20, 512]
        # --------------------------------------------------

        text_batch = (
            text_embeddings[
                start_index:
                end_index
            ]
            .unsqueeze(1)
            .expand(
                -1,
                candidate_pool_size,
                -1
            )
        )


        # --------------------------------------------------
        # Candidate image embeddings
        # [B, 20, 512]
        # --------------------------------------------------

        image_batch = (
            image_embeddings[
                batch_candidate_indices
            ]
        )


        # --------------------------------------------------
        # Flatten pairs
        # --------------------------------------------------

        text_flat = (
            text_batch.reshape(
                -1,
                text_batch.shape[-1]
            )
        )


        image_flat = (
            image_batch.reshape(
                -1,
                image_batch.shape[-1]
            )
        )


        score_flat = (
            batch_retriever_scores.reshape(
                -1
            )
        )


        # --------------------------------------------------
        # Residual reranker
        # --------------------------------------------------

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
                candidate_pool_size
            )
        )


        # --------------------------------------------------
        # Reorder top-20
        # --------------------------------------------------

        reranker_order = torch.argsort(
            final_scores,
            dim=1,
            descending=True
        )


        reranked_indices = torch.gather(
            batch_candidate_indices,
            dim=1,
            index=reranker_order
        )


        reranked_batches.append(
            reranked_indices.cpu()
        )


        print(
            f"Reranked queries: "
            f"{end_index}/"
            f"{len(validation_pairs)}"
        )


reranked_indices = torch.cat(
    reranked_batches,
    dim=0
).to(device)


# --------------------------------------------------
# 15. Reranked metrics
# --------------------------------------------------

reranked_r1 = recall_at_k(
    reranked_indices,
    1
)

reranked_r5 = recall_at_k(
    reranked_indices,
    5
)

reranked_r10 = recall_at_k(
    reranked_indices,
    10
)

reranked_r20 = recall_at_k(
    reranked_indices,
    20
)


print(
    "\nDeepSearch V2 + Residual Reranker V2"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{reranked_r1:.4f}"
)

print(
    f"Recall@5:  "
    f"{reranked_r5:.4f}"
)

print(
    f"Recall@10: "
    f"{reranked_r10:.4f}"
)

print(
    f"Recall@20: "
    f"{reranked_r20:.4f}"
)


# --------------------------------------------------
# 16. Improvements
# --------------------------------------------------

delta_r1 = (
    reranked_r1
    - retriever_r1
)

delta_r5 = (
    reranked_r5
    - retriever_r5
)

delta_r10 = (
    reranked_r10
    - retriever_r10
)


print(
    "\nReranker Improvement"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{delta_r1:+.4f}"
)

print(
    f"Recall@5:  "
    f"{delta_r5:+.4f}"
)

print(
    f"Recall@10: "
    f"{delta_r10:+.4f}"
)


# --------------------------------------------------
# 17. Candidate ceiling
# --------------------------------------------------

print(
    "\nCandidate Pool Ceiling"
)

print(
    "-" * 50
)

print(
    f"Recall@20: "
    f"{retriever_r20:.4f}"
)


# --------------------------------------------------
# 18. Save
# --------------------------------------------------

results = {

    "candidate_pool_size":
        candidate_pool_size,

    "alpha":
        reranker.alpha.item(),

    "checkpoint_epoch":
        reranker_checkpoint[
            "epoch"
        ],

    "held_out_accuracy":
        reranker_checkpoint[
            "validation_accuracy"
        ],

    "retriever":
        {
            "recall_at_1":
                retriever_r1,

            "recall_at_5":
                retriever_r5,

            "recall_at_10":
                retriever_r10,

            "recall_at_20":
                retriever_r20
        },

    "residual_reranker":
        {
            "recall_at_1":
                reranked_r1,

            "recall_at_5":
                reranked_r5,

            "recall_at_10":
                reranked_r10,

            "recall_at_20":
                reranked_r20
        },

    "improvement":
        {
            "recall_at_1":
                delta_r1,

            "recall_at_5":
                delta_r5,

            "recall_at_10":
                delta_r10
        }
}


with open(
    output_file,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=2
    )


print(
    "\nSaved evaluation to:"
)

print(
    output_file
)
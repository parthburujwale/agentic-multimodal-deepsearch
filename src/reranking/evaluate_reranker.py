import json
import os
import sys

import torch

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

from models.deepsearch_reranker import (
    DeepSearchReranker
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
    "deepsearch_reranker_v1/"
    "deepsearch_reranker_v1.pt"
)

processor_path = (
    "../../experiments/"
    "deepsearch_retriever_v2"
)

output_file = (
    "../../experiments/"
    "deepsearch_reranker_v1/"
    "evaluation_results.json"
)

batch_size = 64

candidate_pool_size = 20

reranker_batch_size = 256


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
# 3. Load processor
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


retriever = retriever.to(
    device
)

retriever.eval()


print(
    "Retriever loaded successfully"
)


# --------------------------------------------------
# 5. Load Reranker
# --------------------------------------------------

print(
    "\nLoading DeepSearch Reranker V1..."
)


reranker_checkpoint = torch.load(
    reranker_checkpoint_path,
    map_location=device,
    weights_only=False
)


reranker = DeepSearchReranker(
    backbone_name=(
        reranker_checkpoint[
            "backbone_name"
        ]
    ),

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

    freeze_backbone=True
)


reranker.load_state_dict(
    reranker_checkpoint[
        "model_state_dict"
    ]
)


reranker = reranker.to(
    device
)

reranker.eval()


print(
    "Reranker loaded successfully"
)


# --------------------------------------------------
# 6. Load COCO validation annotations
# --------------------------------------------------

with open(
    captions_file,
    "r"
) as file:

    coco_data = json.load(
        file
    )


# --------------------------------------------------
# 7. First caption for each image
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
# 8. Build aligned validation set
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
    f"\nPrepared "
    f"{len(validation_pairs)} "
    f"validation pairs"
)


# --------------------------------------------------
# 9. Image dataset
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
            pair["file_name"]
        )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        return image


# --------------------------------------------------
# 10. Image preprocessing
# --------------------------------------------------

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
# 11. Encode images
#
# We need two image representations:
#
# 1. Retriever V2 embeddings
# 2. Reranker CLIP embeddings
# --------------------------------------------------

retriever_image_embeddings = []

reranker_image_embeddings = []


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


        retriever_embeddings = (
            retriever.encode_image(
                pixel_values
            )
        )


        reranker_embeddings = (
            reranker.encode_image(
                pixel_values
            )
        )


        retriever_image_embeddings.append(
            retriever_embeddings.cpu()
        )

        reranker_image_embeddings.append(
            reranker_embeddings.cpu()
        )


        encoded_count = min(
            (
                batch_index + 1
            )
            * batch_size,
            len(validation_pairs)
        )


        print(
            f"Encoded images: "
            f"{encoded_count}/"
            f"{len(validation_pairs)}"
        )


retriever_image_embeddings = (
    torch.cat(
        retriever_image_embeddings,
        dim=0
    )
)


reranker_image_embeddings = (
    torch.cat(
        reranker_image_embeddings,
        dim=0
    )
)


print(
    "Retriever image embeddings:",
    retriever_image_embeddings.shape
)

print(
    "Reranker image embeddings:",
    reranker_image_embeddings.shape
)


# --------------------------------------------------
# 12. Encode captions
# --------------------------------------------------

captions = [
    pair["caption"]
    for pair in validation_pairs
]


retriever_text_embeddings = []

reranker_text_embeddings = []


print(
    "\nEncoding validation captions..."
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
            start_index:end_index
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
            ].to(
                device,
                non_blocking=True
            )
        )


        attention_mask = (
            inputs[
                "attention_mask"
            ].to(
                device,
                non_blocking=True
            )
        )


        retriever_embeddings = (
            retriever.encode_text(
                input_ids,
                attention_mask
            )
        )


        reranker_embeddings = (
            reranker.encode_text(
                input_ids,
                attention_mask
            )
        )


        retriever_text_embeddings.append(
            retriever_embeddings.cpu()
        )

        reranker_text_embeddings.append(
            reranker_embeddings.cpu()
        )


        print(
            f"Encoded captions: "
            f"{end_index}/"
            f"{len(captions)}"
        )


retriever_text_embeddings = (
    torch.cat(
        retriever_text_embeddings,
        dim=0
    )
)


reranker_text_embeddings = (
    torch.cat(
        reranker_text_embeddings,
        dim=0
    )
)


print(
    "Retriever text embeddings:",
    retriever_text_embeddings.shape
)

print(
    "Reranker text embeddings:",
    reranker_text_embeddings.shape
)


# --------------------------------------------------
# 13. Move embeddings to GPU
# --------------------------------------------------

retriever_image_embeddings = (
    retriever_image_embeddings.to(
        device
    )
)

retriever_text_embeddings = (
    retriever_text_embeddings.to(
        device
    )
)

reranker_image_embeddings = (
    reranker_image_embeddings.to(
        device
    )
)

reranker_text_embeddings = (
    reranker_text_embeddings.to(
        device
    )
)


# --------------------------------------------------
# 14. Retriever candidate generation
# --------------------------------------------------

print(
    "\nGenerating Retriever V2 candidates..."
)


retriever_similarity = (
    retriever_text_embeddings
    @ retriever_image_embeddings.T
)


retriever_scores, candidate_indices = (
    torch.topk(
        retriever_similarity,
        k=candidate_pool_size,
        dim=1
    )
)


print(
    f"Candidate index shape: "
    f"{candidate_indices.shape}"
)


# --------------------------------------------------
# 15. Utility for Recall@K
# --------------------------------------------------

ground_truth = torch.arange(
    len(validation_pairs),
    device=device
)


def recall_at_k(
    ranked_indices,
    ground_truth,
    k
):

    top_k = ranked_indices[
        :,
        :k
    ]


    correct = (
        top_k
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
# 16. Retriever-only results
# --------------------------------------------------

retriever_r1 = recall_at_k(
    candidate_indices,
    ground_truth,
    1
)

retriever_r5 = recall_at_k(
    candidate_indices,
    ground_truth,
    5
)

retriever_r10 = recall_at_k(
    candidate_indices,
    ground_truth,
    10
)

retriever_r20 = recall_at_k(
    candidate_indices,
    ground_truth,
    20
)


print(
    "\nRetriever V2 Results"
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
# 17. Rerank top-20 candidates
# --------------------------------------------------

print(
    "\nReranking top-20 candidates..."
)


reranked_batches = []


with torch.inference_mode():

    for start_index in range(
        0,
        len(validation_pairs),
        reranker_batch_size
    ):

        end_index = min(
            start_index
            + reranker_batch_size,
            len(validation_pairs)
        )


        batch_candidate_indices = (
            candidate_indices[
                start_index:end_index
            ]
        )


        current_batch_size = (
            end_index
            - start_index
        )


        # --------------------------------------------------
        # Text embeddings:
        # [B, 512]
        #       ↓
        # [B, 20, 512]
        # --------------------------------------------------

        text_batch = (
            reranker_text_embeddings[
                start_index:end_index
            ]
            .unsqueeze(1)
            .expand(
                -1,
                candidate_pool_size,
                -1
            )
        )


        # --------------------------------------------------
        # Candidate image embeddings:
        # [B, 20, 512]
        # --------------------------------------------------

        image_batch = (
            reranker_image_embeddings[
                batch_candidate_indices
            ]
        )


        # --------------------------------------------------
        # Flatten candidate pairs
        # --------------------------------------------------

        text_flat = text_batch.reshape(
            -1,
            text_batch.shape[-1]
        )


        image_flat = image_batch.reshape(
            -1,
            image_batch.shape[-1]
        )


        # --------------------------------------------------
        # Score with trained fusion network
        # --------------------------------------------------

        scores = reranker.fusion(
            text_flat,
            image_flat
        )


        scores = scores.view(
            current_batch_size,
            candidate_pool_size
        )


        # --------------------------------------------------
        # Sort candidates using reranker score
        # --------------------------------------------------

        reranker_order = torch.argsort(
            scores,
            dim=1,
            descending=True
        )


        reranked_indices = torch.gather(
            batch_candidate_indices,
            dim=1,
            index=reranker_order
        )


        reranked_batches.append(
            reranked_indices
        )


        print(
            f"Reranked queries: "
            f"{end_index}/"
            f"{len(validation_pairs)}"
        )


reranked_indices = torch.cat(
    reranked_batches,
    dim=0
)


# --------------------------------------------------
# 18. Reranked metrics
# --------------------------------------------------

reranked_r1 = recall_at_k(
    reranked_indices,
    ground_truth,
    1
)

reranked_r5 = recall_at_k(
    reranked_indices,
    ground_truth,
    5
)

reranked_r10 = recall_at_k(
    reranked_indices,
    ground_truth,
    10
)

reranked_r20 = recall_at_k(
    reranked_indices,
    ground_truth,
    20
)


print(
    "\nDeepSearch V2 + Reranker Results"
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
# 19. Difference caused by reranking
# --------------------------------------------------

print(
    "\nReranker Improvement"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{reranked_r1 - retriever_r1:+.4f}"
)

print(
    f"Recall@5:  "
    f"{reranked_r5 - retriever_r5:+.4f}"
)

print(
    f"Recall@10: "
    f"{reranked_r10 - retriever_r10:+.4f}"
)


# --------------------------------------------------
# 20. Candidate-pool ceiling
# --------------------------------------------------

print(
    "\nCandidate Pool Recall"
)

print(
    "-" * 50
)

print(
    f"Retriever Recall@20: "
    f"{retriever_r20:.4f}"
)

print(
    "This is the maximum recall achievable "
    "by reranking this top-20 candidate pool."
)


# --------------------------------------------------
# 21. Save results
# --------------------------------------------------

results = {

    "candidate_pool_size":
        candidate_pool_size,

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

    "retriever_plus_reranker":
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
                reranked_r1
                - retriever_r1,

            "recall_at_5":
                reranked_r5
                - retriever_r5,

            "recall_at_10":
                reranked_r10
                - retriever_r10
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
    "\nSaved evaluation results to:"
)

print(
    output_file
)
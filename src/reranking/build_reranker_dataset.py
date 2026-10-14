import json
import os
import sys

import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


# --------------------------------------------------
# Import DeepSearch Retriever V2
# --------------------------------------------------

sys.path.append("..")

from models.deepsearch_retriever_v2 import (
    DeepSearchRetrieverV2
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

image_folder = (
    "../../data/coco/train2017"
)

captions_file = (
    "../../data/coco/annotations/"
    "captions_train2017.json"
)

checkpoint_path = (
    "../../experiments/"
    "deepsearch_retriever_v2/"
    "deepsearch_retriever_v2.pt"
)

processor_path = (
    "../../experiments/"
    "deepsearch_retriever_v2"
)

output_file = (
    "../../data/coco/"
    "reranker_dataset_50k.json"
)


num_samples = 50000

image_batch_size = 128
text_batch_size = 256

num_negative_candidates = 20


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
# 3. Load DeepSearch V2 checkpoint
# --------------------------------------------------

print(
    f"Loading checkpoint:"
)

print(
    checkpoint_path
)


checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)


backbone_name = checkpoint[
    "backbone_name"
]

embedding_dim = checkpoint[
    "embedding_dim"
]

hidden_dim = checkpoint[
    "hidden_dim"
]

dropout = checkpoint[
    "dropout"
]


# --------------------------------------------------
# 4. Processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    processor_path
)


# --------------------------------------------------
# 5. Reconstruct DeepSearch V2
# --------------------------------------------------

model = DeepSearchRetrieverV2(
    backbone_name=backbone_name,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    dropout=dropout,
    freeze_backbone=True
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    device
)

model.eval()


print(
    "DeepSearch V2 loaded successfully"
)

print(
    f"Image alpha: "
    f"{model.image_alpha.item():.6f}"
)

print(
    f"Text alpha: "
    f"{model.text_alpha.item():.6f}"
)


# --------------------------------------------------
# 6. Load COCO annotations
# --------------------------------------------------

with open(
    captions_file,
    "r"
) as file:

    coco_data = json.load(file)


image_id_to_file = {}

for image_info in coco_data[
    "images"
]:

    image_id_to_file[
        image_info["id"]
    ] = image_info[
        "file_name"
    ]


# --------------------------------------------------
# 7. Group captions by image
# --------------------------------------------------

image_id_to_captions = {}


for annotation in coco_data[
    "annotations"
]:

    image_id = annotation[
        "image_id"
    ]

    caption = annotation[
        "caption"
    ]


    if image_id not in image_id_to_captions:

        image_id_to_captions[
            image_id
        ] = []


    image_id_to_captions[
        image_id
    ].append(
        caption
    )


# --------------------------------------------------
# 8. Build same 50K unique-image set
# --------------------------------------------------

training_pairs = []


for image_id, captions in (
    image_id_to_captions.items()
):

    training_pairs.append(
        {
            "image_id":
                image_id,

            "file_name":
                image_id_to_file[
                    image_id
                ],

            "caption":
                captions[0]
        }
    )


    if len(training_pairs) >= num_samples:

        break


print(
    f"Prepared "
    f"{len(training_pairs)} "
    f"unique image-caption pairs"
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
        self.image_folder = (
            image_folder
        )


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
# 10. Image collate
# --------------------------------------------------

def image_collate_fn(batch):

    return processor(
        images=batch,
        return_tensors="pt"
    )


# --------------------------------------------------
# 11. Encode all candidate images
# --------------------------------------------------

image_dataset = COCOImageDataset(
    training_pairs,
    image_folder
)


image_loader = DataLoader(
    image_dataset,
    batch_size=image_batch_size,
    shuffle=False,
    collate_fn=image_collate_fn,
    num_workers=4,
    pin_memory=True
)


image_embeddings = []


print(
    "\nEncoding 50K candidate images..."
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
            model.encode_image(
                pixel_values
            )
        )


        image_embeddings.append(
            embeddings.cpu()
        )


        if (
            batch_index + 1
        ) % 25 == 0:

            encoded_count = min(
                (
                    batch_index + 1
                )
                * image_batch_size,
                len(training_pairs)
            )


            print(
                f"Encoded images: "
                f"{encoded_count}/"
                f"{len(training_pairs)}"
            )


image_embeddings = torch.cat(
    image_embeddings,
    dim=0
)


print(
    f"Image embedding shape: "
    f"{image_embeddings.shape}"
)


# Keep the complete search matrix on GPU
image_embeddings = (
    image_embeddings.to(device)
)


# --------------------------------------------------
# 12. Captions
# --------------------------------------------------

captions = [
    pair["caption"]
    for pair in training_pairs
]


# --------------------------------------------------
# 13. Build reranker examples
# --------------------------------------------------

reranker_examples = []


positive_rank_sum = 0

positive_top1 = 0
positive_top5 = 0
positive_top10 = 0
positive_top50 = 0


print(
    "\nRetrieving candidates for reranker..."
)


with torch.inference_mode():

    for start_index in range(
        0,
        len(captions),
        text_batch_size
    ):

        end_index = min(
            start_index
            + text_batch_size,
            len(captions)
        )


        batch_captions = captions[
            start_index:end_index
        ]


        text_inputs = processor(
            text=batch_captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )


        input_ids = (
            text_inputs[
                "input_ids"
            ].to(
                device,
                non_blocking=True
            )
        )


        attention_mask = (
            text_inputs[
                "attention_mask"
            ].to(
                device,
                non_blocking=True
            )
        )


        text_embeddings = (
            model.encode_text(
                input_ids,
                attention_mask
            )
        )


        similarity = (
            text_embeddings
            @ image_embeddings.T
        )


        current_batch_size = (
            end_index
            - start_index
        )


        row_indices = torch.arange(
            current_batch_size,
            device=device
        )


        positive_indices = torch.arange(
            start_index,
            end_index,
            device=device
        )


        # --------------------------------------------------
        # Score of the actual positive image
        # --------------------------------------------------

        positive_scores = similarity[
            row_indices,
            positive_indices
        ]


        # --------------------------------------------------
        # Compute positive rank without sorting all 50K
        # --------------------------------------------------

        positive_ranks = (
            (
                similarity
                > positive_scores.unsqueeze(1)
            )
            .sum(
                dim=1
            )
            + 1
        )


        # --------------------------------------------------
        # Remove positive before mining negatives
        # --------------------------------------------------

        similarity[
            row_indices,
            positive_indices
        ] = float(
            "-inf"
        )


        negative_scores, negative_indices = (
            torch.topk(
                similarity,
                k=num_negative_candidates,
                dim=1
            )
        )


        positive_scores = (
            positive_scores.cpu()
        )

        positive_ranks = (
            positive_ranks.cpu()
        )

        negative_scores = (
            negative_scores.cpu()
        )

        negative_indices = (
            negative_indices.cpu()
        )


        # --------------------------------------------------
        # Save each query
        # --------------------------------------------------

        for local_index in range(
            current_batch_size
        ):

            global_index = (
                start_index
                + local_index
            )


            pair = training_pairs[
                global_index
            ]


            positive_rank = int(
                positive_ranks[
                    local_index
                ]
            )


            positive_score = float(
                positive_scores[
                    local_index
                ]
            )


            positive_rank_sum += (
                positive_rank
            )


            if positive_rank <= 1:

                positive_top1 += 1


            if positive_rank <= 5:

                positive_top5 += 1


            if positive_rank <= 10:

                positive_top10 += 1


            if positive_rank <= 50:

                positive_top50 += 1


            candidates = []


            for rank in range(
                num_negative_candidates
            ):

                negative_index = int(
                    negative_indices[
                        local_index,
                        rank
                    ]
                )


                negative_pair = (
                    training_pairs[
                        negative_index
                    ]
                )


                candidates.append(
                    {
                        "retrieval_rank":
                            rank + 1,

                        "image_id":
                            negative_pair[
                                "image_id"
                            ],

                        "file_name":
                            negative_pair[
                                "file_name"
                            ],

                        "retrieval_score":
                            float(
                                negative_scores[
                                    local_index,
                                    rank
                                ]
                            )
                    }
                )


            reranker_examples.append(
                {
                    "query_index":
                        global_index,

                    "caption":
                        pair[
                            "caption"
                        ],

                    "positive":
                        {
                            "image_id":
                                pair[
                                    "image_id"
                                ],

                            "file_name":
                                pair[
                                    "file_name"
                                ],

                            "retrieval_score":
                                positive_score,

                            "retrieval_rank":
                                positive_rank
                        },

                    "negative_candidates":
                        candidates
                }
            )


        print(
            f"Processed queries: "
            f"{end_index}/"
            f"{len(captions)}"
        )


# --------------------------------------------------
# 14. Retriever diagnostics
# --------------------------------------------------

num_queries = len(
    reranker_examples
)


average_positive_rank = (
    positive_rank_sum
    / num_queries
)


retrieval_r1 = (
    positive_top1
    / num_queries
)

retrieval_r5 = (
    positive_top5
    / num_queries
)

retrieval_r10 = (
    positive_top10
    / num_queries
)

retrieval_r50 = (
    positive_top50
    / num_queries
)


print(
    "\nTraining-set retrieval diagnostics"
)

print(
    "-" * 50
)

print(
    f"Average positive rank: "
    f"{average_positive_rank:.2f}"
)

print(
    f"Recall@1:  "
    f"{retrieval_r1:.4f}"
)

print(
    f"Recall@5:  "
    f"{retrieval_r5:.4f}"
)

print(
    f"Recall@10: "
    f"{retrieval_r10:.4f}"
)

print(
    f"Recall@50: "
    f"{retrieval_r50:.4f}"
)


# --------------------------------------------------
# 15. Save reranker dataset
# --------------------------------------------------

output_data = {

    "retriever":
        "DeepSearch Retriever V2",

    "checkpoint":
        checkpoint_path,

    "num_queries":
        num_queries,

    "num_negative_candidates":
        num_negative_candidates,

    "retriever_diagnostics":
        {
            "average_positive_rank":
                average_positive_rank,

            "recall_at_1":
                retrieval_r1,

            "recall_at_5":
                retrieval_r5,

            "recall_at_10":
                retrieval_r10,

            "recall_at_50":
                retrieval_r50
        },

    "examples":
        reranker_examples
}


with open(
    output_file,
    "w"
) as file:

    json.dump(
        output_data,
        file,
        indent=2
    )


print(
    "\nSaved reranker dataset to:"
)

print(
    output_file
)
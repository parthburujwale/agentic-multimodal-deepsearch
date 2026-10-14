import json
import os
import random
import sys

import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


# --------------------------------------------------
# Import Retriever V2
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

retriever_checkpoint_path = (
    "../../experiments/"
    "deepsearch_retriever_v2/"
    "deepsearch_retriever_v2.pt"
)

processor_path = (
    "../../experiments/"
    "deepsearch_retriever_v2"
)


train_output_file = (
    "../../data/coco/"
    "reranker_v3_train_45k.json"
)

dev_output_file = (
    "../../data/coco/"
    "reranker_v3_dev_5k.json"
)


num_total_samples = 50000

train_size = 45000

dev_size = 5000

num_negative_candidates = 20

image_batch_size = 128

text_batch_size = 256

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

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print(
    f"Using device: {device}"
)


# --------------------------------------------------
# 4. Load Retriever V2
# --------------------------------------------------

print(
    "\nLoading DeepSearch Retriever V2..."
)


checkpoint = torch.load(
    retriever_checkpoint_path,
    map_location=device,
    weights_only=False
)


retriever = DeepSearchRetrieverV2(
    backbone_name=(
        checkpoint[
            "backbone_name"
        ]
    ),

    embedding_dim=(
        checkpoint[
            "embedding_dim"
        ]
    ),

    hidden_dim=(
        checkpoint[
            "hidden_dim"
        ]
    ),

    dropout=(
        checkpoint[
            "dropout"
        ]
    ),

    freeze_backbone=True
)


retriever.load_state_dict(
    checkpoint[
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
    "Retriever V2 loaded successfully"
)


# --------------------------------------------------
# 5. Processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    processor_path
)


# --------------------------------------------------
# 6. Load COCO annotations
# --------------------------------------------------

with open(
    captions_file,
    "r"
) as file:

    coco_data = json.load(
        file
    )


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
# 7. First caption per unique image
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
# 8. Build same 50K unique-image pool
# --------------------------------------------------

all_pairs = []


for image_id, caption in (
    image_id_to_caption.items()
):

    all_pairs.append(
        {
            "image_id":
                image_id,

            "file_name":
                image_id_to_file[
                    image_id
                ],

            "caption":
                caption
        }
    )


    if len(all_pairs) >= (
        num_total_samples
    ):

        break


print(
    f"Prepared "
    f"{len(all_pairs)} "
    f"unique pairs"
)


# --------------------------------------------------
# 9. IMAGE-DISJOINT split
# --------------------------------------------------

indices = list(
    range(
        len(all_pairs)
    )
)


random.shuffle(
    indices
)


train_indices = (
    indices[
        :train_size
    ]
)

dev_indices = (
    indices[
        train_size:
        train_size + dev_size
    ]
)


train_pairs = [
    all_pairs[index]
    for index in train_indices
]


dev_pairs = [
    all_pairs[index]
    for index in dev_indices
]


print(
    f"Train images: "
    f"{len(train_pairs)}"
)

print(
    f"Dev images: "
    f"{len(dev_pairs)}"
)


# --------------------------------------------------
# Verify no image overlap
# --------------------------------------------------

train_image_ids = {
    pair["image_id"]
    for pair in train_pairs
}


dev_image_ids = {
    pair["image_id"]
    for pair in dev_pairs
}


overlap = (
    train_image_ids
    & dev_image_ids
)


print(
    f"Train/dev image overlap: "
    f"{len(overlap)}"
)


if len(overlap) != 0:

    raise RuntimeError(
        "Train and dev images overlap!"
    )


# --------------------------------------------------
# 10. Image Dataset
# --------------------------------------------------

class ImageDataset(Dataset):

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

        file_name = (
            self.pairs[
                index
            ][
                "file_name"
            ]
        )


        image_path = os.path.join(
            self.image_folder,
            file_name
        )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        return image


# --------------------------------------------------
# 11. Image collate
# --------------------------------------------------

def image_collate_fn(batch):

    return processor(
        images=batch,
        return_tensors="pt"
    )


# --------------------------------------------------
# 12. Encode one corpus
# --------------------------------------------------

def encode_images(
    pairs,
    split_name
):

    dataset = ImageDataset(
        pairs,
        image_folder
    )


    loader = DataLoader(
        dataset,
        batch_size=image_batch_size,
        shuffle=False,
        collate_fn=image_collate_fn,
        num_workers=4,
        pin_memory=True
    )


    embeddings = []


    print(
        f"\nEncoding "
        f"{split_name} images..."
    )


    with torch.inference_mode():

        for batch_index, batch in enumerate(
            loader
        ):

            pixel_values = (
                batch[
                    "pixel_values"
                ].to(
                    device,
                    non_blocking=True
                )
            )


            batch_embeddings = (
                retriever.encode_image(
                    pixel_values
                )
            )


            embeddings.append(
                batch_embeddings.cpu()
            )


            if (
                batch_index + 1
            ) % 25 == 0:

                count = min(
                    (
                        batch_index + 1
                    )
                    * image_batch_size,

                    len(pairs)
                )


                print(
                    f"{split_name} images: "
                    f"{count}/"
                    f"{len(pairs)}"
                )


    embeddings = torch.cat(
        embeddings,
        dim=0
    )


    print(
        f"{split_name} image "
        f"embedding shape: "
        f"{embeddings.shape}"
    )


    return embeddings.to(
        device
    )


# --------------------------------------------------
# 13. Mine candidates INSIDE one split only
# --------------------------------------------------

def build_split_dataset(
    pairs,
    image_embeddings,
    split_name
):

    captions = [
        pair["caption"]
        for pair in pairs
    ]


    examples = []


    positive_rank_sum = 0

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    recall_20 = 0


    print(
        f"\nMining {split_name} "
        f"candidates..."
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


            text_embeddings = (
                retriever.encode_text(
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


            # Because pairs and image_embeddings
            # have identical ordering,
            # the positive image index is exactly
            # the query index inside this split.
            positive_indices = torch.arange(
                start_index,
                end_index,
                device=device
            )


            positive_scores = similarity[
                row_indices,
                positive_indices
            ]


            positive_ranks = (
                (
                    similarity
                    > positive_scores.unsqueeze(
                        1
                    )
                )
                .sum(
                    dim=1
                )
                + 1
            )


            # ------------------------------------------
            # Diagnostics
            # ------------------------------------------

            positive_rank_sum += (
                positive_ranks.sum().item()
            )


            recall_1 += (
                positive_ranks <= 1
            ).sum().item()


            recall_5 += (
                positive_ranks <= 5
            ).sum().item()


            recall_10 += (
                positive_ranks <= 10
            ).sum().item()


            recall_20 += (
                positive_ranks <= 20
            ).sum().item()


            # ------------------------------------------
            # Remove positive before mining negatives
            # ------------------------------------------

            similarity[
                row_indices,
                positive_indices
            ] = float(
                "-inf"
            )


            (
                negative_scores,
                negative_indices
            ) = torch.topk(
                similarity,
                k=num_negative_candidates,
                dim=1
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


            # ------------------------------------------
            # Save examples
            # ------------------------------------------

            for local_index in range(
                current_batch_size
            ):

                query_index = (
                    start_index
                    + local_index
                )


                pair = pairs[
                    query_index
                ]


                negatives = []


                for negative_rank in range(
                    num_negative_candidates
                ):

                    candidate_index = int(
                        negative_indices[
                            local_index,
                            negative_rank
                        ]
                    )


                    candidate = pairs[
                        candidate_index
                    ]


                    negatives.append(
                        {
                            "retrieval_rank":
                                negative_rank + 1,

                            "image_id":
                                candidate[
                                    "image_id"
                                ],

                            "file_name":
                                candidate[
                                    "file_name"
                                ],

                            "retrieval_score":
                                float(
                                    negative_scores[
                                        local_index,
                                        negative_rank
                                    ]
                                )
                        }
                    )


                examples.append(
                    {
                        "query_index":
                            query_index,

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
                                    float(
                                        positive_scores[
                                            local_index
                                        ]
                                    ),

                                "retrieval_rank":
                                    int(
                                        positive_ranks[
                                            local_index
                                        ]
                                    )
                            },

                        "negative_candidates":
                            negatives
                    }
                )


            print(
                f"{split_name} queries: "
                f"{end_index}/"
                f"{len(captions)}"
            )


    num_queries = len(
        pairs
    )


    diagnostics = {

        "average_positive_rank":
            positive_rank_sum
            / num_queries,

        "recall_at_1":
            recall_1
            / num_queries,

        "recall_at_5":
            recall_5
            / num_queries,

        "recall_at_10":
            recall_10
            / num_queries,

        "recall_at_20":
            recall_20
            / num_queries
    }


    print(
        f"\n{split_name} diagnostics"
    )

    print(
        "-" * 50
    )

    print(
        f"Average positive rank: "
        f"{diagnostics['average_positive_rank']:.2f}"
    )

    print(
        f"Recall@1:  "
        f"{diagnostics['recall_at_1']:.4f}"
    )

    print(
        f"Recall@5:  "
        f"{diagnostics['recall_at_5']:.4f}"
    )

    print(
        f"Recall@10: "
        f"{diagnostics['recall_at_10']:.4f}"
    )

    print(
        f"Recall@20: "
        f"{diagnostics['recall_at_20']:.4f}"
    )


    return {

        "split":
            split_name,

        "num_queries":
            num_queries,

        "num_negative_candidates":
            num_negative_candidates,

        "random_seed":
            random_seed,

        "retriever":
            "DeepSearch Retriever V2",

        "retriever_diagnostics":
            diagnostics,

        "examples":
            examples
    }


# --------------------------------------------------
# 14. Encode train images
# --------------------------------------------------

train_image_embeddings = (
    encode_images(
        train_pairs,
        "TRAIN"
    )
)


# --------------------------------------------------
# 15. Build TRAIN candidates
#
# Only train images are searched.
# --------------------------------------------------

train_dataset = (
    build_split_dataset(
        train_pairs,
        train_image_embeddings,
        "TRAIN"
    )
)


# --------------------------------------------------
# Free train embedding matrix
# before processing development set.
# --------------------------------------------------

del train_image_embeddings

torch.cuda.empty_cache()


# --------------------------------------------------
# 16. Encode DEV images
# --------------------------------------------------

dev_image_embeddings = (
    encode_images(
        dev_pairs,
        "DEV"
    )
)


# --------------------------------------------------
# 17. Build DEV candidates
#
# Only dev images are searched.
# --------------------------------------------------

dev_dataset = (
    build_split_dataset(
        dev_pairs,
        dev_image_embeddings,
        "DEV"
    )
)


# --------------------------------------------------
# 18. Save
# --------------------------------------------------

with open(
    train_output_file,
    "w"
) as file:

    json.dump(
        train_dataset,
        file,
        indent=2
    )


with open(
    dev_output_file,
    "w"
) as file:

    json.dump(
        dev_dataset,
        file,
        indent=2
    )


print(
    "\nSaved:"
)

print(
    train_output_file
)

print(
    dev_output_file
)


print(
    "\nImage-disjoint dataset "
    "construction complete."
)

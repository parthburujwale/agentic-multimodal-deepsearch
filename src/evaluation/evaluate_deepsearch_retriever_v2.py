import json
import os
import sys

import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


# --------------------------------------------------
# Import our custom V2 model
# --------------------------------------------------

sys.path.append("..")

from models.deepsearch_retriever_v2 import (
    DeepSearchRetrieverV2
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

image_folder = "../../data/coco/val2017"

captions_file = (
    "../../data/coco/annotations/"
    "captions_val2017.json"
)

experiment_dir = (
    "../../experiments/"
    "deepsearch_retriever_v2"
)

checkpoint_path = os.path.join(
    experiment_dir,
    "deepsearch_retriever_v2.pt"
)

batch_size = 64


# --------------------------------------------------
# 2. Select device
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
# 3. Load checkpoint
# --------------------------------------------------

print(
    f"Loading checkpoint from:"
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


print(
    f"Backbone: {backbone_name}"
)

print(
    f"Embedding dimension: "
    f"{embedding_dim}"
)

print(
    f"Hidden dimension: "
    f"{hidden_dim}"
)

print(
    f"Saved image alpha: "
    f"{checkpoint['image_alpha']:.6f}"
)

print(
    f"Saved text alpha: "
    f"{checkpoint['text_alpha']:.6f}"
)


# --------------------------------------------------
# 4. Load processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    experiment_dir
)


# --------------------------------------------------
# 5. Reconstruct DeepSearch Retriever V2
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
    "Model checkpoint loaded successfully"
)

print(
    f"Loaded image alpha: "
    f"{model.image_alpha.item():.6f}"
)

print(
    f"Loaded text alpha: "
    f"{model.text_alpha.item():.6f}"
)


# --------------------------------------------------
# 6. Load COCO validation annotations
# --------------------------------------------------

with open(
    captions_file,
    "r"
) as file:

    coco_data = json.load(file)


# --------------------------------------------------
# 7. Map each image to its first caption
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
# 8. Build aligned validation pairs
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
    f"Prepared "
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
# 10. Image collate function
# --------------------------------------------------

def image_collate_fn(batch):

    inputs = processor(
        images=batch,
        return_tensors="pt"
    )

    return inputs


# --------------------------------------------------
# 11. Encode validation images
# --------------------------------------------------

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
            model.encode_image(
                pixel_values
            )
        )


        image_embeddings.append(
            embeddings.cpu()
        )


        encoded_count = min(
            (
                batch_index + 1
            )
            * batch_size,
            len(validation_pairs)
        )


        print(
            f"Encoded images "
            f"{encoded_count} "
            f"to "
            f"{len(validation_pairs)}"
        )


image_embeddings = torch.cat(
    image_embeddings,
    dim=0
)


print(
    f"Image embedding shape: "
    f"{image_embeddings.shape}"
)


# --------------------------------------------------
# 12. Prepare captions
# --------------------------------------------------

captions = [
    pair["caption"]
    for pair in validation_pairs
]


# --------------------------------------------------
# 13. Encode validation captions
# --------------------------------------------------

text_embeddings = []


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


        embeddings = (
            model.encode_text(
                input_ids,
                attention_mask
            )
        )


        text_embeddings.append(
            embeddings.cpu()
        )


        print(
            f"Encoded captions "
            f"{end_index} "
            f"to "
            f"{len(captions)}"
        )


text_embeddings = torch.cat(
    text_embeddings,
    dim=0
)


print(
    f"Text embedding shape: "
    f"{text_embeddings.shape}"
)


# --------------------------------------------------
# 14. Compute similarity matrix
# --------------------------------------------------

print(
    "\nComputing similarity matrix..."
)


image_embeddings = (
    image_embeddings.to(device)
)

text_embeddings = (
    text_embeddings.to(device)
)


similarity_matrix = (
    text_embeddings
    @ image_embeddings.T
)


print(
    f"Similarity matrix shape: "
    f"{similarity_matrix.shape}"
)


# --------------------------------------------------
# 15. Compute retrieval metrics
# --------------------------------------------------

num_queries = (
    similarity_matrix.shape[0]
)


sorted_indices = torch.argsort(
    similarity_matrix,
    dim=1,
    descending=True
)


ground_truth = torch.arange(
    num_queries,
    device=device
)


def recall_at_k(
    sorted_indices,
    ground_truth,
    k
):

    top_k = sorted_indices[
        :,
        :k
    ]


    correct = (
        top_k
        == ground_truth.unsqueeze(1)
    ).any(
        dim=1
    )


    recall = (
        correct.float()
        .mean()
        .item()
    )


    return recall


recall_1 = recall_at_k(
    sorted_indices,
    ground_truth,
    1
)

recall_5 = recall_at_k(
    sorted_indices,
    ground_truth,
    5
)

recall_10 = recall_at_k(
    sorted_indices,
    ground_truth,
    10
)


# --------------------------------------------------
# 16. Print V2 results
# --------------------------------------------------

print(
    "\nDeepSearch Retriever V2 Results"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{recall_1:.4f}"
)

print(
    f"Recall@5:  "
    f"{recall_5:.4f}"
)

print(
    f"Recall@10: "
    f"{recall_10:.4f}"
)


# --------------------------------------------------
# 17. Existing baselines
# --------------------------------------------------

pretrained_r1 = 0.2868
pretrained_r5 = 0.5320
pretrained_r10 = 0.6466


v1_r1 = 0.2722
v1_r5 = 0.5454
v1_r10 = 0.6788


control_r1 = 0.3312
control_r5 = 0.6016
control_r10 = 0.7208


semihard_r1 = 0.3340
semihard_r5 = 0.6056
semihard_r10 = 0.7192


# --------------------------------------------------
# 18. Compare against pretrained CLIP
# --------------------------------------------------

print(
    "\nPretrained CLIP Baseline"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{pretrained_r1:.4f}"
)

print(
    f"Recall@5:  "
    f"{pretrained_r5:.4f}"
)

print(
    f"Recall@10: "
    f"{pretrained_r10:.4f}"
)


print(
    "\nDifference vs Pretrained CLIP"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{recall_1 - pretrained_r1:+.4f}"
)

print(
    f"Recall@5:  "
    f"{recall_5 - pretrained_r5:+.4f}"
)

print(
    f"Recall@10: "
    f"{recall_10 - pretrained_r10:+.4f}"
)


# --------------------------------------------------
# 19. Compare against DeepSearch V1
# --------------------------------------------------

print(
    "\nDifference vs DeepSearch V1"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{recall_1 - v1_r1:+.4f}"
)

print(
    f"Recall@5:  "
    f"{recall_5 - v1_r5:+.4f}"
)

print(
    f"Recall@10: "
    f"{recall_10 - v1_r10:+.4f}"
)


# --------------------------------------------------
# 20. Compare against CLIP-only control
# --------------------------------------------------

print(
    "\nDifference vs CLIP-Only Control"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{recall_1 - control_r1:+.4f}"
)

print(
    f"Recall@5:  "
    f"{recall_5 - control_r5:+.4f}"
)

print(
    f"Recall@10: "
    f"{recall_10 - control_r10:+.4f}"
)


# --------------------------------------------------
# 21. Compare against semi-hard model
# --------------------------------------------------

print(
    "\nDifference vs Semi-Hard Model"
)

print(
    "-" * 50
)

print(
    f"Recall@1:  "
    f"{recall_1 - semihard_r1:+.4f}"
)

print(
    f"Recall@5:  "
    f"{recall_5 - semihard_r5:+.4f}"
)

print(
    f"Recall@10: "
    f"{recall_10 - semihard_r10:+.4f}"
)
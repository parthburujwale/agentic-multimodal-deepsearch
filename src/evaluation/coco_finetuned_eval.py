import json
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

image_folder = "../../data/coco/val2017"
captions_file = "../../data/coco/annotations/captions_val2017.json"

model_path = "../../experiments/clip_control_extra_epoch_50k"

num_images = 5000
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

print(f"Using device: {device}")


# --------------------------------------------------
# 3. Load fine-tuned CLIP checkpoint
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    model_path
)

model = CLIPModel.from_pretrained(
    model_path
)

model = model.to(device)
model.eval()

print(
    f"Loaded fine-tuned model from: "
    f"{model_path}"
)


# --------------------------------------------------
# 4. Load COCO validation annotations
# --------------------------------------------------

with open(
    captions_file,
    "r"
) as file:

    coco_data = json.load(file)

print(
    f"Loaded {len(coco_data['images'])} images "
    f"and {len(coco_data['annotations'])} captions"
)


# --------------------------------------------------
# 5. Build image -> captions mapping
# --------------------------------------------------

captions_by_image = {}

for annotation in coco_data["annotations"]:

    image_id = annotation["image_id"]
    caption = annotation["caption"]

    if image_id not in captions_by_image:
        captions_by_image[image_id] = []

    captions_by_image[
        image_id
    ].append(caption)


# --------------------------------------------------
# 6. Select evaluation images
# --------------------------------------------------

selected_images = coco_data[
    "images"
][:num_images]

images = []
queries = []

for image_info in selected_images:

    image_id = image_info["id"]
    file_name = image_info["file_name"]

    image_path = os.path.join(
        image_folder,
        file_name
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    caption = captions_by_image[
        image_id
    ][0]

    images.append(image)
    queries.append(caption)

print(
    f"Selected {len(images)} "
    f"validation images"
)


# --------------------------------------------------
# 7. Encode images in batches
# --------------------------------------------------

print("\nEncoding validation images...")

all_image_features = []

for start_idx in range(
    0,
    len(images),
    batch_size
):

    end_idx = min(
        start_idx + batch_size,
        len(images)
    )

    batch_images = images[
        start_idx:end_idx
    ]

    image_inputs = processor(
        images=batch_images,
        return_tensors="pt",
        padding=True
    )

    image_inputs = {
        key: value.to(device)
        for key, value in image_inputs.items()
    }

    with torch.no_grad():

        image_output = model.get_image_features(
            **image_inputs
        )

        batch_features = (
            image_output.pooler_output
        )

    batch_features = (
        batch_features
        / batch_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    all_image_features.append(
        batch_features.cpu()
    )

    print(
        f"Encoded images "
        f"{start_idx} to {end_idx}"
    )


image_features = torch.cat(
    all_image_features,
    dim=0
)

print(
    f"Image embedding shape: "
    f"{image_features.shape}"
)


# --------------------------------------------------
# 8. Encode captions in batches
# --------------------------------------------------

print("\nEncoding validation captions...")

all_text_features = []

for start_idx in range(
    0,
    len(queries),
    batch_size
):

    end_idx = min(
        start_idx + batch_size,
        len(queries)
    )

    batch_queries = queries[
        start_idx:end_idx
    ]

    text_inputs = processor(
        text=batch_queries,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    with torch.no_grad():

        text_output = model.get_text_features(
            **text_inputs
        )

        batch_features = (
            text_output.pooler_output
        )

    batch_features = (
        batch_features
        / batch_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    all_text_features.append(
        batch_features.cpu()
    )

    print(
        f"Encoded captions "
        f"{start_idx} to {end_idx}"
    )


text_features = torch.cat(
    all_text_features,
    dim=0
)

print(
    f"Text embedding shape: "
    f"{text_features.shape}"
)


# --------------------------------------------------
# 9. Compute similarity matrix
# --------------------------------------------------

print("\nComputing similarity matrix...")

similarity_matrix = (
    text_features
    @ image_features.T
)

print(
    f"Similarity matrix shape: "
    f"{similarity_matrix.shape}"
)


# --------------------------------------------------
# 10. Recall@K
# --------------------------------------------------

def recall_at_k(
    similarity_matrix,
    k
):

    correct = 0

    for query_index in range(
        similarity_matrix.shape[0]
    ):

        scores = similarity_matrix[
            query_index
        ]

        top_indices = torch.topk(
            scores,
            k=k
        ).indices

        if query_index in top_indices:
            correct += 1

    return (
        correct
        / similarity_matrix.shape[0]
    )


recall_1 = recall_at_k(
    similarity_matrix,
    1
)

recall_5 = recall_at_k(
    similarity_matrix,
    5
)

recall_10 = recall_at_k(
    similarity_matrix,
    10
)


# --------------------------------------------------
# 11. Print results
# --------------------------------------------------

print("\nFine-Tuned Retrieval Results")
print("-" * 50)

print(f"Recall@1:  {recall_1:.4f}")
print(f"Recall@5:  {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")


# --------------------------------------------------
# 12. Compare with pretrained baseline
# --------------------------------------------------

baseline_recall_1 = 0.2868
baseline_recall_5 = 0.5320
baseline_recall_10 = 0.6466

print("\nPretrained CLIP Baseline")
print("-" * 50)

print(
    f"Recall@1:  "
    f"{baseline_recall_1:.4f}"
)

print(
    f"Recall@5:  "
    f"{baseline_recall_5:.4f}"
)

print(
    f"Recall@10: "
    f"{baseline_recall_10:.4f}"
)


print("\nDifference")
print("-" * 50)

print(
    f"Recall@1:  "
    f"{recall_1 - baseline_recall_1:+.4f}"
)

print(
    f"Recall@5:  "
    f"{recall_5 - baseline_recall_5:+.4f}"
)

print(
    f"Recall@10: "
    f"{recall_10 - baseline_recall_10:+.4f}"
)
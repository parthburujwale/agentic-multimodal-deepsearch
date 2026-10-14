import json
import os

import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

image_folder = "../../data/coco/train2017"

captions_file = (
    "../../data/coco/annotations/"
    "captions_train2017.json"
)

model_path = (
    "../../experiments/"
    "clip_projection_finetune_50k_unique"
)

output_file = (
    "../../data/coco/"
    "hard_negatives_50k.json"
)

num_samples = 50000

image_batch_size = 128
text_batch_size = 256

num_hard_negatives = 5


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
# 3. Load COCO annotations
# --------------------------------------------------

with open(
    captions_file,
    "r"
) as file:

    coco_data = json.load(file)


image_id_to_file = {}

for image_info in coco_data["images"]:

    image_id_to_file[
        image_info["id"]
    ] = image_info["file_name"]


# --------------------------------------------------
# 4. Group captions by unique image
# --------------------------------------------------

image_id_to_captions = {}

for annotation in coco_data["annotations"]:

    image_id = annotation["image_id"]
    caption = annotation["caption"]

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
# 5. Build the same 50K unique-image dataset
# --------------------------------------------------

training_pairs = []

for image_id, captions in image_id_to_captions.items():

    file_name = image_id_to_file[
        image_id
    ]

    caption = captions[0]

    training_pairs.append(
        {
            "image_id": image_id,
            "file_name": file_name,
            "caption": caption
        }
    )

    if len(training_pairs) >= num_samples:
        break


print(
    f"Prepared {len(training_pairs)} "
    f"unique image-caption pairs"
)


# --------------------------------------------------
# 6. Image dataset
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

        return len(self.pairs)


    def __getitem__(
        self,
        index
    ):

        pair = self.pairs[index]

        image_path = os.path.join(
            self.image_folder,
            pair["file_name"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        return image


# --------------------------------------------------
# 7. Load model + processor
# --------------------------------------------------

print(
    f"Loading model from: "
    f"{model_path}"
)

processor = CLIPProcessor.from_pretrained(
    model_path
)

model = CLIPModel.from_pretrained(
    model_path
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# 8. Helper for Transformers output formats
# --------------------------------------------------

def get_feature_tensor(output):

    if isinstance(
        output,
        torch.Tensor
    ):

        return output

    if hasattr(
        output,
        "pooler_output"
    ):

        return output.pooler_output

    raise TypeError(
        "Could not extract embedding "
        "tensor from model output."
    )


# --------------------------------------------------
# 9. Image collate function
# --------------------------------------------------

def image_collate_fn(batch):

    inputs = processor(
        images=batch,
        return_tensors="pt"
    )

    return inputs


# --------------------------------------------------
# 10. Encode all 50K images
# --------------------------------------------------

image_dataset = COCOImageDataset(
    training_pairs,
    image_folder
)

image_loader = DataLoader(
    image_dataset,
    batch_size=image_batch_size,
    shuffle=False,
    collate_fn=image_collate_fn
)


image_embeddings = []

print(
    "\nEncoding images..."
)


with torch.inference_mode():

    for batch_index, batch in enumerate(
        image_loader
    ):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        output = model.get_image_features(
            **batch
        )

        embeddings = get_feature_tensor(
            output
        )

        embeddings = (
            embeddings
            / embeddings.norm(
                dim=-1,
                keepdim=True
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


# Move complete image matrix back to GPU
image_embeddings = (
    image_embeddings.to(device)
)


# --------------------------------------------------
# 11. Mine hard negatives
# --------------------------------------------------

captions = [
    pair["caption"]
    for pair in training_pairs
]


hard_negative_results = []


print(
    "\nMining hard negatives..."
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


        text_inputs = {
            key: value.to(device)
            for key, value
            in text_inputs.items()
        }


        output = model.get_text_features(
            **text_inputs
        )

        text_embeddings = get_feature_tensor(
            output
        )


        text_embeddings = (
            text_embeddings
            / text_embeddings.norm(
                dim=-1,
                keepdim=True
            )
        )


        similarity = (
            text_embeddings
            @ image_embeddings.T
        )


        # --------------------------------------------------
        # Remove the true matching image
        # --------------------------------------------------

        row_indices = torch.arange(
            end_index - start_index,
            device=device
        )

        positive_indices = torch.arange(
            start_index,
            end_index,
            device=device
        )


        similarity[
            row_indices,
            positive_indices
        ] = float("-inf")


        # --------------------------------------------------
        # Retrieve most similar incorrect images
        # --------------------------------------------------

        scores, indices = torch.topk(
            similarity,
            k=num_hard_negatives,
            dim=1
        )


        scores = scores.cpu()
        indices = indices.cpu()


        for local_index in range(
            end_index - start_index
        ):

            global_index = (
                start_index
                + local_index
            )

            positive_pair = training_pairs[
                global_index
            ]


            negatives = []

            for negative_rank in range(
                num_hard_negatives
            ):

                negative_index = int(
                    indices[
                        local_index,
                        negative_rank
                    ]
                )

                negative_pair = training_pairs[
                    negative_index
                ]


                negatives.append(
                    {
                        "rank":
                            negative_rank + 1,

                        "file_name":
                            negative_pair[
                                "file_name"
                            ],

                        "image_id":
                            negative_pair[
                                "image_id"
                            ],

                        "similarity":
                            float(
                                scores[
                                    local_index,
                                    negative_rank
                                ]
                            )
                    }
                )


            hard_negative_results.append(
                {
                    "caption":
                        positive_pair[
                            "caption"
                        ],

                    "positive_file":
                        positive_pair[
                            "file_name"
                        ],

                    "positive_image_id":
                        positive_pair[
                            "image_id"
                        ],

                    "hard_negatives":
                        negatives
                }
            )


        print(
            f"Mined captions: "
            f"{end_index}/"
            f"{len(captions)}"
        )


# --------------------------------------------------
# 12. Save results
# --------------------------------------------------

output_data = {
    "model":
        model_path,

    "num_samples":
        len(training_pairs),

    "num_hard_negatives_per_caption":
        num_hard_negatives,

    "results":
        hard_negative_results
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
    f"\nSaved hard negatives to:"
)

print(
    output_file
)
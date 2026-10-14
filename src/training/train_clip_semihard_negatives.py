import json
import os
import random

import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

image_folder = "../../data/coco/train2017"

hard_negatives_file = (
    "../../data/coco/"
    "hard_negatives_50k.json"
)

model_path = (
    "../../experiments/"
    "clip_projection_finetune_50k_unique"
)

output_dir = (
    "../../experiments/"
    "clip_semihard_negative_50k"
)

batch_size = 32
num_epochs = 1

learning_rate = 1e-5

num_hard_negatives = 2

hard_negative_weight = 0.1


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
# 3. Load hard-negative dataset
# --------------------------------------------------

with open(
    hard_negatives_file,
    "r"
) as file:

    hard_negative_data = json.load(file)


training_samples = (
    hard_negative_data["results"]
)


print(
    f"Loaded {len(training_samples)} "
    f"hard-negative training samples"
)


# --------------------------------------------------
# 4. Dataset
# --------------------------------------------------

class HardNegativeDataset(Dataset):

    def __init__(
        self,
        samples,
        image_folder
    ):

        self.samples = samples
        self.image_folder = image_folder


    def __len__(self):

        return len(self.samples)


    def __getitem__(
        self,
        index
    ):

        sample = self.samples[index]

        caption = sample["caption"]

        positive_path = os.path.join(
            self.image_folder,
            sample["positive_file"]
        )

        positive_image = Image.open(
            positive_path
        ).convert("RGB")


        negative_entries = (
            sample["hard_negatives"]
        )

        # Use semi-hard negatives instead of the
        # absolute hardest negatives.
        #
        # Existing list:
        # index 0 = rank 1
        # index 1 = rank 2
        # index 2 = rank 3
        # index 3 = rank 4
        # index 4 = rank 5
        #
        # We deliberately use ranks 3 and 4.
        selected_negatives = (
            negative_entries[2:4]
        )


        negative_images = []

        for negative in selected_negatives:

            negative_path = os.path.join(
                self.image_folder,
                negative["file_name"]
            )

            negative_image = Image.open(
                negative_path
            ).convert("RGB")

            negative_images.append(
                negative_image
            )


        return (
            caption,
            positive_image,
            negative_images
        )


# --------------------------------------------------
# 5. Load processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    model_path
)


# --------------------------------------------------
# 6. Collate function
# --------------------------------------------------

def collate_fn(batch):

    captions = [
        item[0]
        for item in batch
    ]

    positive_images = [
        item[1]
        for item in batch
    ]


    negative_images = []

    for item in batch:

        negative_images.extend(
            item[2]
        )


    positive_inputs = processor(
        text=captions,
        images=positive_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )


    text_inputs = processor(
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )


    negative_inputs = processor(
        images=negative_images,
        return_tensors="pt"
    )


    return (
        positive_inputs,
        text_inputs,
        negative_inputs
    )


# --------------------------------------------------
# 7. Dataset + DataLoader
# --------------------------------------------------

dataset = HardNegativeDataset(
    training_samples,
    image_folder
)


dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)


print(
    f"Number of training batches: "
    f"{len(dataloader)}"
)


# --------------------------------------------------
# 8. Load current best model
# --------------------------------------------------

print(
    f"Loading model from: "
    f"{model_path}"
)


model = CLIPModel.from_pretrained(
    model_path
)

model = model.to(device)


# --------------------------------------------------
# 9. Freeze entire model
# --------------------------------------------------

for parameter in model.parameters():

    parameter.requires_grad = False


# --------------------------------------------------
# 10. Unfreeze projection heads
# --------------------------------------------------

for parameter in (
    model.visual_projection.parameters()
):

    parameter.requires_grad = True


for parameter in (
    model.text_projection.parameters()
):

    parameter.requires_grad = True


model.logit_scale.requires_grad = True


# --------------------------------------------------
# 11. Parameter counts
# --------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Trainable percentage: "
    f"{100 * trainable_parameters / total_parameters:.4f}%"
)


# --------------------------------------------------
# 12. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ],
    lr=learning_rate
)


# --------------------------------------------------
# 13. Helper for Transformers output formats
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
# 14. Training mode
# --------------------------------------------------

model.train()


# --------------------------------------------------
# 15. Training loop
# --------------------------------------------------

for epoch in range(
    num_epochs
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/"
        f"{num_epochs}"
    )


    running_total_loss = 0.0
    running_clip_loss = 0.0
    running_hard_loss = 0.0


    for batch_index, batch in enumerate(
        dataloader
    ):

        (
            positive_inputs,
            text_inputs,
            negative_inputs
        ) = batch


        positive_inputs = {
            key: value.to(
                device,
                non_blocking=True
            )
            for key, value
            in positive_inputs.items()
        }


        text_inputs = {
            key: value.to(
                device,
                non_blocking=True
            )
            for key, value
            in text_inputs.items()
        }


        negative_inputs = {
            key: value.to(
                device,
                non_blocking=True
            )
            for key, value
            in negative_inputs.items()
        }


        optimizer.zero_grad()


        # --------------------------------------------------
        # Standard CLIP contrastive loss
        # --------------------------------------------------

        outputs = model(
            **positive_inputs,
            return_loss=True
        )


        clip_loss = outputs.loss


        # --------------------------------------------------
        # Text embeddings
        # --------------------------------------------------

        text_output = model.get_text_features(
            **text_inputs
        )


        text_embeddings = get_feature_tensor(
            text_output
        )


        text_embeddings = F.normalize(
            text_embeddings,
            dim=-1
        )


        # --------------------------------------------------
        # Positive image embeddings
        # --------------------------------------------------

        positive_image_output = (
            model.get_image_features(
                pixel_values=(
                    positive_inputs[
                        "pixel_values"
                    ]
                )
            )
        )


        positive_embeddings = (
            get_feature_tensor(
                positive_image_output
            )
        )


        positive_embeddings = F.normalize(
            positive_embeddings,
            dim=-1
        )


        # --------------------------------------------------
        # Hard-negative image embeddings
        # --------------------------------------------------

        negative_image_output = (
            model.get_image_features(
                **negative_inputs
            )
        )


        negative_embeddings = (
            get_feature_tensor(
                negative_image_output
            )
        )


        negative_embeddings = F.normalize(
            negative_embeddings,
            dim=-1
        )


        current_batch_size = (
            text_embeddings.shape[0]
        )


        embedding_dim = (
            negative_embeddings.shape[-1]
        )


        negative_embeddings = (
            negative_embeddings.view(
                current_batch_size,
                num_hard_negatives,
                embedding_dim
            )
        )


        # --------------------------------------------------
        # Build candidate sets:
        #
        # candidate 0 = correct image
        # candidate 1..K = hard negatives
        # --------------------------------------------------

        positive_embeddings = (
            positive_embeddings.unsqueeze(1)
        )


        candidate_embeddings = torch.cat(
            [
                positive_embeddings,
                negative_embeddings
            ],
            dim=1
        )


        # --------------------------------------------------
        # Similarity scores
        # --------------------------------------------------

        logit_scale = (
            model.logit_scale.exp()
        )


        hard_logits = (
            torch.einsum(
                "bd,bkd->bk",
                text_embeddings,
                candidate_embeddings
            )
            * logit_scale
        )


        # Correct image is always candidate 0
        labels = torch.zeros(
            current_batch_size,
            dtype=torch.long,
            device=device
        )


        hard_negative_loss = (
            F.cross_entropy(
                hard_logits,
                labels
            )
        )


        # --------------------------------------------------
        # Combined objective
        # --------------------------------------------------

        total_loss = (
            clip_loss
            + hard_negative_weight
            * hard_negative_loss
        )


        total_loss.backward()


        optimizer.step()


        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        running_total_loss += (
            total_loss.item()
        )

        running_clip_loss += (
            clip_loss.item()
        )

        running_hard_loss += (
            hard_negative_loss.item()
        )


        average_total_loss = (
            running_total_loss
            / (batch_index + 1)
        )


        average_clip_loss = (
            running_clip_loss
            / (batch_index + 1)
        )


        average_hard_loss = (
            running_hard_loss
            / (batch_index + 1)
        )


        if (
            batch_index + 1
        ) % 25 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(dataloader)} "
                f"| Total: "
                f"{average_total_loss:.4f} "
                f"| CLIP: "
                f"{average_clip_loss:.4f} "
                f"| Hard: "
                f"{average_hard_loss:.4f}"
            )


    print(
        f"\nEpoch {epoch + 1} complete"
    )

    print(
        f"Average total loss: "
        f"{average_total_loss:.4f}"
    )

    print(
        f"Average CLIP loss: "
        f"{average_clip_loss:.4f}"
    )

    print(
        f"Average hard-negative loss: "
        f"{average_hard_loss:.4f}"
    )


# --------------------------------------------------
# 16. Save model
# --------------------------------------------------

os.makedirs(
    output_dir,
    exist_ok=True
)


model.save_pretrained(
    output_dir
)

processor.save_pretrained(
    output_dir
)


print(
    f"\nSaved hard-negative model to:"
)

print(
    output_dir
)
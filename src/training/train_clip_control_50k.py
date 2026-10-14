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

model_name = (
    "../../experiments/"
    "clip_projection_finetune_50k_unique"
)

output_dir = (
    "../../experiments/"
    "clip_control_extra_epoch_50k"
)

# Control experiment configuration
num_samples = 50000
batch_size = 32
num_epochs = 1

learning_rate = 1e-5


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
# 4. Build unique image-caption pairs
# --------------------------------------------------

image_id_to_captions = {}

for annotation in coco_data["annotations"]:

    image_id = annotation["image_id"]
    caption = annotation["caption"]

    if image_id not in image_id_to_captions:
        image_id_to_captions[image_id] = []

    image_id_to_captions[image_id].append(
        caption
    )


training_pairs = []

for image_id, captions in image_id_to_captions.items():

    file_name = image_id_to_file[
        image_id
    ]

    # Use one caption per unique image
    caption = captions[0]

    training_pairs.append(
        {
            "file_name": file_name,
            "caption": caption
        }
    )

    if len(training_pairs) >= num_samples:
        break


print(
    f"Prepared {len(training_pairs)} "
    f"unique image-caption training pairs"
)


# --------------------------------------------------
# 5. Dataset
# --------------------------------------------------

class COCORetrievalDataset(Dataset):

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

        caption = pair["caption"]

        return image, caption


# --------------------------------------------------
# 6. Load CLIP processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    model_name
)


# --------------------------------------------------
# 7. Collate function
# --------------------------------------------------

def collate_fn(batch):

    images = [
        item[0]
        for item in batch
    ]

    captions = [
        item[1]
        for item in batch
    ]

    inputs = processor(
        text=captions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    return inputs


# --------------------------------------------------
# 8. Dataset + DataLoader
# --------------------------------------------------

dataset = COCORetrievalDataset(
    training_pairs,
    image_folder
)


dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)


print(
    f"Number of training batches: "
    f"{len(dataloader)}"
)


# --------------------------------------------------
# 9. Load pretrained CLIP
# --------------------------------------------------

model = CLIPModel.from_pretrained(
    model_name
)

model = model.to(device)


# --------------------------------------------------
# 10. Freeze entire CLIP model
# --------------------------------------------------

for parameter in model.parameters():

    parameter.requires_grad = False


# --------------------------------------------------
# 11. Unfreeze projection heads
# --------------------------------------------------

for parameter in (
    model.visual_projection.parameters()
):

    parameter.requires_grad = True


for parameter in (
    model.text_projection.parameters()
):

    parameter.requires_grad = True


# Also allow CLIP's learned temperature
# parameter to adapt
model.logit_scale.requires_grad = True


# --------------------------------------------------
# 12. Count trainable parameters
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
# 13. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ],
    lr=learning_rate
)


model.train()


# --------------------------------------------------
# 14. Training loop
# --------------------------------------------------

for epoch in range(
    num_epochs
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/"
        f"{num_epochs}"
    )

    running_loss = 0.0


    for batch_index, batch in enumerate(
        dataloader
    ):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }


        optimizer.zero_grad()


        outputs = model(
            **batch,
            return_loss=True
        )


        loss = outputs.loss


        loss.backward()


        optimizer.step()


        running_loss += loss.item()


        average_loss = (
            running_loss
            / (batch_index + 1)
        )


        if (batch_index + 1) % 25 == 0:
            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(dataloader)} "
                f"| Batch Loss: "
                f"{loss.item():.4f} "
                f"| Average Loss: "
                f"{average_loss:.4f}"
            )


# --------------------------------------------------
# 15. Save checkpoint
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
    f"\nSaved projection-fine-tuned "
    f"model to: {output_dir}"
)
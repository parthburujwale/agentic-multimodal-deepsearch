import json
import os
import sys

import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


# --------------------------------------------------
# Import our custom retriever
# --------------------------------------------------

sys.path.append("..")

from models.deepsearch_retriever import (
    DeepSearchRetriever
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

image_folder = "../../data/coco/train2017"

captions_file = (
    "../../data/coco/annotations/"
    "captions_train2017.json"
)

backbone_name = (
    "openai/"
    "clip-vit-base-patch32"
)

output_dir = (
    "../../experiments/"
    "deepsearch_retriever_v1"
)


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

num_samples = 50000

batch_size = 64
num_epochs = 3

learning_rate = 1e-4

embedding_dim = 512
hidden_dim = 1024
dropout = 0.1


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
# 5. Build unique image-caption pairs
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
# 6. Dataset
# --------------------------------------------------

class COCORetrievalDataset(Dataset):

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


        caption = pair[
            "caption"
        ]


        return (
            image,
            caption
        )


# --------------------------------------------------
# 7. Processor
# --------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    backbone_name
)


# --------------------------------------------------
# 8. Collate function
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
# 9. Dataset + DataLoader
# --------------------------------------------------

dataset = COCORetrievalDataset(
    training_pairs,
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
# 10. Load custom DeepSearch model
# --------------------------------------------------

model = DeepSearchRetriever(
    backbone_name=backbone_name,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    dropout=dropout,
    freeze_backbones=True
)


model = model.to(
    device
)


# --------------------------------------------------
# 11. Parameter counts
# --------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
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
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ],
    lr=learning_rate,
    weight_decay=0.01
)


# --------------------------------------------------
# 13. Symmetric contrastive loss
# --------------------------------------------------

def contrastive_loss(
    logits_per_text,
    logits_per_image
):

    batch_size = (
        logits_per_text.shape[0]
    )


    labels = torch.arange(
        batch_size,
        device=logits_per_text.device
    )


    text_to_image_loss = (
        F.cross_entropy(
            logits_per_text,
            labels
        )
    )


    image_to_text_loss = (
        F.cross_entropy(
            logits_per_image,
            labels
        )
    )


    loss = (
        text_to_image_loss
        + image_to_text_loss
    ) / 2


    return (
        loss,
        text_to_image_loss,
        image_to_text_loss
    )


# --------------------------------------------------
# 14. Training
# --------------------------------------------------

model.train()


for epoch in range(
    num_epochs
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/"
        f"{num_epochs}"
    )


    running_loss = 0.0

    running_text_loss = 0.0

    running_image_loss = 0.0


    for batch_index, batch in enumerate(
        dataloader
    ):

        batch = {
            key: value.to(
                device,
                non_blocking=True
            )

            for key, value
            in batch.items()
        }


        optimizer.zero_grad()


        outputs = model(
            pixel_values=(
                batch["pixel_values"]
            ),

            input_ids=(
                batch["input_ids"]
            ),

            attention_mask=(
                batch["attention_mask"]
            )
        )


        (
            loss,
            text_loss,
            image_loss
        ) = contrastive_loss(

            outputs[
                "logits_per_text"
            ],

            outputs[
                "logits_per_image"
            ]
        )


        loss.backward()


        optimizer.step()


        # Prevent runaway temperature
        with torch.no_grad():

            model.logit_scale.clamp_(
                0,
                4.6052
            )


        running_loss += (
            loss.item()
        )


        running_text_loss += (
            text_loss.item()
        )


        running_image_loss += (
            image_loss.item()
        )


        average_loss = (
            running_loss
            / (batch_index + 1)
        )


        average_text_loss = (
            running_text_loss
            / (batch_index + 1)
        )


        average_image_loss = (
            running_image_loss
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
                f"{average_loss:.4f} "
                f"| T2I: "
                f"{average_text_loss:.4f} "
                f"| I2T: "
                f"{average_image_loss:.4f}"
            )


    print(
        f"\nEpoch {epoch + 1} complete"
    )


    print(
        f"Average total loss: "
        f"{average_loss:.4f}"
    )


    print(
        f"Average text-to-image loss: "
        f"{average_text_loss:.4f}"
    )


    print(
        f"Average image-to-text loss: "
        f"{average_image_loss:.4f}"
    )


# --------------------------------------------------
# 15. Save custom checkpoint
# --------------------------------------------------

os.makedirs(
    output_dir,
    exist_ok=True
)


checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "backbone_name":
        backbone_name,

    "embedding_dim":
        embedding_dim,

    "hidden_dim":
        hidden_dim,

    "dropout":
        dropout,

    "num_samples":
        num_samples,

    "batch_size":
        batch_size,

    "num_epochs":
        num_epochs,

    "learning_rate":
        learning_rate
}


checkpoint_path = os.path.join(
    output_dir,
    "deepsearch_retriever.pt"
)


torch.save(
    checkpoint,
    checkpoint_path
)


processor.save_pretrained(
    output_dir
)


print(
    f"\nSaved DeepSearch retriever to:"
)

print(
    checkpoint_path
)
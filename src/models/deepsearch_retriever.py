import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel


# --------------------------------------------------
# 1. Custom projection head
# --------------------------------------------------

class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        output_dim=512,
        dropout=0.1
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.LayerNorm(
                input_dim
            ),

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )


    def forward(
        self,
        features
    ):

        embeddings = self.network(
            features
        )

        embeddings = F.normalize(
            embeddings,
            dim=-1
        )

        return embeddings


# --------------------------------------------------
# 2. DeepSearch multimodal retriever
# --------------------------------------------------

class DeepSearchRetriever(nn.Module):

    def __init__(
        self,
        backbone_name=(
            "openai/"
            "clip-vit-base-patch32"
        ),
        embedding_dim=512,
        hidden_dim=1024,
        dropout=0.1,
        freeze_backbones=True
    ):

        super().__init__()


        # --------------------------------------------------
        # Load pretrained CLIP
        # --------------------------------------------------

        clip = CLIPModel.from_pretrained(
            backbone_name
        )


        # We keep the transformer encoders,
        # but DO NOT use CLIP's original
        # projection layers.
        self.vision_encoder = (
            clip.vision_model
        )

        self.text_encoder = (
            clip.text_model
        )


        # --------------------------------------------------
        # Backbone dimensions
        # --------------------------------------------------

        vision_dim = (
            clip.config.vision_config.hidden_size
        )

        text_dim = (
            clip.config.text_config.hidden_size
        )


        print(
            f"Vision backbone dimension: "
            f"{vision_dim}"
        )

        print(
            f"Text backbone dimension: "
            f"{text_dim}"
        )


        # --------------------------------------------------
        # Our custom projection heads
        # --------------------------------------------------

        self.image_projection = (
            ProjectionHead(
                input_dim=vision_dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
        )

        self.text_projection = (
            ProjectionHead(
                input_dim=text_dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
        )


        # --------------------------------------------------
        # Learnable contrastive temperature
        # --------------------------------------------------

        self.logit_scale = nn.Parameter(
            torch.tensor(
                2.6592
            )
        )


        # --------------------------------------------------
        # Freeze pretrained backbones
        # --------------------------------------------------

        if freeze_backbones:

            for parameter in (
                self.vision_encoder.parameters()
            ):

                parameter.requires_grad = False


            for parameter in (
                self.text_encoder.parameters()
            ):

                parameter.requires_grad = False


    # --------------------------------------------------
    # Image encoder
    # --------------------------------------------------

    def encode_image(
        self,
        pixel_values
    ):

        vision_output = (
            self.vision_encoder(
                pixel_values=pixel_values
            )
        )


        pooled_features = (
            vision_output.pooler_output
        )


        image_embeddings = (
            self.image_projection(
                pooled_features
            )
        )


        return image_embeddings


    # --------------------------------------------------
    # Text encoder
    # --------------------------------------------------

    def encode_text(
        self,
        input_ids,
        attention_mask
    ):

        text_output = (
            self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        )


        pooled_features = (
            text_output.pooler_output
        )


        text_embeddings = (
            self.text_projection(
                pooled_features
            )
        )


        return text_embeddings


    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask
    ):

        image_embeddings = (
            self.encode_image(
                pixel_values
            )
        )


        text_embeddings = (
            self.encode_text(
                input_ids,
                attention_mask
            )
        )


        scale = (
            self.logit_scale.exp()
        )


        logits_per_text = (
            scale
            * text_embeddings
            @ image_embeddings.T
        )


        logits_per_image = (
            logits_per_text.T
        )


        return {
            "image_embeddings":
                image_embeddings,

            "text_embeddings":
                text_embeddings,

            "logits_per_text":
                logits_per_text,

            "logits_per_image":
                logits_per_image
        }
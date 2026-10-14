import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel


# --------------------------------------------------
# 1. Residual adapter
# --------------------------------------------------

class ResidualAdapter(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        hidden_dim=1024,
        dropout=0.1
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.LayerNorm(
                embedding_dim
            ),

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )


    def forward(
        self,
        embeddings
    ):

        return self.network(
            embeddings
        )


# --------------------------------------------------
# 2. DeepSearch Retriever V2
# --------------------------------------------------

class DeepSearchRetrieverV2(nn.Module):

    def __init__(
        self,
        backbone_name=(
            "openai/"
            "clip-vit-base-patch32"
        ),
        embedding_dim=512,
        hidden_dim=1024,
        dropout=0.1,
        freeze_backbone=True
    ):

        super().__init__()


        # --------------------------------------------------
        # Load complete pretrained CLIP
        # --------------------------------------------------

        clip = CLIPModel.from_pretrained(
            backbone_name
        )


        # Keep pretrained encoders
        self.vision_encoder = (
            clip.vision_model
        )

        self.text_encoder = (
            clip.text_model
        )


        # Keep pretrained CLIP projections
        self.visual_projection = (
            clip.visual_projection
        )

        self.text_projection = (
            clip.text_projection
        )


        # --------------------------------------------------
        # Our residual adapters
        # --------------------------------------------------

        self.image_adapter = (
            ResidualAdapter(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        )

        self.text_adapter = (
            ResidualAdapter(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        )


        # --------------------------------------------------
        # Learnable residual strengths
        #
        # Start near zero so model initially behaves
        # very similarly to pretrained CLIP.
        # --------------------------------------------------

        self.image_alpha = nn.Parameter(
            torch.tensor(
                0.01
            )
        )

        self.text_alpha = nn.Parameter(
            torch.tensor(
                0.01
            )
        )


        # --------------------------------------------------
        # CLIP temperature parameter
        # --------------------------------------------------

        self.logit_scale = nn.Parameter(
            clip.logit_scale.detach().clone()
        )


        # --------------------------------------------------
        # Freeze pretrained components
        # --------------------------------------------------

        if freeze_backbone:

            for parameter in (
                self.vision_encoder.parameters()
            ):

                parameter.requires_grad = False


            for parameter in (
                self.text_encoder.parameters()
            ):

                parameter.requires_grad = False


            for parameter in (
                self.visual_projection.parameters()
            ):

                parameter.requires_grad = False


            for parameter in (
                self.text_projection.parameters()
            ):

                parameter.requires_grad = False


    # --------------------------------------------------
    # Image encoding
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


        base_embedding = (
            self.visual_projection(
                pooled_features
            )
        )


        base_embedding = (
            F.normalize(
                base_embedding,
                dim=-1
            )
        )


        adapter_update = (
            self.image_adapter(
                base_embedding
            )
        )


        final_embedding = (
            base_embedding
            + self.image_alpha
            * adapter_update
        )


        final_embedding = (
            F.normalize(
                final_embedding,
                dim=-1
            )
        )


        return final_embedding


    # --------------------------------------------------
    # Text encoding
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


        base_embedding = (
            self.text_projection(
                pooled_features
            )
        )


        base_embedding = (
            F.normalize(
                base_embedding,
                dim=-1
            )
        )


        adapter_update = (
            self.text_adapter(
                base_embedding
            )
        )


        final_embedding = (
            base_embedding
            + self.text_alpha
            * adapter_update
        )


        final_embedding = (
            F.normalize(
                final_embedding,
                dim=-1
            )
        )


        return final_embedding


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
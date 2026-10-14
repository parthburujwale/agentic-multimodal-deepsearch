import torch
import torch.nn as nn

from transformers import CLIPModel


# --------------------------------------------------
# 1. Cross-modal fusion module
# --------------------------------------------------

class CrossModalFusion(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        hidden_dim=1024,
        dropout=0.1
    ):

        super().__init__()

        # We combine:
        #
        # text embedding
        # image embedding
        # element-wise product
        # absolute difference
        #
        # 4 * 512 = 2048 dimensions

        fusion_dim = (
            embedding_dim * 4
        )

        self.network = nn.Sequential(

            nn.LayerNorm(
                fusion_dim
            ),

            nn.Linear(
                fusion_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim // 2,
                1
            )
        )


    def forward(
        self,
        text_embeddings,
        image_embeddings
    ):

        elementwise_product = (
            text_embeddings
            * image_embeddings
        )

        absolute_difference = torch.abs(
            text_embeddings
            - image_embeddings
        )

        fused_features = torch.cat(
            [
                text_embeddings,
                image_embeddings,
                elementwise_product,
                absolute_difference
            ],
            dim=-1
        )

        relevance_score = self.network(
            fused_features
        )

        return relevance_score.squeeze(
            -1
        )


# --------------------------------------------------
# 2. DeepSearch multimodal reranker
# --------------------------------------------------

class DeepSearchReranker(nn.Module):

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
        # Load pretrained CLIP
        # --------------------------------------------------

        clip = CLIPModel.from_pretrained(
            backbone_name
        )


        # --------------------------------------------------
        # Keep pretrained CLIP encoders
        # --------------------------------------------------

        self.vision_encoder = (
            clip.vision_model
        )

        self.text_encoder = (
            clip.text_model
        )


        # --------------------------------------------------
        # Keep pretrained CLIP projections
        # --------------------------------------------------

        self.visual_projection = (
            clip.visual_projection
        )

        self.text_projection = (
            clip.text_projection
        )


        # --------------------------------------------------
        # Trainable cross-modal fusion network
        # --------------------------------------------------

        self.fusion = CrossModalFusion(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )


        # --------------------------------------------------
        # Freeze pretrained CLIP components
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
    # Encode image
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
            self.visual_projection(
                pooled_features
            )
        )


        image_embeddings = (
            torch.nn.functional.normalize(
                image_embeddings,
                dim=-1
            )
        )


        return image_embeddings


    # --------------------------------------------------
    # Encode text
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


        text_embeddings = (
            torch.nn.functional.normalize(
                text_embeddings,
                dim=-1
            )
        )


        return text_embeddings


    # --------------------------------------------------
    # Score text-image pairs
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


        relevance_scores = (
            self.fusion(
                text_embeddings,
                image_embeddings
            )
        )


        return relevance_scores
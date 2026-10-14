import torch
import torch.nn as nn

from transformers import CLIPModel


# --------------------------------------------------
# 1. Feed-forward block
# --------------------------------------------------

class FeedForwardBlock(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        hidden_dim=1024,
        dropout=0.1
    ):

        super().__init__()

        self.network = nn.Sequential(

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
            ),

            nn.Dropout(
                dropout
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(x)


# --------------------------------------------------
# 2. Bounded Cross-Attention Reranker V4
# --------------------------------------------------

class DeepSearchCrossAttentionRerankerV4(
    nn.Module
):

    def __init__(
        self,
        backbone_name=(
            "openai/"
            "clip-vit-base-patch32"
        ),
        fusion_dim=512,
        num_heads=8,
        ff_hidden_dim=1024,
        scorer_hidden_dim=512,
        dropout=0.1,
        max_correction=0.02,
        freeze_backbone=True
    ):

        super().__init__()


        # --------------------------------------------------
        # Load pretrained CLIP
        # --------------------------------------------------

        clip = CLIPModel.from_pretrained(
            backbone_name
        )


        self.vision_encoder = (
            clip.vision_model
        )

        self.text_encoder = (
            clip.text_model
        )


        vision_hidden_dim = (
            clip.config
            .vision_config
            .hidden_size
        )

        text_hidden_dim = (
            clip.config
            .text_config
            .hidden_size
        )


        print(
            f"Vision hidden dimension: "
            f"{vision_hidden_dim}"
        )

        print(
            f"Text hidden dimension: "
            f"{text_hidden_dim}"
        )


        # --------------------------------------------------
        # Token projections
        # --------------------------------------------------

        self.vision_token_projection = (
            nn.Linear(
                vision_hidden_dim,
                fusion_dim
            )
        )


        self.text_token_projection = (
            nn.Linear(
                text_hidden_dim,
                fusion_dim
            )
        )


        # --------------------------------------------------
        # Normalization
        # --------------------------------------------------

        self.vision_norm = nn.LayerNorm(
            fusion_dim
        )

        self.text_norm = nn.LayerNorm(
            fusion_dim
        )


        # --------------------------------------------------
        # Cross attention
        #
        # Text tokens = queries
        # Image tokens = keys / values
        # --------------------------------------------------

        self.cross_attention = (
            nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        )


        self.cross_attention_norm = (
            nn.LayerNorm(
                fusion_dim
            )
        )


        # --------------------------------------------------
        # Feed-forward block
        # --------------------------------------------------

        self.feed_forward = (
            FeedForwardBlock(
                embedding_dim=fusion_dim,
                hidden_dim=ff_hidden_dim,
                dropout=dropout
            )
        )


        self.feed_forward_norm = (
            nn.LayerNorm(
                fusion_dim
            )
        )


        # --------------------------------------------------
        # Raw correction scorer
        # --------------------------------------------------

        self.scorer = nn.Sequential(

            nn.Linear(
                fusion_dim,
                scorer_hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                scorer_hidden_dim,
                1
            )
        )


        # --------------------------------------------------
        # HARD BOUND on reranking correction
        #
        # correction is guaranteed to remain:
        #
        # [-max_correction, +max_correction]
        # --------------------------------------------------

        self.max_correction = float(
            max_correction
        )


        # --------------------------------------------------
        # Freeze CLIP
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


    # --------------------------------------------------
    # Mask-aware pooling
    # --------------------------------------------------

    def masked_mean_pool(
        self,
        token_embeddings,
        attention_mask
    ):

        mask = (
            attention_mask
            .unsqueeze(-1)
            .to(
                token_embeddings.dtype
            )
        )


        summed = (
            token_embeddings
            * mask
        ).sum(
            dim=1
        )


        count = (
            mask.sum(
                dim=1
            )
            .clamp(
                min=1.0
            )
        )


        return (
            summed
            / count
        )


    # --------------------------------------------------
    # Cross-modal representation
    # --------------------------------------------------

    def cross_modal_features(
        self,
        pixel_values,
        input_ids,
        attention_mask
    ):

        # --------------------------------------------------
        # Frozen feature extraction
        # --------------------------------------------------

        with torch.no_grad():

            vision_outputs = (
                self.vision_encoder(
                    pixel_values=pixel_values
                )
            )


            text_outputs = (
                self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            )


        vision_tokens = (
            vision_outputs
            .last_hidden_state
        )


        text_tokens = (
            text_outputs
            .last_hidden_state
        )


        # --------------------------------------------------
        # Project tokens
        # --------------------------------------------------

        vision_tokens = (
            self.vision_token_projection(
                vision_tokens
            )
        )


        text_tokens = (
            self.text_token_projection(
                text_tokens
            )
        )


        vision_tokens = (
            self.vision_norm(
                vision_tokens
            )
        )


        text_tokens = (
            self.text_norm(
                text_tokens
            )
        )


        # --------------------------------------------------
        # Cross attention
        # --------------------------------------------------

        attended_visual, attention_weights = (
            self.cross_attention(

                query=text_tokens,

                key=vision_tokens,

                value=vision_tokens,

                need_weights=False
            )
        )


        fused_tokens = (
            self.cross_attention_norm(

                text_tokens
                + attended_visual
            )
        )


        # --------------------------------------------------
        # Feed-forward residual
        # --------------------------------------------------

        ff_output = (
            self.feed_forward(
                fused_tokens
            )
        )


        fused_tokens = (
            self.feed_forward_norm(

                fused_tokens
                + ff_output
            )
        )


        # --------------------------------------------------
        # Pool over valid text tokens
        # --------------------------------------------------

        pooled_features = (
            self.masked_mean_pool(
                fused_tokens,
                attention_mask
            )
        )


        return pooled_features


    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        retriever_scores
    ):

        pooled_features = (
            self.cross_modal_features(

                pixel_values=
                    pixel_values,

                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask
            )
        )


        # --------------------------------------------------
        # Raw unbounded scorer output
        # --------------------------------------------------

        raw_corrections = (
            self.scorer(
                pooled_features
            )
            .squeeze(-1)
        )


        # --------------------------------------------------
        # tanh forces this to [-1, +1]
        # --------------------------------------------------

        normalized_corrections = (
            torch.tanh(
                raw_corrections
            )
        )


        # --------------------------------------------------
        # Hard maximum change of ±0.02
        # --------------------------------------------------

        bounded_corrections = (
            self.max_correction
            * normalized_corrections
        )


        # --------------------------------------------------
        # Preserve Retriever V2 score
        # --------------------------------------------------

        final_scores = (
            retriever_scores
            + bounded_corrections
        )


        return {

            "final_scores":
                final_scores,

            "raw_corrections":
                raw_corrections,

            "bounded_corrections":
                bounded_corrections
        }
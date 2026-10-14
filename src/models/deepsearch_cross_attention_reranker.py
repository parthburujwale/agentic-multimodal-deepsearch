import torch
import torch.nn as nn

from transformers import CLIPModel


# --------------------------------------------------
# 1. Feed-forward block used after cross-attention
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

        return self.network(
            x
        )


# --------------------------------------------------
# 2. DeepSearch Cross-Attention Reranker
# --------------------------------------------------

class DeepSearchCrossAttentionReranker(nn.Module):

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
        initial_alpha=0.01,
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
        # CLIP encoders
        #
        # Vision hidden size for ViT-B/32: 768
        # Text hidden size: 512
        # --------------------------------------------------

        self.vision_encoder = (
            clip.vision_model
        )

        self.text_encoder = (
            clip.text_model
        )


        vision_hidden_dim = (
            clip.config.vision_config.hidden_size
        )

        text_hidden_dim = (
            clip.config.text_config.hidden_size
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
        # Project image patch tokens into common
        # 512-dimensional fusion space
        # --------------------------------------------------

        self.vision_token_projection = (
            nn.Linear(
                vision_hidden_dim,
                fusion_dim
            )
        )


        # --------------------------------------------------
        # Text projection
        #
        # CLIP text tokens are already 512-D for
        # ViT-B/32, but keeping an explicit projection
        # makes the architecture general.
        # --------------------------------------------------

        self.text_token_projection = (
            nn.Linear(
                text_hidden_dim,
                fusion_dim
            )
        )


        # --------------------------------------------------
        # Normalize tokens before attention
        # --------------------------------------------------

        self.text_norm = nn.LayerNorm(
            fusion_dim
        )

        self.vision_norm = nn.LayerNorm(
            fusion_dim
        )


        # --------------------------------------------------
        # Cross-modal attention
        #
        # Queries:
        #     text tokens
        #
        # Keys / Values:
        #     image patch tokens
        #
        # This lets each word/token inspect visual regions.
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
        # Feed-forward transformation after attention
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
        # Relevance scorer
        #
        # Converts fused text-image representation
        # into one correction score.
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
        # Residual reranking strength
        #
        # final score =
        # retriever score
        # +
        # alpha * correction
        # --------------------------------------------------

        self.alpha = nn.Parameter(
            torch.tensor(
                float(initial_alpha)
            )
        )


        # --------------------------------------------------
        # Freeze CLIP encoders
        #
        # The cross-attention components remain trainable.
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
    # 3. Mask-aware text pooling
    # --------------------------------------------------

    def masked_mean_pool(
        self,
        token_embeddings,
        attention_mask
    ):

        # [B, L] -> [B, L, 1]
        mask = (
            attention_mask
            .unsqueeze(-1)
            .to(
                token_embeddings.dtype
            )
        )


        masked_embeddings = (
            token_embeddings
            * mask
        )


        summed_embeddings = (
            masked_embeddings.sum(
                dim=1
            )
        )


        valid_token_count = (
            mask.sum(
                dim=1
            )
            .clamp(
                min=1.0
            )
        )


        pooled_embedding = (
            summed_embeddings
            / valid_token_count
        )


        return pooled_embedding


    # --------------------------------------------------
    # 4. Encode and fuse text + image
    # --------------------------------------------------

    def cross_modal_features(
        self,
        pixel_values,
        input_ids,
        attention_mask
    ):

        # --------------------------------------------------
        # Frozen CLIP token extraction
        # --------------------------------------------------

        with torch.set_grad_enabled(
            any(
                parameter.requires_grad
                for parameter
                in self.vision_encoder.parameters()
            )
        ):

            vision_outputs = (
                self.vision_encoder(
                    pixel_values=pixel_values
                )
            )


        with torch.set_grad_enabled(
            any(
                parameter.requires_grad
                for parameter
                in self.text_encoder.parameters()
            )
        ):

            text_outputs = (
                self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            )


        # --------------------------------------------------
        # Token representations
        #
        # Vision:
        # [B, image_tokens, 768]
        #
        # Text:
        # [B, text_tokens, 512]
        # --------------------------------------------------

        vision_tokens = (
            vision_outputs.last_hidden_state
        )


        text_tokens = (
            text_outputs.last_hidden_state
        )


        # --------------------------------------------------
        # Project into common fusion space
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
        #
        # Every text token can attend to all image tokens.
        # --------------------------------------------------

        attended_visual_features, attention_weights = (
            self.cross_attention(
                query=text_tokens,
                key=vision_tokens,
                value=vision_tokens,
                need_weights=True,
                average_attn_weights=False
            )
        )


        # --------------------------------------------------
        # Residual connection
        # --------------------------------------------------

        fused_tokens = (
            self.cross_attention_norm(
                text_tokens
                + attended_visual_features
            )
        )


        # --------------------------------------------------
        # Feed-forward residual block
        # --------------------------------------------------

        feed_forward_output = (
            self.feed_forward(
                fused_tokens
            )
        )


        fused_tokens = (
            self.feed_forward_norm(
                fused_tokens
                + feed_forward_output
            )
        )


        # --------------------------------------------------
        # Pool only real text tokens
        # --------------------------------------------------

        pooled_features = (
            self.masked_mean_pool(
                fused_tokens,
                attention_mask
            )
        )


        return (
            pooled_features,
            attention_weights
        )


    # --------------------------------------------------
    # 5. Forward
    # --------------------------------------------------

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        retriever_scores
    ):

        (
            pooled_features,
            attention_weights
        ) = self.cross_modal_features(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        # --------------------------------------------------
        # Cross-modal correction
        # --------------------------------------------------

        correction_scores = (
            self.scorer(
                pooled_features
            )
            .squeeze(-1)
        )


        # --------------------------------------------------
        # Residual ranking
        # --------------------------------------------------

        final_scores = (
            retriever_scores
            + self.alpha
            * correction_scores
        )


        return {

            "final_scores":
                final_scores,

            "correction_scores":
                correction_scores,

            "attention_weights":
                attention_weights,

            "alpha":
                self.alpha
        }
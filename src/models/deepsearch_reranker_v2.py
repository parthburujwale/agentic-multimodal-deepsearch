import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Residual reranking fusion network
# --------------------------------------------------

class ResidualRerankingFusion(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        hidden_dim=1024,
        dropout=0.1
    ):

        super().__init__()


        # --------------------------------------------------
        # Features:
        #
        # text embedding             = 512
        # image embedding            = 512
        # element-wise product       = 512
        # absolute difference        = 512
        # retriever similarity score = 1
        #
        # Total = 2049
        # --------------------------------------------------

        fusion_dim = (
            embedding_dim * 4
            + 1
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
        image_embeddings,
        retriever_scores
    ):

        # --------------------------------------------------
        # Pairwise interaction features
        # --------------------------------------------------

        elementwise_product = (
            text_embeddings
            * image_embeddings
        )


        absolute_difference = torch.abs(
            text_embeddings
            - image_embeddings
        )


        retriever_scores = (
            retriever_scores.unsqueeze(
                -1
            )
        )


        # --------------------------------------------------
        # Combine all features
        # --------------------------------------------------

        fused_features = torch.cat(
            [
                text_embeddings,
                image_embeddings,
                elementwise_product,
                absolute_difference,
                retriever_scores
            ],
            dim=-1
        )


        # --------------------------------------------------
        # Learn correction score
        # --------------------------------------------------

        correction_score = (
            self.network(
                fused_features
            )
            .squeeze(-1)
        )


        return correction_score


# --------------------------------------------------
# 2. DeepSearch Residual Reranker V2
# --------------------------------------------------

class DeepSearchRerankerV2(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        hidden_dim=1024,
        dropout=0.1,
        initial_alpha=0.01
    ):

        super().__init__()


        # --------------------------------------------------
        # Trainable fusion network
        # --------------------------------------------------

        self.fusion = (
            ResidualRerankingFusion(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        )


        # --------------------------------------------------
        # Learnable residual strength
        #
        # final_score =
        # retriever_score
        # + alpha * correction_score
        #
        # Start very small so initial ranking stays
        # close to DeepSearch Retriever V2.
        # --------------------------------------------------

        self.alpha = nn.Parameter(
            torch.tensor(
                float(initial_alpha)
            )
        )


    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        text_embeddings,
        image_embeddings,
        retriever_scores
    ):

        correction_scores = (
            self.fusion(
                text_embeddings,
                image_embeddings,
                retriever_scores
            )
        )


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

            "alpha":
                self.alpha
        }
import argparse
import json
import os
import sys

import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


# --------------------------------------------------
# Import project modules
# --------------------------------------------------

sys.path.append("..")

from agents.query_planner import (
    QueryPlanner
)

from models.deepsearch_retriever_v2 import (
    DeepSearchRetrieverV2
)


# --------------------------------------------------
# 1. Image Dataset
# --------------------------------------------------

class ImageCorpusDataset(Dataset):

    def __init__(
        self,
        image_records,
        image_folder
    ):

        self.image_records = (
            image_records
        )

        self.image_folder = (
            image_folder
        )


    def __len__(self):

        return len(
            self.image_records
        )


    def __getitem__(
        self,
        index
    ):

        record = self.image_records[
            index
        ]


        image_path = os.path.join(
            self.image_folder,
            record["file_name"]
        )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        return image


# --------------------------------------------------
# 2. Agentic Retriever
# --------------------------------------------------

class AgenticRetriever:

    def __init__(
        self,
        image_folder,
        captions_file,
        retriever_checkpoint_path,
        retriever_processor_path,
        planner_model_name=(
            "Qwen/"
            "Qwen2.5-1.5B-Instruct"
        ),
        image_batch_size=128,
        device=None
    ):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:

            if torch.cuda.is_available():

                device = torch.device(
                    "cuda"
                )

            elif torch.backends.mps.is_available():

                device = torch.device(
                    "mps"
                )

            else:

                device = torch.device(
                    "cpu"
                )


        self.device = device

        self.image_folder = (
            image_folder
        )

        self.captions_file = (
            captions_file
        )

        self.image_batch_size = (
            image_batch_size
        )


        print(
            f"Using device: "
            f"{self.device}"
        )


        # --------------------------------------------------
        # 3. Load Query Planner
        # --------------------------------------------------

        print(
            "\nLoading Query Planner..."
        )


        self.planner = QueryPlanner(
            model_name=
                planner_model_name,
            device=str(
                self.device
            )
        )


        print(
            "Query Planner loaded"
        )


        # --------------------------------------------------
        # 4. Load Retriever Processor
        # --------------------------------------------------

        print(
            "\nLoading Retriever processor..."
        )


        self.processor = (
            CLIPProcessor.from_pretrained(
                retriever_processor_path
            )
        )


        # --------------------------------------------------
        # 5. Load DeepSearch Retriever V2
        # --------------------------------------------------

        print(
            "\nLoading DeepSearch Retriever V2..."
        )


        checkpoint = torch.load(
            retriever_checkpoint_path,
            map_location=
                self.device,
            weights_only=False
        )


        self.retriever = (
            DeepSearchRetrieverV2(

                backbone_name=
                    checkpoint[
                        "backbone_name"
                    ],

                embedding_dim=
                    checkpoint[
                        "embedding_dim"
                    ],

                hidden_dim=
                    checkpoint[
                        "hidden_dim"
                    ],

                dropout=
                    checkpoint[
                        "dropout"
                    ],

                freeze_backbone=True
            )
        )


        self.retriever.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        self.retriever = (
            self.retriever.to(
                self.device
            )
        )


        self.retriever.eval()


        for parameter in (
            self.retriever.parameters()
        ):

            parameter.requires_grad = False


        print(
            "DeepSearch Retriever V2 loaded"
        )


        # --------------------------------------------------
        # 6. Build searchable image corpus
        # --------------------------------------------------

        self.image_records = (
            self.load_image_corpus()
        )


        print(
            f"\nSearch corpus contains "
            f"{len(self.image_records)} images"
        )


        # --------------------------------------------------
        # 7. Encode corpus once
        # --------------------------------------------------

        self.image_embeddings = (
            self.encode_image_corpus()
        )


        print(
            "\nAgentic Retriever ready."
        )


    # --------------------------------------------------
    # 8. Load COCO image metadata
    # --------------------------------------------------

    def load_image_corpus(
        self
    ):

        with open(
            self.captions_file,
            "r"
        ) as file:

            coco_data = json.load(
                file
            )


        records = []


        for image_info in coco_data[
            "images"
        ]:

            records.append(
                {
                    "image_id":
                        image_info[
                            "id"
                        ],

                    "file_name":
                        image_info[
                            "file_name"
                        ]
                }
            )


        return records


    # --------------------------------------------------
    # 9. Image preprocessing
    # --------------------------------------------------

    def image_collate_fn(
        self,
        batch
    ):

        return self.processor(
            images=batch,
            return_tensors="pt"
        )


    # --------------------------------------------------
    # 10. Encode complete image corpus
    # --------------------------------------------------

    def encode_image_corpus(
        self
    ):

        dataset = ImageCorpusDataset(
            image_records=
                self.image_records,
            image_folder=
                self.image_folder
        )


        dataloader = DataLoader(
            dataset,
            batch_size=
                self.image_batch_size,
            shuffle=False,
            collate_fn=
                self.image_collate_fn,
            num_workers=4,
            pin_memory=True
        )


        embeddings = []


        print(
            "\nEncoding image corpus..."
        )


        with torch.inference_mode():

            for batch_index, batch in enumerate(
                dataloader
            ):

                pixel_values = (
                    batch[
                        "pixel_values"
                    ].to(
                        self.device,
                        non_blocking=True
                    )
                )


                batch_embeddings = (
                    self.retriever.encode_image(
                        pixel_values
                    )
                )


                embeddings.append(
                    batch_embeddings.cpu()
                )


                count = min(
                    (
                        batch_index + 1
                    )
                    * self.image_batch_size,

                    len(
                        self.image_records
                    )
                )


                print(
                    f"Encoded images: "
                    f"{count}/"
                    f"{len(self.image_records)}"
                )


        embeddings = torch.cat(
            embeddings,
            dim=0
        )


        print(
            f"Image embedding shape: "
            f"{embeddings.shape}"
        )


        return embeddings.to(
            self.device
        )


    # --------------------------------------------------
    # 11. Encode one text query
    # --------------------------------------------------

    def encode_query(
        self,
        query
    ):

        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True
        )


        input_ids = (
            inputs[
                "input_ids"
            ].to(
                self.device
            )
        )


        attention_mask = (
            inputs[
                "attention_mask"
            ].to(
                self.device
            )
        )


        with torch.inference_mode():

            embedding = (
                self.retriever.encode_text(
                    input_ids,
                    attention_mask
                )
            )


        return embedding


    # --------------------------------------------------
    # 12. Search one subquery
    # --------------------------------------------------

    def search_one_query(
        self,
        query,
        top_k=20
    ):

        query_embedding = (
            self.encode_query(
                query
            )
        )


        similarities = (
            query_embedding
            @ self.image_embeddings.T
        ).squeeze(0)


        top_scores, top_indices = (
            torch.topk(
                similarities,
                k=min(
                    top_k,
                    len(
                        self.image_records
                    )
                )
            )
        )


        top_scores = (
            top_scores.cpu()
        )

        top_indices = (
            top_indices.cpu()
        )


        results = []


        for rank, (
            score,
            image_index
        ) in enumerate(
            zip(
                top_scores,
                top_indices
            ),
            start=1
        ):

            image_index = int(
                image_index
            )


            record = (
                self.image_records[
                    image_index
                ]
            )


            results.append(
                {
                    "rank":
                        rank,

                    "image_index":
                        image_index,

                    "image_id":
                        record[
                            "image_id"
                        ],

                    "file_name":
                        record[
                            "file_name"
                        ],

                    "retrieval_score":
                        float(
                            score
                        )
                }
            )


        return results


    # --------------------------------------------------
    # 13. Reciprocal Rank Fusion
    # --------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        query_results,
        rrf_k=60
    ):

        fused = {}


        # --------------------------------------------------
        # query_results:
        #
        # {
        #   "original query": [...],
        #   "subquery 1": [...],
        #   "subquery 2": [...]
        # }
        # --------------------------------------------------

        for query, results in (
            query_results.items()
        ):

            for result in results:

                image_index = result[
                    "image_index"
                ]


                if image_index not in fused:

                    fused[
                        image_index
                    ] = {

                        "image_index":
                            image_index,

                        "image_id":
                            result[
                                "image_id"
                            ],

                        "file_name":
                            result[
                                "file_name"
                            ],

                        "rrf_score":
                            0.0,

                        "query_matches":
                            []
                    }


                # ------------------------------------------
                # Standard Reciprocal Rank Fusion
                #
                # score += 1 / (k + rank)
                # ------------------------------------------

                contribution = (
                    1.0
                    / (
                        rrf_k
                        + result[
                            "rank"
                        ]
                    )
                )


                fused[
                    image_index
                ][
                    "rrf_score"
                ] += contribution


                fused[
                    image_index
                ][
                    "query_matches"
                ].append(
                    {
                        "query":
                            query,

                        "rank":
                            result[
                                "rank"
                            ],

                        "retrieval_score":
                            result[
                                "retrieval_score"
                            ],

                        "rrf_contribution":
                            contribution
                    }
                )


        # --------------------------------------------------
        # Convert dictionary → ranked list
        # --------------------------------------------------

        fused_results = list(
            fused.values()
        )


        fused_results.sort(
            key=lambda item:
                item[
                    "rrf_score"
                ],
            reverse=True
        )


        return fused_results


    # --------------------------------------------------
    # 14. Full Agentic Search
    # --------------------------------------------------

    def search(
        self,
        user_query,
        per_query_top_k=20,
        final_top_k=10,
        rrf_k=60,
        include_original_query=True
    ):

        # --------------------------------------------------
        # Step A:
        # Ask LLM to plan retrieval
        # --------------------------------------------------

        print(
            "\nPlanning query..."
        )


        plan = self.planner.plan(
            user_query
        )


        print(
            "\nSearch Plan"
        )

        print(
            "-" * 60
        )

        print(
            json.dumps(
                plan,
                indent=2
            )
        )


        # --------------------------------------------------
        # Step B:
        # Build actual queries that Retriever V2 receives
        #
        # Keep original query as well as decomposed
        # positive subqueries.
        #
        # The original query preserves composition.
        # The subqueries increase recall.
        # --------------------------------------------------

        retrieval_queries = []


        if include_original_query:

            retrieval_queries.append(
                user_query
            )


        retrieval_queries.extend(
            plan[
                "search_queries"
            ]
        )


        # --------------------------------------------------
        # Remove exact duplicates while preserving order
        # --------------------------------------------------

        retrieval_queries = list(
            dict.fromkeys(
                retrieval_queries
            )
        )


        print(
            "\nRetriever Queries"
        )

        print(
            "-" * 60
        )


        for index, query in enumerate(
            retrieval_queries,
            start=1
        ):

            print(
                f"{index}. {query}"
            )


        # --------------------------------------------------
        # Step C:
        # Search independently for each query
        # --------------------------------------------------

        query_results = {}


        for query in retrieval_queries:

            print(
                f"\nSearching: "
                f"{query}"
            )


            results = (
                self.search_one_query(
                    query=query,
                    top_k=
                        per_query_top_k
                )
            )


            query_results[
                query
            ] = results


            print(
                f"Retrieved "
                f"{len(results)} candidates"
            )


        # --------------------------------------------------
        # Step D:
        # Fuse result lists
        # --------------------------------------------------

        fused_results = (
            self.reciprocal_rank_fusion(
                query_results=
                    query_results,
                rrf_k=
                    rrf_k
            )
        )


        final_results = (
            fused_results[
                :final_top_k
            ]
        )


        # --------------------------------------------------
        # Step E:
        # Return evidence package
        #
        # Negative constraints are intentionally carried
        # forward rather than used by Retriever V2.
        # A later visual verifier will evaluate them.
        # --------------------------------------------------

        output = {

            "original_query":
                user_query,

            "plan":
                plan,

            "retrieval_queries":
                retrieval_queries,

            "negative_constraints":
                plan[
                    "negative_constraints"
                ],

            "positive_constraints":
                plan[
                    "positive_constraints"
                ],

            "reasoning_goal":
                plan[
                    "reasoning_goal"
                ],

            "per_query_top_k":
                per_query_top_k,

            "rrf_k":
                rrf_k,

            "results":
                final_results
        }


        return output


# --------------------------------------------------
# 15. Command-line interface
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Agentic Multimodal DeepSearch "
            "retrieval pipeline"
        )
    )


    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help=(
            "Natural-language image "
            "search request"
        )
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=(
            "Number of final fused "
            "results to return"
        )
    )


    parser.add_argument(
        "--per-query-top-k",
        type=int,
        default=20,
        help=(
            "Number of candidates retrieved "
            "for each subquery"
        )
    )


    args = parser.parse_args()


    agent = AgenticRetriever(

        image_folder=(
            "../../data/coco/"
            "val2017"
        ),

        captions_file=(
            "../../data/coco/"
            "annotations/"
            "captions_val2017.json"
        ),

        retriever_checkpoint_path=(
            "../../experiments/"
            "deepsearch_retriever_v2/"
            "deepsearch_retriever_v2.pt"
        ),

        retriever_processor_path=(
            "../../experiments/"
            "deepsearch_retriever_v2"
        )
    )


    output = agent.search(

        user_query=
            args.query,

        per_query_top_k=
            args.per_query_top_k,

        final_top_k=
            args.top_k
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL AGENTIC SEARCH RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        f"\nOriginal query:"
    )

    print(
        output[
            "original_query"
        ]
    )


    print(
        "\nNegative constraints "
        "reserved for later verification:"
    )

    print(
        output[
            "negative_constraints"
        ]
    )


    print(
        "\nTop fused results:"
    )


    for rank, result in enumerate(
        output[
            "results"
        ],
        start=1
    ):

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"FINAL RANK: "
            f"{rank}"
        )

        print(
            f"Image ID: "
            f"{result['image_id']}"
        )

        print(
            f"File: "
            f"{result['file_name']}"
        )

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            "Matched retrieval queries:"
        )


        for match in result[
            "query_matches"
        ]:

            print(
                f"  - "
                f"{match['query']}"
            )

            print(
                f"    rank="
                f"{match['rank']}, "
                f"similarity="
                f"{match['retrieval_score']:.4f}"
            )


if __name__ == "__main__":

    main()
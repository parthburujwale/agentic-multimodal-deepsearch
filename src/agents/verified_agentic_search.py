import argparse
import json
import os
import sys

import torch


sys.path.append("..")


from agents.agentic_retriever import AgenticRetriever
from agents.evidence_verifier import EvidenceVerifier


# ============================================================
# Verified Agentic Search
#
# Pipeline:
#
# DeepSearch V2
#      ↓
# Query Planner
#      ↓
# Weighted RRF
#      ↓
# VLM Evidence Verification
#      ↓
# Strict Anchor Arbitration
#      ↓
# Final Ranking
# ============================================================


class VerifiedAgenticSearch:

    def __init__(
        self,
        image_folder,
        captions_file,
        retriever_checkpoint_path,
        retriever_processor_path,
        planner_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        verifier_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        retrieval_weight=0.25,
        verification_weight=0.75,
        original_query_weight=2.0,
        subquery_weight=1.0,
    ):

        self.image_folder = image_folder

        self.retrieval_weight = retrieval_weight
        self.verification_weight = verification_weight

        self.original_query_weight = original_query_weight
        self.subquery_weight = subquery_weight


        # --------------------------------------------------
        # Normalize final retrieval / verification weights
        # --------------------------------------------------

        weight_sum = (
            self.retrieval_weight
            + self.verification_weight
        )

        if weight_sum <= 0:

            raise ValueError(
                "Retrieval and verification weights "
                "must sum to > 0."
            )


        self.retrieval_weight /= weight_sum
        self.verification_weight /= weight_sum


        print(
            "\n"
            + "=" * 70
        )

        print(
            "INITIALIZING VERIFIED AGENTIC SEARCH"
        )

        print(
            "=" * 70
        )


        # --------------------------------------------------
        # Retriever + planner
        # --------------------------------------------------

        print(
            "\nLoading Agentic Retriever..."
        )


        self.agentic_retriever = AgenticRetriever(

            image_folder=image_folder,

            captions_file=captions_file,

            retriever_checkpoint_path=
                retriever_checkpoint_path,

            retriever_processor_path=
                retriever_processor_path,

            planner_model_name=
                planner_model_name
        )


        # --------------------------------------------------
        # VLM verifier
        # --------------------------------------------------

        print(
            "\nLoading Multimodal Evidence Verifier..."
        )


        self.verifier = EvidenceVerifier(
            model_name=verifier_model_name
        )


        print(
            "\nVerified Agentic Search ready."
        )


    # ========================================================
    # Exact Retriever V2 ranking
    # ========================================================

    def exact_query_ranking(
        self,
        query
    ):

        query_embedding = (
            self.agentic_retriever
            .encode_query(
                query
            )
        )


        similarities = (

            query_embedding

            @

            self.agentic_retriever
            .image_embeddings.T

        ).squeeze(
            0
        )


        ranked_indices = torch.argsort(
            similarities,
            descending=True
        )


        return (
            similarities,
            ranked_indices
        )


    # ========================================================
    # Remove duplicate retrieval queries
    # ========================================================

    def build_retrieval_queries(
        self,
        original_query,
        plan
    ):

        retrieval_queries = [
            original_query
        ]


        retrieval_queries.extend(
            plan[
                "search_queries"
            ]
        )


        retrieval_queries = list(
            dict.fromkeys(
                retrieval_queries
            )
        )


        return retrieval_queries


    # ========================================================
    # Clean redundant negative constraints
    #
    # Example:
    #
    # ["dog swimming", "swimming"]
    #
    # becomes:
    #
    # ["dog swimming"]
    # ========================================================

    def clean_negative_constraints(
        self,
        negative_constraints
    ):

        negative_constraints = list(
            dict.fromkeys(
                negative_constraints
            )
        )


        filtered = []


        for candidate in negative_constraints:

            candidate_normalized = (
                str(candidate)
                .strip()
                .lower()
            )


            redundant = False


            for other in negative_constraints:

                if candidate == other:

                    continue


                other_normalized = (
                    str(other)
                    .strip()
                    .lower()
                )


                if (
                    candidate_normalized
                    in other_normalized

                    and

                    len(other_normalized)
                    >
                    len(candidate_normalized)
                ):

                    redundant = True
                    break


            if not redundant:

                filtered.append(
                    candidate
                )


        return filtered


    # ========================================================
    # Semantic constraints for the VLM
    #
    # Planner search_queries retain useful relations such as:
    #
    # "dog near water"
    # "people in background"
    # ========================================================

    def build_verification_constraints(
        self,
        plan
    ):

        positive_constraints = list(
            dict.fromkeys(
                plan[
                    "search_queries"
                ]
            )
        )


        negative_constraints = (
            self.clean_negative_constraints(
                plan[
                    "negative_constraints"
                ]
            )
        )


        return (
            positive_constraints,
            negative_constraints
        )


    # ========================================================
    # Weighted Reciprocal Rank Fusion
    #
    # Original full query:
    #     weight = 2.0
    #
    # Planner subqueries:
    #     weight = 1.0
    #
    # score(d) =
    #
    # sum_i weight_i / (rrf_k + rank_i(d))
    # ========================================================

    def weighted_rrf(
        self,
        retrieval_queries,
        rrf_k=60
    ):

        rankings = []
        similarities_per_query = []


        for query in retrieval_queries:

            print(
                f"\nSearching: {query}"
            )


            (
                similarities,
                ranking
            ) = self.exact_query_ranking(
                query
            )


            rankings.append(
                ranking
            )

            similarities_per_query.append(
                similarities
            )


        weights = [
            self.original_query_weight
        ]


        weights.extend(

            self.subquery_weight

            for _ in range(
                len(rankings) - 1
            )
        )


        num_images = len(
            self.agentic_retriever
            .image_records
        )


        device = (
            self.agentic_retriever
            .device
        )


        rrf_scores = torch.zeros(
            num_images,
            dtype=torch.float32,
            device=device
        )


        rank_values = torch.arange(
            1,
            num_images + 1,
            dtype=torch.float32,
            device=device
        )


        for (
            ranking,
            weight
        ) in zip(
            rankings,
            weights
        ):

            contributions = (

                float(weight)

                /

                (
                    rrf_k
                    + rank_values
                )
            )


            rrf_scores[
                ranking
            ] += contributions


        final_ranking = torch.argsort(
            rrf_scores,
            descending=True
        )


        return {
            "rankings":
                rankings,

            "similarities":
                similarities_per_query,

            "weights":
                weights,

            "rrf_scores":
                rrf_scores,

            "final_ranking":
                final_ranking,

            "baseline_top1_index":
                int(
                    rankings[
                        0
                    ][
                        0
                    ].item()
                )
        }


    # ========================================================
    # Build candidate dictionaries
    # ========================================================

    def build_candidates(
        self,
        retrieval_queries,
        rrf_output,
        retrieval_top_k
    ):

        candidate_indices = (

            rrf_output[
                "final_ranking"
            ][
                :retrieval_top_k
            ]

            .detach()
            .cpu()
            .tolist()
        )


        candidates = []


        for rrf_rank, image_index in enumerate(
            candidate_indices,
            start=1
        ):

            record = (
                self.agentic_retriever
                .image_records[
                    image_index
                ]
            )


            query_matches = []


            for (
                query,
                ranking,
                similarities
            ) in zip(
                retrieval_queries,
                rrf_output[
                    "rankings"
                ],
                rrf_output[
                    "similarities"
                ]
            ):

                match = (
                    ranking
                    == image_index
                ).nonzero(
                    as_tuple=False
                )


                query_rank = (
                    int(
                        match[
                            0,
                            0
                        ].item()
                    )
                    + 1
                )


                retrieval_score = float(
                    similarities[
                        image_index
                    ].item()
                )


                query_matches.append(
                    {
                        "query":
                            query,

                        "rank":
                            query_rank,

                        "retrieval_score":
                            retrieval_score
                    }
                )


            candidates.append(
                {
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

                    "rrf_rank":
                        rrf_rank,

                    "rrf_score":
                        float(
                            rrf_output[
                                "rrf_scores"
                            ][
                                image_index
                            ].item()
                        ),

                    "query_matches":
                        query_matches
                }
            )


        return candidates


    # ========================================================
    # Normalize RRF scores
    # ========================================================

    def normalize_rrf_scores(
        self,
        candidates
    ):

        if len(candidates) == 0:

            return candidates


        scores = [

            candidate[
                "rrf_score"
            ]

            for candidate in candidates
        ]


        maximum_score = max(
            scores
        )

        minimum_score = min(
            scores
        )


        score_range = (
            maximum_score
            - minimum_score
        )


        for candidate in candidates:

            if score_range == 0:

                normalized = 1.0

            else:

                normalized = (

                    (
                        candidate[
                            "rrf_score"
                        ]
                        - minimum_score
                    )

                    /

                    score_range
                )


            candidate[
                "normalized_rrf_score"
            ] = normalized


        return candidates


    # ========================================================
    # VLM verification
    # ========================================================

    def verify_candidate(
        self,
        candidate,
        original_query,
        positive_constraints,
        negative_constraints
    ):

        image_path = os.path.join(
            self.image_folder,
            candidate[
                "file_name"
            ]
        )


        return self.verifier.verify(

            image_path=
                image_path,

            original_query=
                original_query,

            positive_constraints=
                positive_constraints,

            negative_constraints=
                negative_constraints
        )


    # ========================================================
    # Final retrieval + verification score
    # ========================================================

    def calculate_final_score(
        self,
        retrieval_score,
        verification_score
    ):

        score = (

            self.retrieval_weight
            * retrieval_score

            +

            self.verification_weight
            * verification_score
        )


        return max(
            0.0,
            min(
                1.0,
                score
            )
        )


    # ========================================================
    # Standard VLM ranking
    # ========================================================

    def rank_verified_candidates(
        self,
        verified_candidates
    ):

        verified_candidates.sort(

            key=lambda candidate: (

                candidate[
                    "verification"
                ][
                    "overall_match"
                ],

                candidate[
                    "final_score"
                ],

                candidate[
                    "verification"
                ][
                    "overall_score"
                ],

                candidate[
                    "rrf_score"
                ]
            ),

            reverse=True
        )


        return verified_candidates


    # ========================================================
    # STRICT ANCHOR ARBITRATION
    #
    # Preserve original DeepSearch V2 top-1 only when:
    #
    # 1. It survived into the verification candidate set.
    #
    # 2. VLM says it is a complete semantic match.
    #
    # 3. Its verification score is STRICTLY greater than
    #    the current VLM winner.
    #
    # Equal verifier scores DO NOT trigger the anchor.
    # ========================================================

    def apply_strict_anchor(
        self,
        verified_candidates,
        baseline_top1_index
    ):

        if len(
            verified_candidates
        ) == 0:

            return (
                verified_candidates,
                False
            )


        winner = (
            verified_candidates[
                0
            ]
        )


        baseline_candidate = None


        for candidate in verified_candidates:

            if (
                candidate[
                    "image_index"
                ]
                ==
                baseline_top1_index
            ):

                baseline_candidate = candidate
                break


        if baseline_candidate is None:

            return (
                verified_candidates,
                False
            )


        if (
            baseline_candidate[
                "image_index"
            ]
            ==
            winner[
                "image_index"
            ]
        ):

            return (
                verified_candidates,
                False
            )


        baseline_verification = (
            baseline_candidate[
                "verification"
            ]
        )


        winner_verification = (
            winner[
                "verification"
            ]
        )


        should_anchor = (

            baseline_verification[
                "overall_match"
            ]

            and

            (
                baseline_verification[
                    "overall_score"
                ]

                >

                winner_verification[
                    "overall_score"
                ]
                + 1e-8
            )
        )


        if not should_anchor:

            return (
                verified_candidates,
                False
            )


        reordered = [
            baseline_candidate
        ]


        reordered.extend(

            candidate

            for candidate in verified_candidates

            if (
                candidate[
                    "image_index"
                ]
                !=
                baseline_top1_index
            )
        )


        return (
            reordered,
            True
        )


    # ========================================================
    # Full search
    # ========================================================

    def search(
        self,
        query,
        retrieval_top_k=10,
        final_top_k=5,
        rrf_k=60
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STAGE 1: QUERY PLANNING"
        )

        print(
            "=" * 70
        )


        plan = (
            self.agentic_retriever
            .planner
            .plan(
                query
            )
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


        retrieval_queries = (
            self.build_retrieval_queries(
                original_query=query,
                plan=plan
            )
        )


        print(
            "\nRetriever Queries"
        )

        print(
            "-" * 60
        )


        for index, retrieval_query in enumerate(
            retrieval_queries,
            start=1
        ):

            print(
                f"{index}. "
                f"{retrieval_query}"
            )


        # --------------------------------------------------
        # Weighted RRF
        # --------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STAGE 2: WEIGHTED RRF RETRIEVAL"
        )

        print(
            "=" * 70
        )


        rrf_output = self.weighted_rrf(
            retrieval_queries=
                retrieval_queries,

            rrf_k=
                rrf_k
        )


        print(
            f"\nRRF weights: "
            f"{rrf_output['weights']}"
        )


        candidates = self.build_candidates(

            retrieval_queries=
                retrieval_queries,

            rrf_output=
                rrf_output,

            retrieval_top_k=
                retrieval_top_k
        )


        candidates = (
            self.normalize_rrf_scores(
                candidates
            )
        )


        (
            positive_constraints,
            negative_constraints
        ) = self.build_verification_constraints(
            plan
        )


        print(
            "\nSemantic verification constraints:"
        )

        print(
            "\nPositive:"
        )


        for constraint in positive_constraints:

            print(
                f"  + {constraint}"
            )


        print(
            "\nNegative:"
        )


        for constraint in negative_constraints:

            print(
                f"  - {constraint}"
            )


        # --------------------------------------------------
        # VLM verification
        # --------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STAGE 3: MULTIMODAL EVIDENCE VERIFICATION"
        )

        print(
            "=" * 70
        )


        verified_candidates = []


        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"Verifying candidate "
                f"{index}/"
                f"{len(candidates)}"
            )

            print(
                f"File: "
                f"{candidate['file_name']}"
            )


            verification = self.verify_candidate(

                candidate=
                    candidate,

                original_query=
                    query,

                positive_constraints=
                    positive_constraints,

                negative_constraints=
                    negative_constraints
            )


            final_score = self.calculate_final_score(

                retrieval_score=
                    candidate[
                        "normalized_rrf_score"
                    ],

                verification_score=
                    verification[
                        "overall_score"
                    ]
            )


            candidate[
                "verification"
            ] = verification

            candidate[
                "final_score"
            ] = final_score


            print(
                f"RRF rank: "
                f"{candidate['rrf_rank']}"
            )

            print(
                f"RRF score: "
                f"{candidate['rrf_score']:.6f}"
            )

            print(
                f"Verification score: "
                f"{verification['overall_score']:.4f}"
            )

            print(
                f"Overall match: "
                f"{verification['overall_match']}"
            )

            print(
                f"Combined score: "
                f"{final_score:.4f}"
            )


            verified_candidates.append(
                candidate
            )


        # --------------------------------------------------
        # Standard VLM ranking
        # --------------------------------------------------

        verified_candidates = (
            self.rank_verified_candidates(
                verified_candidates
            )
        )


        pre_anchor_winner = (
            verified_candidates[
                0
            ][
                "image_id"
            ]

            if verified_candidates

            else None
        )


        # --------------------------------------------------
        # Strict anchor arbitration
        # --------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STAGE 4: STRICT ANCHOR ARBITRATION"
        )

        print(
            "=" * 70
        )


        (
            verified_candidates,
            anchor_applied
        ) = self.apply_strict_anchor(

            verified_candidates=
                verified_candidates,

            baseline_top1_index=
                rrf_output[
                    "baseline_top1_index"
                ]
        )


        post_anchor_winner = (
            verified_candidates[
                0
            ][
                "image_id"
            ]

            if verified_candidates

            else None
        )


        print(
            f"\nAnchor applied: "
            f"{anchor_applied}"
        )

        print(
            f"Winner before anchor: "
            f"{pre_anchor_winner}"
        )

        print(
            f"Winner after anchor: "
            f"{post_anchor_winner}"
        )


        final_results = (
            verified_candidates[
                :final_top_k
            ]
        )


        baseline_record = (
            self.agentic_retriever
            .image_records[
                rrf_output[
                    "baseline_top1_index"
                ]
            ]
        )


        return {

            "query":
                query,

            "plan":
                plan,

            "retrieval_queries":
                retrieval_queries,

            "positive_constraints":
                positive_constraints,

            "negative_constraints":
                negative_constraints,

            "reasoning_goal":
                plan[
                    "reasoning_goal"
                ],

            "original_query_weight":
                self.original_query_weight,

            "subquery_weight":
                self.subquery_weight,

            "retrieval_weight":
                self.retrieval_weight,

            "verification_weight":
                self.verification_weight,

            "baseline_top1_image_id":
                baseline_record[
                    "image_id"
                ],

            "anchor_applied":
                anchor_applied,

            "winner_before_anchor":
                pre_anchor_winner,

            "winner_after_anchor":
                post_anchor_winner,

            "num_verified_candidates":
                len(
                    verified_candidates
                ),

            "results":
                final_results
        }


# ============================================================
# Pretty print
# ============================================================


def print_verified_result(
    rank,
    result
):

    verification = (
        result[
            "verification"
        ]
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"FINAL RANK {rank}"
    )

    print(
        "=" * 70
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
        f"Weighted RRF rank: "
        f"{result['rrf_rank']}"
    )

    print(
        f"Weighted RRF score: "
        f"{result['rrf_score']:.6f}"
    )

    print(
        f"Normalized retrieval score: "
        f"{result['normalized_rrf_score']:.4f}"
    )

    print(
        f"Verification score: "
        f"{verification['overall_score']:.4f}"
    )

    print(
        f"Final combined score: "
        f"{result['final_score']:.4f}"
    )

    print(
        f"Complete match: "
        f"{verification['overall_match']}"
    )


    print(
        "\nPositive constraint checks:"
    )


    for check in verification[
        "positive_checks"
    ]:

        status = (
            "PASS"
            if check[
                "satisfied"
            ]
            else "FAIL"
        )


        print(
            f"  [{status}] "
            f"{check['constraint']} "
            f"(confidence="
            f"{check['confidence']:.2f})"
        )


    if len(
        verification[
            "negative_checks"
        ]
    ) > 0:

        print(
            "\nNegative constraint checks:"
        )


        for check in verification[
            "negative_checks"
        ]:

            status = (
                "VIOLATION"
                if check[
                    "present"
                ]
                else "PASS"
            )


            print(
                f"  [{status}] "
                f"{check['constraint']} "
                f"(confidence="
                f"{check['confidence']:.2f})"
            )


    print(
        "\nExplanation:"
    )

    print(
        verification[
            "explanation"
        ]
    )


    print(
        "\nMatched retrieval queries:"
    )


    for match in result[
        "query_matches"
    ]:

        print(
            f"  - "
            f"{match['query']} "
            f"(rank "
            f"{match['rank']}, "
            f"similarity "
            f"{match['retrieval_score']:.4f})"
        )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Anchor-aware Agentic "
            "Multimodal DeepSearch"
        )
    )


    parser.add_argument(
        "--query",
        type=str,
        required=True
    )


    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=10
    )


    parser.add_argument(
        "--final-top-k",
        type=int,
        default=5
    )


    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60
    )


    parser.add_argument(
        "--original-query-weight",
        type=float,
        default=2.0
    )


    parser.add_argument(
        "--subquery-weight",
        type=float,
        default=1.0
    )


    parser.add_argument(
        "--retrieval-weight",
        type=float,
        default=0.25
    )


    parser.add_argument(
        "--verification-weight",
        type=float,
        default=0.75
    )


    parser.add_argument(
        "--output",
        type=str,
        default=(
            "../../experiments/"
            "verified_agentic_search_result.json"
        )
    )


    args = parser.parse_args()


    search_system = VerifiedAgenticSearch(

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
        ),

        retrieval_weight=
            args.retrieval_weight,

        verification_weight=
            args.verification_weight,

        original_query_weight=
            args.original_query_weight,

        subquery_weight=
            args.subquery_weight
    )


    output = search_system.search(

        query=
            args.query,

        retrieval_top_k=
            args.retrieval_top_k,

        final_top_k=
            args.final_top_k,

        rrf_k=
            args.rrf_k
    )


    print(
        "\n\n"
        + "#" * 70
    )

    print(
        "FINAL ANCHOR-AWARE AGENTIC SEARCH RESULTS"
    )

    print(
        "#" * 70
    )


    print(
        "\nQuery:"
    )

    print(
        output[
            "query"
        ]
    )


    print(
        "\nSemantic positive constraints:"
    )

    print(
        output[
            "positive_constraints"
        ]
    )


    print(
        "\nSemantic negative constraints:"
    )

    print(
        output[
            "negative_constraints"
        ]
    )


    print(
        f"\nStrict anchor applied: "
        f"{output['anchor_applied']}"
    )


    for rank, result in enumerate(
        output[
            "results"
        ],
        start=1
    ):

        print_verified_result(
            rank,
            result
        )


    output_directory = os.path.dirname(
        args.output
    )


    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )


    with open(
        args.output,
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )


    print(
        "\nSaved complete result to:"
    )

    print(
        args.output
    )


if __name__ == "__main__":

    main()
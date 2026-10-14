import argparse
import json
import os
import statistics
import sys

import torch


sys.path.append("..")


from agents.agentic_retriever import AgenticRetriever
from agents.evidence_verifier import EvidenceVerifier


# ============================================================
# Benchmark loading
# ============================================================


def load_frozen_benchmark(
    benchmark_file
):

    with open(
        benchmark_file,
        "r"
    ) as file:

        data = json.load(
            file
        )


    benchmark = data[
        "examples"
    ]


    print(
        "\nLoaded frozen benchmark"
    )

    print(
        f"Benchmark name: "
        f"{data.get('benchmark_name')}"
    )

    print(
        f"Benchmark version: "
        f"{data.get('benchmark_version')}"
    )

    print(
        f"Examples: "
        f"{len(benchmark)}"
    )


    return (
        data,
        benchmark
    )


# ============================================================
# Corpus utilities
# ============================================================


def build_image_id_to_index(
    image_records
):

    return {

        record[
            "image_id"
        ]:
            index

        for index, record
        in enumerate(
            image_records
        )
    }


def exact_query_ranking(
    agent,
    query
):

    query_embedding = (
        agent.encode_query(
            query
        )
    )


    similarities = (

        query_embedding

        @

        agent.image_embeddings.T

    ).squeeze(
        0
    )


    ranking = torch.argsort(
        similarities,
        descending=True
    )


    return (
        similarities,
        ranking
    )


def get_target_rank(
    ranking,
    target_index
):

    match = (
        ranking
        == target_index
    ).nonzero(
        as_tuple=False
    )


    if len(match) == 0:

        return None


    return (
        int(
            match[
                0,
                0
            ].item()
        )
        + 1
    )


# ============================================================
# Weighted RRF
# ============================================================


def full_corpus_weighted_rrf(
    rankings,
    weights,
    num_images,
    rrf_k,
    device
):

    scores = torch.zeros(
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

        scores[
            ranking
        ] += (

            float(weight)

            /

            (
                rrf_k
                + rank_values
            )
        )


    ranking = torch.argsort(
        scores,
        descending=True
    )


    return (
        scores,
        ranking
    )


# ============================================================
# Constraints
# ============================================================


def clean_negative_constraints(
    constraints
):

    constraints = list(
        dict.fromkeys(
            constraints
        )
    )


    filtered = []


    for candidate in constraints:

        candidate_norm = (
            str(candidate)
            .strip()
            .lower()
        )


        redundant = False


        for other in constraints:

            if candidate == other:

                continue


            other_norm = (
                str(other)
                .strip()
                .lower()
            )


            if (
                candidate_norm
                in other_norm

                and

                len(other_norm)
                >
                len(candidate_norm)
            ):

                redundant = True
                break


        if not redundant:

            filtered.append(
                candidate
            )


    return filtered


# ============================================================
# VLM reranking
# ============================================================


def verified_reranking(
    verifier,
    agent,
    original_query,
    plan,
    rrf_scores,
    rrf_ranking,
    baseline_top1_index,
    verification_top_k,
    retrieval_weight,
    verification_weight
):

    top_indices = (

        rrf_ranking[
            :verification_top_k
        ]

        .detach()
        .cpu()
        .tolist()
    )


    top_scores = [

        float(
            rrf_scores[
                index
            ].item()
        )

        for index in top_indices
    ]


    maximum = max(
        top_scores
    )

    minimum = min(
        top_scores
    )


    score_range = (
        maximum
        - minimum
    )


    positive_constraints = list(
        dict.fromkeys(
            plan[
                "search_queries"
            ]
        )
    )


    negative_constraints = (
        clean_negative_constraints(
            plan[
                "negative_constraints"
            ]
        )
    )


    verified = []


    for rrf_rank, image_index in enumerate(
        top_indices,
        start=1
    ):

        record = (
            agent.image_records[
                image_index
            ]
        )


        raw_rrf = (
            top_scores[
                rrf_rank - 1
            ]
        )


        if score_range == 0:

            normalized_rrf = 1.0

        else:

            normalized_rrf = (

                (
                    raw_rrf
                    - minimum
                )

                /

                score_range
            )


        image_path = os.path.join(
            agent.image_folder,
            record[
                "file_name"
            ]
        )


        print(
            f"      verifying "
            f"{rrf_rank}/"
            f"{len(top_indices)}: "
            f"{record['file_name']}"
        )


        verification = verifier.verify(

            image_path=
                image_path,

            original_query=
                original_query,

            positive_constraints=
                positive_constraints,

            negative_constraints=
                negative_constraints
        )


        combined_score = (

            retrieval_weight
            * normalized_rrf

            +

            verification_weight
            * verification[
                "overall_score"
            ]
        )


        verified.append(
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
                    raw_rrf,

                "normalized_rrf_score":
                    normalized_rrf,

                "verification":
                    verification,

                "combined_score":
                    combined_score
            }
        )


    verified.sort(

        key=lambda item: (

            item[
                "verification"
            ][
                "overall_match"
            ],

            item[
                "combined_score"
            ],

            item[
                "verification"
            ][
                "overall_score"
            ],

            item[
                "rrf_score"
            ]
        ),

        reverse=True
    )


    # --------------------------------------------------
    # Save ranking before anchor
    # --------------------------------------------------

    pre_anchor = list(
        verified
    )


    # --------------------------------------------------
    # Strict anchor rule
    # --------------------------------------------------

    anchor_candidate = None


    for candidate in verified:

        if (
            candidate[
                "image_index"
            ]
            ==
            baseline_top1_index
        ):

            anchor_candidate = candidate
            break


    anchor_applied = False


    if (
        anchor_candidate is not None
        and len(verified) > 0
    ):

        winner = verified[
            0
        ]


        if (
            anchor_candidate[
                "image_index"
            ]
            !=
            winner[
                "image_index"
            ]
        ):

            anchor_verification = (
                anchor_candidate[
                    "verification"
                ]
            )

            winner_verification = (
                winner[
                    "verification"
                ]
            )


            if (
                anchor_verification[
                    "overall_match"
                ]

                and

                (
                    anchor_verification[
                        "overall_score"
                    ]

                    >

                    winner_verification[
                        "overall_score"
                    ]
                    + 1e-8
                )
            ):

                reordered = [
                    anchor_candidate
                ]


                reordered.extend(

                    candidate

                    for candidate in verified

                    if (
                        candidate[
                            "image_index"
                        ]
                        !=
                        baseline_top1_index
                    )
                )


                verified = reordered
                anchor_applied = True


    return (
        pre_anchor,
        verified,
        positive_constraints,
        negative_constraints,
        anchor_applied
    )


def target_rank_from_verified(
    results,
    target_image_id
):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if (
            result[
                "image_id"
            ]
            == target_image_id
        ):

            return rank


    return None


# ============================================================
# Metrics
# ============================================================


def recall_at_k(
    ranks,
    k
):

    return (

        sum(

            1

            for rank in ranks

            if (
                rank is not None
                and rank <= k
            )
        )

        /

        len(ranks)
    )


def mrr_at_k(
    ranks,
    k
):

    return (

        sum(

            (
                1.0 / rank
            )

            if (
                rank is not None
                and rank <= k
            )

            else 0.0

            for rank in ranks
        )

        /

        len(ranks)
    )


def capped_rank(
    rank,
    evaluation_k
):

    if (
        rank is None
        or rank > evaluation_k
    ):

        return (
            evaluation_k
            + 1
        )


    return rank


def mean_capped_rank(
    ranks,
    evaluation_k
):

    values = [

        capped_rank(
            rank,
            evaluation_k
        )

        for rank in ranks
    ]


    return (
        sum(values)
        / len(values)
    )


def median_capped_rank(
    ranks,
    evaluation_k
):

    values = [

        capped_rank(
            rank,
            evaluation_k
        )

        for rank in ranks
    ]


    return float(
        statistics.median(
            values
        )
    )


def calculate_metrics(
    ranks,
    evaluation_k
):

    return {
        "recall_at_1":
            recall_at_k(
                ranks,
                1
            ),

        "recall_at_5":
            recall_at_k(
                ranks,
                5
            ),

        "recall_at_10":
            recall_at_k(
                ranks,
                10
            ),

        "mrr_at_10":
            mrr_at_k(
                ranks,
                10
            ),

        "mean_capped_rank":
            mean_capped_rank(
                ranks,
                evaluation_k
            ),

        "median_capped_rank":
            median_capped_rank(
                ranks,
                evaluation_k
            )
    }


def compare_rank_lists(
    reference,
    candidate,
    evaluation_k
):

    improved = 0
    unchanged = 0
    worsened = 0


    for (
        old_rank,
        new_rank
    ) in zip(
        reference,
        candidate
    ):

        old_value = capped_rank(
            old_rank,
            evaluation_k
        )

        new_value = capped_rank(
            new_rank,
            evaluation_k
        )


        if new_value < old_value:

            improved += 1

        elif new_value > old_value:

            worsened += 1

        else:

            unchanged += 1


    return {
        "improved":
            improved,

        "unchanged":
            unchanged,

        "worsened":
            worsened
    }


# ============================================================
# Evaluation
# ============================================================


def evaluate(
    args
):

    print(
        "=" * 70
    )

    print(
        "ANCHOR-AWARE FROZEN BENCHMARK EVALUATION"
    )

    print(
        "=" * 70
    )


    (
        benchmark_metadata,
        benchmark
    ) = load_frozen_benchmark(
        args.benchmark_file
    )


    if args.max_queries is not None:

        benchmark = benchmark[
            :args.max_queries
        ]


        print(
            f"\nDEBUG subset: "
            f"{len(benchmark)} queries"
        )


    agent = AgenticRetriever(

        image_folder=
            args.image_folder,

        captions_file=
            args.captions_file,

        retriever_checkpoint_path=
            args.retriever_checkpoint,

        retriever_processor_path=
            args.retriever_processor,

        planner_model_name=
            args.planner_model,

        image_batch_size=
            args.image_batch_size
    )


    verifier = EvidenceVerifier(
        model_name=
            args.verifier_model
    )


    image_id_to_index = (
        build_image_id_to_index(
            agent.image_records
        )
    )


    results = []


    for query_number, example in enumerate(
        benchmark,
        start=1
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"QUERY "
            f"{query_number}/"
            f"{len(benchmark)}"
        )

        print(
            "=" * 70
        )


        query = example[
            "query"
        ]


        target_image_id = (
            example[
                "target_image_id"
            ]
        )


        target_index = (
            image_id_to_index[
                target_image_id
            ]
        )


        print(
            f"\nBenchmark ID: "
            f"{example['benchmark_id']}"
        )

        print(
            f"Target: "
            f"{example['file_name']}"
        )


        # --------------------------------------------------
        # Baseline
        # --------------------------------------------------

        print(
            "\n[1/5] DeepSearch V2"
        )


        (
            _,
            baseline_ranking
        ) = exact_query_ranking(
            agent,
            query
        )


        baseline_rank = (
            get_target_rank(
                baseline_ranking,
                target_index
            )
        )


        baseline_top1_index = int(
            baseline_ranking[
                0
            ].item()
        )


        # --------------------------------------------------
        # Planner
        # --------------------------------------------------

        plan = agent.planner.plan(
            query
        )


        retrieval_queries = [
            query
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


        rankings = []


        for retrieval_query in retrieval_queries:

            (
                _,
                ranking
            ) = exact_query_ranking(
                agent,
                retrieval_query
            )


            rankings.append(
                ranking
            )


        # --------------------------------------------------
        # Equal RRF
        # --------------------------------------------------

        print(
            "[2/5] Equal-weight RRF"
        )


        equal_weights = [
            1.0
            for _ in rankings
        ]


        (
            _,
            equal_ranking
        ) = full_corpus_weighted_rrf(

            rankings=
                rankings,

            weights=
                equal_weights,

            num_images=
                len(
                    agent.image_records
                ),

            rrf_k=
                args.rrf_k,

            device=
                agent.device
        )


        equal_rank = get_target_rank(
            equal_ranking,
            target_index
        )


        # --------------------------------------------------
        # Weighted RRF
        # --------------------------------------------------

        print(
            "[3/5] Weighted RRF"
        )


        weighted_weights = [
            args.original_query_weight
        ]


        weighted_weights.extend(

            args.subquery_weight

            for _ in range(
                len(rankings) - 1
            )
        )


        (
            weighted_scores,
            weighted_ranking
        ) = full_corpus_weighted_rrf(

            rankings=
                rankings,

            weights=
                weighted_weights,

            num_images=
                len(
                    agent.image_records
                ),

            rrf_k=
                args.rrf_k,

            device=
                agent.device
        )


        weighted_rank = get_target_rank(
            weighted_ranking,
            target_index
        )


        # --------------------------------------------------
        # VLM + strict anchor
        # --------------------------------------------------

        print(
            "[4/5] Weighted RRF + VLM"
        )

        print(
            "[5/5] Weighted RRF + VLM + Strict Anchor"
        )


        (
            pre_anchor_results,
            anchor_results,
            positive_constraints,
            negative_constraints,
            anchor_applied
        ) = verified_reranking(

            verifier=
                verifier,

            agent=
                agent,

            original_query=
                query,

            plan=
                plan,

            rrf_scores=
                weighted_scores,

            rrf_ranking=
                weighted_ranking,

            baseline_top1_index=
                baseline_top1_index,

            verification_top_k=
                args.verification_top_k,

            retrieval_weight=
                args.retrieval_weight,

            verification_weight=
                args.verification_weight
        )


        vlm_rank = (
            target_rank_from_verified(
                pre_anchor_results,
                target_image_id
            )
        )


        anchor_rank = (
            target_rank_from_verified(
                anchor_results,
                target_image_id
            )
        )


        print(
            "\nRank comparison:"
        )

        print(
            f"      V2            = "
            f"{baseline_rank}"
        )

        print(
            f"      Equal RRF     = "
            f"{equal_rank}"
        )

        print(
            f"      Weighted RRF  = "
            f"{weighted_rank}"
        )

        print(
            f"      VLM           = "
            f"{vlm_rank}"
        )

        print(
            f"      Strict Anchor = "
            f"{anchor_rank}"
        )

        print(
            f"      Anchor applied = "
            f"{anchor_applied}"
        )


        results.append(
            {
                "benchmark_id":
                    example[
                        "benchmark_id"
                    ],

                "target_image_id":
                    target_image_id,

                "target_file":
                    example[
                        "file_name"
                    ],

                "query":
                    query,

                "plan":
                    plan,

                "baseline_rank":
                    baseline_rank,

                "equal_rrf_rank":
                    equal_rank,

                "weighted_rrf_rank":
                    weighted_rank,

                "weighted_verified_rank":
                    vlm_rank,

                "strict_anchor_rank":
                    anchor_rank,

                "anchor_applied":
                    anchor_applied,

                "semantic_positive_constraints":
                    positive_constraints,

                "semantic_negative_constraints":
                    negative_constraints,

                "verified_results_before_anchor":
                    pre_anchor_results,

                "verified_results_after_anchor":
                    anchor_results
            }
        )


    # ========================================================
    # Aggregate results
    # ========================================================

    baseline_ranks = [
        x["baseline_rank"]
        for x in results
    ]

    equal_ranks = [
        x["equal_rrf_rank"]
        for x in results
    ]

    weighted_ranks = [
        x["weighted_rrf_rank"]
        for x in results
    ]

    vlm_ranks = [
        x["weighted_verified_rank"]
        for x in results
    ]

    anchor_ranks = [
        x["strict_anchor_rank"]
        for x in results
    ]


    k = (
        args.verification_top_k
    )


    metrics = {
        "baseline":
            calculate_metrics(
                baseline_ranks,
                k
            ),

        "equal_rrf":
            calculate_metrics(
                equal_ranks,
                k
            ),

        "weighted_rrf":
            calculate_metrics(
                weighted_ranks,
                k
            ),

        "weighted_rrf_vlm":
            calculate_metrics(
                vlm_ranks,
                k
            ),

        "strict_anchor":
            calculate_metrics(
                anchor_ranks,
                k
            )
    }


    comparisons = {
        "weighted_vs_baseline":
            compare_rank_lists(
                baseline_ranks,
                weighted_ranks,
                k
            ),

        "vlm_vs_weighted":
            compare_rank_lists(
                weighted_ranks,
                vlm_ranks,
                k
            ),

        "strict_anchor_vs_vlm":
            compare_rank_lists(
                vlm_ranks,
                anchor_ranks,
                k
            ),

        "strict_anchor_vs_baseline":
            compare_rank_lists(
                baseline_ranks,
                anchor_ranks,
                k
            )
    }


    print(
        "\n\n"
        + "#" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "#" * 70
    )


    names = {
        "baseline":
            "DEEPSEARCH V2",

        "equal_rrf":
            "EQUAL-WEIGHT RRF",

        "weighted_rrf":
            "WEIGHTED RRF",

        "weighted_rrf_vlm":
            "WEIGHTED RRF + VLM",

        "strict_anchor":
            "WEIGHTED RRF + VLM + STRICT ANCHOR"
    }


    for method in names:

        print(
            f"\n{names[method]}"
        )

        print(
            "-" * 50
        )


        for key, value in (
            metrics[
                method
            ].items()
        ):

            print(
                f"{key}: "
                f"{value:.4f}"
            )


    print(
        "\n"
        + "#" * 70
    )

    print(
        "IMPROVEMENT ANALYSIS"
    )

    print(
        "#" * 70
    )


    for name, values in comparisons.items():

        print(
            f"\n{name}"
        )

        print(
            "-" * 50
        )

        print(
            f"Improved: "
            f"{values['improved']}"
        )

        print(
            f"Unchanged: "
            f"{values['unchanged']}"
        )

        print(
            f"Worsened: "
            f"{values['worsened']}"
        )


    output = {
        "benchmark": {
            "benchmark_file":
                args.benchmark_file,

            "benchmark_version":
                benchmark_metadata.get(
                    "benchmark_version"
                ),

            "num_examples":
                len(
                    benchmark
                )
        },

        "configuration": {
            "rrf_k":
                args.rrf_k,

            "original_query_weight":
                args.original_query_weight,

            "subquery_weight":
                args.subquery_weight,

            "verification_top_k":
                args.verification_top_k,

            "retrieval_weight":
                args.retrieval_weight,

            "verification_weight":
                args.verification_weight
        },

        "metrics":
            metrics,

        "comparisons":
            comparisons,

        "results":
            results
    }


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
        "\nSaved:"
    )

    print(
        args.output
    )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=(
            "../../data/benchmarks/"
            "complex_coco_queries_50.json"
        )
    )


    parser.add_argument(
        "--image-folder",
        type=str,
        default=(
            "../../data/coco/"
            "val2017"
        )
    )


    parser.add_argument(
        "--captions-file",
        type=str,
        default=(
            "../../data/coco/"
            "annotations/"
            "captions_val2017.json"
        )
    )


    parser.add_argument(
        "--retriever-checkpoint",
        type=str,
        default=(
            "../../experiments/"
            "deepsearch_retriever_v2/"
            "deepsearch_retriever_v2.pt"
        )
    )


    parser.add_argument(
        "--retriever-processor",
        type=str,
        default=(
            "../../experiments/"
            "deepsearch_retriever_v2"
        )
    )


    parser.add_argument(
        "--planner-model",
        type=str,
        default=(
            "Qwen/"
            "Qwen2.5-1.5B-Instruct"
        )
    )


    parser.add_argument(
        "--verifier-model",
        type=str,
        default=(
            "Qwen/"
            "Qwen2.5-VL-3B-Instruct"
        )
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
        "--rrf-k",
        type=int,
        default=60
    )


    parser.add_argument(
        "--verification-top-k",
        type=int,
        default=10
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
        "--image-batch-size",
        type=int,
        default=128
    )


    parser.add_argument(
        "--max-queries",
        type=int,
        default=None
    )


    parser.add_argument(
        "--output",
        type=str,
        default=(
            "../../experiments/"
            "anchor_aware_evaluation.json"
        )
    )


    args = parser.parse_args()


    evaluate(
        args
    )


if __name__ == "__main__":

    main()
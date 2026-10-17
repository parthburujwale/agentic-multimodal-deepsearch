import json
import sys
from typing import Any, Dict, List, Optional

import torch


sys.path.append("..")


from agents.iterative_agentic_search import (
    IterativeAgenticSearch,
)


# ============================================================
# Paths
# ============================================================

BENCHMARK_FILE = (
    "../../data/benchmarks/"
    "complex_coco_queries_50.json"
)


OUTPUT_FILE = (
    "../../experiments/"
    "iterative_agentic_dev50_evaluation.json"
)


# ============================================================
# Evaluation configuration
# ============================================================

SEED = 42

RETRIEVAL_TOP_K = 10

FINAL_TOP_K = 10

RRF_K = 60


# ============================================================
# General helpers
# ============================================================


def normalize_image_id(
    value,
):

    if value is None:

        return None


    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return str(
            value
        ).strip()


def load_benchmark() -> List[
    Dict[str, Any]
]:

    with open(
        BENCHMARK_FILE,
        "r",
    ) as file:

        data = json.load(
            file
        )


    if isinstance(
        data,
        list,
    ):

        return data


    if isinstance(
        data,
        dict,
    ):

        for key in [
            "queries",
            "examples",
            "data",
            "benchmark",
            "items",
        ]:

            value = data.get(
                key
            )


            if isinstance(
                value,
                list,
            ):

                return value


    raise ValueError(
        "Unsupported benchmark JSON format."
    )


def get_example_id(
    example: Dict[str, Any],
) -> str:

    for key in [
        "id",
        "example_id",
        "query_id",
        "benchmark_id",
    ]:

        value = example.get(
            key
        )


        if value is not None:

            return str(
                value
            )


    return "unknown"


def get_query(
    example: Dict[str, Any],
) -> str:

    for key in [
        "query",
        "complex_query",
        "text",
        "prompt",
        "search_query",
    ]:

        value = example.get(
            key
        )


        if value:

            return str(
                value
            ).strip()


    raise KeyError(
        f"No query field found in "
        f"{list(example.keys())}"
    )


def get_target_image_id(
    example: Dict[str, Any],
):

    for key in [
        "target_image_id",
        "image_id",
        "target_id",
        "coco_image_id",
        "positive_image_id",
        "ground_truth_image_id",
    ]:

        if (
            key in example
            and
            example[
                key
            ]
            is not None
        ):

            return normalize_image_id(
                example[
                    key
                ]
            )


    for nested_key in [
        "target",
        "ground_truth",
        "positive",
    ]:

        nested = example.get(
            nested_key
        )


        if not isinstance(
            nested,
            dict,
        ):

            continue


        for key in [
            "target_image_id",
            "image_id",
            "target_id",
            "coco_image_id",
        ]:

            if (
                key in nested
                and
                nested[
                    key
                ]
                is not None
            ):

                return normalize_image_id(
                    nested[
                        key
                    ]
                )


    raise KeyError(
        f"No target image ID found in "
        f"{list(example.keys())}"
    )


def get_rank(
    results: List[
        Dict[str, Any]
    ],
    target_image_id,
) -> Optional[int]:

    target = (
        normalize_image_id(
            target_image_id
        )
    )


    for rank, result in enumerate(
        results,
        start=1,
    ):

        candidate = (
            normalize_image_id(
                result.get(
                    "image_id"
                )
            )
        )


        if candidate == target:

            return rank


    return None


def classify_change(
    old_rank: Optional[int],
    new_rank: Optional[int],
) -> str:

    if (
        old_rank is None
        and
        new_rank is None
    ):

        return "unchanged"


    if (
        old_rank is None
        and
        new_rank is not None
    ):

        return "improved"


    if (
        old_rank is not None
        and
        new_rank is None
    ):

        return "worsened"


    if new_rank < old_rank:

        return "improved"


    if new_rank > old_rank:

        return "worsened"


    return "unchanged"


# ============================================================
# Metrics
# ============================================================


class RetrievalMetrics:

    def __init__(
        self,
    ):

        self.total = 0

        self.recall_1 = 0

        self.recall_5 = 0

        self.recall_10 = 0

        self.reciprocal_rank_sum = 0.0


    def update(
        self,
        rank: Optional[int],
    ):

        self.total += 1


        if rank is None:

            return


        if rank <= 1:

            self.recall_1 += 1


        if rank <= 5:

            self.recall_5 += 1


        if rank <= 10:

            self.recall_10 += 1


        if rank <= 10:

            self.reciprocal_rank_sum += (
                1.0
                /
                rank
            )


    def compute(
        self,
    ) -> Dict[str, float]:

        if self.total == 0:

            return {

                "R@1":
                    0.0,

                "R@5":
                    0.0,

                "R@10":
                    0.0,

                "MRR@10":
                    0.0,
            }


        return {

            "R@1":
                self.recall_1
                /
                self.total,

            "R@5":
                self.recall_5
                /
                self.total,

            "R@10":
                self.recall_10
                /
                self.total,

            "MRR@10":
                self.reciprocal_rank_sum
                /
                self.total,
        }


# ============================================================
# DeepSearch V2 baseline
#
# IMPORTANT:
#
# This is the plain exact-query retriever baseline.
#
# No planner
# No RRF
# No verifier
# No iterative search
#
# It directly compares the Retriever V2 text embedding against
# all precomputed image embeddings.
# ============================================================


def run_deepsearch_baseline(
    agent: IterativeAgenticSearch,
    query: str,
    top_k: int = 10,
) -> List[
    Dict[str, Any]
]:

    retriever = (
        agent
        .search_system
        .agentic_retriever
    )


    # --------------------------------------------------------
    # Encode original query with Retriever V2
    # --------------------------------------------------------

    with torch.inference_mode():

        query_embedding = (
            retriever.encode_query(
                query
            )
        )


    # --------------------------------------------------------
    # Query embedding can be [512] or [1, 512]
    # --------------------------------------------------------

    if query_embedding.ndim == 2:

        if query_embedding.shape[0] != 1:

            raise ValueError(
                "Expected a single query embedding, "
                f"got shape {tuple(query_embedding.shape)}"
            )


        query_embedding = (
            query_embedding[
                0
            ]
        )


    if query_embedding.ndim != 1:

        raise ValueError(
            "Unexpected query embedding shape: "
            f"{tuple(query_embedding.shape)}"
        )


    # --------------------------------------------------------
    # Move to same device as image embedding matrix
    # --------------------------------------------------------

    image_embeddings = (
        retriever.image_embeddings
    )


    query_embedding = (
        query_embedding.to(
            image_embeddings.device
        )
    )


    # --------------------------------------------------------
    # Retriever V2 outputs normalized embeddings.
    #
    # Dot product therefore behaves as cosine similarity.
    # --------------------------------------------------------

    similarities = (
        image_embeddings
        @
        query_embedding
    )


    top_k = min(
        int(
            top_k
        ),
        int(
            similarities.shape[
                0
            ]
        ),
    )


    scores, indices = (
        torch.topk(
            similarities,
            k=top_k,
            largest=True,
            sorted=True,
        )
    )


    results = []


    for rank, (
        index,
        score,
    ) in enumerate(
        zip(
            indices.tolist(),
            scores.tolist(),
        ),
        start=1,
    ):

        record = (
            retriever.image_records[
                index
            ]
        )


        # ----------------------------------------------------
        # Expected image_records entries are dictionaries.
        #
        # Preserve whatever metadata already exists.
        # ----------------------------------------------------

        if isinstance(
            record,
            dict,
        ):

            result = dict(
                record
            )

        else:

            result = {

                "record":
                    record,
            }


        result[
            "image_index"
        ] = int(
            index
        )


        result[
            "retrieval_score"
        ] = float(
            score
        )


        result[
            "rank"
        ] = int(
            rank
        )


        # ----------------------------------------------------
        # Defensive image-ID reconstruction.
        #
        # Most likely image_records already contains image_id.
        # These fallbacks keep the evaluator robust.
        # ----------------------------------------------------

        if (
            "image_id"
            not in result
        ):

            for candidate_key in [
                "id",
                "coco_image_id",
                "target_image_id",
            ]:

                if candidate_key in result:

                    result[
                        "image_id"
                    ] = result[
                        candidate_key
                    ]

                    break


        results.append(
            result
        )


    return results


# ============================================================
# Pretty printing
# ============================================================


def print_metrics(
    title: str,
    metrics: Dict[str, float],
):

    print(
        "\n"
        + title
    )


    print(
        "-" * len(
            title
        )
    )


    print(
        f"R@1:    "
        f"{metrics['R@1']:.4f}"
    )


    print(
        f"R@5:    "
        f"{metrics['R@5']:.4f}"
    )


    print(
        f"R@10:   "
        f"{metrics['R@10']:.4f}"
    )


    print(
        f"MRR@10: "
        f"{metrics['MRR@10']:.4f}"
    )


# ============================================================
# Main evaluation
# ============================================================


def main():

    benchmark = (
        load_benchmark()
    )


    print(
        "\n"
        + "#" * 80
    )


    print(
        "ITERATIVE AGENTIC MULTIMODAL DEEPSEARCH"
    )


    print(
        "50-QUERY DEVELOPMENT BENCHMARK"
    )


    print(
        "#" * 80
    )


    print(
        f"\nQueries: "
        f"{len(benchmark)}"
    )


    print(
        f"Seed: "
        f"{SEED}"
    )


    print(
        "\nArchitecture:"
    )


    print(
        "  DeepSearch V2"
    )


    print(
        "  -> planner"
    )


    print(
        "  -> weighted RRF"
    )


    print(
        "  -> VLM verification"
    )


    print(
        "  -> uncertainty router"
    )


    print(
        "  -> evidence-seeking Round 2"
    )


    print(
        "  -> bounded cross-round promotion"
    )


    # ========================================================
    # Initialize once
    # ========================================================

    agent = (
        IterativeAgenticSearch(

            image_folder=
                "../../data/coco/val2017",

            captions_file=
                "../../data/coco/annotations/"
                "captions_val2017.json",

            retriever_checkpoint_path=
                "../../experiments/"
                "deepsearch_retriever_v2/"
                "deepsearch_retriever_v2.pt",

            retriever_processor_path=
                "../../experiments/"
                "deepsearch_retriever_v2",

            min_verification_score=
                0.75,

            max_rounds=
                2,

            router_score_threshold=
                0.85,

            router_score_tolerance=
                1e-6,

            router_tie_tolerance=
                0.01,

            seed=
                SEED,

            fusion_enabled=
                True,

            promotion_enabled=
                True,

            promotion_max_verifier_gap=
                0.20,
        )
    )


    # ========================================================
    # Metrics
    # ========================================================

    baseline_metrics = (
        RetrievalMetrics()
    )


    one_shot_metrics = (
        RetrievalMetrics()
    )


    iterative_metrics = (
        RetrievalMetrics()
    )


    # ========================================================
    # Counters
    # ========================================================

    router_trigger_count = 0

    round_2_count = 0

    promotion_count = 0


    raw_round_2_vs_round_1 = {

        "improved":
            0,

        "unchanged":
            0,

        "worsened":
            0,

        "not_run":
            0,
    }


    iterative_vs_one_shot = {

        "improved":
            0,

        "unchanged":
            0,

        "worsened":
            0,
    }


    iterative_vs_baseline = {

        "improved":
            0,

        "unchanged":
            0,

        "worsened":
            0,
    }


    records = []


    # ========================================================
    # Benchmark loop
    # ========================================================

    for index, example in enumerate(
        benchmark,
        start=1,
    ):

        example_id = (
            get_example_id(
                example
            )
        )


        query = (
            get_query(
                example
            )
        )


        target_image_id = (
            get_target_image_id(
                example
            )
        )


        print(
            "\n\n"
            + "=" * 80
        )


        print(
            f"QUERY "
            f"{index}/"
            f"{len(benchmark)}"
        )


        print(
            "=" * 80
        )


        print(
            f"\nExample ID: "
            f"{example_id}"
        )


        print(
            f"Target image ID: "
            f"{target_image_id}"
        )


        # ====================================================
        # DeepSearch V2 exact-query baseline
        # ====================================================

        baseline_results = (
            run_deepsearch_baseline(

                agent=
                    agent,

                query=
                    query,

                top_k=
                    RETRIEVAL_TOP_K,
            )
        )


        baseline_rank = (
            get_rank(
                baseline_results,
                target_image_id,
            )
        )


        baseline_metrics.update(
            baseline_rank
        )


        # ----------------------------------------------------
        # Sanity check on first query.
        #
        # If image IDs are missing from image_records,
        # catch that immediately instead of silently producing
        # all-None baseline ranks.
        # ----------------------------------------------------

        if (
            index == 1
            and
            baseline_results
            and
            "image_id"
            not in baseline_results[
                0
            ]
        ):

            raise RuntimeError(
                "Baseline image records do not expose "
                "an image_id field. First record keys: "
                f"{list(baseline_results[0].keys())}"
            )


        # ====================================================
        # Iterative system
        #
        # Round 1 = one-shot verified system
        #
        # Final results = bounded-promotion system
        # ====================================================

        output = (
            agent.search(

                query=
                    query,

                retrieval_top_k=
                    RETRIEVAL_TOP_K,

                final_top_k=
                    FINAL_TOP_K,

                rrf_k=
                    RRF_K,
            )
        )


        round_history = (
            output.get(
                "round_history",
                [],
            )
        )


        if not round_history:

            raise RuntimeError(
                f"No round history for "
                f"{example_id}"
            )


        # ====================================================
        # One-shot verified = Round 1
        # ====================================================

        round_1_results = (
            round_history[
                0
            ].get(
                "results",
                [],
            )
        )


        one_shot_rank = (
            get_rank(
                round_1_results,
                target_image_id,
            )
        )


        one_shot_metrics.update(
            one_shot_rank
        )


        # ====================================================
        # Router
        # ====================================================

        round_1_router = (
            round_history[
                0
            ].get(
                "router",
                {},
            )
        )


        router_triggered = bool(
            round_1_router.get(
                "trigger_round_2",
                False,
            )
        )


        if router_triggered:

            router_trigger_count += 1


        # ====================================================
        # Raw Round 2
        # ====================================================

        round_2_rank = None


        if len(
            round_history
        ) >= 2:

            round_2_count += 1


            round_2_results = (
                round_history[
                    1
                ].get(
                    "results",
                    [],
                )
            )


            round_2_rank = (
                get_rank(
                    round_2_results,
                    target_image_id,
                )
            )


            round_2_change = (
                classify_change(
                    one_shot_rank,
                    round_2_rank,
                )
            )


            raw_round_2_vs_round_1[
                round_2_change
            ] += 1


        else:

            round_2_change = (
                "not_run"
            )


            raw_round_2_vs_round_1[
                "not_run"
            ] += 1


        # ====================================================
        # Final iterative ranking
        # ====================================================

        final_results = (
            output.get(
                "results",
                [],
            )
        )


        final_rank = (
            get_rank(
                final_results,
                target_image_id,
            )
        )


        iterative_metrics.update(
            final_rank
        )


        # ====================================================
        # Bounded promotion information
        # ====================================================

        promotion_decision = (
            output.get(
                "promotion_decision",
                {},
            )
        )


        promotion_applied = bool(
            promotion_decision.get(
                "promote",
                False,
            )
        )


        if promotion_applied:

            promotion_count += 1


        # ====================================================
        # Comparisons
        # ====================================================

        final_vs_one_shot = (
            classify_change(
                one_shot_rank,
                final_rank,
            )
        )


        iterative_vs_one_shot[
            final_vs_one_shot
        ] += 1


        final_vs_baseline = (
            classify_change(
                baseline_rank,
                final_rank,
            )
        )


        iterative_vs_baseline[
            final_vs_baseline
        ] += 1


        # ====================================================
        # Save result record
        # ====================================================

        record = {

            "example_id":
                example_id,

            "query":
                query,

            "target_image_id":
                target_image_id,

            "deepsearch_v2_rank":
                baseline_rank,

            "one_shot_rank":
                one_shot_rank,

            "round_2_rank":
                round_2_rank,

            "final_iterative_rank":
                final_rank,

            "router_triggered":
                router_triggered,

            "router_reasons":
                round_1_router.get(
                    "reasons",
                    [],
                ),

            "rounds_executed":
                output.get(
                    "rounds_executed"
                ),

            "selected_mechanism":
                output.get(
                    "selected_round"
                ),

            "promotion_applied":
                promotion_applied,

            "promotion_decision":
                promotion_decision,

            "raw_round_2_vs_round_1":
                round_2_change,

            "final_vs_one_shot":
                final_vs_one_shot,

            "final_vs_baseline":
                final_vs_baseline,

            "baseline_results":
                baseline_results,

            "round_history":
                round_history,

            "cross_round_candidate_memory":
                output.get(
                    "cross_round_candidate_memory",
                    [],
                ),

            "final_results":
                final_results,
        }


        records.append(
            record
        )


        # ====================================================
        # Per-query output
        # ====================================================

        print(
            "\n"
            + "-" * 60
        )


        print(
            "QUERY RESULT"
        )


        print(
            "-" * 60
        )


        print(
            f"DeepSearch V2 rank: "
            f"{baseline_rank}"
        )


        print(
            f"One-shot rank: "
            f"{one_shot_rank}"
        )


        print(
            f"Round-2 rank: "
            f"{round_2_rank}"
        )


        print(
            f"Final iterative rank: "
            f"{final_rank}"
        )


        print(
            f"Router triggered: "
            f"{router_triggered}"
        )


        print(
            f"Promotion applied: "
            f"{promotion_applied}"
        )


        print(
            f"Final vs one-shot: "
            f"{final_vs_one_shot}"
        )


    # ========================================================
    # Final metrics
    # ========================================================

    baseline_final = (
        baseline_metrics.compute()
    )


    one_shot_final = (
        one_shot_metrics.compute()
    )


    iterative_final = (
        iterative_metrics.compute()
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "num_queries":
            len(
                benchmark
            ),

        "seed":
            SEED,

        "architecture":
            (
                "iterative_agentic_search_"
                "bounded_promotion_v1"
            ),

        "deepsearch_v2":
            baseline_final,

        "one_shot_verified":
            one_shot_final,

        "iterative_bounded_promotion":
            iterative_final,

        "router_triggered":
            router_trigger_count,

        "round_2_executed":
            round_2_count,

        "bounded_promotions_applied":
            promotion_count,

        "raw_round_2_vs_round_1":
            raw_round_2_vs_round_1,

        "iterative_vs_one_shot":
            iterative_vs_one_shot,

        "iterative_vs_deepsearch_v2":
            iterative_vs_baseline,

        "configuration":
            {

                "retrieval_top_k":
                    RETRIEVAL_TOP_K,

                "final_top_k":
                    FINAL_TOP_K,

                "rrf_k":
                    RRF_K,

                "router_score_threshold":
                    0.85,

                "router_tie_tolerance":
                    0.01,

                "promotion_max_verifier_gap":
                    0.20,
            },
    }


    output_data = {

        "status":
            "complete",

        "summary":
            summary,

        "records":
            records,
    }


    with open(
        OUTPUT_FILE,
        "w",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
        )


    # ========================================================
    # Console report
    # ========================================================

    print(
        "\n\n"
        + "#" * 80
    )


    print(
        "FINAL 50-QUERY DEVELOPMENT RESULTS"
    )


    print(
        "#" * 80
    )


    print_metrics(
        "DeepSearch V2",
        baseline_final,
    )


    print_metrics(
        "One-shot Verified Agentic",
        one_shot_final,
    )


    print_metrics(
        "Iterative + Bounded Promotion",
        iterative_final,
    )


    print(
        "\n"
        + "-" * 80
    )


    print(
        "ITERATION STATISTICS"
    )


    print(
        "-" * 80
    )


    print(
        f"Router triggered: "
        f"{router_trigger_count}/"
        f"{len(benchmark)}"
    )


    print(
        f"Round 2 executed: "
        f"{round_2_count}/"
        f"{len(benchmark)}"
    )


    print(
        f"Bounded promotions applied: "
        f"{promotion_count}/"
        f"{len(benchmark)}"
    )


    print(
        "\nRaw Round 2 vs Round 1:"
    )


    print(
        f"  Improved:  "
        f"{raw_round_2_vs_round_1['improved']}"
    )


    print(
        f"  Unchanged: "
        f"{raw_round_2_vs_round_1['unchanged']}"
    )


    print(
        f"  Worsened:  "
        f"{raw_round_2_vs_round_1['worsened']}"
    )


    print(
        f"  Not run:   "
        f"{raw_round_2_vs_round_1['not_run']}"
    )


    print(
        "\nFinal Iterative vs One-shot:"
    )


    print(
        f"  Improved:  "
        f"{iterative_vs_one_shot['improved']}"
    )


    print(
        f"  Unchanged: "
        f"{iterative_vs_one_shot['unchanged']}"
    )


    print(
        f"  Worsened:  "
        f"{iterative_vs_one_shot['worsened']}"
    )


    print(
        "\nFinal Iterative vs DeepSearch V2:"
    )


    print(
        f"  Improved:  "
        f"{iterative_vs_baseline['improved']}"
    )


    print(
        f"  Unchanged: "
        f"{iterative_vs_baseline['unchanged']}"
    )


    print(
        f"  Worsened:  "
        f"{iterative_vs_baseline['worsened']}"
    )


    print(
        "\nSaved to:"
    )


    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()
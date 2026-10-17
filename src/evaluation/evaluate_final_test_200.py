import hashlib
import json
import sys
from typing import Any, Dict, List, Optional

import torch


sys.path.append("..")


from agents.iterative_agentic_search import (
    IterativeAgenticSearch,
)


# ============================================================
# FROZEN FINAL TEST CONFIGURATION
# ============================================================

BENCHMARK_FILE = (
    "../../data/benchmarks/"
    "complex_coco_final_test_200.json"
)


OUTPUT_FILE = (
    "../../experiments/"
    "final_test_200_evaluation.json"
)


EXPECTED_BENCHMARK_SHA256 = (
    "723380c31e6842c3388f9c73f1bf251bbf4617bf7bd6f7099f822a25686e720c"
)


EXPECTED_NUM_QUERIES = 200


SEED = 42

RETRIEVAL_TOP_K = 10

FINAL_TOP_K = 10

RRF_K = 60


# ============================================================
# Frozen architecture parameters
# ============================================================

MIN_VERIFICATION_SCORE = 0.75

MAX_ROUNDS = 2

ROUTER_SCORE_THRESHOLD = 0.85

ROUTER_SCORE_TOLERANCE = 1e-6

ROUTER_TIE_TOLERANCE = 0.01

PROMOTION_MAX_VERIFIER_GAP = 0.20


# ============================================================
# Helpers
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


def sha256_file(
    path: str,
) -> str:

    hasher = hashlib.sha256()


    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )


            if not chunk:

                break


            hasher.update(
                chunk
            )


    return hasher.hexdigest()


def verify_frozen_benchmark():

    actual_hash = (
        sha256_file(
            BENCHMARK_FILE
        )
    )


    print(
        "\n"
        + "=" * 80
    )


    print(
        "VERIFYING FROZEN FINAL BENCHMARK"
    )


    print(
        "=" * 80
    )


    print(
        f"\nExpected SHA256:\n"
        f"{EXPECTED_BENCHMARK_SHA256}"
    )


    print(
        f"\nActual SHA256:\n"
        f"{actual_hash}"
    )


    if (
        actual_hash
        !=
        EXPECTED_BENCHMARK_SHA256
    ):

        raise RuntimeError(
            "FINAL BENCHMARK HASH MISMATCH. "
            "The frozen benchmark has changed. "
            "Evaluation aborted."
        )


    print(
        "\nBenchmark hash verified."
    )


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

        records = data


    elif isinstance(
        data,
        dict,
    ):

        records = None


        for key in [
            "queries",
            "examples",
            "records",
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

                records = value

                break


        if records is None:

            raise ValueError(
                "Could not locate benchmark records."
            )


    else:

        raise ValueError(
            "Unsupported benchmark JSON format."
        )


    if len(
        records
    ) != EXPECTED_NUM_QUERIES:

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_NUM_QUERIES} final-test queries, "
            f"but found {len(records)}."
        )


    return records


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
# Exact-query DeepSearch V2 baseline
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


    with torch.inference_mode():

        query_embedding = (
            retriever.encode_query(
                query
            )
        )


    if query_embedding.ndim == 2:

        if query_embedding.shape[0] != 1:

            raise ValueError(
                "Expected a single query embedding, "
                f"got {tuple(query_embedding.shape)}"
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


    image_embeddings = (
        retriever.image_embeddings
    )


    query_embedding = (
        query_embedding.to(
            image_embeddings.device
        )
    )


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
# Output helpers
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
# Main
# ============================================================


def main():

    # ========================================================
    # Verify frozen benchmark before reading/evaluating it
    # ========================================================

    verify_frozen_benchmark()


    benchmark = (
        load_benchmark()
    )


    print(
        "\n"
        + "#" * 80
    )


    print(
        "AGENTIC MULTIMODAL DEEPSEARCH"
    )


    print(
        "FROZEN 200-QUERY FINAL TEST"
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
        "\nFROZEN architecture:"
    )


    print(
        "  DeepSearch Retriever V2"
    )


    print(
        "  -> query planner"
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


    print(
        "\nNo final-test tuning is permitted."
    )


    # ========================================================
    # Initialize frozen system ONCE
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
                MIN_VERIFICATION_SCORE,

            max_rounds=
                MAX_ROUNDS,

            router_score_threshold=
                ROUTER_SCORE_THRESHOLD,

            router_score_tolerance=
                ROUTER_SCORE_TOLERANCE,

            router_tie_tolerance=
                ROUTER_TIE_TOLERANCE,

            seed=
                SEED,

            fusion_enabled=
                True,

            promotion_enabled=
                True,

            promotion_max_verifier_gap=
                PROMOTION_MAX_VERIFIER_GAP,
        )
    )


    # ========================================================
    # Metric accumulators
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
    # Final test loop
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
            f"FINAL TEST QUERY "
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


        # ====================================================
        # DeepSearch V2 baseline
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
                "Baseline results do not expose image_id. "
                f"Keys: {list(baseline_results[0].keys())}"
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


        # ====================================================
        # Frozen iterative system
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
        # One-shot = Round 1
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
        # Final bounded-promotion ranking
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
        # Promotion
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
        # Save only evaluation records.
        #
        # Do NOT use these records for tuning after the run.
        # ====================================================

        record = {

            "example_id":
                example_id,

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
        }


        records.append(
            record
        )


        print(
            "\n"
            + "-" * 60
        )


        print(
            "FINAL TEST RESULT"
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


    summary = {

        "benchmark":
            "Complex COCO Final Test 200",

        "benchmark_version":
            "1.0",

        "benchmark_sha256":
            EXPECTED_BENCHMARK_SHA256,

        "num_queries":
            len(
                benchmark
            ),

        "architecture_status":
            "FROZEN",

        "seed":
            SEED,

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

        "frozen_configuration":
            {

                "seed":
                    SEED,

                "retrieval_top_k":
                    RETRIEVAL_TOP_K,

                "final_top_k":
                    FINAL_TOP_K,

                "rrf_k":
                    RRF_K,

                "min_verification_score":
                    MIN_VERIFICATION_SCORE,

                "max_rounds":
                    MAX_ROUNDS,

                "router_score_threshold":
                    ROUTER_SCORE_THRESHOLD,

                "router_score_tolerance":
                    ROUTER_SCORE_TOLERANCE,

                "router_tie_tolerance":
                    ROUTER_TIE_TOLERANCE,

                "promotion_max_verifier_gap":
                    PROMOTION_MAX_VERIFIER_GAP,
            },
    }


    output_data = {

        "status":
            "FINAL_EVALUATION_COMPLETE",

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
    # Final report
    # ========================================================

    print(
        "\n\n"
        + "#" * 80
    )


    print(
        "FINAL 200-QUERY TEST RESULTS"
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
        "FINAL ITERATION STATISTICS"
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
        "\nBenchmark SHA256:"
    )


    print(
        EXPECTED_BENCHMARK_SHA256
    )


    print(
        "\nSaved final evaluation to:"
    )


    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()
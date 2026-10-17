import json
import sys
from typing import Any, Dict, List, Optional


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
    "iterative_router_targeted_8.json"
)


# ============================================================
# Targeted cases
# ============================================================

TARGET_EXAMPLE_IDS = [
    "coco_complex_005",
    "coco_complex_011",
    "coco_complex_024",
    "coco_complex_032",
    "coco_complex_038",
    "coco_complex_042",
    "coco_complex_043",
    "coco_complex_045",
]


# ============================================================
# Configuration
# ============================================================

SEED = 42

RETRIEVAL_TOP_K = 10

FINAL_TOP_K = 10

RRF_K = 60


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
) -> Optional[str]:

    for key in [
        "id",
        "query_id",
        "example_id",
        "benchmark_id",
    ]:

        value = example.get(
            key
        )


        if value is not None:

            return str(
                value
            )


    return None


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
):

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
    old_rank,
    new_rank,
) -> str:

    # --------------------------------------------------------
    # Neither retrieved
    # --------------------------------------------------------

    if (
        old_rank is None
        and
        new_rank is None
    ):

        return "unchanged"


    # --------------------------------------------------------
    # Newly retrieved
    # --------------------------------------------------------

    if (
        old_rank is None
        and
        new_rank is not None
    ):

        return "improved"


    # --------------------------------------------------------
    # Lost from candidate set
    # --------------------------------------------------------

    if (
        old_rank is not None
        and
        new_rank is None
    ):

        return "worsened"


    # --------------------------------------------------------
    # Both retrieved
    # --------------------------------------------------------

    if new_rank < old_rank:

        return "improved"


    if new_rank > old_rank:

        return "worsened"


    return "unchanged"


# ============================================================
# Main
# ============================================================


def main():

    benchmark = (
        load_benchmark()
    )


    example_map = {}


    for example in benchmark:

        example_id = (
            get_example_id(
                example
            )
        )


        if example_id is not None:

            example_map[
                example_id
            ] = example


    missing_ids = [

        example_id

        for example_id
        in TARGET_EXAMPLE_IDS

        if example_id
        not in example_map
    ]


    if missing_ids:

        raise RuntimeError(
            "Missing targeted benchmark IDs: "
            f"{missing_ids}"
        )


    selected_examples = [

        example_map[
            example_id
        ]

        for example_id
        in TARGET_EXAMPLE_IDS
    ]


    print(
        "\n"
        + "#" * 70
    )


    print(
        "TARGETED ITERATIVE ROUTER + FUSION EVALUATION"
    )


    print(
        "#" * 70
    )


    print(
        f"\nCases: "
        f"{len(selected_examples)}"
    )


    print(
        f"Seed: "
        f"{SEED}"
    )


    # ========================================================
    # Initialize system once
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

            fusion_verification_weight=
                0.70,

            fusion_retrieval_weight=
                0.30,

            round_1_support_weight=
                1.0,

            round_2_support_weight=
                0.70,

            fusion_rrf_k=
                10,
        )
    )


    records = []


    router_triggered_count = 0

    round_2_executed_count = 0

    fused_count = 0


    raw_round_2_counts = {

        "improved":
            0,

        "unchanged":
            0,

        "worsened":
            0,
    }


    final_counts = {

        "improved":
            0,

        "unchanged":
            0,

        "worsened":
            0,
    }


    # ========================================================
    # Evaluate
    # ========================================================

    for index, example in enumerate(
        selected_examples,
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
            f"TARGETED CASE "
            f"{index}/"
            f"{len(selected_examples)}"
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


        print(
            f"\nQuery:\n"
            f"{query}"
        )


        # ====================================================
        # Search
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
        # Round 1
        # ====================================================

        round_1 = (
            round_history[
                0
            ]
        )


        round_1_results = (
            round_1.get(
                "results",
                [],
            )
        )


        round_1_rank = (
            get_rank(
                round_1_results,
                target_image_id,
            )
        )


        round_1_router = (
            round_1.get(
                "router",
                {},
            )
        )


        round_1_router_triggered = bool(
            round_1_router.get(
                "trigger_round_2",
                False,
            )
        )


        round_1_router_reasons = (
            round_1_router.get(
                "reasons",
                [],
            )
        )


        if round_1_router_triggered:

            router_triggered_count += 1


        # ====================================================
        # Round 2
        # ====================================================

        round_2_rank = None

        round_2_results = []

        round_2_vs_round_1 = (
            "not_run"
        )


        if len(
            round_history
        ) >= 2:

            round_2_executed_count += 1


            round_2 = (
                round_history[
                    1
                ]
            )


            round_2_results = (
                round_2.get(
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


            round_2_vs_round_1 = (
                classify_change(
                    round_1_rank,
                    round_2_rank,
                )
            )


            raw_round_2_counts[
                round_2_vs_round_1
            ] += 1


        # ====================================================
        # Final fused results
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


        final_vs_round_1 = (
            classify_change(
                round_1_rank,
                final_rank,
            )
        )


        final_counts[
            final_vs_round_1
        ] += 1


        selected_mechanism = (
            output.get(
                "selected_round"
            )
        )


        if (
            selected_mechanism
            ==
            "cross_round_fusion"
        ):

            fused_count += 1


        candidate_memory = (
            output.get(
                "cross_round_candidate_memory",
                [],
            )
        )


        # ====================================================
        # Save COMPLETE fusion-aware record
        # ====================================================

        record = {

            "example_id":
                example_id,

            "query":
                query,

            "target_image_id":
                target_image_id,

            "round_1_rank":
                round_1_rank,

            "round_2_rank":
                round_2_rank,

            "final_rank":
                final_rank,

            "rounds_executed":
                output.get(
                    "rounds_executed"
                ),

            "selected_round":
                selected_mechanism,

            "selected_retrieval_query":
                output.get(
                    "selected_retrieval_query"
                ),

            "round_1_router_triggered":
                round_1_router_triggered,

            "round_1_router_reasons":
                round_1_router_reasons,

            "round_2_vs_round_1":
                round_2_vs_round_1,

            "final_vs_round_1":
                final_vs_round_1,

            # -----------------------------------------------
            # Preserve original round-level diagnostics
            # -----------------------------------------------

            "round_history":
                round_history,

            # -----------------------------------------------
            # NEW:
            # preserve fusion configuration
            # -----------------------------------------------

            "fusion_configuration":
                output.get(
                    "fusion_configuration",
                    {},
                ),

            # -----------------------------------------------
            # NEW:
            # preserve complete candidate memory
            # -----------------------------------------------

            "cross_round_candidate_memory":
                candidate_memory,

            # -----------------------------------------------
            # NEW:
            # preserve actual final fused ranking
            # -----------------------------------------------

            "final_results":
                final_results,

            # -----------------------------------------------
            # NEW:
            # preserve final sufficiency
            # -----------------------------------------------

            "final_sufficient":
                output.get(
                    "final_sufficient"
                ),

            "final_sufficiency":
                output.get(
                    "final_sufficiency"
                ),

            # -----------------------------------------------
            # NEW:
            # preserve selected final result
            # -----------------------------------------------

            "selected_top_result":
                output.get(
                    "selected_top_result"
                ),
        }


        records.append(
            record
        )


        # ====================================================
        # Print compact result
        # ====================================================

        print(
            "\n"
            + "-" * 60
        )


        print(
            "TARGETED CASE RESULT"
        )


        print(
            "-" * 60
        )


        print(
            f"Round 1 rank: "
            f"{round_1_rank}"
        )


        print(
            f"Round 2 rank: "
            f"{round_2_rank}"
        )


        print(
            f"Final rank: "
            f"{final_rank}"
        )


        print(
            f"Rounds executed: "
            f"{output.get('rounds_executed')}"
        )


        print(
            f"Selected mechanism: "
            f"{selected_mechanism}"
        )


        print(
            f"Router triggered: "
            f"{round_1_router_triggered}"
        )


        print(
            f"Router reasons: "
            f"{round_1_router_reasons}"
        )


        print(
            f"Round 2 vs Round 1: "
            f"{round_2_vs_round_1}"
        )


        print(
            f"Final vs Round 1: "
            f"{final_vs_round_1}"
        )


        if candidate_memory:

            print(
                f"Fusion candidate count: "
                f"{len(candidate_memory)}"
            )


    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "total_cases":
            len(
                selected_examples
            ),

        "router_triggered":
            router_triggered_count,

        "round_2_executed":
            round_2_executed_count,

        "cross_round_fusion_executed":
            fused_count,

        "raw_round_2_effect":
            raw_round_2_counts,

        "final_effect":
            final_counts,

        "seed":
            SEED,

        "retrieval_top_k":
            RETRIEVAL_TOP_K,

        "final_top_k":
            FINAL_TOP_K,

        "rrf_k":
            RRF_K,
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
    # Console summary
    # ========================================================

    print(
        "\n\n"
        + "#" * 70
    )


    print(
        "TARGETED ROUTER + FUSION SUMMARY"
    )


    print(
        "#" * 70
    )


    print(
        f"\nRouter triggered: "
        f"{router_triggered_count}/"
        f"{len(selected_examples)}"
    )


    print(
        f"Round 2 executed: "
        f"{round_2_executed_count}/"
        f"{len(selected_examples)}"
    )


    print(
        f"Cross-round fusion executed: "
        f"{fused_count}/"
        f"{len(selected_examples)}"
    )


    print(
        "\nRaw Round 2 effect:"
    )


    print(
        f"  Improved:  "
        f"{raw_round_2_counts['improved']}"
    )


    print(
        f"  Unchanged: "
        f"{raw_round_2_counts['unchanged']}"
    )


    print(
        f"  Worsened:  "
        f"{raw_round_2_counts['worsened']}"
    )


    print(
        "\nFinal fused effect:"
    )


    print(
        f"  Improved:  "
        f"{final_counts['improved']}"
    )


    print(
        f"  Unchanged: "
        f"{final_counts['unchanged']}"
    )


    print(
        f"  Worsened:  "
        f"{final_counts['worsened']}"
    )


    print(
        "\nSaved to:"
    )


    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()
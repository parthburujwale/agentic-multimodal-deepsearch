import json
import sys
from copy import deepcopy


sys.path.append("..")


from agents.iterative_agentic_search import (
    IterativeAgenticSearch,
)


BENCHMARK_FILE = (
    "../../data/benchmarks/"
    "complex_coco_queries_50.json"
)


OUTPUT_FILE = (
    "../../experiments/"
    "iterative_reproducibility_4_cases.json"
)


TARGET_EXAMPLES = {
    "coco_complex_011",
    "coco_complex_032",
    "coco_complex_043",
    "coco_complex_045",
}


SEED = 42


# ============================================================
# Helpers
# ============================================================


def normalize_image_id(value):

    if value is None:
        return None

    try:
        return int(value)

    except (TypeError, ValueError):
        return str(value).strip()


def load_benchmark():

    with open(
        BENCHMARK_FILE,
        "r",
    ) as file:

        data = json.load(file)

    if isinstance(data, list):
        return data

    for key in [
        "queries",
        "examples",
        "data",
        "benchmark",
        "items",
    ]:

        if (
            isinstance(data, dict)
            and isinstance(
                data.get(key),
                list,
            )
        ):

            return data[key]

    raise ValueError(
        "Unsupported benchmark JSON format."
    )


def get_example_id(example):

    for key in [
        "id",
        "query_id",
        "example_id",
        "benchmark_id",
    ]:

        if key in example:
            return example[key]

    return None


def get_query(example):

    for key in [
        "query",
        "complex_query",
        "text",
        "prompt",
        "search_query",
    ]:

        value = example.get(key)

        if value:
            return str(value).strip()

    raise KeyError(
        f"No query field found: "
        f"{list(example.keys())}"
    )


def get_target_image_id(example):

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
            and example[key] is not None
        ):

            return normalize_image_id(
                example[key]
            )

    for nested_key in [
        "target",
        "ground_truth",
        "positive",
    ]:

        value = example.get(
            nested_key
        )

        if not isinstance(
            value,
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
                key in value
                and value[key] is not None
            ):

                return normalize_image_id(
                    value[key]
                )

    raise KeyError(
        f"No target image ID found: "
        f"{list(example.keys())}"
    )


def get_rank(
    results,
    target_image_id,
):

    target = normalize_image_id(
        target_image_id
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        candidate = normalize_image_id(
            result.get(
                "image_id"
            )
        )

        if candidate == target:
            return rank

    return None


def result_ids(results):

    return [
        normalize_image_id(
            result.get(
                "image_id"
            )
        )
        for result in results
    ]


# ============================================================
# Build compact comparison signature
# ============================================================


def build_signature(
    output,
    target_image_id,
):

    signature = {

        "fixed_original_plan":
            deepcopy(
                output[
                    "fixed_original_plan"
                ]
            ),

        "fixed_positive_constraints":
            deepcopy(
                output[
                    "fixed_positive_constraints"
                ]
            ),

        "fixed_negative_constraints":
            deepcopy(
                output[
                    "fixed_negative_constraints"
                ]
            ),

        "rounds_executed":
            output[
                "rounds_executed"
            ],

        "stop_reason":
            output[
                "stop_reason"
            ],

        "selected_round":
            output[
                "selected_round"
            ],

        "final_sufficient":
            output[
                "final_sufficient"
            ],

        "rounds":
            [],
    }


    for round_record in output[
        "round_history"
    ]:

        results = (
            round_record.get(
                "results",
                []
            )
        )

        round_signature = {

            "round":
                round_record[
                    "round"
                ],

            "retrieval_query":
                round_record[
                    "retrieval_query"
                ],

            "verification_query":
                round_record[
                    "verification_query"
                ],

            "target_rank":
                get_rank(
                    results,
                    target_image_id,
                ),

            "image_ids":
                result_ids(
                    results
                ),

            "router":
                deepcopy(
                    round_record[
                        "router"
                    ]
                ),

            "sufficiency":
                deepcopy(
                    round_record[
                        "sufficiency"
                    ]
                ),

            "top_result":
                deepcopy(
                    round_record[
                        "top_result"
                    ]
                ),

            "verification_scores":
                [
                    float(
                        result
                        .get(
                            "verification",
                            {}
                        )
                        .get(
                            "overall_score",
                            0.0
                        )
                    )
                    for result in results
                ],

            "overall_matches":
                [
                    bool(
                        result
                        .get(
                            "verification",
                            {}
                        )
                        .get(
                            "overall_match",
                            False
                        )
                    )
                    for result in results
                ],

            "final_scores":
                [
                    float(
                        result.get(
                            "final_score",
                            0.0
                        )
                    )
                    for result in results
                ],
        }

        signature[
            "rounds"
        ].append(
            round_signature
        )

    return signature


# ============================================================
# Difference reporting
# ============================================================


def compare_signatures(
    sig_a,
    sig_b,
):

    differences = []

    if (
        sig_a[
            "fixed_original_plan"
        ]
        !=
        sig_b[
            "fixed_original_plan"
        ]
    ):

        differences.append(
            "fixed_original_plan"
        )

    if (
        sig_a[
            "fixed_positive_constraints"
        ]
        !=
        sig_b[
            "fixed_positive_constraints"
        ]
    ):

        differences.append(
            "fixed_positive_constraints"
        )

    if (
        sig_a[
            "fixed_negative_constraints"
        ]
        !=
        sig_b[
            "fixed_negative_constraints"
        ]
    ):

        differences.append(
            "fixed_negative_constraints"
        )

    for key in [
        "rounds_executed",
        "stop_reason",
        "selected_round",
        "final_sufficient",
    ]:

        if sig_a[key] != sig_b[key]:

            differences.append(
                key
            )

    if (
        len(
            sig_a[
                "rounds"
            ]
        )
        !=
        len(
            sig_b[
                "rounds"
            ]
        )
    ):

        differences.append(
            "number_of_rounds"
        )

        return differences

    for index, (
        round_a,
        round_b,
    ) in enumerate(
        zip(
            sig_a["rounds"],
            sig_b["rounds"],
        ),
        start=1,
    ):

        for key in [
            "retrieval_query",
            "verification_query",
            "target_rank",
            "image_ids",
            "router",
            "sufficiency",
            "top_result",
            "verification_scores",
            "overall_matches",
            "final_scores",
        ]:

            if (
                round_a[key]
                !=
                round_b[key]
            ):

                differences.append(
                    f"round_{index}:{key}"
                )

    return differences


# ============================================================
# Main
# ============================================================


def main():

    benchmark = (
        load_benchmark()
    )

    selected = [

        example

        for example in benchmark

        if get_example_id(
            example
        ) in TARGET_EXAMPLES
    ]

    if len(selected) != 4:

        raise RuntimeError(
            "Expected 4 target examples, "
            f"found {len(selected)}."
        )

    print(
        "\n"
        + "#" * 70
    )

    print(
        "ITERATIVE PIPELINE REPRODUCIBILITY TEST"
    )

    print(
        "#" * 70
    )

    print(
        f"\nSeed: {SEED}"
    )

    print(
        "\nCases:"
    )

    for example in selected:

        print(
            f"  {get_example_id(example)}"
        )

    # ========================================================
    # Initialize once.
    #
    # We deliberately use the same loaded model instance for
    # Run A and Run B. The iterative system resets RNG state
    # before its generative calls.
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
        )
    )

    all_results = {

        "seed":
            SEED,

        "run_a":
            {},

        "run_b":
            {},

        "comparison":
            {},
    }

    # ========================================================
    # Two repeated runs
    # ========================================================

    for run_name in [
        "run_a",
        "run_b",
    ]:

        print(
            "\n\n"
            + "#" * 70
        )

        print(
            run_name.upper()
        )

        print(
            "#" * 70
        )

        for index, example in enumerate(
            selected,
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
                "\n"
                + "=" * 80
            )

            print(
                f"{run_name.upper()} "
                f"CASE {index}/4: "
                f"{example_id}"
            )

            print(
                "=" * 80
            )

            output = (
                agent.search(

                    query=
                        query,

                    retrieval_top_k=
                        10,

                    final_top_k=
                        10,

                    rrf_k=
                        60,
                )
            )

            signature = (
                build_signature(
                    output,
                    target_image_id,
                )
            )

            all_results[
                run_name
            ][
                example_id
            ] = (
                signature
            )

            print(
                f"\n{example_id}"
            )

            print(
                f"Rounds: "
                f"{signature['rounds_executed']}"
            )

            print(
                f"Selected round: "
                f"{signature['selected_round']}"
            )

            for round_signature in (
                signature[
                    "rounds"
                ]
            ):

                print(
                    f"  Round "
                    f"{round_signature['round']}: "
                    f"target rank = "
                    f"{round_signature['target_rank']}"
                )

    # ========================================================
    # Compare A and B
    # ========================================================

    print(
        "\n\n"
        + "#" * 70
    )

    print(
        "REPRODUCIBILITY COMPARISON"
    )

    print(
        "#" * 70
    )

    exact_matches = 0

    for example in selected:

        example_id = (
            get_example_id(
                example
            )
        )

        sig_a = (
            all_results[
                "run_a"
            ][
                example_id
            ]
        )

        sig_b = (
            all_results[
                "run_b"
            ][
                example_id
            ]
        )

        differences = (
            compare_signatures(
                sig_a,
                sig_b,
            )
        )

        exact_match = (
            len(differences)
            == 0
        )

        if exact_match:
            exact_matches += 1

        all_results[
            "comparison"
        ][
            example_id
        ] = {

            "exact_match":
                exact_match,

            "differences":
                differences,
        }

        print(
            f"\n{example_id}"
        )

        print(
            f"Exact match: "
            f"{exact_match}"
        )

        if differences:

            print(
                "Differences:"
            )

            for difference in differences:

                print(
                    f"  - {difference}"
                )

    all_results[
        "summary"
    ] = {

        "cases":
            4,

        "exact_matches":
            exact_matches,

        "all_cases_reproducible":
            exact_matches == 4,
    }

    with open(
        OUTPUT_FILE,
        "w",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL REPRODUCIBILITY RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nExact matches: "
        f"{exact_matches}/4"
    )

    print(
        f"All cases reproducible: "
        f"{exact_matches == 4}"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()
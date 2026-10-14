import json
import os


INPUT_FILE = (
    "../../data/benchmarks/"
    "complex_coco_holdout_candidates_150.json"
)

OUTPUT_FILE = (
    "../../data/benchmarks/"
    "complex_coco_holdout_100.json"
)


REJECTED_IDS = {
    6, 11, 13, 14, 15, 24, 25,
    27, 30, 32, 36, 37, 39, 41, 48,
    51, 53, 54, 61, 63, 69, 72, 74, 75,
    76, 78, 79, 81, 82, 83, 87, 89, 92, 96, 98, 100,
    102, 114, 115, 118, 121,
    126, 131, 134, 135, 145, 147, 148, 149, 150
}


def main():

    with open(
        INPUT_FILE,
        "r"
    ) as file:

        data = json.load(
            file
        )


    kept = []


    for example in data[
        "examples"
    ]:

        candidate_number = int(

            example[
                "benchmark_id"
            ].split(
                "_"
            )[
                -1
            ]
        )


        if candidate_number in REJECTED_IDS:

            continue


        kept.append(
            example
        )


    if len(kept) != 100:

        raise ValueError(
            f"Expected exactly 100 examples, "
            f"but found {len(kept)}."
        )


    # --------------------------------------------------
    # Rename IDs so final benchmark has clean
    # consecutive numbering 001-100.
    # --------------------------------------------------

    frozen_examples = []


    for index, example in enumerate(
        kept,
        start=1
    ):

        frozen_example = dict(
            example
        )


        frozen_example[
            "source_candidate_id"
        ] = example[
            "benchmark_id"
        ]


        frozen_example[
            "benchmark_id"
        ] = (
            f"coco_holdout_{index:03d}"
        )


        frozen_examples.append(
            frozen_example
        )


    image_ids = [

        example[
            "target_image_id"
        ]

        for example in frozen_examples
    ]


    if len(
        image_ids
    ) != len(
        set(
            image_ids
        )
    ):

        raise ValueError(
            "Duplicate image IDs found "
            "in frozen benchmark."
        )


    output = {
        "benchmark_name":
            "Complex COCO Holdout Benchmark",

        "benchmark_version":
            "1.0",

        "status":
            "FROZEN",

        "num_examples":
            100,

        "source_dataset":
            "MS COCO val2017 captions",

        "development_benchmark":
            "complex_coco_queries_50.json",

        "image_disjoint_from_development":
            True,

        "selection_policy": (
            "Generated without model outputs. "
            "Manually curated only for caption consistency, "
            "visual verifiability, ambiguity, contradiction, "
            "and subjective or speculative language."
        ),

        "model_outputs_used_for_selection":
            False,

        "examples":
            frozen_examples
    }


    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )


    print(
        "=" * 80
    )

    print(
        "FINAL HOLDOUT BENCHMARK FROZEN"
    )

    print(
        "=" * 80
    )

    print(
        f"\nExamples: "
        f"{len(frozen_examples)}"
    )

    print(
        f"Unique image IDs: "
        f"{len(set(image_ids))}"
    )

    print(
        f"Rejected candidates: "
        f"{len(REJECTED_IDS)}"
    )

    print(
        f"Version: "
        f"{output['benchmark_version']}"
    )

    print(
        f"Status: "
        f"{output['status']}"
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This benchmark is now frozen."
    )

    print(
        "Do not remove examples, change queries, "
        "or tune architecture based on its results."
    )


if __name__ == "__main__":

    main()
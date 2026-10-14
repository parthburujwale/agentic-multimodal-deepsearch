import argparse
import json


DEFAULT_FILE = (
    "../../data/benchmarks/"
    "complex_coco_holdout_candidates_150.json"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_FILE
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1
    )

    parser.add_argument(
        "--count",
        type=int,
        default=25
    )

    args = parser.parse_args()


    with open(
        args.file,
        "r"
    ) as file:

        data = json.load(
            file
        )


    examples = data[
        "examples"
    ]


    start_index = max(
        0,
        args.start - 1
    )

    end_index = min(
        len(examples),
        start_index + args.count
    )


    print(
        "=" * 100
    )

    print(
        f"HOLDOUT CANDIDATES "
        f"{start_index + 1}-"
        f"{end_index}"
    )

    print(
        "=" * 100
    )


    for index in range(
        start_index,
        end_index
    ):

        example = examples[
            index
        ]


        print(
            "\n"
            + "-" * 100
        )

        print(
            f"CANDIDATE {index + 1}"
        )

        print(
            "-" * 100
        )


        print(
            f"ID: "
            f"{example['benchmark_id']}"
        )

        print(
            f"Image ID: "
            f"{example['target_image_id']}"
        )

        print(
            f"File: "
            f"{example['file_name']}"
        )

        print(
            f"Quality: "
            f"{example['quality_score']:.3f}"
        )

        print(
            f"Overlap: "
            f"{example['content_overlap']:.3f}"
        )


        print(
            "\nCaption A:"
        )

        print(
            example[
                "caption_a"
            ]
        )


        print(
            "\nCaption B:"
        )

        print(
            example[
                "caption_b"
            ]
        )


        print(
            "\nQuery:"
        )

        print(
            example[
                "query"
            ]
        )


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Displayed "
        f"{end_index - start_index} "
        f"candidates."
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":

    main()
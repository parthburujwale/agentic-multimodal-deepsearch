import json


EVALUATION_FILE = (
    "../../experiments/"
    "frozen_50_query_ablation.json"
)

CONFIDENCE_FILE = (
    "../../experiments/"
    "retrieval_confidence_analysis.json"
)

BENCHMARK_ID = "coco_complex_024"


def load_json(path):

    with open(path, "r") as file:
        return json.load(file)


def main():

    evaluation = load_json(
        EVALUATION_FILE
    )

    confidence_data = load_json(
        CONFIDENCE_FILE
    )


    result = next(

        item

        for item in evaluation["results"]

        if (
            item["benchmark_id"]
            == BENCHMARK_ID
        )
    )


    confidence = next(

        item

        for item in confidence_data

        if (
            item["benchmark_id"]
            == BENCHMARK_ID
        )
    )


    baseline_top1_image_id = (
        confidence[
            "top1_image_id"
        ]
    )


    target_image_id = (
        result[
            "target_image_id"
        ]
    )


    candidates = (
        result[
            "verified_results"
        ]
    )


    baseline_candidate = None
    target_candidate = None


    for candidate in candidates:

        if (
            candidate["image_id"]
            == baseline_top1_image_id
        ):

            baseline_candidate = candidate


        if (
            candidate["image_id"]
            == target_image_id
        ):

            target_candidate = candidate


    winner = (
        candidates[
            0
        ]
    )


    print(
        "=" * 90
    )

    print(
        "ANCHOR FAILURE ANALYSIS: "
        f"{BENCHMARK_ID}"
    )

    print(
        "=" * 90
    )


    print(
        "\nQuery:"
    )

    print(
        result[
            "query"
        ]
    )


    print(
        "\nRanks:"
    )

    print(
        f"DeepSearch V2 target rank: "
        f"{result['baseline_rank']}"
    )

    print(
        f"Weighted RRF target rank: "
        f"{result['weighted_rrf_rank']}"
    )

    print(
        f"Current VLM target rank: "
        f"{result['weighted_verified_rank']}"
    )


    print(
        "\nDeepSearch confidence:"
    )

    print(
        f"Top1 similarity: "
        f"{confidence['top1_score']:.6f}"
    )

    print(
        f"Top1-top2 margin: "
        f"{confidence['margin_1_2']:.6f}"
    )

    print(
        f"Top1-top5mean margin: "
        f"{confidence['margin_1_5mean']:.6f}"
    )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "ORIGINAL DEEPSEARCH TOP-1"
    )

    print(
        "-" * 90
    )


    print(
        f"Image ID: "
        f"{baseline_top1_image_id}"
    )

    print(
        f"Is benchmark target: "
        f"{baseline_top1_image_id == target_image_id}"
    )


    if baseline_candidate is None:

        print(
            "Not present in verifier candidates."
        )

    else:

        verification = (
            baseline_candidate[
                "verification"
            ]
        )


        print(
            f"RRF rank: "
            f"{baseline_candidate['rrf_rank']}"
        )

        print(
            f"RRF score: "
            f"{baseline_candidate['rrf_score']:.6f}"
        )

        print(
            f"Normalized RRF score: "
            f"{baseline_candidate['normalized_rrf_score']:.6f}"
        )

        print(
            f"Verification score: "
            f"{verification['overall_score']:.6f}"
        )

        print(
            f"Overall match: "
            f"{verification['overall_match']}"
        )

        print(
            f"Combined score: "
            f"{baseline_candidate['combined_score']:.6f}"
        )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "CURRENT VLM WINNER"
    )

    print(
        "-" * 90
    )


    winner_verification = (
        winner[
            "verification"
        ]
    )


    print(
        f"Image ID: "
        f"{winner['image_id']}"
    )

    print(
        f"Is benchmark target: "
        f"{winner['image_id'] == target_image_id}"
    )

    print(
        f"RRF rank: "
        f"{winner['rrf_rank']}"
    )

    print(
        f"RRF score: "
        f"{winner['rrf_score']:.6f}"
    )

    print(
        f"Normalized RRF score: "
        f"{winner['normalized_rrf_score']:.6f}"
    )

    print(
        f"Verification score: "
        f"{winner_verification['overall_score']:.6f}"
    )

    print(
        f"Overall match: "
        f"{winner_verification['overall_match']}"
    )

    print(
        f"Combined score: "
        f"{winner['combined_score']:.6f}"
    )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "TARGET IMAGE"
    )

    print(
        "-" * 90
    )


    if target_candidate is None:

        print(
            "Target not present in verifier top-10."
        )

    else:

        verification = (
            target_candidate[
                "verification"
            ]
        )


        print(
            f"Image ID: "
            f"{target_candidate['image_id']}"
        )

        print(
            f"RRF rank: "
            f"{target_candidate['rrf_rank']}"
        )

        print(
            f"RRF score: "
            f"{target_candidate['rrf_score']:.6f}"
        )

        print(
            f"Normalized RRF score: "
            f"{target_candidate['normalized_rrf_score']:.6f}"
        )

        print(
            f"Verification score: "
            f"{verification['overall_score']:.6f}"
        )

        print(
            f"Overall match: "
            f"{verification['overall_match']}"
        )

        print(
            f"Combined score: "
            f"{target_candidate['combined_score']:.6f}"
        )


    if (
        baseline_candidate is not None
        and target_candidate is not None
    ):

        baseline_verification = (
            baseline_candidate[
                "verification"
            ]
        )

        target_verification = (
            target_candidate[
                "verification"
            ]
        )


        print(
            "\n"
            + "-" * 90
        )

        print(
            "TARGET - BASELINE DIFFERENCES"
        )

        print(
            "-" * 90
        )


        print(
            f"Verification difference: "
            f"{target_verification['overall_score'] - baseline_verification['overall_score']:.6f}"
        )

        print(
            f"Combined difference: "
            f"{target_candidate['combined_score'] - baseline_candidate['combined_score']:.6f}"
        )

        print(
            f"RRF-rank difference: "
            f"{target_candidate['rrf_rank'] - baseline_candidate['rrf_rank']}"
        )


if __name__ == "__main__":

    main()
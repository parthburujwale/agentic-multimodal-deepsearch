import json


EVALUATION_FILE = (
    "../../experiments/"
    "frozen_50_query_ablation.json"
)

CONFIDENCE_FILE = (
    "../../experiments/"
    "retrieval_confidence_analysis.json"
)


REGRESSION_IDS = [
    "coco_complex_001",
    "coco_complex_011",
    "coco_complex_013",
    "coco_complex_032",
    "coco_complex_044",
]


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


    result_lookup = {

        item["benchmark_id"]:
            item

        for item in evaluation["results"]
    }


    confidence_lookup = {

        item["benchmark_id"]:
            item

        for item in confidence_data
    }


    print(
        "=" * 100
    )

    print(
        "REGRESSION VERIFIER SUMMARY"
    )

    print(
        "=" * 100
    )


    for benchmark_id in REGRESSION_IDS:

        result = (
            result_lookup[
                benchmark_id
            ]
        )


        confidence = (
            confidence_lookup[
                benchmark_id
            ]
        )


        baseline_top1_image_id = (
            confidence[
                "top1_image_id"
            ]
        )


        verified_results = (
            result[
                "verified_results"
            ]
        )


        winner = (
            verified_results[
                0
            ]
        )


        baseline_candidate = None


        for candidate in verified_results:

            if (
                candidate[
                    "image_id"
                ]
                == baseline_top1_image_id
            ):

                baseline_candidate = candidate

                break


        print(
            "\n"
            + "=" * 100
        )

        print(
            benchmark_id
        )

        print(
            "=" * 100
        )


        print(
            f"Query:"
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
            f"Baseline rank: "
            f"{result['baseline_rank']}"
        )

        print(
            f"Weighted RRF rank: "
            f"{result['weighted_rrf_rank']}"
        )

        print(
            f"Verified rank: "
            f"{result['weighted_verified_rank']}"
        )


        print(
            "\nRetrieval confidence:"
        )

        print(
            f"Top-1 similarity: "
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
            "\nBASELINE TOP-1 ANALYSIS"
        )

        print(
            "-" * 50
        )


        if baseline_candidate is None:

            print(
                "Baseline top-1 was not present "
                "inside the weighted RRF top-10."
            )

        else:

            baseline_verification = (
                baseline_candidate[
                    "verification"
                ]
            )


            print(
                f"Image ID: "
                f"{baseline_candidate['image_id']}"
            )

            print(
                f"Verification score: "
                f"{baseline_verification['overall_score']:.6f}"
            )

            print(
                f"Overall match: "
                f"{baseline_verification['overall_match']}"
            )

            print(
                f"Combined score: "
                f"{baseline_candidate['combined_score']:.6f}"
            )

            print(
                f"RRF rank: "
                f"{baseline_candidate['rrf_rank']}"
            )


        print(
            "\nWINNER ANALYSIS"
        )

        print(
            "-" * 50
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
            f"RRF rank: "
            f"{winner['rrf_rank']}"
        )


        if baseline_candidate is not None:

            baseline_verification = (
                baseline_candidate[
                    "verification"
                ]
            )


            verification_gap = (

                winner_verification[
                    "overall_score"
                ]

                -

                baseline_verification[
                    "overall_score"
                ]
            )


            combined_gap = (

                winner[
                    "combined_score"
                ]

                -

                baseline_candidate[
                    "combined_score"
                ]
            )


            print(
                f"\nWinner-baseline "
                f"verification gap: "
                f"{verification_gap:.6f}"
            )

            print(
                f"Winner-baseline "
                f"combined gap: "
                f"{combined_gap:.6f}"
            )


if __name__ == "__main__":

    main()
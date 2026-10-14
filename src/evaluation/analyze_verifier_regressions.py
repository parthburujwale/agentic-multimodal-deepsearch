import json


EVALUATION_FILE = (
    "../../experiments/"
    "frozen_50_query_ablation.json"
)

CONFIDENCE_FILE = (
    "../../experiments/"
    "retrieval_confidence_analysis.json"
)


def load_json(path):

    with open(path, "r") as file:
        return json.load(file)


def capped_rank(
    rank,
    k=10
):

    if (
        rank is None
        or rank > k
    ):

        return k + 1

    return rank


def main():

    evaluation = load_json(
        EVALUATION_FILE
    )

    confidence_data = load_json(
        CONFIDENCE_FILE
    )


    confidence_lookup = {

        item["benchmark_id"]:
            item

        for item in confidence_data
    }


    print(
        "=" * 100
    )

    print(
        "VERIFIER REGRESSION ANALYSIS"
    )

    print(
        "=" * 100
    )


    regression_count = 0


    for result in evaluation["results"]:

        benchmark_id = (
            result[
                "benchmark_id"
            ]
        )


        baseline_rank = (
            result[
                "baseline_rank"
            ]
        )


        verified_rank = (
            result[
                "weighted_verified_rank"
            ]
        )


        if (
            capped_rank(
                verified_rank
            )
            >=
            capped_rank(
                baseline_rank
            )
        ):

            continue


        # --------------------------------------------------
        # This block above selects improvements.
        # We want regressions instead, so skip them.
        # --------------------------------------------------


    for result in evaluation["results"]:

        benchmark_id = (
            result[
                "benchmark_id"
            ]
        )


        baseline_rank = (
            result[
                "baseline_rank"
            ]
        )


        verified_rank = (
            result[
                "weighted_verified_rank"
            ]
        )


        if (
            capped_rank(
                verified_rank
            )
            <=
            capped_rank(
                baseline_rank
            )
        ):

            continue


        regression_count += 1


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
            f"Baseline rank: "
            f"{baseline_rank}"
        )

        print(
            f"Weighted RRF rank: "
            f"{result['weighted_rrf_rank']}"
        )

        print(
            f"Verified target rank: "
            f"{verified_rank}"
        )

        print(
            f"Baseline top1 image ID: "
            f"{baseline_top1_image_id}"
        )

        print(
            f"Target image ID: "
            f"{result['target_image_id']}"
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
            "\nQuery:"
        )

        print(
            result[
                "query"
            ]
        )


        print(
            "\nFinal verified candidate order:"
        )


        baseline_candidate = None
        winner_candidate = None


        for rank, candidate in enumerate(
            result[
                "verified_results"
            ],
            start=1
        ):

            verification = (
                candidate[
                    "verification"
                ]
            )


            if rank == 1:

                winner_candidate = (
                    candidate
                )


            if (
                candidate[
                    "image_id"
                ]
                == baseline_top1_image_id
            ):

                baseline_candidate = (
                    candidate
                )


            print(
                f"\nRank {rank}"
            )

            print(
                f"Image ID: "
                f"{candidate['image_id']}"
            )

            print(
                f"File: "
                f"{candidate['file_name']}"
            )

            print(
                f"Original RRF rank: "
                f"{candidate['rrf_rank']}"
            )

            print(
                f"RRF score: "
                f"{candidate['rrf_score']:.6f}"
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
                f"{candidate['combined_score']:.6f}"
            )


        print(
            "\n"
            + "-" * 100
        )

        print(
            "BASELINE TOP-1 ANALYSIS"
        )

        print(
            "-" * 100
        )


        if baseline_candidate is None:

            print(
                "Original DeepSearch top-1 image "
                "was NOT present in the weighted "
                "RRF verification top-10."
            )

        else:

            verification = (
                baseline_candidate[
                    "verification"
                ]
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
            + "-" * 100
        )

        print(
            "WINNER ANALYSIS"
        )

        print(
            "-" * 100
        )


        if winner_candidate is not None:

            verification = (
                winner_candidate[
                    "verification"
                ]
            )


            print(
                f"Winner image ID: "
                f"{winner_candidate['image_id']}"
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
                f"{winner_candidate['combined_score']:.6f}"
            )


            if baseline_candidate is not None:

                baseline_verification = (
                    baseline_candidate[
                        "verification"
                    ]
                )


                verification_gap = (

                    verification[
                        "overall_score"
                    ]

                    -

                    baseline_verification[
                        "overall_score"
                    ]
                )


                combined_gap = (

                    winner_candidate[
                        "combined_score"
                    ]

                    -

                    baseline_candidate[
                        "combined_score"
                    ]
                )


                print(
                    f"Winner-baseline "
                    f"verification gap: "
                    f"{verification_gap:.6f}"
                )

                print(
                    f"Winner-baseline "
                    f"combined gap: "
                    f"{combined_gap:.6f}"
                )


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Regression cases analyzed: "
        f"{regression_count}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":

    main()
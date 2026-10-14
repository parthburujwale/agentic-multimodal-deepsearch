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


def target_rank(
    candidates,
    target_image_id
):

    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        if (
            candidate["image_id"]
            == target_image_id
        ):

            return rank

    return None


def recall_at_k(
    ranks,
    k
):

    hits = sum(
        1
        for rank in ranks
        if (
            rank is not None
            and rank <= k
        )
    )

    return (
        hits
        / len(ranks)
    )


def mrr_at_k(
    ranks,
    k
):

    total = 0.0

    for rank in ranks:

        if (
            rank is not None
            and rank <= k
        ):

            total += (
                1.0 / rank
            )

    return (
        total
        / len(ranks)
    )


def capped(
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


    current_ranks = []
    strict_anchor_ranks = []

    changed_cases = []


    for result in evaluation["results"]:

        benchmark_id = (
            result["benchmark_id"]
        )

        target_image_id = (
            result["target_image_id"]
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


        # --------------------------------------------------
        # Copy current verified ranking
        # --------------------------------------------------

        candidates = [

            dict(candidate)

            for candidate
            in result["verified_results"]
        ]


        current_rank = (
            target_rank(
                candidates,
                target_image_id
            )
        )


        current_ranks.append(
            current_rank
        )


        # --------------------------------------------------
        # Current winner
        # --------------------------------------------------

        winner = (
            candidates[
                0
            ]
        )

        winner_verification = (
            winner[
                "verification"
            ]
        )


        # --------------------------------------------------
        # Find original DeepSearch V2 top-1
        # --------------------------------------------------

        baseline_candidate = None


        for candidate in candidates:

            if (
                candidate["image_id"]
                == baseline_top1_image_id
            ):

                baseline_candidate = candidate
                break


        apply_anchor = False


        if baseline_candidate is not None:

            baseline_verification = (
                baseline_candidate[
                    "verification"
                ]
            )


            # --------------------------------------------------
            # STRICT ANCHOR RULE
            #
            # Preserve original DeepSearch top-1 only if:
            #
            # 1. VLM says it is a complete semantic match
            #
            # 2. Its verification score is STRICTLY greater
            #    than the current winner's verification score.
            #
            # Equal semantic scores are NOT enough to override
            # the fused retrieval ranking.
            # --------------------------------------------------

            if (
                baseline_verification[
                    "overall_match"
                ]

                and

                baseline_verification[
                    "overall_score"
                ]
                >
                winner_verification[
                    "overall_score"
                ]
                + 1e-8
            ):

                apply_anchor = True


        # --------------------------------------------------
        # Move protected anchor to Rank 1
        # --------------------------------------------------

        if (
            apply_anchor

            and baseline_candidate is not None

            and (
                candidates[
                    0
                ][
                    "image_id"
                ]
                !=
                baseline_candidate[
                    "image_id"
                ]
            )
        ):

            reordered = [
                baseline_candidate
            ]


            reordered.extend(

                candidate

                for candidate in candidates

                if (
                    candidate[
                        "image_id"
                    ]
                    !=
                    baseline_candidate[
                        "image_id"
                    ]
                )
            )


            candidates = reordered


        strict_anchor_rank = (
            target_rank(
                candidates,
                target_image_id
            )
        )


        strict_anchor_ranks.append(
            strict_anchor_rank
        )


        if (
            capped(
                strict_anchor_rank
            )
            !=
            capped(
                current_rank
            )
        ):

            changed_cases.append(
                {
                    "benchmark_id":
                        benchmark_id,

                    "baseline_rank":
                        result[
                            "baseline_rank"
                        ],

                    "weighted_rank":
                        result[
                            "weighted_rrf_rank"
                        ],

                    "current_vlm_rank":
                        current_rank,

                    "strict_anchor_rank":
                        strict_anchor_rank,

                    "baseline_top1_image_id":
                        baseline_top1_image_id,

                    "target_image_id":
                        target_image_id,

                    "anchor_applied":
                        apply_anchor
                }
            )


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    current_metrics = {

        "recall_at_1":
            recall_at_k(
                current_ranks,
                1
            ),

        "recall_at_5":
            recall_at_k(
                current_ranks,
                5
            ),

        "recall_at_10":
            recall_at_k(
                current_ranks,
                10
            ),

        "mrr_at_10":
            mrr_at_k(
                current_ranks,
                10
            )
    }


    strict_anchor_metrics = {

        "recall_at_1":
            recall_at_k(
                strict_anchor_ranks,
                1
            ),

        "recall_at_5":
            recall_at_k(
                strict_anchor_ranks,
                5
            ),

        "recall_at_10":
            recall_at_k(
                strict_anchor_ranks,
                10
            ),

        "mrr_at_10":
            mrr_at_k(
                strict_anchor_ranks,
                10
            )
    }


    # --------------------------------------------------
    # Compare against current VLM
    # --------------------------------------------------

    improved = 0
    unchanged = 0
    worsened = 0


    for (
        old_rank,
        new_rank
    ) in zip(
        current_ranks,
        strict_anchor_ranks
    ):

        old_value = capped(
            old_rank
        )

        new_value = capped(
            new_rank
        )


        if new_value < old_value:

            improved += 1

        elif new_value > old_value:

            worsened += 1

        else:

            unchanged += 1


    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print(
        "=" * 90
    )

    print(
        "STRICT ANCHOR RULE OFFLINE EVALUATION"
    )

    print(
        "=" * 90
    )


    print(
        "\nCURRENT WEIGHTED RRF + VLM"
    )

    print(
        "-" * 45
    )


    for key, value in (
        current_metrics.items()
    ):

        print(
            f"{key}: "
            f"{value:.4f}"
        )


    print(
        "\nSTRICT ANCHOR-AWARE VLM"
    )

    print(
        "-" * 45
    )


    for key, value in (
        strict_anchor_metrics.items()
    ):

        print(
            f"{key}: "
            f"{value:.4f}"
        )


    print(
        "\nCHANGE RELATIVE TO CURRENT VLM"
    )

    print(
        "-" * 45
    )


    print(
        f"Improved: "
        f"{improved}"
    )

    print(
        f"Unchanged: "
        f"{unchanged}"
    )

    print(
        f"Worsened: "
        f"{worsened}"
    )


    print(
        "\nCASES WHOSE TARGET RANK CHANGED"
    )

    print(
        "-" * 90
    )


    for case in changed_cases:

        print()

        print(
            case[
                "benchmark_id"
            ]
        )

        print(
            f"DeepSearch V2 rank: "
            f"{case['baseline_rank']}"
        )

        print(
            f"Weighted RRF rank: "
            f"{case['weighted_rank']}"
        )

        print(
            f"Current VLM rank: "
            f"{case['current_vlm_rank']}"
        )

        print(
            f"Strict anchor rank: "
            f"{case['strict_anchor_rank']}"
        )

        print(
            f"Original top1 image: "
            f"{case['baseline_top1_image_id']}"
        )

        print(
            f"Target image: "
            f"{case['target_image_id']}"
        )


if __name__ == "__main__":

    main()
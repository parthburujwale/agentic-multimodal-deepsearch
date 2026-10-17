import json


INPUT_FILE = (
    "../../experiments/"
    "iterative_agentic_dev50_evaluation.json"
)


def rank_value(rank):

    if rank is None or rank > 10:

        return 11

    return rank


def find_baseline_candidate(
    results,
    original_query,
):

    """
    Find the candidate that was rank 1 for the original
    DeepSearch V2 query.

    Each candidate stores query_matches, including its rank
    for every retrieval query.
    """

    for result in results:

        for match in result.get(
            "query_matches",
            [],
        ):

            match_query = str(
                match.get(
                    "query",
                    "",
                )
            ).strip()


            match_rank = (
                match.get(
                    "rank"
                )
            )


            if (
                match_query == original_query
                and match_rank == 1
            ):

                return result


    return None


def evaluate_policy(
    records,
    score_threshold,
):

    triggered = []

    regressions = []

    captured_regressions = []


    reason_counts = {

        "incomplete_match":
            0,

        "low_score":
            0,

        "baseline_missing":
            0,

        "baseline_tied_disagreement":
            0,
    }


    for record in records:

        baseline_rank = (
            record[
                "baseline_rank"
            ]
        )


        one_shot_rank = (
            record[
                "one_shot_rank"
            ]
        )


        is_regression = (

            rank_value(
                one_shot_rank
            )

            >

            rank_value(
                baseline_rank
            )
        )


        if is_regression:

            regressions.append(
                record[
                    "example_id"
                ]
            )


        round_1 = (
            record[
                "round_history"
            ][0]
        )


        results = (
            round_1.get(
                "results",
                [],
            )
        )


        if not results:

            continue


        winner = (
            results[0]
        )


        winner_verification = (
            winner.get(
                "verification",
                {},
            )
        )


        winner_score = float(
            winner_verification.get(
                "overall_score",
                0.0,
            )
        )


        winner_match = bool(
            winner_verification.get(
                "overall_match",
                False,
            )
        )


        original_query = (
            round_1[
                "retrieval_query"
            ]
        )


        baseline_candidate = (
            find_baseline_candidate(
                results,
                original_query,
            )
        )


        # ====================================================
        # Trigger reasons
        # ====================================================

        incomplete_match = (
            not winner_match
        )


        low_score = (
            winner_score
            <= score_threshold
        )


        baseline_missing = (
            baseline_candidate is None
        )


        baseline_tied_disagreement = False


        if baseline_candidate is not None:

            baseline_image_id = (
                baseline_candidate.get(
                    "image_id"
                )
            )


            winner_image_id = (
                winner.get(
                    "image_id"
                )
            )


            baseline_verification = (
                baseline_candidate.get(
                    "verification",
                    {},
                )
            )


            baseline_score = float(
                baseline_verification.get(
                    "overall_score",
                    0.0,
                )
            )


            baseline_match = bool(
                baseline_verification.get(
                    "overall_match",
                    False,
                )
            )


            different_winner = (
                baseline_image_id
                != winner_image_id
            )


            verifier_tie = (

                abs(
                    baseline_score
                    - winner_score
                )

                <= 0.01
            )


            baseline_tied_disagreement = (

                different_winner

                and

                baseline_match

                and

                winner_match

                and

                verifier_tie
            )


        trigger = (

            incomplete_match

            or

            low_score

            or

            baseline_missing

            or

            baseline_tied_disagreement
        )


        if trigger:

            example_id = (
                record[
                    "example_id"
                ]
            )


            triggered.append(
                example_id
            )


            if is_regression:

                captured_regressions.append(
                    example_id
                )


        if incomplete_match:

            reason_counts[
                "incomplete_match"
            ] += 1


        if low_score:

            reason_counts[
                "low_score"
            ] += 1


        if baseline_missing:

            reason_counts[
                "baseline_missing"
            ] += 1


        if baseline_tied_disagreement:

            reason_counts[
                "baseline_tied_disagreement"
            ] += 1


    missed = [

        example

        for example in regressions

        if example
        not in captured_regressions
    ]


    return {

        "threshold":
            score_threshold,

        "triggered":
            triggered,

        "trigger_count":
            len(
                triggered
            ),

        "regressions":
            regressions,

        "captured":
            captured_regressions,

        "missed":
            missed,

        "reason_counts":
            reason_counts,
    }


def main():

    with open(
        INPUT_FILE,
        "r",
    ) as file:

        data = json.load(
            file
        )


    records = (
        data[
            "records"
        ]
    )


    thresholds = [
        0.825,
        0.85,
        0.875,
        0.90,
    ]


    print(
        "\n"
        + "=" * 70
    )


    print(
        "BASELINE-AWARE ITERATIVE ROUTER ANALYSIS"
    )


    print(
        "=" * 70
    )


    for threshold in thresholds:

        result = (
            evaluate_policy(
                records,
                score_threshold=
                    threshold,
            )
        )


        print(
            "\n"
            + "-" * 70
        )


        print(
            f"SCORE THRESHOLD <= "
            f"{threshold:.3f}"
        )


        print(
            "-" * 70
        )


        print(
            f"Would trigger: "
            f"{result['trigger_count']}/"
            f"{len(records)}"
        )


        print(
            f"Captured regressions: "
            f"{len(result['captured'])}/"
            f"{len(result['regressions'])}"
        )


        print(
            f"Captured:"
        )


        print(
            result[
                "captured"
            ]
        )


        print(
            "Missed:"
        )


        print(
            result[
                "missed"
            ]
        )


        print(
            "\nTrigger reason counts:"
        )


        for (
            reason,
            count,
        ) in result[
            "reason_counts"
        ].items():

            print(
                f"  {reason}: "
                f"{count}"
            )


        print(
            "\nTriggered examples:"
        )


        print(
            result[
                "triggered"
            ]
        )


if __name__ == "__main__":

    main()
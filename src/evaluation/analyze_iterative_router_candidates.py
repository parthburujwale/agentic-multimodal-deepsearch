import json


INPUT_FILE = (
    "../../experiments/"
    "iterative_agentic_dev50_evaluation.json"
)


def rank_value(rank):

    if rank is None or rank > 10:
        return 11

    return rank


def main():

    with open(
        INPUT_FILE,
        "r",
    ) as file:

        data = json.load(file)


    records = data["records"]


    policies = {
        "A_score_lt_090": [],
        "B_score_or_verifier_tie": [],
        "C_score_or_small_final_margin": [],
        "D_score_or_multiple_matches": [],
    }


    regressions = []


    for record in records:

        round_1 = (
            record["round_history"][0]
        )

        results = (
            round_1.get(
                "results",
                []
            )
        )


        if not results:
            continue


        top = results[0]

        top_verification = (
            top.get(
                "verification",
                {}
            )
        )


        top_score = float(
            top_verification.get(
                "overall_score",
                0.0
            )
        )


        top_match = bool(
            top_verification.get(
                "overall_match",
                False
            )
        )


        second_verifier_score = 0.0
        second_final_score = 0.0


        if len(results) >= 2:

            second_verifier_score = float(
                results[1]
                .get(
                    "verification",
                    {}
                )
                .get(
                    "overall_score",
                    0.0
                )
            )

            second_final_score = float(
                results[1].get(
                    "final_score",
                    0.0
                )
            )


        top_final_score = float(
            top.get(
                "final_score",
                0.0
            )
        )


        verifier_gap = abs(
            top_score
            - second_verifier_score
        )


        final_gap = abs(
            top_final_score
            - second_final_score
        )


        num_overall_matches = sum(

            1

            for result in results

            if result
            .get(
                "verification",
                {}
            )
            .get(
                "overall_match",
                False
            )
        )


        example_id = (
            record["example_id"]
        )


        baseline_rank = (
            record["baseline_rank"]
        )

        one_shot_rank = (
            record["one_shot_rank"]
        )


        is_regression = (
            rank_value(one_shot_rank)
            >
            rank_value(baseline_rank)
        )


        if is_regression:

            regressions.append(
                example_id
            )


        # ----------------------------------------------
        # Policy A
        # ----------------------------------------------

        trigger_a = (
            (not top_match)
            or
            top_score < 0.90
        )


        # ----------------------------------------------
        # Policy B
        # ----------------------------------------------

        trigger_b = (
            trigger_a

            or

            verifier_gap <= 0.01
        )


        # ----------------------------------------------
        # Policy C
        # ----------------------------------------------

        trigger_c = (
            trigger_a

            or

            final_gap <= 0.03
        )


        # ----------------------------------------------
        # Policy D
        # ----------------------------------------------

        trigger_d = (
            trigger_a

            or

            num_overall_matches >= 2
        )


        if trigger_a:
            policies[
                "A_score_lt_090"
            ].append(example_id)


        if trigger_b:
            policies[
                "B_score_or_verifier_tie"
            ].append(example_id)


        if trigger_c:
            policies[
                "C_score_or_small_final_margin"
            ].append(example_id)


        if trigger_d:
            policies[
                "D_score_or_multiple_matches"
            ].append(example_id)


    print(
        "\nKnown one-shot regressions:"
    )

    print(
        regressions
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "CANDIDATE ITERATIVE ROUTER POLICIES"
    )

    print(
        "=" * 70
    )


    for name, triggered in policies.items():

        captured = [

            example

            for example in regressions

            if example in triggered
        ]


        missed = [

            example

            for example in regressions

            if example not in triggered
        ]


        print(
            f"\n{name}"
        )

        print(
            "-" * 60
        )

        print(
            f"Would trigger: "
            f"{len(triggered)}/"
            f"{len(records)}"
        )

        print(
            f"Captured regressions: "
            f"{len(captured)}/"
            f"{len(regressions)}"
        )

        print(
            f"Captured: {captured}"
        )

        print(
            f"Missed:   {missed}"
        )


if __name__ == "__main__":

    main()
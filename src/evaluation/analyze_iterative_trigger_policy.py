import json
import statistics


INPUT_FILE = (
    "../../experiments/"
    "iterative_agentic_dev50_evaluation.json"
)


def main():

    with open(
        INPUT_FILE,
        "r",
    ) as file:

        data = json.load(
            file
        )


    records = data[
        "records"
    ]


    rows = []


    for record in records:

        round_1 = (
            record[
                "round_history"
            ][0]
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

        verification = (
            top.get(
                "verification",
                {}
            )
        )


        score = float(
            verification.get(
                "overall_score",
                0.0
            )
        )


        overall_match = bool(
            verification.get(
                "overall_match",
                False
            )
        )


        second_score = 0.0


        if len(results) >= 2:

            second_score = float(
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


        margin = (
            score
            - second_score
        )


        rows.append(
            {
                "example_id":
                    record[
                        "example_id"
                    ],

                "baseline_rank":
                    record[
                        "baseline_rank"
                    ],

                "one_shot_rank":
                    record[
                        "one_shot_rank"
                    ],

                "top_score":
                    score,

                "second_score":
                    second_score,

                "margin":
                    margin,

                "overall_match":
                    overall_match,
            }
        )


    scores = [
        row[
            "top_score"
        ]
        for row in rows
    ]


    margins = [
        row[
            "margin"
        ]
        for row in rows
    ]


    print(
        "\n"
        + "=" * 70
    )

    print(
        "ROUND 1 VERIFICATION CONFIDENCE ANALYSIS"
    )

    print(
        "=" * 70
    )


    print(
        f"\nQueries: "
        f"{len(rows)}"
    )


    print(
        f"Mean top score: "
        f"{statistics.mean(scores):.4f}"
    )


    print(
        f"Median top score: "
        f"{statistics.median(scores):.4f}"
    )


    print(
        f"Min top score: "
        f"{min(scores):.4f}"
    )


    print(
        f"Max top score: "
        f"{max(scores):.4f}"
    )


    print(
        f"\nMean top-vs-second margin: "
        f"{statistics.mean(margins):.4f}"
    )


    print(
        f"Median margin: "
        f"{statistics.median(margins):.4f}"
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "POTENTIAL SCORE THRESHOLDS"
    )

    print(
        "=" * 70
    )


    thresholds = [
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]


    for threshold in thresholds:

        would_trigger = sum(

            1

            for row in rows

            if (
                not row[
                    "overall_match"
                ]

                or

                row[
                    "top_score"
                ]
                < threshold
            )
        )


        print(
            f"\nThreshold {threshold:.2f}: "
            f"{would_trigger}/"
            f"{len(rows)} "
            "would trigger Round 2"
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOW-MARGIN CASES"
    )

    print(
        "=" * 70
    )


    sorted_rows = sorted(
        rows,
        key=lambda row:
            row[
                "margin"
            ]
    )


    for row in sorted_rows[:15]:

        print(
            "\n"
            f"Example: "
            f"{row['example_id']}"
        )

        print(
            f"One-shot target rank: "
            f"{row['one_shot_rank']}"
        )

        print(
            f"Top verifier score: "
            f"{row['top_score']:.4f}"
        )

        print(
            f"Second verifier score: "
            f"{row['second_score']:.4f}"
        )

        print(
            f"Margin: "
            f"{row['margin']:.4f}"
        )

        print(
            f"Overall match: "
            f"{row['overall_match']}"
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "ONE-SHOT REGRESSIONS VS DEEPSEARCH V2"
    )

    print(
        "=" * 70
    )


    regression_count = 0


    for row in rows:

        baseline_rank = (
            row[
                "baseline_rank"
            ]
        )


        one_shot_rank = (
            row[
                "one_shot_rank"
            ]
        )


        baseline_value = (
            baseline_rank
            if baseline_rank is not None
            and baseline_rank <= 10
            else 11
        )


        agentic_value = (
            one_shot_rank
            if one_shot_rank is not None
            and one_shot_rank <= 10
            else 11
        )


        if agentic_value > baseline_value:

            regression_count += 1


            print(
                "\n"
                f"Example: "
                f"{row['example_id']}"
            )

            print(
                f"DeepSearch V2 rank: "
                f"{baseline_rank}"
            )

            print(
                f"One-shot rank: "
                f"{one_shot_rank}"
            )

            print(
                f"Top verifier score: "
                f"{row['top_score']:.4f}"
            )

            print(
                f"Second verifier score: "
                f"{row['second_score']:.4f}"
            )

            print(
                f"Verifier margin: "
                f"{row['margin']:.4f}"
            )

            print(
                f"Overall match: "
                f"{row['overall_match']}"
            )


    print(
        f"\nTotal regressions: "
        f"{regression_count}"
    )


if __name__ == "__main__":

    main()
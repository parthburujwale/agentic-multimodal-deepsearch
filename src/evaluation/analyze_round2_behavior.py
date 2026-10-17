import json


INPUT_FILE = (
    "../../experiments/"
    "iterative_router_targeted_8.json"
)


def normalize_image_id(value):

    if value is None:
        return None

    try:
        return int(value)

    except (TypeError, ValueError):
        return str(value).strip()


def get_ids(results):

    return [
        normalize_image_id(
            result.get("image_id")
        )
        for result in results
    ]


def overlap_at_k(
    results_a,
    results_b,
    k=10,
):

    ids_a = set(
        get_ids(results_a[:k])
    )

    ids_b = set(
        get_ids(results_b[:k])
    )

    if not ids_a and not ids_b:
        return 1.0

    union = (
        ids_a | ids_b
    )

    intersection = (
        ids_a & ids_b
    )

    if not union:
        return 0.0

    return (
        len(intersection)
        / len(union)
    )


def top_k_common_count(
    results_a,
    results_b,
    k=10,
):

    ids_a = set(
        get_ids(results_a[:k])
    )

    ids_b = set(
        get_ids(results_b[:k])
    )

    return len(
        ids_a & ids_b
    )


def get_verification_score(result):

    if result is None:
        return 0.0

    verification = (
        result.get(
            "verification",
            {}
        )
    )

    return float(
        verification.get(
            "overall_score",
            0.0
        )
    )


def get_final_score(result):

    if result is None:
        return 0.0

    return float(
        result.get(
            "final_score",
            0.0
        )
    )


def main():

    with open(
        INPUT_FILE,
        "r",
    ) as file:

        data = json.load(file)


    records = data[
        "records"
    ]


    triggered_records = [

        record

        for record in records

        if record[
            "round_1_router_triggered"
        ]
    ]


    overlaps = []

    common_counts = []


    print(
        "\n"
        + "=" * 80
    )

    print(
        "ROUND 2 BEHAVIOR ANALYSIS"
    )

    print(
        "=" * 80
    )


    for record in records:

        print(
            "\n\n"
            + "#" * 80
        )

        print(
            record[
                "example_id"
            ]
        )

        print(
            "#" * 80
        )


        print(
            f"\nTarget image ID: "
            f"{record['target_image_id']}"
        )


        print(
            f"Router triggered: "
            f"{record['round_1_router_triggered']}"
        )


        print(
            f"Router reasons: "
            f"{record['round_1_router_reasons']}"
        )


        print(
            f"\nRound 1 target rank: "
            f"{record['round_1_rank']}"
        )

        print(
            f"Round 2 target rank: "
            f"{record['round_2_rank']}"
        )

        print(
            f"Final target rank: "
            f"{record['final_rank']}"
        )

        print(
            f"Selected round: "
            f"{record['selected_round']}"
        )


        history = (
            record[
                "round_history"
            ]
        )


        round_1 = (
            history[0]
        )


        print(
            "\nROUND 1 RETRIEVAL QUERY:"
        )

        print(
            round_1[
                "retrieval_query"
            ]
        )


        if len(history) < 2:

            print(
                "\nRound 2 was not executed."
            )

            continue


        round_2 = (
            history[1]
        )


        print(
            "\nROUND 2 RETRIEVAL QUERY:"
        )

        print(
            round_2[
                "retrieval_query"
            ]
        )


        results_1 = (
            round_1.get(
                "results",
                []
            )
        )

        results_2 = (
            round_2.get(
                "results",
                []
            )
        )


        overlap = (
            overlap_at_k(
                results_1,
                results_2,
                k=10,
            )
        )


        common = (
            top_k_common_count(
                results_1,
                results_2,
                k=10,
            )
        )


        overlaps.append(
            overlap
        )

        common_counts.append(
            common
        )


        print(
            f"\nTop-10 common images: "
            f"{common}/10"
        )


        print(
            f"Top-10 Jaccard overlap: "
            f"{overlap:.4f}"
        )


        top_1 = (
            results_1[0]
            if results_1
            else None
        )

        top_2 = (
            results_2[0]
            if results_2
            else None
        )


        if top_1 is not None:

            print(
                "\nRound 1 winner:"
            )

            print(
                f"  Image ID: "
                f"{top_1.get('image_id')}"
            )

            print(
                f"  Verification: "
                f"{get_verification_score(top_1):.4f}"
            )

            print(
                f"  Final score: "
                f"{get_final_score(top_1):.4f}"
            )


        if top_2 is not None:

            print(
                "\nRound 2 winner:"
            )

            print(
                f"  Image ID: "
                f"{top_2.get('image_id')}"
            )

            print(
                f"  Verification: "
                f"{get_verification_score(top_2):.4f}"
            )

            print(
                f"  Final score: "
                f"{get_final_score(top_2):.4f}"
            )


        print(
            "\nRound 1 top-10 image IDs:"
        )

        print(
            get_ids(
                results_1[:10]
            )
        )


        print(
            "\nRound 2 top-10 image IDs:"
        )

        print(
            get_ids(
                results_2[:10]
            )
        )


    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "AGGREGATE ROUND 2 ANALYSIS"
    )

    print(
        "=" * 80
    )


    print(
        f"\nTriggered cases: "
        f"{len(triggered_records)}"
    )


    if overlaps:

        print(
            f"Average Top-10 Jaccard overlap: "
            f"{sum(overlaps) / len(overlaps):.4f}"
        )


    if common_counts:

        print(
            f"Average common Top-10 images: "
            f"{sum(common_counts) / len(common_counts):.2f}/10"
        )


    same_query_count = 0


    for record in triggered_records:

        history = (
            record[
                "round_history"
            ]
        )


        if len(history) < 2:
            continue


        query_1 = (
            history[0][
                "retrieval_query"
            ]
            .strip()
            .lower()
        )


        query_2 = (
            history[1][
                "retrieval_query"
            ]
            .strip()
            .lower()
        )


        if query_1 == query_2:

            same_query_count += 1


    print(
        f"Exact duplicate Round-2 queries: "
        f"{same_query_count}"
    )


if __name__ == "__main__":

    main()
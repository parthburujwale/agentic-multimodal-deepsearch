import json


INPUT_FILE = (
    "../../experiments/"
    "iterative_router_targeted_8.json"
)


def normalize_image_id(
    value,
):

    if value is None:
        return None

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):

        return str(
            value
        ).strip()


def find_target_image_id(
    record,
):

    for key in [
        "target_image_id",
        "target_id",
        "image_id",
        "ground_truth_image_id",
    ]:

        if (
            key in record
            and
            record[key] is not None
        ):

            return normalize_image_id(
                record[key]
            )


    example = record.get(
        "example",
        {}
    )


    if isinstance(
        example,
        dict,
    ):

        for key in [
            "target_image_id",
            "target_id",
            "image_id",
            "ground_truth_image_id",
        ]:

            if (
                key in example
                and
                example[key] is not None
            ):

                return normalize_image_id(
                    example[key]
                )


    return None


def get_case_id(
    record,
):

    for key in [
        "id",
        "example_id",
        "query_id",
        "benchmark_id",
    ]:

        if key in record:
            return str(
                record[key]
            )


    example = record.get(
        "example",
        {}
    )


    if isinstance(
        example,
        dict,
    ):

        for key in [
            "id",
            "example_id",
            "query_id",
        ]:

            if key in example:
                return str(
                    example[key]
                )


    return "unknown"


def get_output(
    record,
):

    for key in [
        "output",
        "search_output",
        "iterative_output",
        "result",
    ]:

        value = record.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):

            return value


    return record


def target_rank(
    results,
    target_image_id,
):

    target_image_id = (
        normalize_image_id(
            target_image_id
        )
    )


    for rank, result in enumerate(
        results,
        start=1,
    ):

        image_id = (
            normalize_image_id(
                result.get(
                    "image_id"
                )
            )
        )


        if image_id == target_image_id:
            return rank


    return None


def load_records():

    with open(
        INPUT_FILE,
        "r",
    ) as file:

        data = json.load(
            file
        )


    if isinstance(
        data,
        list,
    ):

        return data


    for key in [
        "results",
        "cases",
        "examples",
        "records",
    ]:

        if (
            isinstance(
                data,
                dict,
            )
            and isinstance(
                data.get(
                    key
                ),
                list,
            )
        ):

            return data[
                key
            ]


    raise ValueError(
        "Could not locate targeted "
        "evaluation records."
    )


def main():

    records = (
        load_records()
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "CROSS-ROUND FUSION ANALYSIS"
    )

    print(
        "=" * 80
    )


    for record in records:

        case_id = (
            get_case_id(
                record
            )
        )


        output = (
            get_output(
                record
            )
        )


        target_image_id = (
            find_target_image_id(
                record
            )
        )


        round_history = (
            output.get(
                "round_history",
                []
            )
        )


        final_results = (
            output.get(
                "final_results",
                output.get(
                    "results",
                    []
                )
            )
        )


        candidate_memory = (
            output.get(
                "cross_round_candidate_memory",
                []
            )
        )


        print(
            "\n\n"
            + "#" * 80
        )

        print(
            case_id
        )

        print(
            "#" * 80
        )


        print(
            f"\nTarget image ID: "
            f"{target_image_id}"
        )


        for round_record in (
            round_history
        ):

            round_number = (
                round_record.get(
                    "round"
                )
            )


            results = (
                round_record.get(
                    "results",
                    []
                )
            )


            rank = (
                target_rank(
                    results,
                    target_image_id,
                )
            )


            print(
                f"Round {round_number} "
                f"target rank: "
                f"{rank}"
            )


        final_rank = (
            target_rank(
                final_results,
                target_image_id,
            )
        )


        print(
            f"Final fused target rank: "
            f"{final_rank}"
        )


        print(
            f"Selected mechanism: "
            f"{output.get('selected_round')}"
        )


        if not candidate_memory:

            print(
                "\nNo cross-round fusion "
                "was executed."
            )

            continue


        print(
            "\nTop fused candidates:"
        )


        for rank, item in enumerate(
            candidate_memory[
                :10
            ],
            start=1,
        ):

            image_id = (
                normalize_image_id(
                    item.get(
                        "image_id"
                    )
                )
            )


            marker = (
                "  <-- TARGET"
                if (
                    image_id
                    ==
                    target_image_id
                )
                else ""
            )


            print(
                "\n"
                f"Fusion Rank {rank}"
                f"{marker}"
            )


            print(
                f"  Image ID: "
                f"{image_id}"
            )


            print(
                f"  Round ranks: "
                f"{item.get('round_ranks')}"
            )


            print(
                f"  Round count: "
                f"{item.get('round_count')}"
            )


            print(
                f"  Retrieval support raw: "
                f"{float(item.get('retrieval_support_raw', 0.0)):.6f}"
            )


            print(
                f"  Retrieval support normalized: "
                f"{float(item.get('retrieval_support', 0.0)):.4f}"
            )


            print(
                f"  Best verifier: "
                f"{float(item.get('best_verification_score', 0.0)):.4f}"
            )


            print(
                f"  Overall match: "
                f"{item.get('best_overall_match')}"
            )


            print(
                f"  Fusion score: "
                f"{float(item.get('fusion_score', 0.0)):.4f}"
            )


            print(
                f"  Per-round verifier scores: "
                f"{item.get('round_verification_scores')}"
            )


            print(
                f"  Per-round final scores: "
                f"{item.get('round_final_scores')}"
            )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "END OF FUSION ANALYSIS"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
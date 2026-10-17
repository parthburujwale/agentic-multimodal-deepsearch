from query_refiner import QueryRefiner


def main():

    refiner = QueryRefiner(
        model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST 1: MISSING POSITIVE CONSTRAINT"
    )

    print(
        "=" * 70
    )


    original_query = (
        "Find an image of a dog near water "
        "with people in the background, "
        "but the dog should not be swimming."
    )


    refined = refiner.refine(

        original_query=
            original_query,

        missing_positive_constraints=[
            "people visible in the background"
        ],

        violated_negative_constraints=[],
    )


    print(
        "\nOriginal:"
    )

    print(
        original_query
    )


    print(
        "\nRefined:"
    )

    print(
        refined
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST 2: NEGATIVE CONSTRAINT VIOLATION"
    )

    print(
        "=" * 70
    )


    refined = refiner.refine(

        original_query=
            original_query,

        missing_positive_constraints=[],

        violated_negative_constraints=[
            "dog swimming"
        ],
    )


    print(
        "\nOriginal:"
    )

    print(
        original_query
    )


    print(
        "\nRefined:"
    )

    print(
        refined
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST 3: BOTH TYPES OF FAILURE"
    )

    print(
        "=" * 70
    )


    refined = refiner.refine(

        original_query=
            original_query,

        missing_positive_constraints=[
            "people visible in the background"
        ],

        violated_negative_constraints=[
            "dog swimming"
        ],
    )


    print(
        "\nOriginal:"
    )

    print(
        original_query
    )


    print(
        "\nRefined:"
    )

    print(
        refined
    )


if __name__ == "__main__":

    main()
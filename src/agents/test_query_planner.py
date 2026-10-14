import json
import sys


sys.path.append("..")


from agents.query_planner import (
    QueryPlanner
)


# --------------------------------------------------
# Load planner
# --------------------------------------------------

planner = QueryPlanner()


# --------------------------------------------------
# Test queries
# --------------------------------------------------

test_queries = [

    (
        "Find an image of a dog near water "
        "with people in the background, "
        "but the dog should not be swimming."
    ),

    (
        "Find a red car parked beside a building."
    ),

    (
        "Find an airplane on the ground with "
        "airport vehicles nearby but no passengers."
    ),

    (
        "Find a person riding a bicycle near "
        "a city street with cars in the background."
    )
]


# --------------------------------------------------
# Generate plans
# --------------------------------------------------

for index, query in enumerate(
    test_queries,
    start=1
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"TEST QUERY {index}"
    )

    print(
        "=" * 70
    )

    print(
        query
    )


    plan = planner.plan(
        query
    )


    print(
        "\nPLAN:"
    )


    print(
        json.dumps(
            plan,
            indent=2
        )
    )
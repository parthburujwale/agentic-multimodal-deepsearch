import json
import sys


sys.path.append("..")


from agents.agentic_retriever import (
    AgenticRetriever
)


# --------------------------------------------------
# Build agent
# --------------------------------------------------

agent = AgenticRetriever(

    image_folder=(
        "../../data/coco/"
        "val2017"
    ),

    captions_file=(
        "../../data/coco/"
        "annotations/"
        "captions_val2017.json"
    ),

    retriever_checkpoint_path=(
        "../../experiments/"
        "deepsearch_retriever_v2/"
        "deepsearch_retriever_v2.pt"
    ),

    retriever_processor_path=(
        "../../experiments/"
        "deepsearch_retriever_v2"
    )
)


# --------------------------------------------------
# Complex query
# --------------------------------------------------

query = (
    "Find an image of a dog near water "
    "with people in the background, "
    "but the dog should not be swimming."
)


# --------------------------------------------------
# Agentic retrieval
# --------------------------------------------------

output = agent.search(

    user_query=query,

    per_query_top_k=20,

    final_top_k=10,

    rrf_k=60
)


# --------------------------------------------------
# Print compact result
# --------------------------------------------------

print(
    "\n"
    + "=" * 70
)

print(
    "AGENTIC RETRIEVAL OUTPUT"
)

print(
    "=" * 70
)


print(
    "\nPlan:"
)

print(
    json.dumps(
        output[
            "plan"
        ],
        indent=2
    )
)


print(
    "\nFinal Results:"
)


for rank, result in enumerate(
    output[
        "results"
    ],
    start=1
):

    print(
        f"\nRank {rank}"
    )

    print(
        f"Image ID: "
        f"{result['image_id']}"
    )

    print(
        f"File: "
        f"{result['file_name']}"
    )

    print(
        f"RRF score: "
        f"{result['rrf_score']:.6f}"
    )

    print(
        "Matched queries:"
    )


    for match in result[
        "query_matches"
    ]:

        print(
            f"  {match['query']} "
            f"-> rank "
            f"{match['rank']}"
        )
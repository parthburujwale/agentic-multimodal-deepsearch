import json
import os
import sys


sys.path.append("..")


from agents.evidence_verifier import (
    EvidenceVerifier
)


# --------------------------------------------------
# Image from Agentic Retriever rank 1
# --------------------------------------------------

image_path = (
    "../../data/coco/"
    "val2017/"
    "000000331075.jpg"
)


# --------------------------------------------------
# Original user request
# --------------------------------------------------

original_query = (
    "Find an image of a dog near water "
    "with people in the background, "
    "but the dog should not be swimming."
)


# --------------------------------------------------
# Planner constraints
# --------------------------------------------------

positive_constraints = [
    "dog",
    "near water",
    "people",
    "background"
]


negative_constraints = [
    "dog swimming",
    "swimming"
]


# --------------------------------------------------
# Load verifier
# --------------------------------------------------

verifier = EvidenceVerifier()


# --------------------------------------------------
# Verify image
# --------------------------------------------------

result = verifier.verify(

    image_path=
        image_path,

    original_query=
        original_query,

    positive_constraints=
        positive_constraints,

    negative_constraints=
        negative_constraints
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print(
    "\n"
    + "=" * 70
)

print(
    "MULTIMODAL EVIDENCE VERIFICATION"
)

print(
    "=" * 70
)


print(
    f"\nImage:"
)

print(
    os.path.basename(
        image_path
    )
)


print(
    "\nVerification:"
)


print(
    json.dumps(
        result,
        indent=2
    )
)
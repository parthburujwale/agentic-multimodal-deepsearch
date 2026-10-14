import argparse
import json
import math
import os
import random
import re
from collections import defaultdict


# ============================================================
# Stopwords
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "two",
    "up",
    "was",
    "were",
    "with",
    "while",
    "who",
    "one",
    "some",
}


# ============================================================
# Phrases that often indicate:
#
# - speculation
# - subjective interpretation
# - weak visual grounding
#
# We want the benchmark to focus on information that should
# actually be visible in the image.
# ============================================================

VAGUE_OR_SPECULATIVE_PHRASES = {
    "probably",
    "possibly",
    "perhaps",
    "maybe",
    "might be",
    "may be",
    "could be",
    "could have",
    "appears to be",
    "seems to be",
    "looks like it might",
    "looks like they might",
    "beautiful",
    "nice",
    "nicely",
    "serenity",
    "stately environment",
    "for dear life",
    "healthy dinner",
    "watchful eye",
    "protect her baby",
    "curmudgeonly",
}


# ============================================================
# IDs previously rejected during development benchmark
# construction because they were ambiguous, speculative,
# contradictory, or otherwise undesirable.
#
# These are excluded from the holdout candidate pool too.
# ============================================================

KNOWN_REJECTED_IMAGE_IDS = {
    442456,
    479099,
    545129,
    388927,
    16010,
    68286,
    344268,
    126216,
    377814,
    538067,
    181303,
    219271,
    393569,
    341094,
    445365,
    241602,
    262895,
}


# ============================================================
# Text utilities
# ============================================================


def normalize_text(text):

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def tokenize_content_words(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )


    return {
        word

        for word in words

        if (
            word not in STOPWORDS
            and len(word) >= 2
        )
    }


def contains_vague_phrase(text):

    lowered = text.lower()


    return any(
        phrase in lowered
        for phrase in VAGUE_OR_SPECULATIVE_PHRASES
    )


# ============================================================
# Basic caption quality filters
# ============================================================


def caption_is_usable(text):

    text = normalize_text(
        text
    )


    if len(text) < 25:

        return False


    if len(text) > 220:

        return False


    content_words = tokenize_content_words(
        text
    )


    if len(content_words) < 5:

        return False


    if contains_vague_phrase(
        text
    ):

        return False


    # --------------------------------------------------------
    # Require some alphabetic content.
    # --------------------------------------------------------

    if not re.search(
        r"[A-Za-z]",
        text
    ):

        return False


    # --------------------------------------------------------
    # Reject extreme punctuation/noise.
    # --------------------------------------------------------

    alpha_count = len(
        re.findall(
            r"[A-Za-z]",
            text
        )
    )


    if alpha_count < 15:

        return False


    return True


# ============================================================
# Content overlap
#
# Jaccard overlap between caption content-word sets.
#
# Very high overlap means the two captions are nearly
# redundant and do not create an interesting compositional
# query.
# ============================================================


def content_overlap(
    words_a,
    words_b
):

    union = (
        words_a
        | words_b
    )


    if len(union) == 0:

        return 1.0


    intersection = (
        words_a
        & words_b
    )


    return (
        len(intersection)
        / len(union)
    )


# ============================================================
# Pair quality
#
# We want:
#
# 1. Both captions to contain useful visual information.
# 2. Some shared semantic grounding.
# 3. Each caption to contribute unique information.
# 4. Neither caption to simply duplicate the other.
#
# This is only used for benchmark construction.
# It does NOT use retrieval/model performance.
# ============================================================


def score_caption_pair(
    caption_a,
    caption_b
):

    words_a = tokenize_content_words(
        caption_a
    )

    words_b = tokenize_content_words(
        caption_b
    )


    overlap = content_overlap(
        words_a,
        words_b
    )


    unique_a = (
        words_a
        - words_b
    )

    unique_b = (
        words_b
        - words_a
    )

    shared = (
        words_a
        & words_b
    )


    # --------------------------------------------------------
    # Require complementary information on BOTH sides.
    # --------------------------------------------------------

    if len(unique_a) < 2:

        return None


    if len(unique_b) < 2:

        return None


    # --------------------------------------------------------
    # Very redundant captions are not useful for our complex
    # compositional retrieval benchmark.
    # --------------------------------------------------------

    if overlap > 0.65:

        return None


    # --------------------------------------------------------
    # Information score
    # --------------------------------------------------------

    information = (
        len(words_a)
        + len(words_b)
    )


    complementarity = (
        len(unique_a)
        + len(unique_b)
    )


    balance = min(
        len(unique_a),
        len(unique_b)
    )


    # --------------------------------------------------------
    # A small shared-content bonus helps prevent choosing two
    # captions that describe completely unrelated interpretations
    # of the same image.
    # --------------------------------------------------------

    shared_bonus = min(
        len(shared),
        4
    )


    # --------------------------------------------------------
    # Penalize excessive overlap.
    # --------------------------------------------------------

    redundancy_penalty = (
        4.0
        * overlap
    )


    quality_score = (

        0.45
        * information

        +

        0.45
        * complementarity

        +

        0.35
        * balance

        +

        0.30
        * shared_bonus

        -

        redundancy_penalty
    )


    return {
        "quality_score":
            quality_score,

        "content_overlap":
            overlap,

        "unique_a":
            len(unique_a),

        "unique_b":
            len(unique_b),

        "shared_words":
            len(shared),
    }


# ============================================================
# Load COCO captions
# ============================================================


def load_coco_captions(
    captions_file
):

    with open(
        captions_file,
        "r"
    ) as file:

        coco = json.load(
            file
        )


    captions_by_image = defaultdict(
        list
    )


    for annotation in coco[
        "annotations"
    ]:

        image_id = int(
            annotation[
                "image_id"
            ]
        )


        caption = normalize_text(
            annotation[
                "caption"
            ]
        )


        captions_by_image[
            image_id
        ].append(
            caption
        )


    return captions_by_image


# ============================================================
# Load development benchmark IDs
# ============================================================


def load_excluded_development_ids(
    development_benchmark
):

    with open(
        development_benchmark,
        "r"
    ) as file:

        data = json.load(
            file
        )


    excluded = set()


    for example in data[
        "examples"
    ]:

        excluded.add(
            int(
                example[
                    "target_image_id"
                ]
            )
        )


    return excluded


# ============================================================
# Generate all valid caption pairs for one image
# ============================================================


def generate_pairs_for_image(
    image_id,
    captions
):

    usable_captions = [

        caption

        for caption in captions

        if caption_is_usable(
            caption
        )
    ]


    candidates = []


    for i in range(
        len(usable_captions)
    ):

        for j in range(
            i + 1,
            len(usable_captions)
        ):

            caption_a = (
                usable_captions[
                    i
                ]
            )

            caption_b = (
                usable_captions[
                    j
                ]
            )


            score = score_caption_pair(
                caption_a,
                caption_b
            )


            if score is None:

                continue


            candidates.append(
                {
                    "image_id":
                        image_id,

                    "caption_a":
                        caption_a,

                    "caption_b":
                        caption_b,

                    **score,
                }
            )


    return candidates


# ============================================================
# Build image-level candidate pool
#
# Keep only the best caption pair for each image.
#
# This guarantees every benchmark example corresponds to a
# unique target image.
# ============================================================


def build_candidate_pool(
    captions_by_image,
    excluded_ids
):

    pool = []


    for (
        image_id,
        captions
    ) in captions_by_image.items():

        if image_id in excluded_ids:

            continue


        if image_id in KNOWN_REJECTED_IMAGE_IDS:

            continue


        pairs = generate_pairs_for_image(
            image_id,
            captions
        )


        if len(pairs) == 0:

            continue


        best_pair = max(
            pairs,
            key=lambda item:
                item[
                    "quality_score"
                ]
        )


        pool.append(
            best_pair
        )


    return pool


# ============================================================
# Deterministic selection
#
# We take candidates from a high-quality pool but add a small
# seeded random component so the holdout is not simply the
# exact next 150 examples under one handcrafted quality score.
#
# No retrieval/model output is used here.
# ============================================================


def select_candidates(
    candidate_pool,
    num_candidates,
    seed,
    high_quality_pool_size
):

    candidate_pool = sorted(

        candidate_pool,

        key=lambda item:
            item[
                "quality_score"
            ],

        reverse=True
    )


    high_quality_pool_size = min(
        high_quality_pool_size,
        len(candidate_pool)
    )


    high_quality_pool = (
        candidate_pool[
            :high_quality_pool_size
        ]
    )


    if len(
        high_quality_pool
    ) < num_candidates:

        raise ValueError(
            "Not enough valid holdout candidates. "
            f"Requested {num_candidates}, "
            f"but only "
            f"{len(high_quality_pool)} "
            f"were available."
        )


    rng = random.Random(
        seed
    )


    selected = rng.sample(
        high_quality_pool,
        num_candidates
    )


    # --------------------------------------------------------
    # Present strongest examples first for easier manual
    # inspection.
    # --------------------------------------------------------

    selected.sort(

        key=lambda item:
            item[
                "quality_score"
            ],

        reverse=True
    )


    return selected


# ============================================================
# Convert candidate into benchmark format
# ============================================================


def make_benchmark_example(
    candidate,
    example_number
):

    image_id = (
        candidate[
            "image_id"
        ]
    )


    caption_a = (
        candidate[
            "caption_a"
        ]
    )


    caption_b = (
        candidate[
            "caption_b"
        ]
    )


    query = (

        "Find an image matching both of these "
        "visual descriptions: "

        f"{caption_a}; "

        f"{caption_b}"
    )


    return {
        "benchmark_id":
            (
                "coco_holdout_candidate_"
                f"{example_number:03d}"
            ),

        "target_image_id":
            image_id,

        "file_name":
            f"{image_id:012d}.jpg",

        "caption_a":
            caption_a,

        "caption_b":
            caption_b,

        "query":
            query,

        "quality_score":
            round(
                candidate[
                    "quality_score"
                ],
                6
            ),

        "content_overlap":
            round(
                candidate[
                    "content_overlap"
                ],
                6
            ),

        "unique_content_a":
            candidate[
                "unique_a"
            ],

        "unique_content_b":
            candidate[
                "unique_b"
            ],

        "shared_content_words":
            candidate[
                "shared_words"
            ],
    }


# ============================================================
# Validation
# ============================================================


def validate_candidate_set(
    examples,
    excluded_development_ids
):

    image_ids = [

        int(
            example[
                "target_image_id"
            ]
        )

        for example in examples
    ]


    duplicate_ids = {

        image_id

        for image_id in image_ids

        if (
            image_ids.count(
                image_id
            )
            > 1
        )
    }


    overlap_with_development = (

        set(
            image_ids
        )

        &

        excluded_development_ids
    )


    if duplicate_ids:

        raise ValueError(
            "Duplicate target image IDs found: "
            f"{sorted(duplicate_ids)}"
        )


    if overlap_with_development:

        raise ValueError(
            "HOLDOUT LEAKAGE DETECTED. "
            "These image IDs already occur in the "
            "development benchmark: "
            f"{sorted(overlap_with_development)}"
        )


    return {
        "num_examples":
            len(examples),

        "num_unique_images":
            len(
                set(
                    image_ids
                )
            ),

        "development_overlap":
            len(
                overlap_with_development
            ),
    }


# ============================================================
# Pretty printing for manual inspection
# ============================================================


def print_examples(
    examples
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "COMPLEX COCO HOLDOUT CANDIDATES"
    )

    print(
        "=" * 100
    )


    for index, example in enumerate(
        examples,
        start=1
    ):

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"Candidate {index}"
        )

        print(
            f"Benchmark ID: "
            f"{example['benchmark_id']}"
        )

        print(
            f"Target image: "
            f"{example['file_name']}"
        )

        print(
            f"Quality score: "
            f"{example['quality_score']:.3f}"
        )

        print(
            f"Content overlap: "
            f"{example['content_overlap']:.3f}"
        )


        print(
            "\nCaption A:"
        )

        print(
            example[
                "caption_a"
            ]
        )


        print(
            "\nCaption B:"
        )

        print(
            example[
                "caption_b"
            ]
        )


        print(
            "\nComplex query:"
        )

        print(
            example[
                "query"
            ]
        )


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate an image-disjoint candidate pool "
            "for the final Complex COCO holdout benchmark."
        )
    )


    parser.add_argument(
        "--captions-file",
        type=str,
        default=(
            "../../data/coco/"
            "annotations/"
            "captions_val2017.json"
        )
    )


    parser.add_argument(
        "--development-benchmark",
        type=str,
        default=(
            "../../data/benchmarks/"
            "complex_coco_queries_50.json"
        )
    )


    parser.add_argument(
        "--output",
        type=str,
        default=(
            "../../data/benchmarks/"
            "complex_coco_holdout_candidates_150.json"
        )
    )


    parser.add_argument(
        "--num-candidates",
        type=int,
        default=150
    )


    parser.add_argument(
        "--high-quality-pool-size",
        type=int,
        default=1200
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=20261014
    )


    args = parser.parse_args()


    print(
        "=" * 100
    )

    print(
        "BUILDING COMPLEX COCO HOLDOUT CANDIDATE SET"
    )

    print(
        "=" * 100
    )


    print(
        "\nLoading frozen development benchmark..."
    )


    excluded_development_ids = (
        load_excluded_development_ids(
            args.development_benchmark
        )
    )


    print(
        f"Development image IDs excluded: "
        f"{len(excluded_development_ids)}"
    )


    print(
        "\nLoading COCO captions..."
    )


    captions_by_image = (
        load_coco_captions(
            args.captions_file
        )
    )


    print(
        f"COCO images with captions: "
        f"{len(captions_by_image)}"
    )


    print(
        "\nGenerating image-level candidate pool..."
    )


    candidate_pool = (
        build_candidate_pool(
            captions_by_image=
                captions_by_image,

            excluded_ids=
                excluded_development_ids
        )
    )


    print(
        f"Usable candidate images: "
        f"{len(candidate_pool)}"
    )


    selected_candidates = (
        select_candidates(
            candidate_pool=
                candidate_pool,

            num_candidates=
                args.num_candidates,

            seed=
                args.seed,

            high_quality_pool_size=
                args.high_quality_pool_size
        )
    )


    examples = [

        make_benchmark_example(
            candidate,
            example_number=index
        )

        for index, candidate in enumerate(
            selected_candidates,
            start=1
        )
    ]


    validation = (
        validate_candidate_set(
            examples=
                examples,

            excluded_development_ids=
                excluded_development_ids
        )
    )


    output = {
        "benchmark_name":
            (
                "Complex COCO Holdout "
                "Candidate Benchmark"
            ),

        "benchmark_version":
            "candidate-v1",

        "status":
            "UNFROZEN_CANDIDATES",

        "purpose":
            (
                "Candidate pool for manual construction "
                "of an untouched final test benchmark."
            ),

        "source_dataset":
            "MS COCO val2017 captions",

        "development_benchmark":
            os.path.basename(
                args.development_benchmark
            ),

        "development_images_excluded":
            len(
                excluded_development_ids
            ),

        "seed":
            args.seed,

        "selection": {
            "num_candidates":
                args.num_candidates,

            "high_quality_pool_size":
                args.high_quality_pool_size,

            "model_outputs_used_for_selection":
                False,
        },

        "validation":
            validation,

        "examples":
            examples
    }


    output_directory = (
        os.path.dirname(
            args.output
        )
    )


    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )


    with open(
        args.output,
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )


    print_examples(
        examples
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "HOLDOUT CANDIDATE SET SAVED"
    )

    print(
        "=" * 100
    )


    print(
        f"\nCandidates: "
        f"{len(examples)}"
    )

    print(
        f"Unique target images: "
        f"{validation['num_unique_images']}"
    )

    print(
        f"Overlap with development set: "
        f"{validation['development_overlap']}"
    )

    print(
        f"\nFile:"
    )

    print(
        args.output
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT run retrieval, VLM, or any other model "
        "against these candidates yet."
    )

    print(
        "First manually inspect the candidate queries, "
        "remove ambiguous or contradictory examples, "
        "and freeze exactly 100."
    )


if __name__ == "__main__":

    main()
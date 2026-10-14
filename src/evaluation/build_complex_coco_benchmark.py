import argparse
import json
import os
import random
import re


# --------------------------------------------------
# 1. Common stop words
#
# These words carry little visual information.
# --------------------------------------------------

STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "from",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "this",
    "that",
    "these",
    "those",
    "there",
    "it",
    "its",
    "as",
    "into",
    "near",
    "next",
    "over",
    "under",
    "around",
    "while",
    "during",
    "through",
    "up",
    "down",
    "out"
}


# --------------------------------------------------
# 2. Vague / subjective phrases
#
# These descriptions do not provide reliable visual
# evidence for retrieval evaluation.
# --------------------------------------------------

VAGUE_PHRASES = [
    "looks incredible",
    "looks amazing",
    "looks beautiful",
    "looks nice",
    "looks good",
    "nice picture",
    "beautiful picture",
    "interesting picture",
    "a picture of",
    "a photo of",
    "an image of",
    "this is a picture",
    "this is an image",
    "this is a photo",
    "a scene of",
    "some sort of",
    "something that",
    "an open air zone",
    "a view of the area",
    "serenity",
    "stately environment",
    "may have",
    "probably",
    "could be",
    "could have",
    "beautiful"
]


# --------------------------------------------------
# 3. Manually rejected images
#
# These were identified during benchmark inspection
# as ambiguous, contradictory, speculative, or
# unsuitable for compositional visual retrieval.
# --------------------------------------------------

MANUALLY_REJECTED_IMAGE_IDS = {
    # Previous rejects
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

    # Final benchmark cleanup
    219271,   # contradictory/disconnected captions
    393569,   # nonvisual living-arrangement inference
    341094,   # general knowledge rather than image evidence
    445365,   # inferred family relation / protective intent
    241602,   # subjective ambience description
    262895,   # subjective personality description
}


# --------------------------------------------------
# 4. Tokenize text
# --------------------------------------------------

def tokenize(
    text
):

    return re.findall(
        r"[a-z0-9]+",
        text.lower()
    )


# --------------------------------------------------
# 5. Extract content words
# --------------------------------------------------

def content_words(
    text
):

    words = tokenize(
        text
    )


    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


# --------------------------------------------------
# 6. Check whether caption contains vague language
# --------------------------------------------------

def contains_vague_phrase(
    caption
):

    caption_lower = (
        caption.lower()
    )


    for phrase in VAGUE_PHRASES:

        if phrase in caption_lower:

            return True


    return False


# --------------------------------------------------
# 7. Caption quality test
# --------------------------------------------------

def caption_is_usable(
    caption,
    min_words=5,
    max_words=22,
    min_content_words=4
):

    words = tokenize(
        caption
    )


    content = content_words(
        caption
    )


    # --------------------------------------------------
    # Reject captions that are too short
    # --------------------------------------------------

    if len(
        words
    ) < min_words:

        return False


    # --------------------------------------------------
    # Reject excessively long captions
    # --------------------------------------------------

    if len(
        words
    ) > max_words:

        return False


    # --------------------------------------------------
    # Require enough visually meaningful words
    # --------------------------------------------------

    if len(
        content
    ) < min_content_words:

        return False


    # --------------------------------------------------
    # Reject vague / subjective captions
    # --------------------------------------------------

    if contains_vague_phrase(
        caption
    ):

        return False


    # --------------------------------------------------
    # Reject captions dominated by repeated words
    # --------------------------------------------------

    unique_ratio = (
        len(
            set(
                words
            )
        )
        /
        len(
            words
        )
    )


    if unique_ratio < 0.60:

        return False


    return True


# --------------------------------------------------
# 8. Jaccard similarity of content words
# --------------------------------------------------

def content_jaccard(
    caption_a,
    caption_b
):

    words_a = set(
        content_words(
            caption_a
        )
    )


    words_b = set(
        content_words(
            caption_b
        )
    )


    union = (
        words_a
        | words_b
    )


    if len(
        union
    ) == 0:

        return 1.0


    return (
        len(
            words_a
            & words_b
        )
        /
        len(
            union
        )
    )


# --------------------------------------------------
# 9. Number of unique visual concepts added
# --------------------------------------------------

def unique_content_counts(
    caption_a,
    caption_b
):

    words_a = set(
        content_words(
            caption_a
        )
    )


    words_b = set(
        content_words(
            caption_b
        )
    )


    unique_a = (
        words_a
        - words_b
    )


    unique_b = (
        words_b
        - words_a
    )


    return (
        len(
            unique_a
        ),
        len(
            unique_b
        )
    )


# --------------------------------------------------
# 10. Score one caption pair
# --------------------------------------------------

def score_caption_pair(
    caption_a,
    caption_b
):

    content_a = content_words(
        caption_a
    )

    content_b = content_words(
        caption_b
    )


    overlap = content_jaccard(
        caption_a,
        caption_b
    )


    (
        unique_a,
        unique_b
    ) = unique_content_counts(
        caption_a,
        caption_b
    )


    # --------------------------------------------------
    # Reject nearly identical captions
    # --------------------------------------------------

    if overlap > 0.65:

        return None


    # --------------------------------------------------
    # Require each caption to contribute
    # at least two unique visual concepts
    # --------------------------------------------------

    if unique_a < 2:

        return None


    if unique_b < 2:

        return None


    # --------------------------------------------------
    # Information score
    # --------------------------------------------------

    information_score = (
        len(
            set(
                content_a
            )
        )
        +
        len(
            set(
                content_b
            )
        )
    )


    # --------------------------------------------------
    # Complementarity score
    # --------------------------------------------------

    complementarity_score = (
        1.0
        - overlap
    )


    # --------------------------------------------------
    # Balanced uniqueness
    # --------------------------------------------------

    balance_score = min(
        unique_a,
        unique_b
    )


    # --------------------------------------------------
    # Final heuristic quality score
    # --------------------------------------------------

    quality_score = (

        0.40
        * information_score

        +

        3.0
        * complementarity_score

        +

        0.75
        * balance_score
    )


    return {
        "quality_score":
            quality_score,

        "content_overlap":
            overlap,

        "unique_content_a":
            unique_a,

        "unique_content_b":
            unique_b,

        "content_words_a":
            len(
                set(
                    content_a
                )
            ),

        "content_words_b":
            len(
                set(
                    content_b
                )
            )
    }


# --------------------------------------------------
# 11. Convert two captions into a complex query
# --------------------------------------------------

def build_complex_query(
    caption_a,
    caption_b
):

    caption_a = (
        caption_a
        .strip()
        .rstrip(".")
    )


    caption_b = (
        caption_b
        .strip()
        .rstrip(".")
    )


    return (
        "Find an image matching both of these "
        "visual descriptions: "
        f"{caption_a}; "
        f"{caption_b}."
    )


# --------------------------------------------------
# 12. Load COCO captions
# --------------------------------------------------

def load_coco(
    captions_file
):

    with open(
        captions_file,
        "r"
    ) as file:

        coco_data = json.load(
            file
        )


    image_lookup = {

        image[
            "id"
        ]:
            image[
                "file_name"
            ]

        for image
        in coco_data[
            "images"
        ]
    }


    captions_by_image = {}


    for annotation in coco_data[
        "annotations"
    ]:

        image_id = annotation[
            "image_id"
        ]


        caption = (
            annotation[
                "caption"
            ]
            .strip()
        )


        captions_by_image.setdefault(
            image_id,
            []
        ).append(
            caption
        )


    return (
        image_lookup,
        captions_by_image
    )


# --------------------------------------------------
# 13. Generate all usable candidate pairs
# --------------------------------------------------

def generate_candidates(
    captions_file
):

    (
        image_lookup,
        captions_by_image
    ) = load_coco(
        captions_file
    )


    candidates = []


    for (
        image_id,
        captions
    ) in captions_by_image.items():


        # --------------------------------------------------
        # Skip manually rejected noisy images
        # --------------------------------------------------

        if (
            image_id
            in MANUALLY_REJECTED_IMAGE_IDS
        ):

            continue


        # --------------------------------------------------
        # Filter individual captions
        # --------------------------------------------------

        usable_captions = [

            caption

            for caption in captions

            if caption_is_usable(
                caption
            )
        ]


        if len(
            usable_captions
        ) < 2:

            continue


        best_pair = None
        best_pair_score = None


        # --------------------------------------------------
        # Evaluate every pair belonging to this image
        # --------------------------------------------------

        for i in range(
            len(
                usable_captions
            )
        ):

            for j in range(
                i + 1,
                len(
                    usable_captions
                )
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


                pair_stats = (
                    score_caption_pair(
                        caption_a,
                        caption_b
                    )
                )


                if pair_stats is None:

                    continue


                if (
                    best_pair_score is None

                    or pair_stats[
                        "quality_score"
                    ]
                    > best_pair_score
                ):

                    best_pair_score = (
                        pair_stats[
                            "quality_score"
                        ]
                    )


                    best_pair = {
                        "caption_a":
                            caption_a,

                        "caption_b":
                            caption_b,

                        **pair_stats
                    }


        if best_pair is None:

            continue


        candidates.append(
            {
                "image_id":
                    image_id,

                "target_image_id":
                    image_id,

                "file_name":
                    image_lookup[
                        image_id
                    ],

                "caption_a":
                    best_pair[
                        "caption_a"
                    ],

                "caption_b":
                    best_pair[
                        "caption_b"
                    ],

                "query":
                    build_complex_query(
                        best_pair[
                            "caption_a"
                        ],
                        best_pair[
                            "caption_b"
                        ]
                    ),

                "quality_score":
                    best_pair[
                        "quality_score"
                    ],

                "content_overlap":
                    best_pair[
                        "content_overlap"
                    ],

                "unique_content_a":
                    best_pair[
                        "unique_content_a"
                    ],

                "unique_content_b":
                    best_pair[
                        "unique_content_b"
                    ],

                "content_words_a":
                    best_pair[
                        "content_words_a"
                    ],

                "content_words_b":
                    best_pair[
                        "content_words_b"
                    ]
            }
        )


    return candidates


# --------------------------------------------------
# 14. Select final benchmark
# --------------------------------------------------

def select_benchmark(
    candidates,
    benchmark_size,
    seed
):

    # --------------------------------------------------
    # Sort candidates by quality
    # --------------------------------------------------

    candidates.sort(

        key=lambda item:
            item[
                "quality_score"
            ],

        reverse=True
    )


    # --------------------------------------------------
    # Draw deterministically from a high-quality pool
    #
    # This avoids choosing only one type of scene.
    # --------------------------------------------------

    pool_size = min(

        len(
            candidates
        ),

        benchmark_size * 5
    )


    high_quality_pool = (
        candidates[
            :pool_size
        ]
    )


    random_generator = (
        random.Random(
            seed
        )
    )


    random_generator.shuffle(
        high_quality_pool
    )


    selected = (
        high_quality_pool[
            :benchmark_size
        ]
    )


    # --------------------------------------------------
    # Assign permanent benchmark IDs
    # --------------------------------------------------

    for index, example in enumerate(
        selected,
        start=1
    ):

        example[
            "benchmark_id"
        ] = (
            f"coco_complex_"
            f"{index:03d}"
        )


    return selected


# --------------------------------------------------
# 15. Print benchmark for manual inspection
# --------------------------------------------------

def print_benchmark(
    benchmark
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FROZEN COMPLEX COCO BENCHMARK"
    )

    print(
        "=" * 70
    )


    for index, example in enumerate(
        benchmark,
        start=1
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Example {index}"
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
            "\nQuery:"
        )

        print(
            example[
                "query"
            ]
        )


# --------------------------------------------------
# 16. Save benchmark
# --------------------------------------------------

def save_benchmark(
    benchmark,
    output_path,
    captions_file,
    seed
):

    output_directory = (
        os.path.dirname(
            output_path
        )
    )


    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )


    output = {

        "benchmark_name":
            "Complex COCO Retrieval Benchmark",

        "benchmark_version":
            "1.1",

        "description":
            (
                "Fixed compositional image retrieval "
                "benchmark created from pairs of "
                "human-written COCO captions after "
                "automatic and manual quality filtering."
            ),

        "source_captions_file":
            captions_file,

        "seed":
            seed,

        "num_examples":
            len(
                benchmark
            ),

        "generation_policy": {

            "min_caption_words":
                5,

            "max_caption_words":
                22,

            "min_content_words":
                4,

            "max_pair_content_overlap":
                0.65,

            "min_unique_content_per_caption":
                2,

            "vague_caption_filter":
                True,

            "manual_rejection_filter":
                True,

            "manually_rejected_image_ids":
                sorted(
                    MANUALLY_REJECTED_IMAGE_IDS
                )
        },

        "examples":
            benchmark
    }


    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )


# --------------------------------------------------
# 17. Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed complex-query "
            "COCO retrieval benchmark."
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
        "--benchmark-size",
        type=int,
        default=50
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )


    parser.add_argument(
        "--output",
        type=str,
        default=(
            "../../data/benchmarks/"
            "complex_coco_queries_50.json"
        )
    )


    args = parser.parse_args()


    # --------------------------------------------------
    # Generate candidate examples
    # --------------------------------------------------

    print(
        "Generating candidate caption pairs..."
    )


    candidates = generate_candidates(
        args.captions_file
    )


    print(
        f"Usable candidate images: "
        f"{len(candidates)}"
    )


    if len(
        candidates
    ) < args.benchmark_size:

        raise ValueError(
            "Not enough usable examples "
            "to create requested benchmark."
        )


    # --------------------------------------------------
    # Select benchmark
    # --------------------------------------------------

    benchmark = select_benchmark(

        candidates=
            candidates,

        benchmark_size=
            args.benchmark_size,

        seed=
            args.seed
    )


    # --------------------------------------------------
    # Print examples
    # --------------------------------------------------

    print_benchmark(
        benchmark
    )


    # --------------------------------------------------
    # Save benchmark
    # --------------------------------------------------

    save_benchmark(

        benchmark=
            benchmark,

        output_path=
            args.output,

        captions_file=
            args.captions_file,

        seed=
            args.seed
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "BENCHMARK SAVED"
    )

    print(
        "=" * 70
    )


    print(
        f"\nExamples: "
        f"{len(benchmark)}"
    )


    print(
        "\nFile:"
    )

    print(
        args.output
    )


    print(
        "\nBenchmark version: 1.1"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "After manual inspection approves this "
        "benchmark, do not regenerate it for "
        "future model comparisons."
    )


if __name__ == "__main__":

    main()
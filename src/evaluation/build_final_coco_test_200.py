import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Set


# ============================================================
# Configuration
# ============================================================

SEED = 20261017

NUM_FINAL_QUERIES = 200


CAPTIONS_FILE = (
    "../../data/coco/annotations/"
    "captions_val2017.json"
)


DEV50_FILE = (
    "../../data/benchmarks/"
    "complex_coco_queries_50.json"
)


OLD_HOLDOUT100_FILE = (
    "../../data/benchmarks/"
    "complex_coco_holdout_100.json"
)


OUTPUT_FILE = (
    "../../data/benchmarks/"
    "complex_coco_final_test_200.json"
)


MANIFEST_FILE = (
    "../../data/benchmarks/"
    "complex_coco_final_test_200_manifest.json"
)


HASH_FILE = (
    "../../data/benchmarks/"
    "complex_coco_final_test_200.sha256"
)


# ============================================================
# Helpers
# ============================================================


def normalize_image_id(
    value,
):

    if value is None:

        return None


    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return str(
            value
        ).strip()


def load_json(
    path: str,
):

    with open(
        path,
        "r",
    ) as file:

        return json.load(
            file
        )


def save_json(
    path: str,
    data,
):

    directory = (
        os.path.dirname(
            path
        )
    )


    if directory:

        os.makedirs(
            directory,
            exist_ok=True,
        )


    with open(
        path,
        "w",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def extract_records(
    data,
) -> List[
    Dict[str, Any]
]:

    if isinstance(
        data,
        list,
    ):

        return data


    if isinstance(
        data,
        dict,
    ):

        for key in [
            "queries",
            "examples",
            "records",
            "data",
            "benchmark",
            "items",
        ]:

            value = data.get(
                key
            )


            if isinstance(
                value,
                list,
            ):

                return value


    raise ValueError(
        "Could not locate benchmark records."
    )


def get_target_image_id(
    record: Dict[str, Any],
):

    for key in [
        "target_image_id",
        "image_id",
        "target_id",
        "coco_image_id",
        "positive_image_id",
        "ground_truth_image_id",
    ]:

        if (
            key in record
            and
            record[
                key
            ]
            is not None
        ):

            return normalize_image_id(
                record[
                    key
                ]
            )


    for nested_key in [
        "target",
        "ground_truth",
        "positive",
    ]:

        nested = record.get(
            nested_key
        )


        if not isinstance(
            nested,
            dict,
        ):

            continue


        for key in [
            "target_image_id",
            "image_id",
            "target_id",
            "coco_image_id",
        ]:

            if (
                key in nested
                and
                nested[
                    key
                ]
                is not None
            ):

                return normalize_image_id(
                    nested[
                        key
                    ]
                )


    raise KeyError(
        "Could not find target image ID "
        f"in record keys: {list(record.keys())}"
    )


def get_target_ids_from_benchmark(
    path: str,
) -> Set[int]:

    data = (
        load_json(
            path
        )
    )


    records = (
        extract_records(
            data
        )
    )


    target_ids = set()


    for record in records:

        target_id = (
            get_target_image_id(
                record
            )
        )


        target_ids.add(
            target_id
        )


    return target_ids


def clean_caption(
    caption: str,
) -> str:

    caption = str(
        caption
    ).strip()


    caption = " ".join(
        caption.split()
    )


    return caption


def build_query(
    caption_a: str,
    caption_b: str,
) -> str:

    return (
        "Find an image matching both of these "
        "visual descriptions: "
        f"{caption_a}; "
        f"{caption_b}."
    )


def sha256_file(
    path: str,
) -> str:

    hasher = (
        hashlib.sha256()
    )


    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )


            if not chunk:

                break


            hasher.update(
                chunk
            )


    return hasher.hexdigest()


# ============================================================
# Main
# ============================================================


def main():

    random.seed(
        SEED
    )


    print(
        "\n"
        + "=" * 80
    )


    print(
        "BUILDING UNTOUCHED FINAL COCO TEST SET"
    )


    print(
        "=" * 80
    )


    print(
        f"\nRequested queries: "
        f"{NUM_FINAL_QUERIES}"
    )


    print(
        f"Random seed: "
        f"{SEED}"
    )


    # ========================================================
    # Load previous benchmark targets
    # ========================================================

    dev50_ids = (
        get_target_ids_from_benchmark(
            DEV50_FILE
        )
    )


    holdout100_ids = (
        get_target_ids_from_benchmark(
            OLD_HOLDOUT100_FILE
        )
    )


    excluded_ids = (
        dev50_ids
        |
        holdout100_ids
    )


    print(
        f"\nDevelopment target IDs: "
        f"{len(dev50_ids)}"
    )


    print(
        f"Old holdout target IDs: "
        f"{len(holdout100_ids)}"
    )


    print(
        f"Unique excluded target IDs: "
        f"{len(excluded_ids)}"
    )


    overlap_existing = (
        dev50_ids
        &
        holdout100_ids
    )


    print(
        f"Existing dev50/holdout100 overlap: "
        f"{len(overlap_existing)}"
    )


    # ========================================================
    # Load COCO captions
    # ========================================================

    coco = (
        load_json(
            CAPTIONS_FILE
        )
    )


    annotations = (
        coco.get(
            "annotations",
            []
        )
    )


    images = (
        coco.get(
            "images",
            []
        )
    )


    if not annotations:

        raise RuntimeError(
            "No COCO caption annotations found."
        )


    # ========================================================
    # Image ID -> captions
    # ========================================================

    captions_by_image = (
        defaultdict(
            list
        )
    )


    for annotation in annotations:

        image_id = (
            normalize_image_id(
                annotation.get(
                    "image_id"
                )
            )
        )


        caption = (
            clean_caption(
                annotation.get(
                    "caption",
                    "",
                )
            )
        )


        if (
            image_id is None
            or
            not caption
        ):

            continue


        if (
            caption
            not in
            captions_by_image[
                image_id
            ]
        ):

            captions_by_image[
                image_id
            ].append(
                caption
            )


    # ========================================================
    # Image metadata
    # ========================================================

    image_metadata = {}


    for image in images:

        image_id = (
            normalize_image_id(
                image.get(
                    "id"
                )
            )
        )


        if image_id is None:

            continue


        image_metadata[
            image_id
        ] = image


    # ========================================================
    # Eligible final-test targets
    # ========================================================

    eligible_ids = []


    for image_id, captions in (
        captions_by_image.items()
    ):

        if image_id in excluded_ids:

            continue


        # Need at least two distinct captions to construct
        # a two-description compositional query.

        if len(
            captions
        ) < 2:

            continue


        eligible_ids.append(
            image_id
        )


    eligible_ids = sorted(
        eligible_ids
    )


    print(
        f"\nEligible unused target images: "
        f"{len(eligible_ids)}"
    )


    if len(
        eligible_ids
    ) < NUM_FINAL_QUERIES:

        raise RuntimeError(
            "Not enough eligible images to create "
            f"{NUM_FINAL_QUERIES} test queries."
        )


    # ========================================================
    # Deterministic sampling
    # ========================================================

    selected_ids = (
        random.sample(
            eligible_ids,
            NUM_FINAL_QUERIES,
        )
    )


    # ========================================================
    # Construct benchmark
    # ========================================================

    benchmark_records = []


    for index, image_id in enumerate(
        selected_ids,
        start=1,
    ):

        captions = list(
            captions_by_image[
                image_id
            ]
        )


        # ----------------------------------------------------
        # Deterministic caption choice.
        #
        # Shuffle with the globally fixed RNG and use first two
        # unique captions.
        # ----------------------------------------------------

        random.shuffle(
            captions
        )


        caption_a = (
            captions[
                0
            ]
        )


        caption_b = (
            captions[
                1
            ]
        )


        query = (
            build_query(
                caption_a,
                caption_b,
            )
        )


        metadata = (
            image_metadata.get(
                image_id,
                {},
            )
        )


        record = {

            "id":
                (
                    "coco_final_"
                    f"{index:03d}"
                ),

            "query":
                query,

            "target_image_id":
                image_id,

            "caption_1":
                caption_a,

            "caption_2":
                caption_b,

            "file_name":
                metadata.get(
                    "file_name"
                ),

            "benchmark_split":
                "final_test",

            "benchmark_version":
                "1.0",
        }


        benchmark_records.append(
            record
        )


    # ========================================================
    # Validation
    # ========================================================

    final_ids = {

        normalize_image_id(
            record[
                "target_image_id"
            ]
        )

        for record
        in benchmark_records
    }


    if len(
        final_ids
    ) != NUM_FINAL_QUERIES:

        raise RuntimeError(
            "Duplicate target image IDs detected."
        )


    dev_overlap = (
        final_ids
        &
        dev50_ids
    )


    holdout_overlap = (
        final_ids
        &
        holdout100_ids
    )


    if dev_overlap:

        raise RuntimeError(
            "Final test overlaps development set: "
            f"{sorted(dev_overlap)}"
        )


    if holdout_overlap:

        raise RuntimeError(
            "Final test overlaps old holdout set: "
            f"{sorted(holdout_overlap)}"
        )


    # ========================================================
    # Freeze benchmark
    # ========================================================

    output_data = {

        "benchmark_name":
            "Complex COCO Final Test 200",

        "version":
            "1.0",

        "status":
            "FROZEN",

        "purpose":
            (
                "Untouched final evaluation set for "
                "Agentic Multimodal DeepSearch."
            ),

        "construction":
            {

                "source_dataset":
                    "MS COCO 2017 validation",

                "retrieval_corpus":
                    "COCO val2017",

                "num_queries":
                    NUM_FINAL_QUERIES,

                "random_seed":
                    SEED,

                "query_template":
                    (
                        "Find an image matching both of these "
                        "visual descriptions: "
                        "<caption_1>; <caption_2>."
                    ),

                "target_overlap_with_dev50":
                    0,

                "target_overlap_with_old_holdout100":
                    0,

                "manual_selection":
                    False,

                "manual_tuning":
                    False,
            },

        "queries":
            benchmark_records,
    }


    save_json(
        OUTPUT_FILE,
        output_data,
    )


    benchmark_hash = (
        sha256_file(
            OUTPUT_FILE
        )
    )


    # ========================================================
    # Manifest
    # ========================================================

    manifest = {

        "benchmark_file":
            os.path.basename(
                OUTPUT_FILE
            ),

        "benchmark_name":
            "Complex COCO Final Test 200",

        "version":
            "1.0",

        "status":
            "FROZEN",

        "num_queries":
            NUM_FINAL_QUERIES,

        "random_seed":
            SEED,

        "sha256":
            benchmark_hash,

        "excluded_benchmarks":
            [

                os.path.basename(
                    DEV50_FILE
                ),

                os.path.basename(
                    OLD_HOLDOUT100_FILE
                ),
            ],

        "dev50_target_overlap":
            len(
                dev_overlap
            ),

        "holdout100_target_overlap":
            len(
                holdout_overlap
            ),

        "instruction":
            (
                "Do not modify this benchmark or tune "
                "the retrieval architecture using its "
                "individual examples or results."
            ),
    }


    save_json(
        MANIFEST_FILE,
        manifest,
    )


    with open(
        HASH_FILE,
        "w",
    ) as file:

        file.write(
            benchmark_hash
            +
            "  "
            +
            os.path.basename(
                OUTPUT_FILE
            )
            +
            "\n"
        )


    # ========================================================
    # Final report
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )


    print(
        "FINAL BENCHMARK CREATED AND FROZEN"
    )


    print(
        "=" * 80
    )


    print(
        f"\nQueries: "
        f"{len(benchmark_records)}"
    )


    print(
        f"Unique targets: "
        f"{len(final_ids)}"
    )


    print(
        f"Overlap with dev50: "
        f"{len(dev_overlap)}"
    )


    print(
        f"Overlap with old holdout100: "
        f"{len(holdout_overlap)}"
    )


    print(
        f"\nSHA256:"
    )


    print(
        benchmark_hash
    )


    print(
        "\nBenchmark:"
    )


    print(
        OUTPUT_FILE
    )


    print(
        "\nManifest:"
    )


    print(
        MANIFEST_FILE
    )


    print(
        "\nHash file:"
    )


    print(
        HASH_FILE
    )


    print(
        "\nIMPORTANT:"
    )


    print(
        "Do not inspect individual benchmark examples "
        "for architecture tuning."
    )


    print(
        "Do not modify the frozen JSON after evaluation."
    )


if __name__ == "__main__":

    main()
import json
import sys

import torch


sys.path.append("..")


from agents.agentic_retriever import AgenticRetriever


BENCHMARK_FILE = (
    "../../data/benchmarks/"
    "complex_coco_queries_50.json"
)

RESULTS_FILE = (
    "../../experiments/"
    "frozen_50_query_ablation.json"
)


def load_json(
    path
):

    with open(
        path,
        "r"
    ) as file:

        return json.load(
            file
        )


def main():

    # --------------------------------------------------
    # Load frozen benchmark
    # --------------------------------------------------

    benchmark_data = load_json(
        BENCHMARK_FILE
    )


    benchmark = (
        benchmark_data[
            "examples"
        ]
    )


    # --------------------------------------------------
    # Load previous evaluation results
    # --------------------------------------------------

    evaluation = load_json(
        RESULTS_FILE
    )


    previous_results = {

        item[
            "benchmark_id"
        ]:
            item

        for item in evaluation[
            "results"
        ]
    }


    # --------------------------------------------------
    # Load DeepSearch V2
    # --------------------------------------------------

    agent = AgenticRetriever(

        image_folder=
            "../../data/coco/val2017",

        captions_file=
            (
                "../../data/coco/"
                "annotations/"
                "captions_val2017.json"
            ),

        retriever_checkpoint_path=
            (
                "../../experiments/"
                "deepsearch_retriever_v2/"
                "deepsearch_retriever_v2.pt"
            ),

        retriever_processor_path=
            (
                "../../experiments/"
                "deepsearch_retriever_v2"
            ),

        planner_model_name=
            "Qwen/Qwen2.5-1.5B-Instruct",

        image_batch_size=
            128
    )


    output_rows = []


    print(
        "\n"
        + "=" * 100
    )

    print(
        "DEEPSEARCH V2 CONFIDENCE ANALYSIS"
    )

    print(
        "=" * 100
    )


    for example in benchmark:

        benchmark_id = (
            example[
                "benchmark_id"
            ]
        )


        query = (
            example[
                "query"
            ]
        )


        target_image_id = (
            example[
                "target_image_id"
            ]
        )


        # --------------------------------------------------
        # Encode original full query
        # --------------------------------------------------

        query_embedding = (
            agent.encode_query(
                query
            )
        )


        similarities = (
            query_embedding
            @ agent.image_embeddings.T
        ).squeeze(
            0
        )


        values, indices = torch.topk(
            similarities,
            k=10
        )


        values = (
            values
            .detach()
            .cpu()
            .tolist()
        )


        indices = (
            indices
            .detach()
            .cpu()
            .tolist()
        )


        # --------------------------------------------------
        # Confidence features
        # --------------------------------------------------

        top1_score = (
            values[
                0
            ]
        )


        top2_score = (
            values[
                1
            ]
        )


        top5_mean = (
            sum(
                values[
                    :5
                ]
            )
            / 5
        )


        margin_1_2 = (
            top1_score
            - top2_score
        )


        margin_1_5mean = (
            top1_score
            - top5_mean
        )


        top1_record = (
            agent.image_records[
                indices[
                    0
                ]
            ]
        )


        top1_image_id = (
            top1_record[
                "image_id"
            ]
        )


        top1_correct = (
            top1_image_id
            == target_image_id
        )


        old_result = (
            previous_results[
                benchmark_id
            ]
        )


        baseline_rank = (
            old_result[
                "baseline_rank"
            ]
        )


        weighted_rank = (
            old_result[
                "weighted_rrf_rank"
            ]
        )


        verified_rank = (
            old_result[
                "weighted_verified_rank"
            ]
        )


        # --------------------------------------------------
        # Determine final pipeline behavior
        # --------------------------------------------------

        capped_baseline = (
            baseline_rank
            if (
                baseline_rank is not None
                and baseline_rank <= 10
            )
            else 11
        )


        capped_verified = (
            verified_rank
            if (
                verified_rank is not None
                and verified_rank <= 10
            )
            else 11
        )


        if (
            capped_verified
            < capped_baseline
        ):

            outcome = "IMPROVED"

        elif (
            capped_verified
            > capped_baseline
        ):

            outcome = "WORSENED"

        else:

            outcome = "UNCHANGED"


        row = {

            "benchmark_id":
                benchmark_id,

            "baseline_rank":
                baseline_rank,

            "weighted_rank":
                weighted_rank,

            "verified_rank":
                verified_rank,

            "pipeline_outcome":
                outcome,

            "top1_score":
                top1_score,

            "top2_score":
                top2_score,

            "margin_1_2":
                margin_1_2,

            "margin_1_5mean":
                margin_1_5mean,

            "top1_image_id":
                top1_image_id,

            "target_image_id":
                target_image_id,

            "top1_correct":
                top1_correct
        }


        output_rows.append(
            row
        )


    # --------------------------------------------------
    # Sort by confidence margin
    # --------------------------------------------------

    output_rows.sort(

        key=lambda item:
            item[
                "margin_1_2"
            ],

        reverse=True
    )


    # --------------------------------------------------
    # Print all examples
    # --------------------------------------------------

    print(
        "\n"
        "ID                 "
        "Base   W-RRF   VLM   "
        "Top1      Margin12   Margin1-5   "
        "Outcome"
    )


    print(
        "-" * 100
    )


    for row in output_rows:

        print(

            f"{row['benchmark_id']:<18} "

            f"{str(row['baseline_rank']):<6} "

            f"{str(row['weighted_rank']):<7} "

            f"{str(row['verified_rank']):<5} "

            f"{row['top1_score']:<9.5f} "

            f"{row['margin_1_2']:<10.5f} "

            f"{row['margin_1_5mean']:<11.5f} "

            f"{row['pipeline_outcome']}"
        )


    # --------------------------------------------------
    # Group statistics
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "GROUP STATISTICS"
    )

    print(
        "=" * 100
    )


    for group_name in [
        "IMPROVED",
        "UNCHANGED",
        "WORSENED"
    ]:

        group = [

            row

            for row in output_rows

            if (
                row[
                    "pipeline_outcome"
                ]
                == group_name
            )
        ]


        if len(
            group
        ) == 0:

            continue


        mean_margin_12 = (

            sum(
                row[
                    "margin_1_2"
                ]

                for row in group
            )

            / len(
                group
            )
        )


        mean_margin_15 = (

            sum(
                row[
                    "margin_1_5mean"
                ]

                for row in group
            )

            / len(
                group
            )
        )


        mean_top1 = (

            sum(
                row[
                    "top1_score"
                ]

                for row in group
            )

            / len(
                group
            )
        )


        print(
            f"\n{group_name}"
        )

        print(
            f"Count: "
            f"{len(group)}"
        )

        print(
            f"Mean top-1 similarity: "
            f"{mean_top1:.6f}"
        )

        print(
            f"Mean top1-top2 margin: "
            f"{mean_margin_12:.6f}"
        )

        print(
            f"Mean top1-top5mean margin: "
            f"{mean_margin_15:.6f}"
        )


    # --------------------------------------------------
    # Specifically inspect regression cases
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "REGRESSION CASES"
    )

    print(
        "=" * 100
    )


    for row in output_rows:

        if (
            row[
                "pipeline_outcome"
            ]
            == "WORSENED"
        ):

            print(
                f"\n"
                f"{row['benchmark_id']}"
            )

            print(
                f"Baseline rank: "
                f"{row['baseline_rank']}"
            )

            print(
                f"Weighted RRF rank: "
                f"{row['weighted_rank']}"
            )

            print(
                f"VLM rank: "
                f"{row['verified_rank']}"
            )

            print(
                f"Top-1 similarity: "
                f"{row['top1_score']:.6f}"
            )

            print(
                f"Top1-top2 margin: "
                f"{row['margin_1_2']:.6f}"
            )

            print(
                f"Top1-top5mean margin: "
                f"{row['margin_1_5mean']:.6f}"
            )


    # --------------------------------------------------
    # Save analysis
    # --------------------------------------------------

    output_path = (
        "../../experiments/"
        "retrieval_confidence_analysis.json"
    )


    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            output_rows,
            file,
            indent=2
        )


    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    main()
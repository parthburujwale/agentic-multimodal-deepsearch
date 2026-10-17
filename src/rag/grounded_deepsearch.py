import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


# ============================================================
# Project paths
# ============================================================

THIS_FILE = Path(
    __file__
).resolve()

SRC_DIR = THIS_FILE.parents[
    1
]

PROJECT_ROOT = THIS_FILE.parents[
    2
]


if str(
    SRC_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            SRC_DIR
        ),
    )


# ============================================================
# Project imports
# ============================================================

from agents.iterative_agentic_search import (
    IterativeAgenticSearch,
)

from rag.evidence_builder import (
    EvidenceBuilder,
)

from rag.multimodal_answer_generator import (
    MultimodalAnswerGenerator,
)


# ============================================================
# Frozen retrieval configuration
#
# These values match the architecture used for the frozen
# 200-query final evaluation.
# ============================================================

SEED = 42

RETRIEVAL_TOP_K = 10

FINAL_TOP_K = 10

RRF_K = 60

MIN_VERIFICATION_SCORE = 0.75

MAX_ROUNDS = 2

ROUTER_SCORE_THRESHOLD = 0.85

ROUTER_SCORE_TOLERANCE = 1e-6

ROUTER_TIE_TOLERANCE = 0.01

PROMOTION_MAX_VERIFIER_GAP = 0.20


# ============================================================
# Frozen paths
# ============================================================

IMAGE_FOLDER = (
    PROJECT_ROOT
    /
    "data"
    /
    "coco"
    /
    "val2017"
)


CAPTIONS_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "coco"
    /
    "annotations"
    /
    "captions_val2017.json"
)


RETRIEVER_CHECKPOINT = (
    PROJECT_ROOT
    /
    "experiments"
    /
    "deepsearch_retriever_v2"
    /
    "deepsearch_retriever_v2.pt"
)


RETRIEVER_PROCESSOR = (
    PROJECT_ROOT
    /
    "experiments"
    /
    "deepsearch_retriever_v2"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    /
    "experiments"
    /
    "grounded_deepsearch_result.json"
)


# ============================================================
# JSON helper
# ============================================================


def save_json(
    path: str,
    data: Dict[str, Any],
):

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )


# ============================================================
# Grounded DeepSearch
# ============================================================


class GroundedDeepSearch:
    """
    End-to-end Agentic Multimodal DeepSearch.

    Pipeline:

        query
            ↓
        frozen iterative retrieval
            ↓
        verified final ranking
            ↓
        evidence builder
            ↓
        structured multimodal evidence
            ↓
        grounded VLM answer generation

    The frozen retrieval architecture is not modified here.
    """

    def __init__(
        self,
        max_evidence_images: int = 5,
        max_answer_images: int = 3,
    ):

        self.max_evidence_images = (
            max_evidence_images
        )

        self.max_answer_images = (
            max_answer_images
        )

        # ====================================================
        # Stage 1:
        # Frozen iterative retrieval system
        # ====================================================

        print(
            "\n"
            + "=" * 80
        )

        print(
            "INITIALIZING FROZEN ITERATIVE RETRIEVAL SYSTEM"
        )

        print(
            "=" * 80
        )

        self.search_system = (
            IterativeAgenticSearch(

                image_folder=
                    str(
                        IMAGE_FOLDER
                    ),

                captions_file=
                    str(
                        CAPTIONS_FILE
                    ),

                retriever_checkpoint_path=
                    str(
                        RETRIEVER_CHECKPOINT
                    ),

                retriever_processor_path=
                    str(
                        RETRIEVER_PROCESSOR
                    ),

                min_verification_score=
                    MIN_VERIFICATION_SCORE,

                max_rounds=
                    MAX_ROUNDS,

                router_score_threshold=
                    ROUTER_SCORE_THRESHOLD,

                router_score_tolerance=
                    ROUTER_SCORE_TOLERANCE,

                router_tie_tolerance=
                    ROUTER_TIE_TOLERANCE,

                seed=
                    SEED,

                fusion_enabled=
                    True,

                promotion_enabled=
                    True,

                promotion_max_verifier_gap=
                    PROMOTION_MAX_VERIFIER_GAP,
            )
        )

        # ====================================================
        # Stage 2:
        # Deterministic evidence builder
        # ====================================================

        print(
            "\n"
            + "=" * 80
        )

        print(
            "INITIALIZING EVIDENCE BUILDER"
        )

        print(
            "=" * 80
        )

        self.evidence_builder = (
            EvidenceBuilder(

                image_folder=
                    str(
                        IMAGE_FOLDER
                    ),

                max_evidence_images=
                    self.max_evidence_images,
            )
        )

        # ====================================================
        # Stage 3:
        # Grounded answer generator
        # ====================================================

        print(
            "\n"
            + "=" * 80
        )

        print(
            "INITIALIZING MULTIMODAL ANSWER GENERATOR"
        )

        print(
            "=" * 80
        )

        self.answer_generator = (
            MultimodalAnswerGenerator(

                model_name=
                    "Qwen/Qwen2.5-VL-3B-Instruct",

                max_images=
                    self.max_answer_images,

                max_new_tokens=
                    300,
            )
        )


    # ========================================================
    # Run frozen retrieval
    # ========================================================

    def retrieve(
        self,
        query: str,
    ) -> Dict[str, Any]:

        print(
            "\n\n"
            + "#"
            * 80
        )

        print(
            "STAGE 1: AGENTIC MULTIMODAL RETRIEVAL"
        )

        print(
            "#"
            * 80
        )

        search_output = (
            self.search_system.search(

                query=
                    query,

                retrieval_top_k=
                    RETRIEVAL_TOP_K,

                final_top_k=
                    FINAL_TOP_K,

                rrf_k=
                    RRF_K,
            )
        )

        return search_output


    # ========================================================
    # Build grounded evidence
    # ========================================================

    def build_evidence(
        self,
        search_output: Dict[str, Any],
    ) -> Dict[str, Any]:

        print(
            "\n\n"
            + "#"
            * 80
        )

        print(
            "STAGE 2: BUILDING GROUNDED EVIDENCE"
        )

        print(
            "#"
            * 80
        )

        evidence_package = (
            self.evidence_builder.build(
                search_output
            )
        )

        print(
            f"\nEvidence images: "
            f"{evidence_package.get('evidence_count')}"
        )

        print(
            f"Evidence confidence: "
            f"{evidence_package.get('confidence', {}).get('level')}"
        )

        print(
            f"Final retrieval sufficient: "
            f"{evidence_package.get('search_metadata', {}).get('final_sufficient')}"
        )

        primary = (
            evidence_package.get(
                "primary_evidence"
            )
        )

        if primary is not None:

            print(
                f"Primary image ID: "
                f"{primary.get('image_id')}"
            )

            print(
                f"Primary verifier score: "
                f"{primary.get('verification', {}).get('overall_score')}"
            )

        return evidence_package


    # ========================================================
    # Generate grounded answer
    # ========================================================

    def generate_answer(
        self,
        evidence_package: Dict[str, Any],
    ) -> Dict[str, Any]:

        print(
            "\n\n"
            + "#"
            * 80
        )

        print(
            "STAGE 3: GROUNDED MULTIMODAL ANSWER GENERATION"
        )

        print(
            "#"
            * 80
        )

        answer = (
            self.answer_generator.generate(
                evidence_package
            )
        )

        return answer


    # ========================================================
    # Full end-to-end query
    # ========================================================

    def search(
        self,
        query: str,
    ) -> Dict[str, Any]:

        query = str(
            query
        ).strip()

        if not query:

            raise ValueError(
                "Query cannot be empty."
            )

        print(
            "\n\n"
            + "="
            * 80
        )

        print(
            "AGENTIC MULTIMODAL DEEPSEARCH"
        )

        print(
            "="
            * 80
        )

        print(
            "\nUser query:"
        )

        print(
            query
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        search_output = (
            self.retrieve(
                query
            )
        )

        # ----------------------------------------------------
        # Evidence packaging
        # ----------------------------------------------------

        evidence_package = (
            self.build_evidence(
                search_output
            )
        )

        # ----------------------------------------------------
        # Answer generation
        # ----------------------------------------------------

        grounded_answer = (
            self.generate_answer(
                evidence_package
            )
        )

        # ====================================================
        # Compact final result
        # ====================================================

        final_result = {

            "query":
                query,

            "answer":
                grounded_answer.get(
                    "answer"
                ),

            "complete_match":
                grounded_answer.get(
                    "complete_match"
                ),

            "confidence":
                grounded_answer.get(
                    "confidence"
                ),

            "best_evidence":
                grounded_answer.get(
                    "best_evidence"
                ),

            "supporting_evidence":
                grounded_answer.get(
                    "supporting_evidence",
                    [],
                ),

            "satisfied_requirements":
                grounded_answer.get(
                    "satisfied_requirements",
                    [],
                ),

            "missing_requirements":
                grounded_answer.get(
                    "missing_requirements",
                    [],
                ),

            "violated_requirements":
                grounded_answer.get(
                    "violated_requirements",
                    [],
                ),

            "respected_negative_requirements":
                grounded_answer.get(
                    "respected_negative_requirements",
                    [],
                ),

            "retrieval": {

                "rounds_executed":
                    search_output.get(
                        "rounds_executed"
                    ),

                "stop_reason":
                    search_output.get(
                        "stop_reason"
                    ),

                "selected_round":
                    search_output.get(
                        "selected_round"
                    ),

                "final_sufficient":
                    search_output.get(
                        "final_sufficient"
                    ),

                "anchor_applied":
                    search_output.get(
                        "anchor_applied"
                    ),
            },

            "evidence_references":
                grounded_answer.get(
                    "evidence_references",
                    [],
                ),

            # ------------------------------------------------
            # Preserve structured evidence for auditability.
            #
            # This makes the final output traceable without
            # requiring the enormous raw iterative-search JSON.
            # ------------------------------------------------

            "evidence":
                evidence_package.get(
                    "evidence",
                    [],
                ),

            "pipeline_version":
                "grounded_deepsearch_v1",

            "retrieval_architecture":
                "frozen_iterative_bounded_promotion",

            "retrieval_seed":
                SEED,
        }

        return final_result


# ============================================================
# Pretty-print final result
# ============================================================


def print_final_result(
    result: Dict[str, Any],
):

    print(
        "\n\n"
        + "#"
        * 80
    )

    print(
        "FINAL GROUNDED DEEPSEARCH RESULT"
    )

    print(
        "#"
        * 80
    )

    print(
        "\nQuery:"
    )

    print(
        result.get(
            "query"
        )
    )

    print(
        "\nAnswer:"
    )

    print(
        result.get(
            "answer"
        )
    )

    print(
        "\nComplete match:"
    )

    print(
        result.get(
            "complete_match"
        )
    )

    print(
        "\nConfidence:"
    )

    print(
        result.get(
            "confidence"
        )
    )

    print(
        "\nRetrieval rounds:"
    )

    print(
        result.get(
            "retrieval",
            {},
        ).get(
            "rounds_executed"
        )
    )

    print(
        "\nSelected retrieval mechanism:"
    )

    print(
        result.get(
            "retrieval",
            {},
        ).get(
            "selected_round"
        )
    )

    # --------------------------------------------------------
    # Requirements
    # --------------------------------------------------------

    print(
        "\nSatisfied requirements:"
    )

    satisfied = result.get(
        "satisfied_requirements",
        [],
    )

    if satisfied:

        for item in satisfied:

            print(
                f"  + {item}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nMissing requirements:"
    )

    missing = result.get(
        "missing_requirements",
        [],
    )

    if missing:

        for item in missing:

            print(
                f"  - {item}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nViolated requirements:"
    )

    violated = result.get(
        "violated_requirements",
        [],
    )

    if violated:

        for item in violated:

            print(
                f"  - {item}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nForbidden concepts not detected:"
    )

    respected = result.get(
        "respected_negative_requirements",
        [],
    )

    if respected:

        for item in respected:

            print(
                f"  + {item}"
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # Evidence references
    # --------------------------------------------------------

    print(
        "\nEvidence:"
    )

    references = result.get(
        "evidence_references",
        [],
    )

    for item in references:

        print(
            (
                f"  Evidence "
                f"{item.get('evidence_number')}: "
                f"image_id="
                f"{item.get('image_id')} | "
                f"score="
                f"{item.get('verification_score')} | "
                f"overall_match="
                f"{item.get('overall_match')}"
            )
        )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Agentic Multimodal "
            "DeepSearch pipeline."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help=(
            "Natural-language multimodal search query."
        ),
    )

    parser.add_argument(
        "--output",
        default=
            str(
                DEFAULT_OUTPUT
            ),
        help=(
            "JSON path for the final grounded result."
        ),
    )

    parser.add_argument(
        "--max-evidence-images",
        type=int,
        default=5,
        help=(
            "Number of final retrieval results retained "
            "in the evidence package."
        ),
    )

    parser.add_argument(
        "--max-answer-images",
        type=int,
        default=3,
        help=(
            "Number of evidence images shown to the "
            "multimodal answer generator."
        ),
    )

    args = parser.parse_args()

    system = (
        GroundedDeepSearch(

            max_evidence_images=
                args.max_evidence_images,

            max_answer_images=
                args.max_answer_images,
        )
    )

    result = (
        system.search(
            args.query
        )
    )

    save_json(
        args.output,
        result,
    )

    print_final_result(
        result
    )

    print(
        "\nSaved to:"
    )

    print(
        args.output
    )


if __name__ == "__main__":

    main()
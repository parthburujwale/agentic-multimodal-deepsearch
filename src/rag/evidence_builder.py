import argparse
import json
import os
from typing import Any, Dict, List, Optional


# ============================================================
# Evidence Builder
# ============================================================


class EvidenceBuilder:
    """
    Convert the full Agentic Multimodal DeepSearch output into a
    compact evidence package suitable for grounded multimodal RAG.

    This component does NOT:
        - perform retrieval
        - rerank images
        - change the frozen search results
        - run another model

    It simply converts the final retrieval output into a structured
    representation that can later be consumed by a multimodal answer
    generator.
    """

    def __init__(
        self,
        image_folder: Optional[str] = None,
        max_evidence_images: int = 5,
    ):
        self.image_folder = image_folder
        self.max_evidence_images = max_evidence_images


    # ========================================================
    # Basic helpers
    # ========================================================

    @staticmethod
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


    @staticmethod
    def safe_float(
        value,
        default=0.0,
    ) -> float:
        try:
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return float(
                default
            )


    @staticmethod
    def safe_bool(
        value,
    ) -> bool:
        return bool(
            value
        )


    # ========================================================
    # Image path handling
    # ========================================================

    def build_image_path(
        self,
        file_name: Optional[str],
    ) -> Optional[str]:
        """
        Construct the image path when an image folder is available.
        """

        if not file_name:
            return None

        if self.image_folder is None:
            return file_name

        return os.path.join(
            self.image_folder,
            file_name,
        )


    # ========================================================
    # Verification extraction
    # ========================================================

    def build_positive_evidence(
        self,
        verification: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Preserve all positive verifier checks in a compact form.
        """

        checks = verification.get(
            "positive_checks",
            [],
        )

        evidence = []

        for check in checks:
            if not isinstance(
                check,
                dict,
            ):
                continue

            evidence.append(
                {
                    "constraint":
                        str(
                            check.get(
                                "constraint",
                                "",
                            )
                        ).strip(),

                    "satisfied":
                        self.safe_bool(
                            check.get(
                                "satisfied",
                                False,
                            )
                        ),

                    "confidence":
                        self.safe_float(
                            check.get(
                                "confidence",
                                0.0,
                            )
                        ),
                }
            )

        return evidence


    def build_negative_evidence(
        self,
        verification: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Preserve all negative verifier checks.

        For a negative constraint:
            present=False
        means the forbidden concept was NOT detected.
        """

        checks = verification.get(
            "negative_checks",
            [],
        )

        evidence = []

        for check in checks:
            if not isinstance(
                check,
                dict,
            ):
                continue

            evidence.append(
                {
                    "constraint":
                        str(
                            check.get(
                                "constraint",
                                "",
                            )
                        ).strip(),

                    "present":
                        self.safe_bool(
                            check.get(
                                "present",
                                False,
                            )
                        ),

                    "confidence":
                        self.safe_float(
                            check.get(
                                "confidence",
                                0.0,
                            )
                        ),
                }
            )

        return evidence


    # ========================================================
    # Retrieval support
    # ========================================================

    def build_query_matches(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert raw retrieval-query matches into a compact structure.

        These can later be used to explain why an image entered the
        candidate set.
        """

        raw_matches = result.get(
            "query_matches",
            [],
        )

        query_matches = []

        for match in raw_matches:
            if not isinstance(
                match,
                dict,
            ):
                continue

            query_matches.append(
                {
                    "query":
                        str(
                            match.get(
                                "query",
                                "",
                            )
                        ).strip(),

                    "rank":
                        match.get(
                            "rank"
                        ),

                    "retrieval_score":
                        self.safe_float(
                            match.get(
                                "retrieval_score",
                                0.0,
                            )
                        ),
                }
            )

        return query_matches


    # ========================================================
    # Single evidence item
    # ========================================================

    def build_evidence_item(
        self,
        result: Dict[str, Any],
        rank: int,
    ) -> Dict[str, Any]:
        """
        Build one compact evidence record from one final search result.
        """

        verification = result.get(
            "verification",
            {},
        )

        if not isinstance(
            verification,
            dict,
        ):
            verification = {}

        file_name = result.get(
            "file_name"
        )

        positive_checks = (
            self.build_positive_evidence(
                verification
            )
        )

        negative_checks = (
            self.build_negative_evidence(
                verification
            )
        )

        satisfied_positive = sum(
            1
            for check in positive_checks
            if check.get(
                "satisfied",
                False,
            )
        )

        total_positive = len(
            positive_checks
        )

        negative_violations = sum(
            1
            for check in negative_checks
            if check.get(
                "present",
                False,
            )
        )

        total_negative = len(
            negative_checks
        )

        evidence_item = {
            "rank":
                int(
                    rank
                ),

            "image_index":
                result.get(
                    "image_index"
                ),

            "image_id":
                self.normalize_image_id(
                    result.get(
                        "image_id"
                    )
                ),

            "file_name":
                file_name,

            "image_path":
                self.build_image_path(
                    file_name
                ),

            # -----------------------------------------------
            # Final retrieval / ranking support
            # -----------------------------------------------

            "rrf_rank":
                result.get(
                    "rrf_rank"
                ),

            "rrf_score":
                self.safe_float(
                    result.get(
                        "rrf_score",
                        0.0,
                    )
                ),

            "normalized_rrf_score":
                self.safe_float(
                    result.get(
                        "normalized_rrf_score",
                        0.0,
                    )
                ),

            "final_score":
                self.safe_float(
                    result.get(
                        "final_score",
                        0.0,
                    )
                ),

            "query_matches":
                self.build_query_matches(
                    result
                ),

            # -----------------------------------------------
            # Semantic verification
            # -----------------------------------------------

            "verification": {
                "overall_match":
                    self.safe_bool(
                        verification.get(
                            "overall_match",
                            False,
                        )
                    ),

                "overall_score":
                    self.safe_float(
                        verification.get(
                            "overall_score",
                            0.0,
                        )
                    ),

                "positive_score":
                    self.safe_float(
                        verification.get(
                            "positive_score",
                            0.0,
                        )
                    ),

                "negative_violation_score":
                    self.safe_float(
                        verification.get(
                            "negative_violation_score",
                            0.0,
                        )
                    ),

                "positive_checks":
                    positive_checks,

                "negative_checks":
                    negative_checks,

                "positive_constraints_satisfied":
                    satisfied_positive,

                "positive_constraints_total":
                    total_positive,

                "negative_violations":
                    negative_violations,

                "negative_constraints_total":
                    total_negative,

                "explanation":
                    str(
                        verification.get(
                            "explanation",
                            "",
                        )
                    ).strip(),
            },
        }

        return evidence_item


    # ========================================================
    # Confidence summary
    # ========================================================

    def compute_confidence_summary(
        self,
        evidence: List[Dict[str, Any]],
        final_sufficient: bool,
    ) -> Dict[str, Any]:
        """
        Create a simple non-model confidence summary.

        This is descriptive metadata only.
        It does not change ranking or retrieval decisions.
        """

        if not evidence:
            return {
                "level":
                    "low",

                "score":
                    0.0,

                "reason":
                    "No evidence images are available.",
            }

        top = evidence[
            0
        ]

        verification = top.get(
            "verification",
            {},
        )

        top_score = self.safe_float(
            verification.get(
                "overall_score",
                0.0,
            )
        )

        overall_match = bool(
            verification.get(
                "overall_match",
                False,
            )
        )

        negative_violations = int(
            verification.get(
                "negative_violations",
                0,
            )
        )

        if (
            final_sufficient
            and
            overall_match
            and
            top_score >= 0.85
            and
            negative_violations == 0
        ):
            level = "high"

        elif (
            overall_match
            and
            top_score >= 0.65
            and
            negative_violations == 0
        ):
            level = "medium"

        else:
            level = "low"

        return {
            "level":
                level,

            "score":
                top_score,

            "final_sufficient":
                bool(
                    final_sufficient
                ),

            "top_result_overall_match":
                overall_match,

            "negative_violations":
                negative_violations,
        }


    # ========================================================
    # Full evidence package
    # ========================================================

    def build(
        self,
        search_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert a full IterativeAgenticSearch output into a compact
        evidence package.
        """

        if not isinstance(
            search_output,
            dict,
        ):
            raise TypeError(
                "search_output must be a dictionary."
            )

        original_query = str(
            search_output.get(
                "original_query",
                "",
            )
        ).strip()

        if not original_query:
            raise ValueError(
                "Search output does not contain an original_query."
            )

        raw_results = search_output.get(
            "results",
            [],
        )

        if not isinstance(
            raw_results,
            list,
        ):
            raise TypeError(
                "search_output['results'] must be a list."
            )

        selected_results = raw_results[
            :self.max_evidence_images
        ]

        evidence = []

        for rank, result in enumerate(
            selected_results,
            start=1,
        ):
            if not isinstance(
                result,
                dict,
            ):
                continue

            evidence.append(
                self.build_evidence_item(
                    result=result,
                    rank=rank,
                )
            )

        final_sufficient = bool(
            search_output.get(
                "final_sufficient",
                False,
            )
        )

        package = {
            "query":
                original_query,

            "search_metadata": {
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

                "selected_retrieval_query":
                    search_output.get(
                        "selected_retrieval_query"
                    ),

                "verification_query":
                    search_output.get(
                        "verification_query"
                    ),

                "final_sufficient":
                    final_sufficient,

                "anchor_applied":
                    bool(
                        search_output.get(
                            "anchor_applied",
                            False,
                        )
                    ),
            },

            "constraints": {
                "positive":
                    list(
                        search_output.get(
                            "positive_constraints",
                            [],
                        )
                        or
                        search_output.get(
                            "fixed_positive_constraints",
                            [],
                        )
                        or
                        []
                    ),

                "negative":
                    list(
                        search_output.get(
                            "negative_constraints",
                            [],
                        )
                        or
                        search_output.get(
                            "fixed_negative_constraints",
                            [],
                        )
                        or
                        []
                    ),
            },

            "evidence_count":
                len(
                    evidence
                ),

            "evidence":
                evidence,

            "confidence":
                self.compute_confidence_summary(
                    evidence=evidence,
                    final_sufficient=final_sufficient,
                ),
        }

        # ----------------------------------------------------
        # Convenience pointer to the strongest final evidence
        # ----------------------------------------------------

        if evidence:
            package[
                "primary_evidence"
            ] = evidence[
                0
            ]

        else:
            package[
                "primary_evidence"
            ] = None

        return package


# ============================================================
# CLI helpers
# ============================================================


def load_json(
    path: str,
) -> Dict[str, Any]:
    with open(
        path,
        "r",
    ) as file:
        return json.load(
            file
        )


def save_json(
    path: str,
    data: Dict[str, Any],
):
    output_dir = os.path.dirname(
        path
    )

    if output_dir:
        os.makedirs(
            output_dir,
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
        )


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a grounded multimodal evidence package "
            "from an IterativeAgenticSearch result."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to an IterativeAgenticSearch JSON output."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path where the compact evidence package "
            "should be written."
        ),
    )

    parser.add_argument(
        "--image-folder",
        default=None,
        help=(
            "Optional folder containing the retrieved images."
        ),
    )

    parser.add_argument(
        "--max-evidence-images",
        type=int,
        default=5,
        help=(
            "Maximum number of final ranked images to "
            "include as RAG evidence."
        ),
    )

    args = parser.parse_args()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MULTIMODAL DEEPSEARCH EVIDENCE BUILDER"
    )

    print(
        "=" * 80
    )

    print(
        f"\nInput:"
        f"\n{args.input}"
    )

    print(
        f"\nOutput:"
        f"\n{args.output}"
    )

    search_output = load_json(
        args.input
    )

    builder = EvidenceBuilder(
        image_folder=
            args.image_folder,

        max_evidence_images=
            args.max_evidence_images,
    )

    package = builder.build(
        search_output
    )

    save_json(
        args.output,
        package,
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "EVIDENCE PACKAGE"
    )

    print(
        "-" * 80
    )

    print(
        f"Query: "
        f"{package['query']}"
    )

    print(
        f"Evidence images: "
        f"{package['evidence_count']}"
    )

    print(
        f"Confidence: "
        f"{package['confidence']['level']}"
    )

    primary = package.get(
        "primary_evidence"
    )

    if primary is not None:
        print(
            f"Primary image ID: "
            f"{primary['image_id']}"
        )

        print(
            f"Primary verifier score: "
            f"{primary['verification']['overall_score']:.4f}"
        )

        print(
            f"Primary overall match: "
            f"{primary['verification']['overall_match']}"
        )

        print(
            "\nVerifier explanation:"
        )

        print(
            primary[
                "verification"
            ][
                "explanation"
            ]
        )

    print(
        "\nSaved evidence package to:"
    )

    print(
        args.output
    )


if __name__ == "__main__":
    main()
    
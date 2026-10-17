import os


# ============================================================
# Deterministic CUDA configuration
# ============================================================

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)


import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional


sys.path.append("..")


from agents.reproducibility import (
    configure_reproducibility,
    reset_random_state,
)

from agents.verified_agentic_search import (
    VerifiedAgenticSearch,
)

from agents.evidence_sufficiency import (
    EvidenceSufficiencyChecker,
)

from agents.query_refiner import (
    QueryRefiner,
)


# ============================================================
# Iterative Agentic Multimodal DeepSearch
#
# ROUND 1
#   full original objective
#        |
#        v
#   verified ranking
#        |
#        v
# uncertainty / ambiguity router
#        |
#        v
# ROUND 2
#   evidence-seeking retrieval
#        |
#        v
# cross-round evidence memory
#        |
#        v
# BOUNDED PROMOTION
#
#
# IMPORTANT:
#
# Round 1 is the conservative base ranking.
#
# Round 2 is NOT allowed to globally rerank all candidates.
#
# It may only promote the original Round-1 rank-2 candidate
# to rank 1 when strong corroborating evidence is present.
#
# Verification always uses the original semantic objective.
# ============================================================


class IterativeAgenticSearch:

    def __init__(
        self,
        image_folder: str,
        captions_file: str,
        retriever_checkpoint_path: str,
        retriever_processor_path: str,
        planner_model_name: str = (
            "Qwen/Qwen2.5-1.5B-Instruct"
        ),
        verifier_model_name: str = (
            "Qwen/Qwen2.5-VL-3B-Instruct"
        ),
        refiner_model_name: str = (
            "Qwen/Qwen2.5-1.5B-Instruct"
        ),
        retrieval_weight: float = 0.25,
        verification_weight: float = 0.75,
        original_query_weight: float = 2.0,
        subquery_weight: float = 1.0,
        min_verification_score: float = 0.75,
        max_rounds: int = 2,
        router_score_threshold: float = 0.85,
        router_score_tolerance: float = 1e-6,
        router_tie_tolerance: float = 0.01,
        seed: int = 42,

        # ----------------------------------------------------
        # Diagnostic cross-round memory configuration
        # ----------------------------------------------------

        fusion_enabled: bool = True,

        fusion_verification_weight: float = 0.70,

        fusion_retrieval_weight: float = 0.30,

        round_1_support_weight: float = 1.0,

        round_2_support_weight: float = 0.70,

        fusion_rrf_k: int = 10,

        # ----------------------------------------------------
        # Bounded promotion configuration
        # ----------------------------------------------------

        promotion_enabled: bool = True,

        promotion_max_verifier_gap: float = 0.20,
    ):

        if max_rounds < 1:

            raise ValueError(
                "max_rounds must be at least 1."
            )


        self.seed = int(
            seed
        )


        self.max_rounds = int(
            max_rounds
        )


        self.router_score_threshold = float(
            router_score_threshold
        )


        self.router_score_tolerance = float(
            router_score_tolerance
        )


        self.router_tie_tolerance = float(
            router_tie_tolerance
        )


        # ====================================================
        # Diagnostic fusion-memory configuration
        # ====================================================

        self.fusion_enabled = bool(
            fusion_enabled
        )


        fusion_weight_sum = (

            float(
                fusion_verification_weight
            )

            +

            float(
                fusion_retrieval_weight
            )
        )


        if fusion_weight_sum <= 0:

            raise ValueError(
                "Fusion diagnostic weights must sum "
                "to a positive value."
            )


        self.fusion_verification_weight = (

            float(
                fusion_verification_weight
            )

            /
            fusion_weight_sum
        )


        self.fusion_retrieval_weight = (

            float(
                fusion_retrieval_weight
            )

            /
            fusion_weight_sum
        )


        self.round_1_support_weight = float(
            round_1_support_weight
        )


        self.round_2_support_weight = float(
            round_2_support_weight
        )


        self.fusion_rrf_k = int(
            fusion_rrf_k
        )


        # ====================================================
        # Bounded promotion configuration
        # ====================================================

        self.promotion_enabled = bool(
            promotion_enabled
        )


        self.promotion_max_verifier_gap = float(
            promotion_max_verifier_gap
        )


        configure_reproducibility(
            seed=self.seed
        )


        print(
            "\n"
            + "#" * 70
        )


        print(
            "INITIALIZING ITERATIVE AGENTIC SEARCH"
        )


        print(
            "#" * 70
        )


        print(
            "\nReproducibility:"
        )


        print(
            f"  seed: {self.seed}"
        )


        print(
            "  deterministic algorithms: enabled"
        )


        print(
            "  eager attention: enabled"
        )


        print(
            "  TF32: disabled"
        )


        # ====================================================
        # Verified search system
        # ====================================================

        self.search_system = (
            VerifiedAgenticSearch(

                image_folder=
                    image_folder,

                captions_file=
                    captions_file,

                retriever_checkpoint_path=
                    retriever_checkpoint_path,

                retriever_processor_path=
                    retriever_processor_path,

                planner_model_name=
                    planner_model_name,

                verifier_model_name=
                    verifier_model_name,

                retrieval_weight=
                    retrieval_weight,

                verification_weight=
                    verification_weight,

                original_query_weight=
                    original_query_weight,

                subquery_weight=
                    subquery_weight,
            )
        )


        # ====================================================
        # Sufficiency checker
        # ====================================================

        print(
            "\nInitializing evidence "
            "sufficiency checker..."
        )


        self.sufficiency_checker = (
            EvidenceSufficiencyChecker(

                min_verification_score=
                    min_verification_score
            )
        )


        # ====================================================
        # Evidence-seeking refiner
        # ====================================================

        print(
            "\nInitializing evidence-seeking "
            "query refiner..."
        )


        reset_random_state(
            self.seed
        )


        self.query_refiner = (
            QueryRefiner(

                model_name=
                    refiner_model_name
            )
        )


        print(
            "\nBounded promotion:"
        )


        print(
            f"  enabled: "
            f"{self.promotion_enabled}"
        )


        print(
            f"  maximum verifier gap: "
            f"{self.promotion_max_verifier_gap:.4f}"
        )


        print(
            "\nCross-round memory:"
        )


        print(
            "  retained for diagnostics"
        )


        print(
            f"  Round-1 support weight: "
            f"{self.round_1_support_weight:.4f}"
        )


        print(
            f"  Round-2 support weight: "
            f"{self.round_2_support_weight:.4f}"
        )


        print(
            "\n"
            + "#" * 70
        )


        print(
            "ITERATIVE AGENTIC SEARCH READY"
        )


        print(
            "#" * 70
        )


    # ========================================================
    # RNG reset
    # ========================================================

    def reset_rng(
        self,
    ):

        reset_random_state(
            self.seed
        )


    # ========================================================
    # Image ID helper
    # ========================================================

    @staticmethod
    def normalize_image_id(
        image_id,
    ):

        if image_id is None:

            return None


        try:

            return int(
                image_id
            )

        except (
            TypeError,
            ValueError,
        ):

            return str(
                image_id
            ).strip()


    # ========================================================
    # Safe bool
    # ========================================================

    @staticmethod
    def safe_bool(
        value,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):

            return value


        if isinstance(
            value,
            str,
        ):

            normalized = (
                value
                .strip()
                .lower()
            )


            if normalized in {
                "true",
                "1",
                "yes",
            }:

                return True


            if normalized in {
                "false",
                "0",
                "no",
                "",
            }:

                return False


        return bool(
            value
        )


    # ========================================================
    # Top result
    # ========================================================

    @staticmethod
    def get_top_result(
        search_output: Dict[str, Any],
    ) -> Optional[
        Dict[str, Any]
    ]:

        results = (
            search_output.get(
                "results",
                [],
            )
        )


        if not results:

            return None


        return results[
            0
        ]


    # ========================================================
    # Find result by image ID
    # ========================================================

    def find_result_by_image_id(
        self,
        results: List[
            Dict[str, Any]
        ],
        image_id,
    ) -> Optional[
        Dict[str, Any]
    ]:

        target = (
            self.normalize_image_id(
                image_id
            )
        )


        for result in results:

            candidate = (
                self.normalize_image_id(
                    result.get(
                        "image_id"
                    )
                )
            )


            if candidate == target:

                return result


        return None


    # ========================================================
    # Find rank
    # ========================================================

    def find_rank(
        self,
        results: List[
            Dict[str, Any]
        ],
        image_id,
    ) -> Optional[int]:

        target = (
            self.normalize_image_id(
                image_id
            )
        )


        for rank, result in enumerate(
            results,
            start=1,
        ):

            candidate = (
                self.normalize_image_id(
                    result.get(
                        "image_id"
                    )
                )
            )


            if candidate == target:

                return rank


        return None


    # ========================================================
    # Baseline candidate
    # ========================================================

    def find_baseline_candidate(
        self,
        search_output: Dict[str, Any],
    ) -> Optional[
        Dict[str, Any]
    ]:

        baseline_image_id = (
            self.normalize_image_id(
                search_output.get(
                    "baseline_top1_image_id"
                )
            )
        )


        if baseline_image_id is None:

            return None


        return (
            self.find_result_by_image_id(

                search_output.get(
                    "results",
                    [],
                ),

                baseline_image_id,
            )
        )


    # ========================================================
    # Compact result summary
    # ========================================================

    def summarize_result(
        self,
        result: Optional[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        if result is None:

            return {

                "image_id":
                    None,

                "file_name":
                    None,

                "final_score":
                    0.0,

                "overall_match":
                    False,

                "verification_score":
                    0.0,
            }


        verification = (
            result.get(
                "verification",
                {},
            )
        )


        return {

            "image_id":
                result.get(
                    "image_id"
                ),

            "file_name":
                result.get(
                    "file_name"
                ),

            "final_score":
                float(
                    result.get(
                        "final_score",
                        0.0,
                    )
                ),

            "overall_match":
                self.safe_bool(
                    verification.get(
                        "overall_match",
                        False,
                    )
                ),

            "verification_score":
                float(
                    verification.get(
                        "overall_score",
                        0.0,
                    )
                ),
        }


    # ========================================================
    # Sufficiency dictionary
    # ========================================================

    @staticmethod
    def sufficiency_to_dict(
        decision,
    ) -> Dict[str, Any]:

        return {

            "sufficient":
                decision.sufficient,

            "reason":
                decision.reason,

            "overall_match":
                decision.overall_match,

            "verification_score":
                decision.verification_score,

            "satisfied_positive_constraints":
                decision
                .satisfied_positive_constraints,

            "total_positive_constraints":
                decision
                .total_positive_constraints,

            "violated_negative_constraints":
                decision
                .violated_negative_constraints,

            "total_negative_constraints":
                decision
                .total_negative_constraints,

            "missing_positive_constraints":
                decision
                .missing_positive_constraints,

            "violated_negative_constraint_names":
                decision
                .violated_negative_constraint_names,
        }


    # ========================================================
    # Check explicit negative violation
    # ========================================================

    def has_negative_violation(
        self,
        result: Optional[
            Dict[str, Any]
        ],
    ) -> bool:

        if result is None:

            return True


        verification = (
            result.get(
                "verification",
                {},
            )
        )


        try:

            violation_score = float(
                verification.get(
                    "negative_violation_score",
                    0.0,
                )
            )

        except Exception:

            violation_score = 0.0


        if violation_score > 0.0:

            return True


        negative_checks = (
            verification.get(
                "negative_checks",
                [],
            )
        )


        if isinstance(
            negative_checks,
            list,
        ):

            for check in negative_checks:

                if not isinstance(
                    check,
                    dict,
                ):

                    continue


                if self.safe_bool(
                    check.get(
                        "present",
                        False,
                    )
                ):

                    return True


        return False


    # ========================================================
    # Baseline-aware router
    # ========================================================

    def evaluate_iteration_router(
        self,
        search_output: Dict[str, Any],
    ) -> Dict[str, Any]:

        results = (
            search_output.get(
                "results",
                [],
            )
        )


        if not results:

            return {

                "trigger_round_2":
                    True,

                "reasons":
                    [
                        "no_verified_candidates"
                    ],

                "winner_image_id":
                    None,

                "winner_overall_match":
                    False,

                "winner_verification_score":
                    0.0,

                "baseline_top1_image_id":
                    search_output.get(
                        "baseline_top1_image_id"
                    ),

                "baseline_candidate_present":
                    False,

                "baseline_candidate_overall_match":
                    False,

                "baseline_candidate_verification_score":
                    None,

                "baseline_winner_disagreement":
                    False,

                "baseline_verifier_tie":
                    False,
            }


        winner = (
            results[
                0
            ]
        )


        winner_verification = (
            winner.get(
                "verification",
                {},
            )
        )


        winner_score = float(
            winner_verification.get(
                "overall_score",
                0.0,
            )
        )


        winner_match = (
            self.safe_bool(
                winner_verification.get(
                    "overall_match",
                    False,
                )
            )
        )


        winner_image_id = (
            self.normalize_image_id(
                winner.get(
                    "image_id"
                )
            )
        )


        baseline_top1_image_id = (
            self.normalize_image_id(
                search_output.get(
                    "baseline_top1_image_id"
                )
            )
        )


        baseline_candidate = (
            self.find_baseline_candidate(
                search_output
            )
        )


        baseline_candidate_present = (
            baseline_candidate
            is not None
        )


        incomplete_match = (
            not winner_match
        )


        low_score = (

            winner_score

            <=

            (
                self.router_score_threshold
                +
                self.router_score_tolerance
            )
        )


        baseline_missing = (
            not baseline_candidate_present
        )


        baseline_score = None

        baseline_match = False

        baseline_winner_disagreement = False

        baseline_verifier_tie = False

        tied_disagreement = False


        if baseline_candidate is not None:

            baseline_verification = (
                baseline_candidate.get(
                    "verification",
                    {},
                )
            )


            baseline_score = float(
                baseline_verification.get(
                    "overall_score",
                    0.0,
                )
            )


            baseline_match = (
                self.safe_bool(
                    baseline_verification.get(
                        "overall_match",
                        False,
                    )
                )
            )


            baseline_image_id = (
                self.normalize_image_id(
                    baseline_candidate.get(
                        "image_id"
                    )
                )
            )


            baseline_winner_disagreement = (

                baseline_image_id
                !=
                winner_image_id
            )


            baseline_verifier_tie = (

                abs(
                    baseline_score
                    -
                    winner_score
                )

                <=

                self.router_tie_tolerance
            )


            tied_disagreement = (

                baseline_winner_disagreement

                and

                baseline_match

                and

                winner_match

                and

                baseline_verifier_tie
            )


        reasons = []


        if incomplete_match:

            reasons.append(
                "incomplete_semantic_match"
            )


        if low_score:

            reasons.append(
                "low_verification_score"
            )


        if baseline_missing:

            reasons.append(
                "baseline_top1_missing"
            )


        if tied_disagreement:

            reasons.append(
                "baseline_agentic_tied_disagreement"
            )


        return {

            "trigger_round_2":
                bool(
                    reasons
                ),

            "reasons":
                reasons,

            "winner_image_id":
                winner_image_id,

            "winner_overall_match":
                winner_match,

            "winner_verification_score":
                winner_score,

            "baseline_top1_image_id":
                baseline_top1_image_id,

            "baseline_candidate_present":
                baseline_candidate_present,

            "baseline_candidate_overall_match":
                baseline_match,

            "baseline_candidate_verification_score":
                baseline_score,

            "baseline_winner_disagreement":
                baseline_winner_disagreement,

            "baseline_verifier_tie":
                baseline_verifier_tie,
        }


    # ========================================================
    # Router display
    # ========================================================

    def print_router_decision(
        self,
        router: Dict[str, Any],
    ):

        print(
            "\n"
            + "=" * 70
        )


        print(
            "BASELINE-AWARE ITERATION ROUTER"
        )


        print(
            "=" * 70
        )


        print(
            f"Trigger another round: "
            f"{router['trigger_round_2']}"
        )


        print(
            f"Winner image ID: "
            f"{router['winner_image_id']}"
        )


        print(
            f"Winner overall match: "
            f"{router['winner_overall_match']}"
        )


        print(
            f"Winner verifier score: "
            f"{router['winner_verification_score']:.4f}"
        )


        print(
            f"\nDeepSearch V2 top-1 image ID: "
            f"{router['baseline_top1_image_id']}"
        )


        print(
            f"Baseline candidate present: "
            f"{router['baseline_candidate_present']}"
        )


        baseline_score = (
            router.get(
                "baseline_candidate_verification_score"
            )
        )


        if baseline_score is not None:

            print(
                f"Baseline verifier score: "
                f"{baseline_score:.4f}"
            )


            print(
                f"Baseline overall match: "
                f"{router['baseline_candidate_overall_match']}"
            )


        print(
            f"Baseline/winner disagreement: "
            f"{router['baseline_winner_disagreement']}"
        )


        print(
            f"Verifier tie: "
            f"{router['baseline_verifier_tie']}"
        )


        print(
            "\nRouter reasons:"
        )


        if router[
            "reasons"
        ]:

            for reason in (
                router[
                    "reasons"
                ]
            ):

                print(
                    f"  - {reason}"
                )

        else:

            print(
                "  None"
            )


        print(
            "=" * 70
        )


    # ========================================================
    # Refinement strategy
    # ========================================================

    @staticmethod
    def choose_refinement_mode(
        sufficiency: Dict[str, Any],
        router: Dict[str, Any],
    ) -> str:

        missing_positive = (
            sufficiency.get(
                "missing_positive_constraints",
                [],
            )
        )


        violated_negative = (
            sufficiency.get(
                "violated_negative_constraint_names",
                [],
            )
        )


        if (
            missing_positive
            or
            violated_negative
        ):

            return "missing_evidence"


        return "ambiguity_resolution"


    # ========================================================
    # Run one search round
    # ========================================================

    def run_round(
        self,
        round_number: int,
        retrieval_query: str,
        original_query: str,
        original_plan: Dict[str, Any],
        retrieval_top_k: int,
        final_top_k: int,
        rrf_k: int,
    ) -> Dict[str, Any]:

        print(
            "\n\n"
            + "#" * 70
        )


        print(
            f"ITERATIVE SEARCH ROUND "
            f"{round_number}"
        )


        print(
            "#" * 70
        )


        print(
            "\nRetrieval query:"
        )


        print(
            retrieval_query
        )


        print(
            "\nFixed verification objective:"
        )


        print(
            original_query
        )


        internal_final_top_k = max(
            final_top_k,
            retrieval_top_k,
        )


        self.reset_rng()


        search_output = (
            self.search_system.search(

                query=
                    retrieval_query,

                retrieval_top_k=
                    retrieval_top_k,

                final_top_k=
                    internal_final_top_k,

                rrf_k=
                    rrf_k,

                verification_query=
                    original_query,

                verification_plan=
                    original_plan,
            )
        )


        top_result = (
            self.get_top_result(
                search_output
            )
        )


        if top_result is None:

            sufficiency = {

                "sufficient":
                    False,

                "reason":
                    (
                        "No retrieval candidates "
                        "were returned."
                    ),

                "overall_match":
                    False,

                "verification_score":
                    0.0,

                "satisfied_positive_constraints":
                    0,

                "total_positive_constraints":
                    0,

                "violated_negative_constraints":
                    0,

                "total_negative_constraints":
                    0,

                "missing_positive_constraints":
                    [],

                "violated_negative_constraint_names":
                    [],
            }


            router = (
                self.evaluate_iteration_router(
                    search_output
                )
            )


            return {

                "round":
                    round_number,

                "retrieval_query":
                    retrieval_query,

                "verification_query":
                    original_query,

                "search_output":
                    search_output,

                "top_result":
                    None,

                "sufficiency":
                    sufficiency,

                "router":
                    router,
            }


        decision = (
            self.sufficiency_checker.evaluate(
                top_result
            )
        )


        print(
            "\n"
            + "=" * 70
        )


        print(
            f"ROUND {round_number} "
            f"EVIDENCE SUFFICIENCY"
        )


        print(
            "=" * 70
        )


        self.sufficiency_checker.print_decision(
            decision
        )


        router = (
            self.evaluate_iteration_router(
                search_output
            )
        )


        self.print_router_decision(
            router
        )


        return {

            "round":
                round_number,

            "retrieval_query":
                retrieval_query,

            "verification_query":
                original_query,

            "search_output":
                search_output,

            "top_result":
                top_result,

            "sufficiency":
                self.sufficiency_to_dict(
                    decision
                ),

            "router":
                router,
        }


    # ========================================================
    # Diagnostic cross-round support
    # ========================================================

    def calculate_round_support(
        self,
        rank: int,
        round_number: int,
    ) -> float:

        if rank <= 0:

            return 0.0


        if round_number == 1:

            round_weight = (
                self.round_1_support_weight
            )

        else:

            round_weight = (
                self.round_2_support_weight
            )


        return (

            round_weight

            /

            (
                self.fusion_rrf_k
                +
                rank
            )
        )


    # ========================================================
    # Normalize support
    # ========================================================

    @staticmethod
    def minmax_normalize(
        values: List[float],
    ) -> List[float]:

        if not values:

            return []


        minimum = min(
            values
        )

        maximum = max(
            values
        )


        if math.isclose(
            minimum,
            maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):

            return [
                1.0
                for _ in values
            ]


        denominator = (
            maximum
            -
            minimum
        )


        return [
            (
                value
                -
                minimum
            )
            /
            denominator

            for value in values
        ]


    # ========================================================
    # Cross-round candidate memory
    #
    # IMPORTANT:
    # This memory is now DIAGNOSTIC.
    #
    # It does not directly determine final ordering.
    # ========================================================

    def build_candidate_memory(
        self,
        rounds: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:

        memory = {}


        for round_record in rounds:

            round_number = int(
                round_record[
                    "round"
                ]
            )


            results = (
                round_record[
                    "search_output"
                ].get(
                    "results",
                    [],
                )
            )


            for rank, result in enumerate(
                results,
                start=1,
            ):

                image_id = (
                    self.normalize_image_id(
                        result.get(
                            "image_id"
                        )
                    )
                )


                if image_id is None:

                    continue


                verification = (
                    result.get(
                        "verification",
                        {},
                    )
                )


                verification_score = float(
                    verification.get(
                        "overall_score",
                        0.0,
                    )
                )


                overall_match = (
                    self.safe_bool(
                        verification.get(
                            "overall_match",
                            False,
                        )
                    )
                )


                support = (
                    self.calculate_round_support(

                        rank=
                            rank,

                        round_number=
                            round_number,
                    )
                )


                if image_id not in memory:

                    memory[
                        image_id
                    ] = {

                        "image_id":
                            image_id,

                        "file_name":
                            result.get(
                                "file_name"
                            ),

                        "representative_result":
                            result,

                        "round_ranks":
                            {},

                        "round_final_scores":
                            {},

                        "round_verification_scores":
                            {},

                        "round_overall_matches":
                            {},

                        "retrieval_support_raw":
                            0.0,

                        "best_verification_score":
                            verification_score,

                        "best_overall_match":
                            overall_match,

                        "best_round":
                            round_number,
                    }


                item = memory[
                    image_id
                ]


                item[
                    "round_ranks"
                ][
                    str(
                        round_number
                    )
                ] = rank


                item[
                    "round_final_scores"
                ][
                    str(
                        round_number
                    )
                ] = float(
                    result.get(
                        "final_score",
                        0.0,
                    )
                )


                item[
                    "round_verification_scores"
                ][
                    str(
                        round_number
                    )
                ] = (
                    verification_score
                )


                item[
                    "round_overall_matches"
                ][
                    str(
                        round_number
                    )
                ] = (
                    overall_match
                )


                item[
                    "retrieval_support_raw"
                ] += (
                    support
                )


                if (
                    verification_score
                    >
                    item[
                        "best_verification_score"
                    ]
                ):

                    item[
                        "best_verification_score"
                    ] = (
                        verification_score
                    )


                    item[
                        "best_overall_match"
                    ] = (
                        overall_match
                    )


                    item[
                        "best_round"
                    ] = (
                        round_number
                    )


                    item[
                        "representative_result"
                    ] = (
                        result
                    )


        candidates = list(
            memory.values()
        )


        raw_supports = [
            item[
                "retrieval_support_raw"
            ]
            for item in candidates
        ]


        normalized_supports = (
            self.minmax_normalize(
                raw_supports
            )
        )


        for item, normalized_support in zip(
            candidates,
            normalized_supports,
        ):

            item[
                "retrieval_support"
            ] = float(
                normalized_support
            )


            item[
                "fusion_score"
            ] = (

                self.fusion_verification_weight

                *

                float(
                    item[
                        "best_verification_score"
                    ]
                )

                +

                self.fusion_retrieval_weight

                *

                normalized_support
            )


            item[
                "round_count"
            ] = len(
                item[
                    "round_ranks"
                ]
            )


        # ----------------------------------------------------
        # Diagnostic ordering only
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: (
                float(
                    item[
                        "fusion_score"
                    ]
                ),
                int(
                    item[
                        "round_count"
                    ]
                ),
                float(
                    item[
                        "retrieval_support"
                    ]
                ),
                str(
                    item[
                        "image_id"
                    ]
                ),
            ),
            reverse=True,
        )


        return candidates


    # ========================================================
    # Bounded promotion decision
    #
    # Only R1 rank 2 can challenge R1 rank 1.
    # ========================================================

    def evaluate_bounded_promotion(
        self,
        round_1_results: List[
            Dict[str, Any]
        ],
        round_2_results: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        decision = {

            "promotion_enabled":
                self.promotion_enabled,

            "promote":
                False,

            "reason":
                None,

            "round_1_winner_image_id":
                None,

            "challenger_image_id":
                None,

            "challenger_round_1_rank":
                None,

            "challenger_round_2_rank":
                None,

            "round_1_winner_round_2_rank":
                None,

            "winner_verification_score":
                None,

            "challenger_verification_score":
                None,

            "verifier_gap":
                None,

            "max_allowed_verifier_gap":
                self.promotion_max_verifier_gap,

            "challenger_negative_violation":
                None,
        }


        if not self.promotion_enabled:

            decision[
                "reason"
            ] = (
                "promotion_disabled"
            )

            return decision


        if len(
            round_1_results
        ) < 2:

            decision[
                "reason"
            ] = (
                "round_1_has_fewer_than_two_candidates"
            )

            return decision


        if not round_2_results:

            decision[
                "reason"
            ] = (
                "round_2_has_no_candidates"
            )

            return decision


        round_1_winner = (
            round_1_results[
                0
            ]
        )


        challenger = (
            round_1_results[
                1
            ]
        )


        winner_image_id = (
            self.normalize_image_id(
                round_1_winner.get(
                    "image_id"
                )
            )
        )


        challenger_image_id = (
            self.normalize_image_id(
                challenger.get(
                    "image_id"
                )
            )
        )


        decision[
            "round_1_winner_image_id"
        ] = winner_image_id


        decision[
            "challenger_image_id"
        ] = challenger_image_id


        decision[
            "challenger_round_1_rank"
        ] = 2


        challenger_round_2_rank = (
            self.find_rank(
                round_2_results,
                challenger_image_id,
            )
        )


        winner_round_2_rank = (
            self.find_rank(
                round_2_results,
                winner_image_id,
            )
        )


        decision[
            "challenger_round_2_rank"
        ] = challenger_round_2_rank


        decision[
            "round_1_winner_round_2_rank"
        ] = winner_round_2_rank


        # ----------------------------------------------------
        # Rule 1:
        # R1 rank-2 must become R2 rank-1.
        # ----------------------------------------------------

        if challenger_round_2_rank != 1:

            decision[
                "reason"
            ] = (
                "challenger_did_not_become_round_2_rank_1"
            )

            return decision


        # ----------------------------------------------------
        # Rule 2:
        # Original R1 winner must disappear from R2 top-k.
        # ----------------------------------------------------

        if winner_round_2_rank is not None:

            decision[
                "reason"
            ] = (
                "round_1_winner_still_present_in_round_2"
            )

            return decision


        # ----------------------------------------------------
        # Use the challenger's Round-1 verification score
        # because Round 1 evaluates it in the conservative
        # original ranking context.
        #
        # Round 2 still uses the same semantic objective, but
        # this avoids arbitrary score replacement.
        # ----------------------------------------------------

        winner_verification = (
            round_1_winner.get(
                "verification",
                {},
            )
        )


        challenger_verification = (
            challenger.get(
                "verification",
                {},
            )
        )


        winner_score = float(
            winner_verification.get(
                "overall_score",
                0.0,
            )
        )


        challenger_score = float(
            challenger_verification.get(
                "overall_score",
                0.0,
            )
        )


        verifier_gap = (
            winner_score
            -
            challenger_score
        )


        decision[
            "winner_verification_score"
        ] = winner_score


        decision[
            "challenger_verification_score"
        ] = challenger_score


        decision[
            "verifier_gap"
        ] = verifier_gap


        # ----------------------------------------------------
        # Rule 3:
        # Challenger cannot be dramatically worse according
        # to semantic verification.
        #
        # A negative gap means challenger is actually better.
        # ----------------------------------------------------

        if (
            verifier_gap
            >
            self.promotion_max_verifier_gap
        ):

            decision[
                "reason"
            ] = (
                "verifier_gap_too_large"
            )

            return decision


        # ----------------------------------------------------
        # Rule 4:
        # Explicit negative-constraint violations block
        # promotion.
        # ----------------------------------------------------

        negative_violation = (
            self.has_negative_violation(
                challenger
            )
        )


        decision[
            "challenger_negative_violation"
        ] = (
            negative_violation
        )


        if negative_violation:

            decision[
                "reason"
            ] = (
                "challenger_has_negative_constraint_violation"
            )

            return decision


        decision[
            "promote"
        ] = True


        decision[
            "reason"
        ] = (
            "strong_cross_round_rank2_to_rank1_corroboration"
        )


        return decision


    # ========================================================
    # Apply bounded promotion
    # ========================================================

    def apply_bounded_promotion(
        self,
        round_1_results: List[
            Dict[str, Any]
        ],
        round_2_results: List[
            Dict[str, Any]
        ],
    ):

        decision = (
            self.evaluate_bounded_promotion(

                round_1_results=
                    round_1_results,

                round_2_results=
                    round_2_results,
            )
        )


        # ----------------------------------------------------
        # Round 1 remains the base ranking.
        # ----------------------------------------------------

        final_results = [
            dict(
                result
            )
            for result in round_1_results
        ]


        if (
            decision[
                "promote"
            ]
        ):

            # ------------------------------------------------
            # Swap only positions 1 and 2.
            #
            # Nothing else is reordered.
            # ------------------------------------------------

            final_results[
                0
            ], final_results[
                1
            ] = (
                final_results[
                    1
                ],
                final_results[
                    0
                ],
            )


        return (
            final_results,
            decision,
        )


    # ========================================================
    # Print promotion decision
    # ========================================================

    def print_promotion_decision(
        self,
        decision: Dict[str, Any],
    ):

        print(
            "\n"
            + "#" * 70
        )


        print(
            "BOUNDED CROSS-ROUND PROMOTION"
        )


        print(
            "#" * 70
        )


        print(
            f"\nPromote challenger: "
            f"{decision['promote']}"
        )


        print(
            f"Reason: "
            f"{decision['reason']}"
        )


        print(
            f"\nRound-1 winner: "
            f"{decision['round_1_winner_image_id']}"
        )


        print(
            f"Round-1 rank-2 challenger: "
            f"{decision['challenger_image_id']}"
        )


        print(
            f"Challenger Round-2 rank: "
            f"{decision['challenger_round_2_rank']}"
        )


        print(
            f"Round-1 winner Round-2 rank: "
            f"{decision['round_1_winner_round_2_rank']}"
        )


        if (
            decision[
                "winner_verification_score"
            ]
            is not None
        ):

            print(
                f"\nWinner verifier: "
                f"{decision['winner_verification_score']:.4f}"
            )


        if (
            decision[
                "challenger_verification_score"
            ]
            is not None
        ):

            print(
                f"Challenger verifier: "
                f"{decision['challenger_verification_score']:.4f}"
            )


        if (
            decision[
                "verifier_gap"
            ]
            is not None
        ):

            print(
                f"Verifier gap: "
                f"{decision['verifier_gap']:.4f}"
            )


        print(
            f"Maximum allowed verifier gap: "
            f"{decision['max_allowed_verifier_gap']:.4f}"
        )


        print(
            f"Negative violation: "
            f"{decision['challenger_negative_violation']}"
        )


    # ========================================================
    # Main iterative search
    # ========================================================

    def search(
        self,
        query: str,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:

        original_query = (
            query
        )


        current_retrieval_query = (
            query
        )


        # ====================================================
        # Fixed original objective
        # ====================================================

        print(
            "\n"
            + "#" * 70
        )


        print(
            "BUILDING FIXED ORIGINAL SEARCH OBJECTIVE"
        )


        print(
            "#" * 70
        )


        self.reset_rng()


        original_plan = (
            self.search_system
            .agentic_retriever
            .planner
            .plan(
                original_query
            )
        )


        print(
            "\nFixed original query:"
        )


        print(
            original_query
        )


        print(
            "\nFixed original plan:"
        )


        print(
            json.dumps(
                original_plan,
                indent=2,
            )
        )


        (
            original_positive_constraints,
            original_negative_constraints,
        ) = (
            self.search_system
            .build_verification_constraints(
                original_plan
            )
        )


        rounds: List[
            Dict[str, Any]
        ] = []


        stop_reason = None


        # ====================================================
        # Iterative loop
        # ====================================================

        for round_number in range(
            1,
            self.max_rounds + 1,
        ):

            round_record = (
                self.run_round(

                    round_number=
                        round_number,

                    retrieval_query=
                        current_retrieval_query,

                    original_query=
                        original_query,

                    original_plan=
                        original_plan,

                    retrieval_top_k=
                        retrieval_top_k,

                    final_top_k=
                        final_top_k,

                    rrf_k=
                        rrf_k,
                )
            )


            rounds.append(
                round_record
            )


            sufficiency = (
                round_record[
                    "sufficiency"
                ]
            )


            router = (
                round_record[
                    "router"
                ]
            )


            if (
                round_number
                >=
                self.max_rounds
            ):

                if sufficiency[
                    "sufficient"
                ]:

                    stop_reason = (
                        "sufficient_evidence_at_max_round"
                    )

                else:

                    stop_reason = (
                        "maximum_rounds_reached"
                    )


                break


            if (
                sufficiency[
                    "sufficient"
                ]

                and

                not router[
                    "trigger_round_2"
                ]
            ):

                stop_reason = (
                    "sufficient_evidence"
                )


                print(
                    "\n"
                    + "#" * 70
                )


                print(
                    "SUFFICIENT AND STABLE "
                    "EVIDENCE FOUND"
                )


                print(
                    f"Stopping after round "
                    f"{round_number}."
                )


                print(
                    "#" * 70
                )


                break


            # =================================================
            # Evidence-seeking refinement
            # =================================================

            missing_positive = list(
                sufficiency.get(
                    "missing_positive_constraints",
                    [],
                )
            )


            violated_negative = list(
                sufficiency.get(
                    "violated_negative_constraint_names",
                    [],
                )
            )


            refinement_mode = (
                self.choose_refinement_mode(

                    sufficiency=
                        sufficiency,

                    router=
                        router,
                )
            )


            winner = (
                round_record[
                    "top_result"
                ]
            )


            winner_verification = {}


            if winner is not None:

                winner_verification = (
                    winner.get(
                        "verification",
                        {},
                    )
                )


            baseline_candidate = (
                self.find_baseline_candidate(
                    round_record[
                        "search_output"
                    ]
                )
            )


            baseline_verification = {}


            if baseline_candidate is not None:

                baseline_verification = (
                    baseline_candidate.get(
                        "verification",
                        {},
                    )
                )


            print(
                "\n"
                + "=" * 70
            )


            print(
                "ROUND 2 REFINEMENT STRATEGY"
            )


            print(
                "=" * 70
            )


            print(
                f"Mode: "
                f"{refinement_mode}"
            )


            self.reset_rng()


            refined_query = (
                self.query_refiner.refine(

                    original_query=
                        original_query,

                    missing_positive_constraints=
                        missing_positive,

                    violated_negative_constraints=
                        violated_negative,

                    previous_query=
                        current_retrieval_query,

                    mode=
                        refinement_mode,

                    positive_constraints=
                        original_positive_constraints,

                    router_reasons=
                        router.get(
                            "reasons",
                            [],
                        ),

                    winner_verification=
                        winner_verification,

                    baseline_verification=
                        baseline_verification,
                )
            )


            print(
                "\nPrevious retrieval query:"
            )


            print(
                current_retrieval_query
            )


            print(
                "\nEvidence-seeking "
                "Round-2 retrieval query:"
            )


            print(
                refined_query
            )


            normalized_current = (
                self.query_refiner
                .normalize_text(
                    current_retrieval_query
                )
            )


            normalized_refined = (
                self.query_refiner
                .normalize_text(
                    refined_query
                )
            )


            if (
                normalized_current
                ==
                normalized_refined
            ):

                stop_reason = (
                    "refinement_produced_same_query"
                )


                break


            current_retrieval_query = (
                refined_query
            )


        # ====================================================
        # Round history
        # ====================================================

        round_history = []


        for round_record in rounds:

            round_history.append(
                {

                    "round":
                        round_record[
                            "round"
                        ],

                    "retrieval_query":
                        round_record[
                            "retrieval_query"
                        ],

                    "verification_query":
                        round_record[
                            "verification_query"
                        ],

                    "top_result":
                        self.summarize_result(
                            round_record[
                                "top_result"
                            ]
                        ),

                    "sufficiency":
                        round_record[
                            "sufficiency"
                        ],

                    "router":
                        round_record[
                            "router"
                        ],

                    "results":
                        round_record[
                            "search_output"
                        ].get(
                            "results",
                            [],
                        ),
                }
            )


        # ====================================================
        # Cross-round memory
        # ====================================================

        candidate_memory = []


        if (
            self.fusion_enabled
            and
            len(
                rounds
            )
            > 1
        ):

            candidate_memory = (
                self.build_candidate_memory(
                    rounds
                )
            )


        # ====================================================
        # FINAL DECISION:
        # Bounded promotion instead of unrestricted fusion
        # ====================================================

        promotion_decision = {

            "promotion_enabled":
                self.promotion_enabled,

            "promote":
                False,

            "reason":
                "round_2_not_executed",

            "round_1_winner_image_id":
                None,

            "challenger_image_id":
                None,

            "challenger_round_1_rank":
                None,

            "challenger_round_2_rank":
                None,

            "round_1_winner_round_2_rank":
                None,

            "winner_verification_score":
                None,

            "challenger_verification_score":
                None,

            "verifier_gap":
                None,

            "max_allowed_verifier_gap":
                self.promotion_max_verifier_gap,

            "challenger_negative_violation":
                None,
        }


        round_1_results = (
            rounds[
                0
            ][
                "search_output"
            ].get(
                "results",
                [],
            )
        )


        if len(
            rounds
        ) > 1:

            round_2_results = (
                rounds[
                    1
                ][
                    "search_output"
                ].get(
                    "results",
                    [],
                )
            )


            (
                final_ranked_results,
                promotion_decision,
            ) = (
                self.apply_bounded_promotion(

                    round_1_results=
                        round_1_results,

                    round_2_results=
                        round_2_results,
                )
            )


            self.print_promotion_decision(
                promotion_decision
            )


            selected_round = (
                "bounded_promotion"
            )


            if promotion_decision[
                "promote"
            ]:

                selected_retrieval_query = (
                    "round_1_base_with_round_2_promotion"
                )

            else:

                selected_retrieval_query = (
                    "round_1_preserved_after_round_2"
                )


        else:

            final_ranked_results = [
                dict(
                    result
                )
                for result in round_1_results
            ]


            selected_round = 1


            selected_retrieval_query = (
                rounds[
                    0
                ][
                    "retrieval_query"
                ]
            )


        selected_results = (
            final_ranked_results[
                :final_top_k
            ]
        )


        selected_top_result = (

            selected_results[
                0
            ]

            if selected_results

            else None
        )


        # ====================================================
        # Final sufficiency
        # ====================================================

        final_sufficient = False

        final_sufficiency = None


        if selected_top_result is not None:

            final_decision = (
                self.sufficiency_checker.evaluate(
                    selected_top_result
                )
            )


            final_sufficiency = (
                self.sufficiency_to_dict(
                    final_decision
                )
            )


            final_sufficient = (
                final_decision.sufficient
            )


        # ====================================================
        # Final output
        # ====================================================

        output = {

            "original_query":
                original_query,

            "fixed_original_plan":
                original_plan,

            "fixed_positive_constraints":
                original_positive_constraints,

            "fixed_negative_constraints":
                original_negative_constraints,

            "reproducibility":
                {

                    "seed":
                        self.seed,

                    "deterministic_algorithms":
                        True,

                    "eager_attention":
                        True,

                    "tf32_enabled":
                        False,
                },

            "router_configuration":
                {

                    "score_threshold":
                        self.router_score_threshold,

                    "score_tolerance":
                        self.router_score_tolerance,

                    "tie_tolerance":
                        self.router_tie_tolerance,
                },

            # ------------------------------------------------
            # Kept only because evaluator/analysis scripts
            # already expect this configuration.
            # ------------------------------------------------

            "fusion_configuration":
                {

                    "enabled":
                        self.fusion_enabled,

                    "used_for_final_ranking":
                        False,

                    "verification_weight":
                        self.fusion_verification_weight,

                    "retrieval_weight":
                        self.fusion_retrieval_weight,

                    "round_1_support_weight":
                        self.round_1_support_weight,

                    "round_2_support_weight":
                        self.round_2_support_weight,

                    "fusion_rrf_k":
                        self.fusion_rrf_k,
                },

            "promotion_configuration":
                {

                    "enabled":
                        self.promotion_enabled,

                    "max_verifier_gap":
                        self.promotion_max_verifier_gap,

                    "allowed_challenger_round_1_rank":
                        2,

                    "required_challenger_round_2_rank":
                        1,

                    "require_round_1_winner_absent_from_round_2":
                        True,

                    "block_negative_constraint_violations":
                        True,
                },

            "promotion_decision":
                promotion_decision,

            "max_rounds":
                self.max_rounds,

            "rounds_executed":
                len(
                    rounds
                ),

            "stop_reason":
                stop_reason,

            "selected_round":
                selected_round,

            "selected_retrieval_query":
                selected_retrieval_query,

            "verification_query":
                original_query,

            "final_sufficient":
                final_sufficient,

            "final_sufficiency":
                final_sufficiency,

            "round_history":
                round_history,

            # ------------------------------------------------
            # Retain memory for scientific analysis.
            # It no longer determines final ranking.
            # ------------------------------------------------

            "cross_round_candidate_memory":
                candidate_memory,

            "results":
                selected_results,

            "positive_constraints":
                original_positive_constraints,

            "negative_constraints":
                original_negative_constraints,

            "selected_top_result":
                selected_top_result,
        }


        return output


# ============================================================
# Final summary
# ============================================================


def print_final_summary(
    output: Dict[str, Any],
):

    print(
        "\n\n"
        + "#" * 70
    )


    print(
        "FINAL ITERATIVE AGENTIC SEARCH SUMMARY"
    )


    print(
        "#" * 70
    )


    print(
        "\nOriginal semantic objective:"
    )


    print(
        output[
            "original_query"
        ]
    )


    print(
        f"\nRounds executed: "
        f"{output['rounds_executed']}"
    )


    print(
        f"Stop reason: "
        f"{output['stop_reason']}"
    )


    print(
        f"Selected mechanism: "
        f"{output['selected_round']}"
    )


    promotion = (
        output.get(
            "promotion_decision",
            {}
        )
    )


    print(
        f"Promotion applied: "
        f"{promotion.get('promote', False)}"
    )


    print(
        f"Promotion reason: "
        f"{promotion.get('reason')}"
    )


    print(
        f"Final evidence sufficient: "
        f"{output['final_sufficient']}"
    )


    print(
        "\n"
        + "-" * 70
    )


    print(
        "FINAL TOP RESULTS"
    )


    print(
        "-" * 70
    )


    for rank, result in enumerate(
        output[
            "results"
        ],
        start=1,
    ):

        verification = (
            result.get(
                "verification",
                {},
            )
        )


        print(
            f"\nRank {rank}"
        )


        print(
            f"Image ID: "
            f"{result.get('image_id')}"
        )


        print(
            f"File: "
            f"{result.get('file_name')}"
        )


        print(
            f"Overall match: "
            f"{verification.get('overall_match')}"
        )


        print(
            f"Verification score: "
            f"{float(verification.get('overall_score', 0.0)):.4f}"
        )


        print(
            f"Final score: "
            f"{float(result.get('final_score', 0.0)):.4f}"
        )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Deterministic iterative Agentic Multimodal "
            "DeepSearch with bounded cross-round promotion"
        )
    )


    parser.add_argument(
        "--query",
        type=str,
        required=True,
    )


    parser.add_argument(
        "--image-folder",
        type=str,
        default="../../data/coco/val2017",
    )


    parser.add_argument(
        "--captions-file",
        type=str,
        default=(
            "../../data/coco/annotations/"
            "captions_val2017.json"
        ),
    )


    parser.add_argument(
        "--retriever-checkpoint",
        type=str,
        default=(
            "../../experiments/"
            "deepsearch_retriever_v2/"
            "deepsearch_retriever_v2.pt"
        ),
    )


    parser.add_argument(
        "--retriever-processor",
        type=str,
        default=(
            "../../experiments/"
            "deepsearch_retriever_v2"
        ),
    )


    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=10,
    )


    parser.add_argument(
        "--final-top-k",
        type=int,
        default=5,
    )


    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )


    parser.add_argument(
        "--retrieval-weight",
        type=float,
        default=0.25,
    )


    parser.add_argument(
        "--verification-weight",
        type=float,
        default=0.75,
    )


    parser.add_argument(
        "--original-query-weight",
        type=float,
        default=2.0,
    )


    parser.add_argument(
        "--subquery-weight",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--min-verification-score",
        type=float,
        default=0.75,
    )


    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2,
    )


    parser.add_argument(
        "--router-score-threshold",
        type=float,
        default=0.85,
    )


    parser.add_argument(
        "--router-score-tolerance",
        type=float,
        default=1e-6,
    )


    parser.add_argument(
        "--router-tie-tolerance",
        type=float,
        default=0.01,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    parser.add_argument(
        "--promotion-max-verifier-gap",
        type=float,
        default=0.20,
    )


    parser.add_argument(
        "--output",
        type=str,
        default=(
            "../../experiments/"
            "iterative_agentic_search_result.json"
        ),
    )


    args = parser.parse_args()


    search_system = (
        IterativeAgenticSearch(

            image_folder=
                args.image_folder,

            captions_file=
                args.captions_file,

            retriever_checkpoint_path=
                args.retriever_checkpoint,

            retriever_processor_path=
                args.retriever_processor,

            retrieval_weight=
                args.retrieval_weight,

            verification_weight=
                args.verification_weight,

            original_query_weight=
                args.original_query_weight,

            subquery_weight=
                args.subquery_weight,

            min_verification_score=
                args.min_verification_score,

            max_rounds=
                args.max_rounds,

            router_score_threshold=
                args.router_score_threshold,

            router_score_tolerance=
                args.router_score_tolerance,

            router_tie_tolerance=
                args.router_tie_tolerance,

            seed=
                args.seed,

            promotion_enabled=
                True,

            promotion_max_verifier_gap=
                args.promotion_max_verifier_gap,
        )
    )


    output = (
        search_system.search(

            query=
                args.query,

            retrieval_top_k=
                args.retrieval_top_k,

            final_top_k=
                args.final_top_k,

            rrf_k=
                args.rrf_k,
        )
    )


    print_final_summary(
        output
    )


    output_directory = (
        os.path.dirname(
            args.output
        )
    )


    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True,
        )


    with open(
        args.output,
        "w",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )


    print(
        "\nSaved iterative search result to:"
    )


    print(
        args.output
    )


if __name__ == "__main__":

    main()
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SufficiencyDecision:
    sufficient: bool

    reason: str

    overall_match: bool

    verification_score: float

    satisfied_positive_constraints: int

    total_positive_constraints: int

    violated_negative_constraints: int

    total_negative_constraints: int

    missing_positive_constraints: List[str]

    violated_negative_constraint_names: List[str]


class EvidenceSufficiencyChecker:
    """
    Deterministic evidence-sufficiency gate.

    This component consumes the validated output produced by
    EvidenceVerifier.

    Expected verifier schema:

        {
            "positive_checks": [
                {
                    "constraint": "...",
                    "satisfied": True/False,
                    "confidence": 0.0-1.0
                }
            ],

            "negative_checks": [
                {
                    "constraint": "...",
                    "present": True/False,
                    "confidence": 0.0-1.0
                }
            ],

            "positive_score": ...,
            "negative_violation_score": ...,
            "overall_score": ...,
            "overall_match": True/False
        }

    Search stops only when:

        1. overall_match is True
        2. every positive constraint is satisfied
        3. no negative constraint is present
        4. overall_score exceeds the configured threshold
    """

    def __init__(
        self,
        min_verification_score: float = 0.75,
    ):

        self.min_verification_score = (
            min_verification_score
        )


    # ============================================================
    # Safe boolean conversion
    # ============================================================

    @staticmethod
    def _safe_bool(
        value: Any,
        default: bool = False,
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
                value.strip().lower()
            )

            if normalized in {
                "true",
                "yes",
                "1",
            }:

                return True


            if normalized in {
                "false",
                "no",
                "0",
            }:

                return False


        if value is None:

            return default


        return bool(
            value
        )


    # ============================================================
    # Safe floating-point conversion
    # ============================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return default


    # ============================================================
    # Positive constraints
    # ============================================================

    def _analyze_positive_checks(
        self,
        verification: Dict[str, Any],
    ):

        positive_checks = (
            verification.get(
                "positive_checks",
                [],
            )
        )


        if not isinstance(
            positive_checks,
            list,
        ):

            positive_checks = []


        total = len(
            positive_checks
        )


        satisfied_count = 0

        missing_constraints = []


        for check in positive_checks:

            if not isinstance(
                check,
                dict,
            ):

                continue


            constraint = str(
                check.get(
                    "constraint",
                    "unknown constraint",
                )
            )


            satisfied = (
                self._safe_bool(
                    check.get(
                        "satisfied",
                        False,
                    ),
                    default=False,
                )
            )


            if satisfied:

                satisfied_count += 1

            else:

                missing_constraints.append(
                    constraint
                )


        return (
            satisfied_count,
            total,
            missing_constraints,
        )


    # ============================================================
    # Negative constraints
    # ============================================================

    def _analyze_negative_checks(
        self,
        verification: Dict[str, Any],
    ):

        negative_checks = (
            verification.get(
                "negative_checks",
                [],
            )
        )


        if not isinstance(
            negative_checks,
            list,
        ):

            negative_checks = []


        total = len(
            negative_checks
        )


        violation_count = 0

        violated_constraints = []


        for check in negative_checks:

            if not isinstance(
                check,
                dict,
            ):

                continue


            constraint = str(
                check.get(
                    "constraint",
                    "unknown constraint",
                )
            )


            present = (
                self._safe_bool(
                    check.get(
                        "present",
                        False,
                    ),
                    default=False,
                )
            )


            if present:

                violation_count += 1

                violated_constraints.append(
                    constraint
                )


        return (
            violation_count,
            total,
            violated_constraints,
        )


    # ============================================================
    # Main decision
    # ============================================================

    def evaluate(
        self,
        result: Dict[str, Any],
    ) -> SufficiencyDecision:

        """
        Evaluate the top-ranked result returned by
        VerifiedAgenticSearch.
        """

        verification = (
            result.get(
                "verification",
                result,
            )
        )


        # ========================================================
        # Main verifier outputs
        # ========================================================

        overall_match = (
            self._safe_bool(
                verification.get(
                    "overall_match",
                    False,
                ),
                default=False,
            )
        )


        verification_score = (
            self._safe_float(
                verification.get(
                    "overall_score",
                    0.0,
                ),
                default=0.0,
            )
        )


        # ========================================================
        # Constraint-level evidence
        # ========================================================

        (
            positive_satisfied,
            positive_total,
            missing_positive,
        ) = (
            self._analyze_positive_checks(
                verification
            )
        )


        (
            negative_violations,
            negative_total,
            violated_negative,
        ) = (
            self._analyze_negative_checks(
                verification
            )
        )


        # ========================================================
        # Decision criteria
        # ========================================================

        positives_complete = (

            positive_total == 0

            or

            positive_satisfied
            == positive_total
        )


        negatives_clean = (

            negative_violations
            == 0
        )


        score_good = (

            verification_score
            >= self.min_verification_score
        )


        sufficient = (

            overall_match

            and

            positives_complete

            and

            negatives_clean

            and

            score_good
        )


        # ========================================================
        # Human-readable explanation
        # ========================================================

        failure_reasons = []


        if not overall_match:

            failure_reasons.append(
                "the verifier does not consider "
                "the image a complete semantic match"
            )


        if not positives_complete:

            if missing_positive:

                failure_reasons.append(
                    "missing required evidence: "
                    + ", ".join(
                        missing_positive
                    )
                )

            else:

                failure_reasons.append(
                    "one or more required positive "
                    "constraints are not satisfied"
                )


        if not negatives_clean:

            if violated_negative:

                failure_reasons.append(
                    "undesired evidence is present: "
                    + ", ".join(
                        violated_negative
                    )
                )

            else:

                failure_reasons.append(
                    "one or more negative "
                    "constraints are violated"
                )


        if not score_good:

            failure_reasons.append(
                "verification confidence "
                f"{verification_score:.2f} is below "
                f"the required threshold of "
                f"{self.min_verification_score:.2f}"
            )


        if sufficient:

            reason = (

                "Evidence is sufficient: "
                f"{positive_satisfied}/"
                f"{positive_total} positive constraints "
                "are satisfied, "

                f"{negative_violations}/"
                f"{negative_total} negative constraints "
                "are violated, "

                "the verifier considers the candidate "
                "a complete semantic match, and "

                f"the verification score "
                f"{verification_score:.2f} exceeds "
                f"the threshold "
                f"{self.min_verification_score:.2f}."
            )

        else:

            reason = (
                "Evidence is insufficient because "
                + "; ".join(
                    failure_reasons
                )
                + "."
            )


        return SufficiencyDecision(

            sufficient=
                sufficient,

            reason=
                reason,

            overall_match=
                overall_match,

            verification_score=
                verification_score,

            satisfied_positive_constraints=
                positive_satisfied,

            total_positive_constraints=
                positive_total,

            violated_negative_constraints=
                negative_violations,

            total_negative_constraints=
                negative_total,

            missing_positive_constraints=
                missing_positive,

            violated_negative_constraint_names=
                violated_negative,
        )


    # ============================================================
    # Pretty printing
    # ============================================================

    @staticmethod
    def print_decision(
        decision: SufficiencyDecision,
    ):

        print(
            "=" * 70
        )

        print(
            "EVIDENCE SUFFICIENCY DECISION"
        )

        print(
            "=" * 70
        )


        print(
            f"Sufficient: "
            f"{decision.sufficient}"
        )


        print(
            f"Overall match: "
            f"{decision.overall_match}"
        )


        print(
            f"Verification score: "
            f"{decision.verification_score:.4f}"
        )


        print(
            "Positive constraints: "
            f"{decision.satisfied_positive_constraints}"
            "/"
            f"{decision.total_positive_constraints}"
        )


        print(
            "Negative violations: "
            f"{decision.violated_negative_constraints}"
            "/"
            f"{decision.total_negative_constraints}"
        )


        if (
            decision
            .missing_positive_constraints
        ):

            print(
                "\nMissing positive constraints:"
            )

            for item in (
                decision
                .missing_positive_constraints
            ):

                print(
                    f"  - {item}"
                )


        if (
            decision
            .violated_negative_constraint_names
        ):

            print(
                "\nViolated negative constraints:"
            )

            for item in (
                decision
                .violated_negative_constraint_names
            ):

                print(
                    f"  - {item}"
                )


        print(
            "\nReason:"
        )

        print(
            decision.reason
        )


        print(
            "=" * 70
        )
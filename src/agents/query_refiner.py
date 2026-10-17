import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# Controlled Evidence-Seeking Query Refiner
#
# Core principle:
#
# The LLM DOES NOT decide which concepts should be searched.
#
# Instead:
#
# MISSING EVIDENCE MODE
#   1. Take actual missing positive constraints.
#   2. Add at most ONE supporting positive constraint.
#   3. Generate a short retrieval phrase.
#
# AMBIGUITY RESOLUTION MODE
#   1. Compare verifier evidence between:
#        - current agentic winner
#        - DeepSearch baseline candidate
#   2. Select at most TWO constraints that are most
#      discriminative between the candidates.
#   3. Generate a short retrieval phrase only from those.
#
#
# The original semantic objective remains unchanged.
# Final verification is always against the original query.
# ============================================================


class QueryRefiner:

    def __init__(
        self,
        model_name: str = (
            "Qwen/Qwen2.5-1.5B-Instruct"
        ),
        device: Optional[str] = None,
        max_similarity_to_previous: float = 0.72,
        max_query_words: int = 15,
        max_focus_constraints: int = 2,
    ):

        if device is None:

            if torch.cuda.is_available():

                device = "cuda"

            elif (
                hasattr(
                    torch.backends,
                    "mps",
                )
                and
                torch.backends.mps.is_available()
            ):

                device = "mps"

            else:

                device = "cpu"


        self.device = device

        self.max_similarity_to_previous = float(
            max_similarity_to_previous
        )

        self.max_query_words = int(
            max_query_words
        )

        self.max_focus_constraints = int(
            max_focus_constraints
        )


        print(
            f"Loading controlled query refiner: "
            f"{model_name}"
        )

        print(
            f"Using device: "
            f"{self.device}"
        )


        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )


        if self.device == "cuda":

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
                .to(
                    self.device
                )
            )

        else:

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_name,
                    attn_implementation="eager",
                )
                .to(
                    self.device
                )
            )


        self.model.eval()


        print(
            "Controlled Query Refiner loaded."
        )


    # ========================================================
    # Text normalization
    # ========================================================

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:

        text = str(
            text
        ).lower().strip()


        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )


        text = re.sub(
            r"\s+",
            " ",
            text,
        )


        return text.strip()


    # ========================================================
    # Deduplicate
    # ========================================================

    @staticmethod
    def deduplicate_strings(
        values: List[str],
    ) -> List[str]:

        output = []

        seen = set()


        for value in values:

            value = str(
                value
            ).strip()


            if not value:

                continue


            key = (
                value
                .lower()
                .strip()
            )


            if key in seen:

                continue


            seen.add(
                key
            )

            output.append(
                value
            )


        return output


    # ========================================================
    # Similarity
    # ========================================================

    def query_similarity(
        self,
        query_a: str,
        query_b: str,
    ) -> float:

        return SequenceMatcher(
            None,
            self.normalize_text(
                query_a
            ),
            self.normalize_text(
                query_b
            ),
        ).ratio()


    # ========================================================
    # Clean query
    # ========================================================

    @staticmethod
    def clean_query(
        query: str,
    ) -> str:

        query = str(
            query
        ).strip()


        query = query.strip(
            "\"'"
        )


        prefixes = [
            "refined query:",
            "search query:",
            "retrieval query:",
            "focused query:",
            "query:",
            "new query:",
        ]


        lower_query = query.lower()


        for prefix in prefixes:

            if lower_query.startswith(
                prefix
            ):

                query = (
                    query[
                        len(prefix):
                    ]
                    .strip()
                )

                break


        query = re.sub(
            r"\s+",
            " ",
            query,
        )


        return query.strip(
            " \n\t\"'."
        )


    # ========================================================
    # JSON extraction
    # ========================================================

    @staticmethod
    def extract_json(
        text: str,
    ) -> Optional[
        Dict[str, Any]
    ]:

        text = str(
            text
        ).strip()


        try:

            parsed = json.loads(
                text
            )


            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except Exception:

            pass


        cleaned = re.sub(
            r"```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )


        cleaned = cleaned.replace(
            "```",
            "",
        )


        start = cleaned.find(
            "{"
        )

        end = cleaned.rfind(
            "}"
        )


        if (
            start == -1
            or
            end == -1
            or
            end <= start
        ):

            return None


        candidate = cleaned[
            start:
            end + 1
        ]


        try:

            parsed = json.loads(
                candidate
            )


            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except Exception:

            return None


        return None


    # ========================================================
    # Safe boolean
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
    # Positive verification checks
    # ========================================================

    @staticmethod
    def positive_checks(
        verification: Optional[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:

        if not isinstance(
            verification,
            dict,
        ):

            return []


        checks = verification.get(
            "positive_checks",
            [],
        )


        if not isinstance(
            checks,
            list,
        ):

            return []


        return [
            item
            for item in checks
            if isinstance(
                item,
                dict,
            )
        ]


    # ========================================================
    # Build confidence map
    # ========================================================

    def build_confidence_map(
        self,
        verification: Optional[
            Dict[str, Any]
        ],
    ) -> Dict[
        str,
        Dict[str, Any]
    ]:

        output = {}


        for check in (
            self.positive_checks(
                verification
            )
        ):

            raw_constraint = str(
                check.get(
                    "constraint",
                    "",
                )
            ).strip()


            normalized = (
                self.normalize_text(
                    raw_constraint
                )
            )


            if not normalized:

                continue


            try:

                confidence = float(
                    check.get(
                        "confidence",
                        0.0,
                    )
                )

            except Exception:

                confidence = 0.0


            satisfied = (
                self.safe_bool(
                    check.get(
                        "satisfied",
                        False,
                    )
                )
            )


            output[
                normalized
            ] = {

                "constraint":
                    raw_constraint,

                "confidence":
                    confidence,

                "satisfied":
                    satisfied,
            }


        return output


    # ========================================================
    # Constraint content score
    #
    # Favor visually specific constraints.
    # ========================================================

    def visual_specificity_score(
        self,
        constraint: str,
    ) -> float:

        normalized = (
            self.normalize_text(
                constraint
            )
        )


        words = (
            normalized.split()
        )


        score = float(
            len(words)
        )


        # ----------------------------------------------------
        # Favor explicit colors, counts, relationships, and
        # visually distinctive object terms.
        # ----------------------------------------------------

        discriminative_tokens = {
            "red",
            "green",
            "blue",
            "yellow",
            "black",
            "white",
            "brown",
            "orange",
            "purple",
            "three",
            "two",
            "four",
            "beside",
            "next",
            "near",
            "under",
            "above",
            "behind",
            "around",
            "inside",
            "outside",
            "turtle",
            "hydrant",
            "stemware",
            "flowers",
            "fence",
            "stool",
            "stools",
            "blender",
            "cooler",
            "coolers",
        }


        for word in words:

            if word in (
                discriminative_tokens
            ):

                score += 1.5


        return score


    # ========================================================
    # Select ambiguity constraints
    #
    # This is the core new behavior.
    #
    # Constraint ranking combines:
    #
    # 1. Confidence disagreement between candidates
    # 2. Satisfaction disagreement
    # 3. Visual specificity
    #
    # At most TWO constraints survive.
    # ========================================================

    def select_ambiguity_constraints(
        self,
        positive_constraints: List[str],
        winner_verification: Optional[
            Dict[str, Any]
        ],
        baseline_verification: Optional[
            Dict[str, Any]
        ],
    ) -> List[str]:

        positive_constraints = (
            self.deduplicate_strings(
                positive_constraints
            )
        )


        winner_map = (
            self.build_confidence_map(
                winner_verification
            )
        )


        baseline_map = (
            self.build_confidence_map(
                baseline_verification
            )
        )


        scored = []


        for constraint in (
            positive_constraints
        ):

            normalized = (
                self.normalize_text(
                    constraint
                )
            )


            winner_info = (
                winner_map.get(
                    normalized
                )
            )


            baseline_info = (
                baseline_map.get(
                    normalized
                )
            )


            winner_confidence = (
                winner_info[
                    "confidence"
                ]
                if winner_info
                else 0.0
            )


            baseline_confidence = (
                baseline_info[
                    "confidence"
                ]
                if baseline_info
                else 0.0
            )


            winner_satisfied = (
                winner_info[
                    "satisfied"
                ]
                if winner_info
                else False
            )


            baseline_satisfied = (
                baseline_info[
                    "satisfied"
                ]
                if baseline_info
                else False
            )


            confidence_gap = abs(
                winner_confidence
                -
                baseline_confidence
            )


            satisfaction_gap = (

                1.0

                if (
                    winner_satisfied
                    !=
                    baseline_satisfied
                )

                else 0.0
            )


            specificity = (
                self.visual_specificity_score(
                    constraint
                )
            )


            # ------------------------------------------------
            # Confidence disagreement dominates.
            #
            # Satisfaction disagreement is even stronger.
            #
            # Specificity breaks ties.
            # ------------------------------------------------

            score = (
                confidence_gap
                * 20.0

                +

                satisfaction_gap
                * 10.0

                +

                specificity
                * 0.5
            )


            scored.append(
                {
                    "constraint":
                        constraint,

                    "score":
                        score,

                    "confidence_gap":
                        confidence_gap,

                    "satisfaction_gap":
                        satisfaction_gap,

                    "specificity":
                        specificity,
                }
            )


        scored.sort(
            key=lambda item: (
                item[
                    "score"
                ],
                item[
                    "specificity"
                ],
            ),
            reverse=True,
        )


        selected = [
            item[
                "constraint"
            ]
            for item in scored[
                :self.max_focus_constraints
            ]
        ]


        print(
            "\nAmbiguity constraint ranking:"
        )


        for item in scored:

            print(
                "  "
                f"{item['constraint']} | "
                f"score={item['score']:.4f} | "
                f"confidence_gap="
                f"{item['confidence_gap']:.4f} | "
                f"satisfaction_gap="
                f"{item['satisfaction_gap']:.1f} | "
                f"specificity="
                f"{item['specificity']:.2f}"
            )


        print(
            "\nSelected ambiguity focus:"
        )


        for item in selected:

            print(
                f"  + {item}"
            )


        return selected


    # ========================================================
    # Select missing-evidence constraints
    # ========================================================

    def select_missing_constraints(
        self,
        missing_positive_constraints: List[str],
        positive_constraints: List[str],
    ) -> List[str]:

        missing = (
            self.deduplicate_strings(
                missing_positive_constraints
            )
        )


        positives = (
            self.deduplicate_strings(
                positive_constraints
            )
        )


        selected = []


        # ----------------------------------------------------
        # Missing evidence has absolute priority.
        # ----------------------------------------------------

        for constraint in missing:

            selected.append(
                constraint
            )


            if (
                len(selected)
                >=
                self.max_focus_constraints
            ):

                break


        # ----------------------------------------------------
        # If there is only one missing constraint, add at most
        # ONE useful contextual positive constraint.
        # ----------------------------------------------------

        if (
            len(selected)
            <
            self.max_focus_constraints
        ):

            candidate_context = []


            for constraint in positives:

                normalized = (
                    self.normalize_text(
                        constraint
                    )
                )


                if any(

                    normalized
                    ==
                    self.normalize_text(
                        item
                    )

                    for item in selected
                ):

                    continue


                candidate_context.append(
                    (
                        self.visual_specificity_score(
                            constraint
                        ),
                        constraint,
                    )
                )


            candidate_context.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                ),
                reverse=True,
            )


            if candidate_context:

                selected.append(
                    candidate_context[
                        0
                    ][
                        1
                    ]
                )


        return selected[
            :self.max_focus_constraints
        ]


    # ========================================================
    # Build deterministic fallback phrase
    # ========================================================

    def deterministic_phrase(
        self,
        selected_constraints: List[str],
    ) -> str:

        selected_constraints = (
            self.deduplicate_strings(
                selected_constraints
            )
        )


        if not selected_constraints:

            return (
                "distinctive visual details"
            )


        phrase = ", ".join(
            selected_constraints[
                :self.max_focus_constraints
            ]
        )


        words = phrase.split()


        if (
            len(words)
            >
            self.max_query_words
        ):

            phrase = " ".join(
                words[
                    :self.max_query_words
                ]
            )


        return self.clean_query(
            phrase
        )


    # ========================================================
    # Build controlled LLM prompt
    #
    # The LLM only compresses already-selected constraints.
    # ========================================================

    def build_messages(
        self,
        mode: str,
        selected_constraints: List[str],
        previous_query: str,
    ):

        system_prompt = """
You are a retrieval-query compression module.

You are given one or two PRESELECTED visual constraints.

You are NOT allowed to choose new concepts.

Convert only those constraints into one concise visual search phrase.

STRICT RULES:
- Use only concepts contained in the supplied constraints.
- Do not add new objects.
- Do not add new colors.
- Do not add new counts.
- Do not add new scene details.
- Do not mention missing evidence.
- Do not mention candidate images.
- Do not explain anything.
- Do not say "find an image".
- Do not say "image with".
- Do not repeat the full original request.
- Maximum 15 words.
- Return ONLY JSON.

Format:

{
  "refined_query": "..."
}
""".strip()


        user_prompt = f"""
MODE:
{mode}

PRESELECTED VISUAL CONSTRAINTS:
{json.dumps(selected_constraints, indent=2)}

PREVIOUS RETRIEVAL QUERY:
{previous_query}

Compress ONLY the supplied visual constraints into one short search phrase.
""".strip()


        return [
            {
                "role":
                    "system",

                "content":
                    system_prompt,
            },
            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ]


    # ========================================================
    # Generate compression
    # ========================================================

    def generate_query(
        self,
        messages,
    ) -> Optional[str]:

        prompt = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


        inputs = (
            self.tokenizer(
                prompt,
                return_tensors="pt",
            )
        )


        inputs = {
            key:
                value.to(
                    self.device
                )
            for key, value
            in inputs.items()
        }


        with torch.inference_mode():

            output_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=
                        self.tokenizer.eos_token_id,
                )
            )


        generated_ids = (
            output_ids[
                0,
                inputs[
                    "input_ids"
                ].shape[1]:
            ]
        )


        generated_text = (
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
        )


        parsed = (
            self.extract_json(
                generated_text
            )
        )


        if not parsed:

            return None


        query = (
            parsed.get(
                "refined_query"
            )
        )


        if query is None:

            return None


        return self.clean_query(
            query
        )


    # ========================================================
    # Validate generated query
    # ========================================================

    def validate_generated_query(
        self,
        query: str,
        selected_constraints: List[str],
        previous_query: str,
    ) -> Tuple[
        bool,
        str,
    ]:

        if not query:

            return (
                False,
                "empty_query",
            )


        normalized = (
            self.normalize_text(
                query
            )
        )


        if not normalized:

            return (
                False,
                "empty_normalized_query",
            )


        words = (
            normalized.split()
        )


        if (
            len(words)
            >
            self.max_query_words
        ):

            return (
                False,
                "query_too_long",
            )


        forbidden_prefixes = [
            "find an image",
            "search for an image",
            "image with",
            "look for an image",
            "the query",
            "refined query",
        ]


        for prefix in (
            forbidden_prefixes
        ):

            if normalized.startswith(
                prefix
            ):

                return (
                    False,
                    "meta_language",
                )


        similarity = (
            self.query_similarity(
                query,
                previous_query,
            )
        )


        if (
            similarity
            >=
            self.max_similarity_to_previous
        ):

            return (
                False,
                "too_similar_to_previous",
            )


        # ----------------------------------------------------
        # Generated query must overlap substantially with the
        # selected constraints.
        # ----------------------------------------------------

        selected_text = (
            self.normalize_text(
                " ".join(
                    selected_constraints
                )
            )
        )


        selected_words = set(
            selected_text.split()
        )


        query_words = set(
            normalized.split()
        )


        if not selected_words:

            return (
                False,
                "no_selected_constraint_terms",
            )


        overlap = (
            selected_words
            &
            query_words
        )


        overlap_ratio = (
            len(overlap)
            /
            max(
                1,
                len(query_words),
            )
        )


        if overlap_ratio < 0.50:

            return (
                False,
                "insufficient_constraint_overlap",
            )


        return (
            True,
            "valid",
        )


    # ========================================================
    # Main refinement API
    # ========================================================

    def refine(
        self,
        original_query: str,
        missing_positive_constraints: Optional[
            List[str]
        ] = None,
        violated_negative_constraints: Optional[
            List[str]
        ] = None,
        previous_query: Optional[str] = None,
        mode: str = "missing_evidence",
        positive_constraints: Optional[
            List[str]
        ] = None,
        router_reasons: Optional[
            List[str]
        ] = None,
        winner_verification: Optional[
            Dict[str, Any]
        ] = None,
        baseline_verification: Optional[
            Dict[str, Any]
        ] = None,
    ) -> str:

        missing_positive_constraints = list(
            missing_positive_constraints
            or []
        )


        violated_negative_constraints = list(
            violated_negative_constraints
            or []
        )


        positive_constraints = list(
            positive_constraints
            or []
        )


        router_reasons = list(
            router_reasons
            or []
        )


        previous_query = (
            previous_query
            or original_query
        )


        if mode not in {
            "missing_evidence",
            "ambiguity_resolution",
        }:

            raise ValueError(
                f"Unsupported refinement mode: "
                f"{mode}"
            )


        print(
            "\n"
            + "=" * 70
        )


        print(
            "CONTROLLED EVIDENCE-SEEKING REFINER"
        )


        print(
            "=" * 70
        )


        print(
            f"Mode: "
            f"{mode}"
        )


        # ====================================================
        # Select evidence BEFORE invoking the LLM.
        # ====================================================

        if (
            mode
            ==
            "missing_evidence"
        ):

            selected_constraints = (
                self.select_missing_constraints(

                    missing_positive_constraints=
                        missing_positive_constraints,

                    positive_constraints=
                        positive_constraints,
                )
            )

        else:

            selected_constraints = (
                self.select_ambiguity_constraints(

                    positive_constraints=
                        positive_constraints,

                    winner_verification=
                        winner_verification,

                    baseline_verification=
                        baseline_verification,
                )
            )


        # ====================================================
        # Safety fallback if selection was empty.
        # ====================================================

        if not selected_constraints:

            selected_constraints = (
                positive_constraints[
                    :self.max_focus_constraints
                ]
            )


        print(
            "\nSelected retrieval evidence:"
        )


        for constraint in (
            selected_constraints
        ):

            print(
                f"  + {constraint}"
            )


        fallback = (
            self.deterministic_phrase(
                selected_constraints
            )
        )


        # ====================================================
        # LLM only compresses selected evidence.
        # ====================================================

        messages = (
            self.build_messages(

                mode=
                    mode,

                selected_constraints=
                    selected_constraints,

                previous_query=
                    previous_query,
            )
        )


        candidate = (
            self.generate_query(
                messages
            )
        )


        if candidate is None:

            print(
                "\nLLM compression unavailable."
            )

            print(
                "Using deterministic phrase:"
            )

            print(
                fallback
            )

            return fallback


        valid, reason = (
            self.validate_generated_query(

                query=
                    candidate,

                selected_constraints=
                    selected_constraints,

                previous_query=
                    previous_query,
            )
        )


        print(
            f"\nGenerated query:"
        )

        print(
            candidate
        )


        print(
            f"Validation: "
            f"{valid}"
        )


        print(
            f"Validation reason: "
            f"{reason}"
        )


        if not valid:

            print(
                "\nUsing deterministic "
                "constraint-only fallback:"
            )

            print(
                fallback
            )

            return fallback


        print(
            "\nAccepted evidence-seeking query:"
        )

        print(
            candidate
        )


        return candidate
    
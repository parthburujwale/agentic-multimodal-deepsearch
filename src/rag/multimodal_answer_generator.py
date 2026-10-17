import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


# ============================================================
# Multimodal Answer Generator
# ============================================================


class MultimodalAnswerGenerator:
    """
    Generate a grounded natural-language answer from:

        original query
        +
        frozen retrieval evidence
        +
        retrieved images

    IMPORTANT:

    Retrieval decisions and verifier decisions remain authoritative.

    The VLM is used for natural-language synthesis only.

    It is NOT allowed to redefine:
        - which positive requirements are missing
        - which negative requirements are violated
        - which candidate is the primary evidence
        - whether the retrieval system declared evidence sufficient
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = None,
        max_images: int = 3,
        max_new_tokens: int = 300,
    ):
        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is None:

            if torch.cuda.is_available():
                device = "cuda"

            elif torch.backends.mps.is_available():
                device = "mps"

            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name
        self.max_images = max_images
        self.max_new_tokens = max_new_tokens

        print(
            f"Loading multimodal answer generator: "
            f"{model_name}"
        )

        print(
            f"Using device: "
            f"{device}"
        )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name
            )
        )

        # ----------------------------------------------------
        # Model dtype
        # ----------------------------------------------------

        if device == "cuda":
            model_dtype = torch.bfloat16

        else:
            model_dtype = torch.float32

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                attn_implementation="eager",
            )
        )

        self.model = self.model.to(
            device
        )

        self.model.eval()


    # ========================================================
    # JSON helpers
    # ========================================================

    @staticmethod
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


    @staticmethod
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


    # ========================================================
    # Evidence-image loading
    # ========================================================

    def load_evidence_images(
        self,
        evidence_package: Dict[str, Any],
    ):

        raw_evidence = evidence_package.get(
            "evidence",
            [],
        )

        selected = raw_evidence[
            :self.max_images
        ]

        images = []
        usable_evidence = []

        for item in selected:

            image_path = item.get(
                "image_path"
            )

            if not image_path:

                print(
                    "Skipping evidence item with "
                    "no image_path."
                )

                continue

            if not os.path.exists(
                image_path
            ):

                print(
                    f"WARNING: Image not found: "
                    f"{image_path}"
                )

                continue

            image = (
                Image.open(
                    image_path
                )
                .convert(
                    "RGB"
                )
            )

            images.append(
                image
            )

            usable_evidence.append(
                item
            )

        if not images:

            raise RuntimeError(
                "No evidence images could be loaded."
            )

        return (
            images,
            usable_evidence,
        )


    # ========================================================
    # Structured verifier helpers
    # ========================================================

    @staticmethod
    def get_verification(
        evidence_item: Dict[str, Any],
    ) -> Dict[str, Any]:

        verification = evidence_item.get(
            "verification",
            {},
        )

        if not isinstance(
            verification,
            dict,
        ):

            return {}

        return verification


    def get_missing_requirements(
        self,
        evidence_item: Dict[str, Any],
    ) -> List[str]:
        """
        Missing positive requirements are determined ONLY from
        verifier positive_checks.

        The answer-generation VLM cannot redefine these.
        """

        verification = self.get_verification(
            evidence_item
        )

        checks = verification.get(
            "positive_checks",
            [],
        )

        missing = []

        for check in checks:

            if not isinstance(
                check,
                dict,
            ):
                continue

            satisfied = bool(
                check.get(
                    "satisfied",
                    False,
                )
            )

            constraint = str(
                check.get(
                    "constraint",
                    "",
                )
            ).strip()

            if (
                constraint
                and
                not satisfied
            ):

                missing.append(
                    constraint
                )

        return list(
            dict.fromkeys(
                missing
            )
        )


    def get_satisfied_requirements(
        self,
        evidence_item: Dict[str, Any],
    ) -> List[str]:

        verification = self.get_verification(
            evidence_item
        )

        checks = verification.get(
            "positive_checks",
            [],
        )

        satisfied = []

        for check in checks:

            if not isinstance(
                check,
                dict,
            ):
                continue

            is_satisfied = bool(
                check.get(
                    "satisfied",
                    False,
                )
            )

            constraint = str(
                check.get(
                    "constraint",
                    "",
                )
            ).strip()

            if (
                constraint
                and
                is_satisfied
            ):

                satisfied.append(
                    constraint
                )

        return list(
            dict.fromkeys(
                satisfied
            )
        )


    def get_violated_requirements(
        self,
        evidence_item: Dict[str, Any],
    ) -> List[str]:
        """
        A negative requirement is violated ONLY when
        verifier negative_checks reports present=True.
        """

        verification = self.get_verification(
            evidence_item
        )

        checks = verification.get(
            "negative_checks",
            [],
        )

        violated = []

        for check in checks:

            if not isinstance(
                check,
                dict,
            ):
                continue

            present = bool(
                check.get(
                    "present",
                    False,
                )
            )

            constraint = str(
                check.get(
                    "constraint",
                    "",
                )
            ).strip()

            if (
                constraint
                and
                present
            ):

                violated.append(
                    constraint
                )

        return list(
            dict.fromkeys(
                violated
            )
        )


    def get_respected_negative_requirements(
        self,
        evidence_item: Dict[str, Any],
    ) -> List[str]:

        verification = self.get_verification(
            evidence_item
        )

        checks = verification.get(
            "negative_checks",
            [],
        )

        respected = []

        for check in checks:

            if not isinstance(
                check,
                dict,
            ):
                continue

            present = bool(
                check.get(
                    "present",
                    False,
                )
            )

            constraint = str(
                check.get(
                    "constraint",
                    "",
                )
            ).strip()

            if (
                constraint
                and
                not present
            ):

                respected.append(
                    constraint
                )

        return list(
            dict.fromkeys(
                respected
            )
        )


    # ========================================================
    # Evidence text
    # ========================================================

    def build_evidence_text(
        self,
        usable_evidence: List[
            Dict[str, Any]
        ],
    ) -> str:

        sections = []

        for index, item in enumerate(
            usable_evidence,
            start=1,
        ):

            verification = (
                self.get_verification(
                    item
                )
            )

            positive_lines = []

            for check in verification.get(
                "positive_checks",
                [],
            ):

                positive_lines.append(
                    (
                        f"- "
                        f"{check.get('constraint', '')}: "
                        f"satisfied="
                        f"{bool(check.get('satisfied', False))}, "
                        f"confidence="
                        f"{float(check.get('confidence', 0.0)):.2f}"
                    )
                )

            negative_lines = []

            for check in verification.get(
                "negative_checks",
                [],
            ):

                negative_lines.append(
                    (
                        f"- "
                        f"{check.get('constraint', '')}: "
                        f"present="
                        f"{bool(check.get('present', False))}, "
                        f"confidence="
                        f"{float(check.get('confidence', 0.0)):.2f}"
                    )
                )

            if not positive_lines:
                positive_lines = [
                    "- None"
                ]

            if not negative_lines:
                negative_lines = [
                    "- None"
                ]

            section = (
                f"EVIDENCE {index}\n"
                f"Image ID: "
                f"{item.get('image_id')}\n"
                f"Final rank: "
                f"{item.get('rank')}\n"
                f"Final score: "
                f"{float(item.get('final_score', 0.0)):.4f}\n"
                f"Verifier score: "
                f"{float(verification.get('overall_score', 0.0)):.4f}\n"
                f"Overall semantic match: "
                f"{bool(verification.get('overall_match', False))}\n\n"
                f"Positive checks:\n"
                f"{chr(10).join(positive_lines)}\n\n"
                f"Negative checks:\n"
                f"{chr(10).join(negative_lines)}\n\n"
                f"Verifier explanation:\n"
                f"{verification.get('explanation', '')}"
            )

            sections.append(
                section
            )

        separator = (
            "\n\n"
            + "=" * 60
            + "\n\n"
        )

        return separator.join(
            sections
        )


    # ========================================================
    # Prompt
    # ========================================================

    def build_messages(
        self,
        evidence_package: Dict[str, Any],
        usable_evidence: List[
            Dict[str, Any]
        ],
    ):

        query = str(
            evidence_package.get(
                "query",
                "",
            )
        ).strip()

        final_sufficient = bool(
            evidence_package.get(
                "search_metadata",
                {},
            ).get(
                "final_sufficient",
                False,
            )
        )

        evidence_text = (
            self.build_evidence_text(
                usable_evidence
            )
        )

        primary = usable_evidence[
            0
        ]

        satisfied = (
            self.get_satisfied_requirements(
                primary
            )
        )

        missing = (
            self.get_missing_requirements(
                primary
            )
        )

        violated = (
            self.get_violated_requirements(
                primary
            )
        )

        respected_negative = (
            self.get_respected_negative_requirements(
                primary
            )
        )

        system_prompt = """
You are the natural-language explanation component of a
grounded multimodal DeepSearch system.

You receive retrieved images and AUTHORITATIVE structured
verification results.

The verification fields are ground truth for this generation
stage. You must NOT reinterpret their Boolean meanings.

In particular:

- positive check satisfied=true means the requested property
  was detected.

- positive check satisfied=false means the requested property
  was not established.

- negative check present=true means the forbidden property IS
  visible and therefore violates the request.

- negative check present=false means the forbidden property was
  NOT detected and therefore does NOT violate the request.

Your job is to write a concise explanation.

Do not modify retrieval rankings.
Do not invent visual details.
Do not contradict the structured verifier.
Do not expose RRF, routers, embeddings, or internal algorithms.

If final_sufficient=false, explicitly say that no complete match
was verified.

Return ONLY valid JSON with exactly:

{
  "answer": string
}
""".strip()

        user_text = f"""
Original request:

{query}

Retrieval final_sufficient:
{final_sufficient}

The primary candidate is Evidence 1.

AUTHORITATIVE interpretation of Evidence 1:

Satisfied positive requirements:
{json.dumps(satisfied)}

Missing positive requirements:
{json.dumps(missing)}

Violated negative requirements:
{json.dumps(violated)}

Forbidden concepts NOT detected:
{json.dumps(respected_negative)}

Full structured evidence:

{evidence_text}

Write a concise grounded explanation.

If no complete match was verified, describe Evidence 1 as the
strongest partial candidate and explain what it satisfies and
what is still missing.

Do NOT label a forbidden requirement as violated unless it
appears in the authoritative violated-negative list above.
""".strip()

        content = []

        for _ in usable_evidence:

            content.append(
                {
                    "type":
                        "image",
                }
            )

        content.append(
            {
                "type":
                    "text",

                "text":
                    user_text,
            }
        )

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
                    content,
            },
        ]


    # ========================================================
    # JSON extraction
    # ========================================================

    @staticmethod
    def extract_json(
        generated_text: str,
    ) -> Dict[str, Any]:

        generated_text = (
            generated_text.strip()
        )

        cleaned = re.sub(
            r"```(?:json)?",
            "",
            generated_text,
            flags=re.IGNORECASE,
        )

        cleaned = (
            cleaned.replace(
                "```",
                "",
            )
            .strip()
        )

        try:

            result = json.loads(
                cleaned
            )

            if isinstance(
                result,
                dict,
            ):
                return result

        except json.JSONDecodeError:
            pass

        start = cleaned.find(
            "{"
        )

        end = cleaned.rfind(
            "}"
        )

        if (
            start != -1
            and
            end != -1
            and
            end > start
        ):

            candidate = cleaned[
                start:
                end + 1
            ]

            try:

                result = json.loads(
                    candidate
                )

                if isinstance(
                    result,
                    dict,
                ):
                    return result

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Answer generator returned malformed JSON."
            "\n\n"
            f"Raw output:\n"
            f"{generated_text}"
        )


    # ========================================================
    # Deterministic grounding
    # ========================================================

    def build_grounded_result(
        self,
        generated_answer: str,
        evidence_package: Dict[str, Any],
        usable_evidence: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Construct the final response deterministically.

        The VLM controls ONLY the prose answer.

        All structured grounding fields are derived from the
        frozen verifier output.
        """

        primary = usable_evidence[
            0
        ]

        verification = (
            self.get_verification(
                primary
            )
        )

        final_sufficient = bool(
            evidence_package.get(
                "search_metadata",
                {},
            ).get(
                "final_sufficient",
                False,
            )
        )

        missing = (
            self.get_missing_requirements(
                primary
            )
        )

        violated = (
            self.get_violated_requirements(
                primary
            )
        )

        satisfied = (
            self.get_satisfied_requirements(
                primary
            )
        )

        respected_negative = (
            self.get_respected_negative_requirements(
                primary
            )
        )

        overall_match = bool(
            verification.get(
                "overall_match",
                False,
            )
        )

        verification_score = float(
            verification.get(
                "overall_score",
                0.0,
            )
        )

        # ----------------------------------------------------
        # Deterministic confidence
        # ----------------------------------------------------

        if (
            final_sufficient
            and
            overall_match
            and
            not missing
            and
            not violated
            and
            verification_score >= 0.85
        ):

            confidence = "high"

        elif (
            final_sufficient
            and
            overall_match
            and
            not violated
        ):

            confidence = "medium"

        else:

            confidence = "low"

        # ----------------------------------------------------
        # Best evidence is deterministic.
        #
        # The final ranking already selected Evidence 1.
        # ----------------------------------------------------

        best_evidence = 1

        supporting_evidence = [
            1
        ]

        # ----------------------------------------------------
        # Ensure insufficiency is explicit.
        # ----------------------------------------------------

        answer = str(
            generated_answer
        ).strip()

        if not answer:

            answer = (
                "The retrieved evidence was insufficient "
                "to establish a complete match."
            )

        if not final_sufficient:

            lower = answer.lower()

            abstention_language = [
                "no complete match",
                "no retrieved image",
                "no image fully",
                "not a complete match",
                "no fully matching",
                "does not fully",
                "doesn't fully",
                "partial candidate",
            ]

            has_abstention = any(
                phrase in lower
                for phrase
                in abstention_language
            )

            if not has_abstention:

                answer = (
                    "No retrieved image was verified as a "
                    "complete match. "
                    +
                    answer
                )

        # ----------------------------------------------------
        # Evidence references
        # ----------------------------------------------------

        evidence_references = []

        for index, item in enumerate(
            usable_evidence,
            start=1,
        ):

            item_verification = (
                self.get_verification(
                    item
                )
            )

            evidence_references.append(
                {
                    "evidence_number":
                        index,

                    "image_id":
                        item.get(
                            "image_id"
                        ),

                    "file_name":
                        item.get(
                            "file_name"
                        ),

                    "image_path":
                        item.get(
                            "image_path"
                        ),

                    "final_rank":
                        item.get(
                            "rank"
                        ),

                    "verification_score":
                        item_verification.get(
                            "overall_score"
                        ),

                    "overall_match":
                        item_verification.get(
                            "overall_match"
                        ),
                }
            )

        return {
            "query":
                evidence_package.get(
                    "query"
                ),

            "answer":
                answer,

            "best_evidence":
                best_evidence,

            "supporting_evidence":
                supporting_evidence,

            "satisfied_requirements":
                satisfied,

            "missing_requirements":
                missing,

            "violated_requirements":
                violated,

            "respected_negative_requirements":
                respected_negative,

            "confidence":
                confidence,

            "grounded":
                True,

            "complete_match":
                bool(
                    final_sufficient
                    and
                    overall_match
                    and
                    not missing
                    and
                    not violated
                ),

            "retrieval_final_sufficient":
                final_sufficient,

            "primary_verification_score":
                verification_score,

            "evidence_references":
                evidence_references,
        }


    # ========================================================
    # Generation
    # ========================================================

    def generate(
        self,
        evidence_package: Dict[str, Any],
    ) -> Dict[str, Any]:

        images, usable_evidence = (
            self.load_evidence_images(
                evidence_package
            )
        )

        messages = (
            self.build_messages(
                evidence_package=
                    evidence_package,

                usable_evidence=
                    usable_evidence,
            )
        )

        prompt_text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = (
            self.processor(
                text=[
                    prompt_text
                ],

                images=
                    images,

                padding=
                    True,

                return_tensors=
                    "pt",
            )
        )

        inputs = {
            key:
                value.to(
                    self.device
                )

            if hasattr(
                value,
                "to",
            )

            else value

            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            generated_ids = (
                self.model.generate(
                    **inputs,

                    max_new_tokens=
                        self.max_new_tokens,

                    do_sample=
                        False,
                )
            )

        input_length = (
            inputs[
                "input_ids"
            ].shape[
                1
            ]
        )

        generated_only = (
            generated_ids[
                :,
                input_length:
            ]
        )

        generated_text = (
            self.processor.batch_decode(
                generated_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[
                0
            ]
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "RAW MULTIMODAL ANSWER"
        )

        print(
            "=" * 80
        )

        print(
            generated_text
        )

        parsed = (
            self.extract_json(
                generated_text
            )
        )

        generated_answer = str(
            parsed.get(
                "answer",
                "",
            )
        ).strip()

        return (
            self.build_grounded_result(
                generated_answer=
                    generated_answer,

                evidence_package=
                    evidence_package,

                usable_evidence=
                    usable_evidence,
            )
        )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a grounded multimodal answer "
            "from a DeepSearch evidence package."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to evidence package JSON."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path to save generated answer JSON."
        ),
    )

    parser.add_argument(
        "--model",
        default=
            "Qwen/Qwen2.5-VL-3B-Instruct",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    evidence_package = (
        MultimodalAnswerGenerator.load_json(
            args.input
        )
    )

    generator = (
        MultimodalAnswerGenerator(
            model_name=
                args.model,

            max_images=
                args.max_images,

            max_new_tokens=
                args.max_new_tokens,
        )
    )

    result = (
        generator.generate(
            evidence_package
        )
    )

    MultimodalAnswerGenerator.save_json(
        args.output,
        result,
    )

    print(
        "\n"
        + "#" * 80
    )

    print(
        "GROUNDED MULTIMODAL DEEPSEARCH ANSWER"
    )

    print(
        "#" * 80
    )

    print(
        "\nQuery:"
    )

    print(
        result[
            "query"
        ]
    )

    print(
        "\nAnswer:"
    )

    print(
        result[
            "answer"
        ]
    )

    print(
        "\nComplete match:"
    )

    print(
        result[
            "complete_match"
        ]
    )

    print(
        "\nConfidence:"
    )

    print(
        result[
            "confidence"
        ]
    )

    print(
        "\nBest evidence:"
    )

    print(
        result[
            "best_evidence"
        ]
    )

    print(
        "\nSatisfied requirements:"
    )

    for item in result[
        "satisfied_requirements"
    ]:

        print(
            f"  + {item}"
        )

    print(
        "\nMissing requirements:"
    )

    if result[
        "missing_requirements"
    ]:

        for item in result[
            "missing_requirements"
        ]:

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

    if result[
        "violated_requirements"
    ]:

        for item in result[
            "violated_requirements"
        ]:

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

    if result[
        "respected_negative_requirements"
    ]:

        for item in result[
            "respected_negative_requirements"
        ]:

            print(
                f"  + {item}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nSaved to:"
    )

    print(
        args.output
    )


if __name__ == "__main__":

    main()
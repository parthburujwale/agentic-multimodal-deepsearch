import json
import re

import torch

from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)


# --------------------------------------------------
# 1. Multimodal Evidence Verifier
# --------------------------------------------------

class EvidenceVerifier:

    def __init__(
        self,
        model_name=(
            "Qwen/"
            "Qwen2.5-VL-3B-Instruct"
        ),
        device=None
    ):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:

            if torch.cuda.is_available():

                device = "cuda"

            elif torch.backends.mps.is_available():

                device = "mps"

            else:

                device = "cpu"


        self.device = device
        self.model_name = model_name


        print(
            f"Loading evidence verifier: "
            f"{model_name}"
        )

        print(
            f"Using device: "
            f"{device}"
        )


        # --------------------------------------------------
        # Processor
        # --------------------------------------------------

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name
            )
        )


        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model_dtype = (
            torch.bfloat16
            if device == "cuda"
            else torch.float32
        )


        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_name,
                dtype=model_dtype,

                # Force eager attention for reproducible
                # multimodal verification.
                attn_implementation="eager",
            )
        )


        self.model = self.model.to(
            device
        )


        self.model.eval()


    # --------------------------------------------------
    # 2. Build verification instructions
    # --------------------------------------------------

    def build_prompt(
        self,
        original_query,
        positive_constraints,
        negative_constraints
    ):

        positive_text = (
            "\n".join(
                [
                    f"- {constraint}"
                    for constraint
                    in positive_constraints
                ]
            )
        )


        if len(
            negative_constraints
        ) > 0:

            negative_text = (
                "\n".join(
                    [
                        f"- {constraint}"
                        for constraint
                        in negative_constraints
                    ]
                )
            )

        else:

            negative_text = (
                "None"
            )


        prompt = f"""
You are the visual evidence verification component of an
image retrieval system.

Inspect the provided image carefully.

Original user request:
{original_query}

Visual requirements that SHOULD be satisfied:
{positive_text}

Visual concepts that should NOT be present:
{negative_text}

Determine whether the image satisfies the request.

Return ONLY valid JSON using exactly this structure:

{{
  "positive_checks": [
    {{
      "constraint": "constraint text",
      "satisfied": true,
      "confidence": 0.0
    }}
  ],
  "negative_checks": [
    {{
      "constraint": "constraint text",
      "present": false,
      "confidence": 0.0
    }}
  ],
  "positive_score": 0.0,
  "negative_violation_score": 0.0,
  "overall_score": 0.0,
  "overall_match": true,
  "explanation": "short explanation"
}}

Rules:

1. Check every supplied positive constraint.

2. Check every supplied negative constraint.

3. A positive constraint is satisfied only if there is
   visible evidence in the image.

4. For negative constraints, "present" means the unwanted
   concept IS visible.

5. Do not assume things that are not visibly supported.

6. confidence values must be between 0 and 1.

7. positive_score must summarize how well the positive
   requirements are satisfied, from 0 to 1.

8. negative_violation_score must represent how strongly
   unwanted content appears, from 0 to 1.

9. overall_score must be between 0 and 1, where larger means
   the image better satisfies the complete request.

10. overall_match should be true only when the image is a
    strong overall match.

11. Keep the explanation short.

12. Return JSON only.
""".strip()


        return prompt


    # --------------------------------------------------
    # 3. Extract JSON
    # --------------------------------------------------

    def extract_json(
        self,
        generated_text
    ):

        generated_text = (
            generated_text.strip()
        )


        try:

            return json.loads(
                generated_text
            )

        except json.JSONDecodeError:

            pass


        match = re.search(
            r"\{.*\}",
            generated_text,
            flags=re.DOTALL
        )


        if match is None:

            raise ValueError(
                "Verifier did not return JSON."
                "\n\n"
                f"Raw output:\n"
                f"{generated_text}"
            )


        try:

            return json.loads(
                match.group(0)
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Verifier returned malformed JSON."
                "\n\n"
                f"Raw output:\n"
                f"{generated_text}"
            ) from error


    # --------------------------------------------------
    # 4. Clamp numeric score
    # --------------------------------------------------

    def clamp_score(
        self,
        value
    ):

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            value = 0.0


        return max(
            0.0,
            min(
                1.0,
                value
            )
        )


    # --------------------------------------------------
    # 5. Validate verifier output
    #
    # Important:
    # The VLM provides the visual judgments, but WE
    # deterministically calculate the final scores.
    # --------------------------------------------------

    def validate_result(
        self,
        result,
        positive_constraints,
        negative_constraints
    ):

        # --------------------------------------------------
        # Safely retrieve model-produced checks
        # --------------------------------------------------

        model_positive_checks = result.get(
            "positive_checks",
            []
        )

        model_negative_checks = result.get(
            "negative_checks",
            []
        )


        # --------------------------------------------------
        # Index checks by normalized constraint text
        # --------------------------------------------------

        positive_lookup = {}

        for check in model_positive_checks:

            constraint = str(
                check.get(
                    "constraint",
                    ""
                )
            ).strip()


            if not constraint:

                continue


            positive_lookup[
                constraint.lower()
            ] = check


        negative_lookup = {}

        for check in model_negative_checks:

            constraint = str(
                check.get(
                    "constraint",
                    ""
                )
            ).strip()


            if not constraint:

                continue


            negative_lookup[
                constraint.lower()
            ] = check


        # --------------------------------------------------
        # Reconstruct EVERY positive constraint
        # --------------------------------------------------

        cleaned_positive_checks = []


        for constraint in positive_constraints:

            key = (
                str(
                    constraint
                )
                .strip()
                .lower()
            )


            check = positive_lookup.get(
                key
            )


            if check is None:

                cleaned_positive_checks.append(
                    {
                        "constraint":
                            constraint,

                        "satisfied":
                            False,

                        "confidence":
                            0.0,

                        "missing_from_model_output":
                            True
                    }
                )

                continue


            cleaned_positive_checks.append(
                {
                    "constraint":
                        constraint,

                    "satisfied":
                        bool(
                            check.get(
                                "satisfied",
                                False
                            )
                        ),

                    "confidence":
                        self.clamp_score(
                            check.get(
                                "confidence",
                                0.0
                            )
                        ),

                    "missing_from_model_output":
                        False
                }
            )


        # --------------------------------------------------
        # Reconstruct EVERY negative constraint
        # --------------------------------------------------

        cleaned_negative_checks = []


        for constraint in negative_constraints:

            key = (
                str(
                    constraint
                )
                .strip()
                .lower()
            )


            check = negative_lookup.get(
                key
            )


            if check is None:

                cleaned_negative_checks.append(
                    {
                        "constraint":
                            constraint,

                        "present":
                            False,

                        "confidence":
                            0.0,

                        "missing_from_model_output":
                            True
                    }
                )

                continue


            cleaned_negative_checks.append(
                {
                    "constraint":
                        constraint,

                    "present":
                        bool(
                            check.get(
                                "present",
                                False
                            )
                        ),

                    "confidence":
                        self.clamp_score(
                            check.get(
                                "confidence",
                                0.0
                            )
                        ),

                    "missing_from_model_output":
                        False
                }
            )


        # --------------------------------------------------
        # Positive score
        #
        # satisfied=True contributes confidence
        # satisfied=False contributes 0
        # --------------------------------------------------

        positive_values = []


        for check in cleaned_positive_checks:

            if check[
                "satisfied"
            ]:

                positive_values.append(
                    check[
                        "confidence"
                    ]
                )

            else:

                positive_values.append(
                    0.0
                )


        if len(
            positive_values
        ) > 0:

            positive_score = (
                sum(
                    positive_values
                )
                / len(
                    positive_values
                )
            )

        else:

            positive_score = 1.0


        # --------------------------------------------------
        # Negative violation score
        #
        # present=True contributes confidence
        # present=False contributes 0
        # --------------------------------------------------

        negative_values = []


        for check in cleaned_negative_checks:

            if check[
                "present"
            ]:

                negative_values.append(
                    check[
                        "confidence"
                    ]
                )

            else:

                negative_values.append(
                    0.0
                )


        if len(
            negative_values
        ) > 0:

            negative_violation_score = (
                sum(
                    negative_values
                )
                / len(
                    negative_values
                )
            )

        else:

            negative_violation_score = 0.0


        # --------------------------------------------------
        # All positives satisfied?
        # --------------------------------------------------

        all_positive_satisfied = all(

            check[
                "satisfied"
            ]

            for check
            in cleaned_positive_checks
        )


        # --------------------------------------------------
        # Any negative violation?
        # --------------------------------------------------

        any_negative_violation = any(

            check[
                "present"
            ]

            for check
            in cleaned_negative_checks
        )


        # --------------------------------------------------
        # Deterministic overall score
        # --------------------------------------------------

        overall_score = (
            positive_score
            * (
                1.0
                - negative_violation_score
            )
        )


        overall_score = self.clamp_score(
            overall_score
        )


        # --------------------------------------------------
        # Final match decision
        # --------------------------------------------------

        overall_match = (
            all_positive_satisfied
            and not any_negative_violation
        )


        # --------------------------------------------------
        # Preserve VLM explanation only as text
        # --------------------------------------------------

        explanation = str(
            result.get(
                "explanation",
                ""
            )
        ).strip()


        return {
            "positive_checks":
                cleaned_positive_checks,

            "negative_checks":
                cleaned_negative_checks,

            "positive_score":
                positive_score,

            "negative_violation_score":
                negative_violation_score,

            "overall_score":
                overall_score,

            "overall_match":
                overall_match,

            "explanation":
                explanation
        }


    # --------------------------------------------------
    # 6. Verify one image
    # --------------------------------------------------

    def verify(
        self,
        image_path,
        original_query,
        positive_constraints,
        negative_constraints
    ):

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        prompt = self.build_prompt(
            original_query=
                original_query,

            positive_constraints=
                positive_constraints,

            negative_constraints=
                negative_constraints
        )


        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]


        formatted_text = (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )


        inputs = self.processor(
            text=[
                formatted_text
            ],
            images=[
                image
            ],
            padding=True,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }


        input_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )


        with torch.inference_mode():

            generated_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False
                )
            )


        generated_tokens = (
            generated_ids[
                0,
                input_length:
            ]
        )


        generated_text = (
            self.processor
            .tokenizer
            .decode(
                generated_tokens,
                skip_special_tokens=True
            )
        )


        result = self.extract_json(
            generated_text
        )


        result = self.validate_result(
            result,
            positive_constraints=
                positive_constraints,
            negative_constraints=
                negative_constraints
        )


        return result
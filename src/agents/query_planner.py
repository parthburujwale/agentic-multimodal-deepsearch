import argparse
import json
import re

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# --------------------------------------------------
# 1. Query Planner
# --------------------------------------------------

class QueryPlanner:

    def __init__(
        self,
        model_name=(
            "Qwen/"
            "Qwen2.5-1.5B-Instruct"
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
            f"Loading query planner: "
            f"{model_name}"
        )

        print(
            f"Using device: "
            f"{device}"
        )

        # --------------------------------------------------
        # Tokenizer
        # --------------------------------------------------

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
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
            AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=model_dtype
            )
        )

        self.model = self.model.to(
            device
        )

        self.model.eval()

    # --------------------------------------------------
    # 2. System instructions
    # --------------------------------------------------

    def build_messages(
        self,
        query
    ):

        system_prompt = """
You are the query-planning component of an agentic
multimodal image search system.

Your job is to convert a complex natural-language image
search request into a small set of focused POSITIVE visual
retrieval queries plus explicit constraints.

Return ONLY valid JSON.

The JSON must have exactly these fields:

{
  "original_query": string,
  "search_queries": list of strings,
  "positive_constraints": list of strings,
  "negative_constraints": list of strings,
  "reasoning_goal": string
}

Rules:

1. Preserve the meaning of the original request.

2. Generate between 1 and 4 search queries.

3. search_queries must describe VISUAL CONTENT THAT SHOULD
   BE PRESENT in the desired image.

4. NEVER create a search query based only on negation.
   Examples of INVALID search queries:
   "no swimming"
   "no passengers"
   "without cars"
   "not indoors"

5. Negative requirements belong ONLY in
   negative_constraints.

6. Break complex requests into simpler positive visual
   concepts when useful.

7. Keep important relationships between objects.
   For example:
   "dog near water"
   "person riding bicycle"
   "car beside building"

8. positive_constraints must contain ALL important objects,
   attributes, actions, and relationships that the requested
   image should contain.

9. negative_constraints must contain ONLY things that the
   user explicitly said should not be present or should not
   be happening.

10. Do not invent constraints.

11. Search queries should be natural image descriptions that
    work well with a text-to-image retrieval model.

12. reasoning_goal should state what must be verified after
    retrieval, including both positive and negative
    requirements.

13. Do not include markdown.

14. Do not include explanations outside the JSON.

Example:

User:
Find a red car parked beside a building with no people.

Output:
{
  "original_query": "Find a red car parked beside a building with no people.",
  "search_queries": [
    "red car parked beside a building",
    "red car near a building"
  ],
  "positive_constraints": [
    "red car",
    "parked car",
    "building",
    "car beside building"
  ],
  "negative_constraints": [
    "people"
  ],
  "reasoning_goal": "Verify that a red car is parked beside a building and that no people are visible."
}
""".strip()

        user_prompt = (
            f"User image-search request:\n"
            f"{query}"
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        return messages

    # --------------------------------------------------
    # 3. Extract JSON from model output
    # --------------------------------------------------

    def extract_json(
        self,
        generated_text
    ):

        generated_text = (
            generated_text.strip()
        )

        # --------------------------------------------------
        # First attempt:
        # direct JSON parsing
        # --------------------------------------------------

        try:

            return json.loads(
                generated_text
            )

        except json.JSONDecodeError:

            pass

        # --------------------------------------------------
        # Second attempt:
        # find first {...} block
        # --------------------------------------------------

        match = re.search(
            r"\{.*\}",
            generated_text,
            flags=re.DOTALL
        )

        if match is None:

            raise ValueError(
                "Planner did not return "
                "a JSON object.\n\n"
                f"Raw output:\n"
                f"{generated_text}"
            )

        json_text = match.group(0)

        try:

            return json.loads(
                json_text
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Planner returned malformed JSON."
                "\n\n"
                f"Raw output:\n"
                f"{generated_text}"
            ) from error

    # --------------------------------------------------
    # 4. Detect whether a phrase represents negation
    # --------------------------------------------------

    def is_negative_phrase(
        self,
        text
    ):

        text = str(
            text
        ).strip().lower()

        negative_patterns = [
            r"\bno\b",
            r"\bnot\b",
            r"\bwithout\b",
            r"\bexclude\b",
            r"\bexcluding\b",
            r"\babsence of\b"
        ]

        for pattern in negative_patterns:

            if re.search(
                pattern,
                text
            ):

                return True

        return False

    # --------------------------------------------------
    # 5. Convert a negative phrase into the visual concept
    #    that should be absent.
    #
    # Examples:
    #
    # "no passengers"          -> "passengers"
    # "not swimming"           -> "swimming"
    # "dog not swimming"       -> "dog swimming"
    # "dog should not be swimming" -> "dog swimming"
    # "without people"         -> "people"
    # --------------------------------------------------

    def clean_negative_phrase(
        self,
        text
    ):

        text = str(
            text
        ).strip()

        # ------------------------------------------
        # "no passengers"
        # ------------------------------------------

        match = re.match(
            r"^\s*no\s+(.+)$",
            text,
            flags=re.IGNORECASE
        )

        if match:

            return (
                match.group(1)
                .strip()
            )

        # ------------------------------------------
        # "without people"
        # ------------------------------------------

        match = re.match(
            r"^\s*without\s+(.+)$",
            text,
            flags=re.IGNORECASE
        )

        if match:

            return (
                match.group(1)
                .strip()
            )

        # ------------------------------------------
        # "not swimming"
        # ------------------------------------------

        match = re.match(
            r"^\s*not\s+(.+)$",
            text,
            flags=re.IGNORECASE
        )

        if match:

            return (
                match.group(1)
                .strip()
            )

        # ------------------------------------------
        # "dog should not be swimming"
        # "dog not swimming"
        #
        # becomes:
        #
        # "dog swimming"
        # ------------------------------------------

        match = re.match(
            (
                r"^\s*(.+?)\s+"
                r"(?:should\s+)?"
                r"not\s+"
                r"(?:be\s+)?"
                r"(.+)$"
            ),
            text,
            flags=re.IGNORECASE
        )

        if match:

            subject = (
                match.group(1)
                .strip()
            )

            negative_action = (
                match.group(2)
                .strip()
            )

            return (
                f"{subject} "
                f"{negative_action}"
            )

        # ------------------------------------------
        # Fallback
        # ------------------------------------------

        cleaned = re.sub(
            (
                r"\b"
                r"(no|not|without|exclude|excluding)"
                r"\b"
            ),
            "",
            text,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned
        ).strip()

        return cleaned

    # --------------------------------------------------
    # 6. Validate + sanitize planner output
    # --------------------------------------------------

    def validate_plan(
        self,
        plan,
        original_query
    ):

        required_keys = [
            "original_query",
            "search_queries",
            "positive_constraints",
            "negative_constraints",
            "reasoning_goal"
        ]

        # --------------------------------------------------
        # Ensure required fields exist
        # --------------------------------------------------

        for key in required_keys:

            if key not in plan:

                raise ValueError(
                    f"Missing required field: "
                    f"{key}"
                )

        # --------------------------------------------------
        # Preserve exact original query
        # --------------------------------------------------

        plan[
            "original_query"
        ] = original_query

        # --------------------------------------------------
        # Validate list fields
        # --------------------------------------------------

        list_fields = [
            "search_queries",
            "positive_constraints",
            "negative_constraints"
        ]

        for field in list_fields:

            if not isinstance(
                plan[field],
                list
            ):

                raise ValueError(
                    f"{field} must be a list."
                )

        # --------------------------------------------------
        # Collect negative concepts
        # --------------------------------------------------

        cleaned_negative_constraints = []

        # Start with negatives explicitly produced
        # by the LLM.
        for item in plan[
            "negative_constraints"
        ]:

            item = str(
                item
            ).strip()

            if not item:

                continue

            if self.is_negative_phrase(
                item
            ):

                item = (
                    self.clean_negative_phrase(
                        item
                    )
                )

            if item:

                cleaned_negative_constraints.append(
                    item
                )

        # --------------------------------------------------
        # Clean search queries
        #
        # Any negated search query is removed from
        # retrieval and moved into negative constraints.
        # --------------------------------------------------

        cleaned_search_queries = []

        for query in plan[
            "search_queries"
        ]:

            query = str(
                query
            ).strip()

            if not query:

                continue

            if self.is_negative_phrase(
                query
            ):

                negative_concept = (
                    self.clean_negative_phrase(
                        query
                    )
                )

                if negative_concept:

                    cleaned_negative_constraints.append(
                        negative_concept
                    )

                continue

            cleaned_search_queries.append(
                query
            )

        # --------------------------------------------------
        # Clean positive constraints
        #
        # Negated "positive" constraints are actually
        # negative constraints.
        # --------------------------------------------------

        cleaned_positive_constraints = []

        for constraint in plan[
            "positive_constraints"
        ]:

            constraint = str(
                constraint
            ).strip()

            if not constraint:

                continue

            if self.is_negative_phrase(
                constraint
            ):

                negative_concept = (
                    self.clean_negative_phrase(
                        constraint
                    )
                )

                if negative_concept:

                    cleaned_negative_constraints.append(
                        negative_concept
                    )

                continue

            cleaned_positive_constraints.append(
                constraint
            )

        # --------------------------------------------------
        # Remove duplicates while preserving order
        # --------------------------------------------------

        cleaned_search_queries = list(
            dict.fromkeys(
                cleaned_search_queries
            )
        )

        cleaned_positive_constraints = list(
            dict.fromkeys(
                cleaned_positive_constraints
            )
        )

        cleaned_negative_constraints = list(
            dict.fromkeys(
                cleaned_negative_constraints
            )
        )

        # --------------------------------------------------
        # Maximum 4 retrieval queries
        # --------------------------------------------------

        cleaned_search_queries = (
            cleaned_search_queries[
                :4
            ]
        )

        # --------------------------------------------------
        # Always keep at least one positive query
        # --------------------------------------------------

        if len(
            cleaned_search_queries
        ) == 0:

            cleaned_search_queries = [
                original_query
            ]

        # --------------------------------------------------
        # Store sanitized results
        # --------------------------------------------------

        plan[
            "search_queries"
        ] = cleaned_search_queries

        plan[
            "positive_constraints"
        ] = cleaned_positive_constraints

        plan[
            "negative_constraints"
        ] = cleaned_negative_constraints

        plan[
            "reasoning_goal"
        ] = str(
            plan[
                "reasoning_goal"
            ]
        ).strip()

        return plan

    # --------------------------------------------------
    # 7. Generate search plan
    # --------------------------------------------------

    def plan(
        self,
        query,
        max_new_tokens=400
    ):

        messages = self.build_messages(
            query
        )

        # --------------------------------------------------
        # Convert messages to model input
        # --------------------------------------------------

        formatted_prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = self.tokenizer(
            formatted_prompt,
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

        # --------------------------------------------------
        # Generate
        # --------------------------------------------------

        with torch.inference_mode():

            generated_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=
                        max_new_tokens,
                    do_sample=False,
                    pad_token_id=(
                        self.tokenizer
                        .eos_token_id
                    )
                )
            )

        # --------------------------------------------------
        # Remove original prompt tokens
        # --------------------------------------------------

        new_tokens = (
            generated_ids[
                0,
                input_length:
            ]
        )

        generated_text = (
            self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            )
        )

        # --------------------------------------------------
        # Parse + validate JSON
        # --------------------------------------------------

        plan = self.extract_json(
            generated_text
        )

        plan = self.validate_plan(
            plan,
            original_query=query
        )

        return plan


# --------------------------------------------------
# 8. Command-line interface
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Agentic multimodal "
            "DeepSearch query planner"
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help=(
            "Natural-language image "
            "search request"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "Qwen/"
            "Qwen2.5-1.5B-Instruct"
        ),
        help=(
            "Hugging Face instruction "
            "model used for planning"
        )
    )

    args = parser.parse_args()

    planner = QueryPlanner(
        model_name=args.model
    )

    plan = planner.plan(
        args.query
    )

    print(
        "\nGenerated Search Plan"
    )

    print(
        "-" * 60
    )

    print(
        json.dumps(
            plan,
            indent=2
        )
    )


if __name__ == "__main__":

    main()

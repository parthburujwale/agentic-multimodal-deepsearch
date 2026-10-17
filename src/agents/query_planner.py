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

        device=None,

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

                dtype=model_dtype,

                # Use standard eager attention for

                # reproducible evaluation.

                attn_implementation="eager",

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

        query,

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

4. NEVER create a search query for something the user said

   should be absent.

   For example, if the user says:

   "no boat should be visible"

   then:

   VALID negative constraint:

   "boat visible"

   INVALID search query:

   "boat visible"

5. NEVER create a search query based only on negation.

   Examples of INVALID search queries:

   "no swimming"

   "no passengers"

   "without cars"

   "not indoors"

6. Negative requirements belong ONLY in

   negative_constraints.

7. A concept must NEVER appear simultaneously as:

   - a positive retrieval query

   and

   - a negative constraint.

8. Break complex requests into simpler positive visual

   concepts when useful.

9. Keep important relationships between objects.

   Examples:

   "dog near water"

   "person riding bicycle"

   "car beside building"

10. positive_constraints must contain all important objects,

    attributes, actions, and relationships that SHOULD be

    present.

11. positive_constraints must NOT contain anything the user

    explicitly requested to be absent.

12. negative_constraints must contain ONLY things that the

    user explicitly said should not be present or should not

    be happening.

13. Do not invent constraints.

14. Search queries should be natural image descriptions that

    work well with a text-to-image retrieval model.

15. reasoning_goal should state what must be verified after

    retrieval, including both positive and negative

    requirements.

16. Do not include markdown.

17. Do not include explanations outside the JSON.

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

Example with several negative requirements:

User:

Find a dog beside water with a red bicycle nearby,

but the dog should not be swimming and no boat should

be visible.

Output:

{

  "original_query": "Find a dog beside water with a red bicycle nearby, but the dog should not be swimming and no boat should be visible.",

  "search_queries": [

    "dog beside water",

    "red bicycle near dog"

  ],

  "positive_constraints": [

    "dog beside water",

    "red bicycle nearby"

  ],

  "negative_constraints": [

    "dog swimming",

    "boat visible"

  ],

  "reasoning_goal": "Verify that a dog is beside water with a red bicycle nearby, while the dog is not swimming and no boat is visible."

}

""".strip()

        user_prompt = (

            f"User image-search request:\n"

            f"{query}"

        )

        messages = [

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

        return messages

    # --------------------------------------------------

    # 3. Extract JSON from model output

    # --------------------------------------------------

    def extract_json(

        self,

        generated_text,

    ):

        """
        Parse planner output conservatively.

        The planner normally returns valid JSON, but an instruction
        model can occasionally produce a malformed echo of the
        original_query field while the remaining planner fields are
        valid. The original query is already known to plan(), so this
        method may safely recover the JSON tail beginning at
        search_queries. Other malformed planner structures still fail
        loudly rather than being silently invented or repaired.
        """

        generated_text = str(

            generated_text

        ).strip()

        # --------------------------------------------------

        # Remove optional Markdown fences. The system prompt asks

        # for JSON only, but instruction models sometimes wrap the

        # object in ```json ... ```.

        # --------------------------------------------------

        cleaned_text = re.sub(

            r"^\s*```(?:json)?\s*",

            "",

            generated_text,

            flags=re.IGNORECASE,

        )

        cleaned_text = re.sub(

            r"\s*```\s*$",

            "",

            cleaned_text,

        ).strip()

        # --------------------------------------------------

        # Attempt 1: direct JSON parsing

        # --------------------------------------------------

        try:

            parsed = json.loads(

                cleaned_text

            )

            if not isinstance(

                parsed,

                dict,

            ):

                raise ValueError(

                    "Planner JSON must be an object."

                )

            return parsed

        except json.JSONDecodeError:

            pass

        # --------------------------------------------------

        # Attempt 2: extract the first complete-looking JSON object

        # from surrounding text.

        # --------------------------------------------------

        object_start = cleaned_text.find(

            "{"

        )

        object_end = cleaned_text.rfind(

            "}"

        )

        if (

            object_start != -1

            and

            object_end > object_start

        ):

            json_text = cleaned_text[

                object_start:

                object_end + 1

            ]

            try:

                parsed = json.loads(

                    json_text

                )

                if not isinstance(

                    parsed,

                    dict,

                ):

                    raise ValueError(

                        "Planner JSON must be an object."

                    )

                return parsed

            except json.JSONDecodeError:

                pass

        # --------------------------------------------------

        # Attempt 3: conservative recovery for the specific failure

        # mode where original_query is malformed but every planner

        # field after it is valid JSON.

        #

        # The exact original query is authoritative and is restored

        # by plan() immediately after this function returns.

        # --------------------------------------------------

        search_key_match = re.search(

            r'"search_queries"\s*:',

            cleaned_text,

        )

        if (

            search_key_match is not None

            and

            object_end != -1

            and

            object_end > search_key_match.start()

        ):

            recovered_text = (

                '{"original_query": "", '

                +

                cleaned_text[

                    search_key_match.start():

                    object_end + 1

                ]

            )

            try:

                parsed = json.loads(

                    recovered_text

                )

                if not isinstance(

                    parsed,

                    dict,

                ):

                    raise ValueError(

                        "Recovered planner JSON must be an object."

                    )

                print(

                    "Planner warning: recovered valid planner "

                    "fields after malformed original_query echo."

                )

                return parsed

            except json.JSONDecodeError:

                pass

        # --------------------------------------------------

        # Do not invent planner semantics if the structured fields

        # themselves are malformed. Fail loudly so the caller knows

        # that generation did not satisfy the contract.

        # --------------------------------------------------

        raise ValueError(

            "Planner returned malformed JSON."

            "\n\n"

            f"Raw output:\n"

            f"{generated_text}"

        )

    # --------------------------------------------------

    # 4. Detect explicit negation

    # --------------------------------------------------

    def is_negative_phrase(

        self,

        text,

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

            r"\babsence of\b",

        ]

        for pattern in negative_patterns:

            if re.search(

                pattern,

                text,

            ):

                return True

        return False

    # --------------------------------------------------

    # 5. Convert explicit negation into the visual

    #    concept that should be absent

    # --------------------------------------------------

    def clean_negative_phrase(

        self,

        text,

    ):

        text = str(

            text

        ).strip()

        # ----------------------------------------------

        # "no passengers"

        # ----------------------------------------------

        match = re.match(

            r"^\s*no\s+(.+)$",

            text,

            flags=re.IGNORECASE,

        )

        if match:

            return (

                match.group(1)

                .strip()

            )

        # ----------------------------------------------

        # "without people"

        # ----------------------------------------------

        match = re.match(

            r"^\s*without\s+(.+)$",

            text,

            flags=re.IGNORECASE,

        )

        if match:

            return (

                match.group(1)

                .strip()

            )

        # ----------------------------------------------

        # "not swimming"

        # ----------------------------------------------

        match = re.match(

            r"^\s*not\s+(.+)$",

            text,

            flags=re.IGNORECASE,

        )

        if match:

            return (

                match.group(1)

                .strip()

            )

        # ----------------------------------------------

        # "dog should not be swimming"

        # "dog must not be swimming"

        # "dog not swimming"

        #

        # becomes:

        #

        # "dog swimming"

        # ----------------------------------------------

        match = re.match(

            (

                r"^\s*(.+?)\s+"

                r"(?:should\s+|must\s+)?"

                r"not\s+"

                r"(?:be\s+)?"

                r"(.+)$"

            ),

            text,

            flags=re.IGNORECASE,

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

        # ----------------------------------------------

        # Fallback

        # ----------------------------------------------

        cleaned = re.sub(

            (

                r"\b"

                r"(no|not|without|exclude|excluding)"

                r"\b"

            ),

            "",

            text,

            flags=re.IGNORECASE,

        )

        cleaned = re.sub(

            r"\s+",

            " ",

            cleaned,

        ).strip()

        return cleaned

    # --------------------------------------------------

    # 6. Normalize visual concepts for deterministic

    #    contradiction detection

    # --------------------------------------------------

    @staticmethod

    def normalize_concept(

        text,

    ):

        text = str(

            text

        ).lower()

        text = re.sub(

            r"[^a-z0-9\s]",

            " ",

            text,

        )

        text = re.sub(

            r"\s+",

            " ",

            text,

        ).strip()

        # Words that do not materially identify the

        # visual concept itself.

        removable_words = {

            "a",

            "an",

            "the",

            "is",

            "are",

            "be",

            "being",

            "to",

            "of",

        }

        tokens = [

            token

            for token in text.split()

            if token not in removable_words

        ]

        return " ".join(

            tokens

        )

    # --------------------------------------------------

    # 7. Determine whether a positive-looking phrase

    #    conflicts with a negative constraint

    # --------------------------------------------------

    def concepts_conflict(

        self,

        positive_text,

        negative_text,

    ):

        positive = (

            self.normalize_concept(

                positive_text

            )

        )

        negative = (

            self.normalize_concept(

                negative_text

            )

        )

        if (

            not positive

            or

            not negative

        ):

            return False

        # ----------------------------------------------

        # Exact normalized match

        #

        # "boat visible"

        # vs

        # "boat visible"

        # ----------------------------------------------

        if positive == negative:

            return True

        positive_tokens = set(

            positive.split()

        )

        negative_tokens = set(

            negative.split()

        )

        if (

            not positive_tokens

            or

            not negative_tokens

        ):

            return False

        # ----------------------------------------------

        # Negative concept fully contained in the

        # positive phrase.

        #

        # negative:

        #   "boat"

        #

        # positive:

        #   "boat near water"

        #

        # -> conflict

        # ----------------------------------------------

        if negative_tokens.issubset(

            positive_tokens

        ):

            return True

        # ----------------------------------------------

        # Positive phrase is simply a shorter version

        # of the forbidden concept.

        #

        # negative:

        #   "the dog swimming"

        #

        # positive:

        #   "dog swimming"

        #

        # -> conflict

        # ----------------------------------------------

        if positive_tokens.issubset(

            negative_tokens

        ):

            return True

        return False

    # --------------------------------------------------

    # 8. Remove phrases that contradict negative

    #    constraints

    # --------------------------------------------------

    def remove_negative_conflicts(

        self,

        positive_items,

        negative_constraints,

        item_type,

    ):

        cleaned = []

        for item in positive_items:

            conflicts = []

            for negative in negative_constraints:

                if self.concepts_conflict(

                    positive_text=

                        item,

                    negative_text=

                        negative,

                ):

                    conflicts.append(

                        negative

                    )

            if conflicts:

                print(

                    f"Planner sanitizer removed "

                    f"{item_type}: "

                    f"'{item}'"

                )

                print(

                    "  Conflicts with negative "

                    "constraint(s): "

                    + ", ".join(

                        conflicts

                    )

                )

                continue

            cleaned.append(

                item

            )

        return cleaned

    # --------------------------------------------------

    # 9. Validate + sanitize planner output

    # --------------------------------------------------

    def validate_plan(

        self,

        plan,

        original_query,

    ):

        required_keys = [

            "original_query",

            "search_queries",

            "positive_constraints",

            "negative_constraints",

            "reasoning_goal",

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

        # Validate field types

        # --------------------------------------------------

        list_fields = [

            "search_queries",

            "positive_constraints",

            "negative_constraints",

        ]

        for field in list_fields:

            if not isinstance(

                plan[field],

                list,

            ):

                raise ValueError(

                    f"{field} must be a list."

                )

        # --------------------------------------------------

        # Collect negative concepts

        # --------------------------------------------------

        cleaned_negative_constraints = []

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

        # Explicitly negated queries are removed from

        # retrieval and converted to negative concepts.

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

        # Explicitly negated "positive" constraints are

        # actually negative constraints.

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

        # Remove duplicates BEFORE contradiction checking

        # --------------------------------------------------

        cleaned_negative_constraints = list(

            dict.fromkeys(

                cleaned_negative_constraints

            )

        )

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

        # --------------------------------------------------

        # CRITICAL CONTRADICTION FILTER

        #

        # A phrase may look positive because it contains no

        # words like "no" or "not", while still representing

        # something the user explicitly prohibited.

        #

        # Example:

        #

        # search query:

        #   "boat visible"

        #

        # negative constraint:

        #   "boat visible"

        #

        # This phrase MUST NOT survive into retrieval.

        # --------------------------------------------------

        cleaned_search_queries = (

            self.remove_negative_conflicts(

                positive_items=

                    cleaned_search_queries,

                negative_constraints=

                    cleaned_negative_constraints,

                item_type=

                    "search query",

            )

        )

        cleaned_positive_constraints = (

            self.remove_negative_conflicts(

                positive_items=

                    cleaned_positive_constraints,

                negative_constraints=

                    cleaned_negative_constraints,

                item_type=

                    "positive constraint",

            )

        )

        # --------------------------------------------------

        # Final deduplication

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

        # Maximum four positive subqueries

        # --------------------------------------------------

        cleaned_search_queries = (

            cleaned_search_queries[

                :4

            ]

        )

        # --------------------------------------------------

        # Important:

        #

        # Do NOT insert the full original query here merely

        # because all subqueries disappeared.

        #

        # VerifiedAgenticSearch already adds the original

        # query separately as the first retrieval query.

        #

        # An empty subquery list is therefore valid and is

        # safer than accidentally reintroducing a prohibited

        # concept as a positive subquery.

        # --------------------------------------------------

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

    # 10. Generate search plan

    # --------------------------------------------------

    def plan(

        self,

        query,

        max_new_tokens=400,

    ):

        messages = (

            self.build_messages(

                query

            )

        )

        # --------------------------------------------------

        # Convert messages to model input

        # --------------------------------------------------

        formatted_prompt = (

            self.tokenizer.apply_chat_template(

                messages,

                tokenize=False,

                add_generation_prompt=True,

            )

        )

        inputs = self.tokenizer(

            formatted_prompt,

            return_tensors="pt",

        )

        inputs = {

            key:

                value.to(

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

                    ),

                )

            )

        # --------------------------------------------------

        # Remove prompt tokens

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

                skip_special_tokens=True,

            )

        )

        # --------------------------------------------------

        # Parse + sanitize

        # --------------------------------------------------

        plan = (

            self.extract_json(

                generated_text

            )

        )

        # --------------------------------------------------

        # The user-supplied query is authoritative. Never rely on

        # the language model to reproduce it byte-for-byte. This

        # also makes recovery from a malformed original_query echo

        # safe without changing the planner's semantic outputs.

        # --------------------------------------------------

        plan[

            "original_query"

        ] = query

        plan = (

            self.validate_plan(

                plan,

                original_query=query,

            )

        )

        return plan

# --------------------------------------------------

# 11. Command-line interface

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

        ),

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

        ),

    )

    args = (

        parser.parse_args()

    )

    planner = (

        QueryPlanner(

            model_name=

                args.model

        )

    )

    plan = (

        planner.plan(

            args.query

        )

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

            indent=2,

        )

    )

if __name__ == "__main__":

    main()
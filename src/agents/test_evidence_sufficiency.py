from evidence_sufficiency import (
    EvidenceSufficiencyChecker,
)


def test_sufficient_case():

    checker = EvidenceSufficiencyChecker(
        min_verification_score=0.75
    )


    result = {
        "verification": {

            "overall_match": True,

            "overall_score": 0.91,

            "positive_checks": [

                {
                    "constraint":
                        "dog near water",

                    "satisfied":
                        True,

                    "confidence":
                        0.95,
                },

                {
                    "constraint":
                        "people in background",

                    "satisfied":
                        True,

                    "confidence":
                        0.90,
                },
            ],

            "negative_checks": [

                {
                    "constraint":
                        "dog swimming",

                    "present":
                        False,

                    "confidence":
                        0.95,
                },
            ],
        }
    }


    decision = checker.evaluate(
        result
    )


    checker.print_decision(
        decision
    )


    assert (
        decision.sufficient
        is True
    )

    assert (
        decision.satisfied_positive_constraints
        == 2
    )

    assert (
        decision.total_positive_constraints
        == 2
    )

    assert (
        decision.violated_negative_constraints
        == 0
    )

    assert (
        decision.total_negative_constraints
        == 1
    )



def test_missing_positive_case():

    checker = EvidenceSufficiencyChecker(
        min_verification_score=0.75
    )


    result = {
        "verification": {

            "overall_match": False,

            "overall_score": 0.55,

            "positive_checks": [

                {
                    "constraint":
                        "red bus",

                    "satisfied":
                        True,

                    "confidence":
                        0.95,
                },

                {
                    "constraint":
                        "McDonalds in background",

                    "satisfied":
                        False,

                    "confidence":
                        0.90,
                },
            ],

            "negative_checks": [],
        }
    }


    decision = checker.evaluate(
        result
    )


    checker.print_decision(
        decision
    )


    assert (
        decision.sufficient
        is False
    )

    assert (
        decision.satisfied_positive_constraints
        == 1
    )

    assert (
        decision.total_positive_constraints
        == 2
    )

    assert (
        "McDonalds in background"
        in decision.missing_positive_constraints
    )



def test_negative_violation_case():

    checker = EvidenceSufficiencyChecker(
        min_verification_score=0.75
    )


    result = {
        "verification": {

            "overall_match": False,

            "overall_score": 0.70,

            "positive_checks": [

                {
                    "constraint":
                        "dog near water",

                    "satisfied":
                        True,

                    "confidence":
                        0.95,
                },
            ],

            "negative_checks": [

                {
                    "constraint":
                        "dog swimming",

                    "present":
                        True,

                    "confidence":
                        0.95,
                },
            ],
        }
    }


    decision = checker.evaluate(
        result
    )


    checker.print_decision(
        decision
    )


    assert (
        decision.sufficient
        is False
    )

    assert (
        decision.violated_negative_constraints
        == 1
    )

    assert (
        decision.total_negative_constraints
        == 1
    )

    assert (
        "dog swimming"
        in
        decision
        .violated_negative_constraint_names
    )



def main():

    print(
        "\nTEST 1: SUFFICIENT EVIDENCE\n"
    )

    test_sufficient_case()


    print(
        "\nTEST 2: MISSING POSITIVE EVIDENCE\n"
    )

    test_missing_positive_case()


    print(
        "\nTEST 3: NEGATIVE CONSTRAINT VIOLATION\n"
    )

    test_negative_violation_case()


    print(
        "\nAll evidence sufficiency tests passed."
    )


if __name__ == "__main__":

    main()
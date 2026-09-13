from scripts.audit_person_name_behavior import audit, paired_text_sha256


def test_name_categories_are_mutually_exclusive_and_row_based():
    pairs = [
        ("alice met bob", "alice went home"),
        ("carol spoke", "dave answered"),
        ("eve waited", "nobody came"),
        ("the room was quiet", "frank entered"),
        ("the rain stopped", "the sun appeared"),
    ]
    result = audit(
        pairs,
        {0: ["alice", "bob"], 1: ["carol"], 2: ["eve"]},
        {0: ["alice"], 1: ["dave"], 3: ["frank"]},
        system="fixture",
        digest=paired_text_sha256(pairs),
    )

    assert result["counts"] == {
        "named_target_rows": 3,
        "unnamed_target_rows": 2,
        "named_target_to_any_name": 2,
        "exact_name_recovery": 1,
        "name_substitution": 1,
        "name_deletion": 1,
        "false_name_insertion": 1,
    }
    assert result["rates"] == {
        "named_target_to_any_name": 2 / 3,
        "exact_name_recovery": 1 / 3,
        "name_substitution": 1 / 3,
        "name_deletion": 1 / 3,
        "false_name_insertion": 1 / 2,
    }

from src.stats import compute_metrics, holm_adjust


def paired_rows() -> list[dict[str, str]]:
    rows = []
    for q_id, correct in (("q1", "true"), ("q2", "false"), ("q3", "true")):
        for lang in ("kk", "en"):
            rows.append(
                {
                    "model": "fixture/model-a",
                    "q_id": q_id,
                    "rep": "0",
                    "lang": lang,
                    "is_correct": correct if lang == "kk" else "false",
                }
            )
    return rows


def test_mcnemar_exact_known_p_value() -> None:
    metrics = compute_metrics(paired_rows(), bootstrap_samples=100, seed=42)
    comparison = next(
        row
        for row in metrics
        if row["metric"] == "mcnemar_exact"
        and row["lang_a"] == "kk"
        and row["lang_b"] == "en"
    )
    assert comparison["p_value"] == 0.5
    assert comparison["effect_size"] == 2 / 3
    assert comparison["n_questions"] == 3


def test_holm_adjustment_on_three_p_values() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_question_bootstrap_is_reproducible_and_reports_small_n() -> None:
    first = compute_metrics(paired_rows(), bootstrap_samples=200, seed=42)
    second = compute_metrics(paired_rows(), bootstrap_samples=200, seed=42)
    accuracy = next(row for row in first if row["metric"] == "accuracy")

    assert first == second
    assert accuracy["accuracy"] == 2 / 3
    assert accuracy["n_questions"] == 3
    assert accuracy["preliminary"] == "true"


def test_pairs_use_question_accuracy_even_with_disjoint_repetitions() -> None:
    rows = [
        {"model": "fixture/m", "q_id": "q1", "rep": "0", "lang": "kk", "is_correct": "true"},
        {"model": "fixture/m", "q_id": "q1", "rep": "1", "lang": "kk", "is_correct": "false"},
        {"model": "fixture/m", "q_id": "q1", "rep": "2", "lang": "en", "is_correct": "false"},
    ]

    comparison = next(
        row for row in compute_metrics(rows, bootstrap_samples=20) if row["metric"] == "mcnemar_exact"
    )

    assert comparison["n"] == 1
    assert comparison["n_questions"] == 1
    assert comparison["effect_size"] == 0.5
    assert comparison["p_value"] == 1.0


def test_copying_repetitions_preserves_question_comparison() -> None:
    rows = paired_rows()
    repeated_rows = rows + [{**row, "rep": "1"} for row in rows]

    original = compute_metrics(rows, bootstrap_samples=20)
    repeated = compute_metrics(repeated_rows, bootstrap_samples=20)

    assert [row for row in repeated if row["metric"] == "mcnemar_exact"] == [
        row for row in original if row["metric"] == "mcnemar_exact"
    ]


def test_holm_adjustment_is_per_model_across_language_pairs() -> None:
    rows = []
    for model, outcomes in (
        ("model-a", {"kk": True, "ru": False, "en": False}),
        ("model-b", {"kk": True, "ru": True, "en": True}),
    ):
        for q_number in range(10):
            for lang in ("kk", "ru", "en"):
                rows.append(
                    {
                        "model": model,
                        "q_id": f"q{q_number}",
                        "rep": "0",
                        "lang": lang,
                        "is_correct": str(outcomes[lang]).lower(),
                    }
                )

    comparisons = [
        row for row in compute_metrics(rows, bootstrap_samples=20)
        if row["metric"] == "mcnemar_exact"
    ]
    for model in ("model-a", "model-b"):
        model_rows = [row for row in comparisons if row["model"] == model]
        expected = holm_adjust([float(row["p_value"]) for row in model_rows])
        assert [row["p_value_holm"] for row in model_rows] == expected

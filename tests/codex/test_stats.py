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


def test_pairs_are_aligned_by_question_and_repetition() -> None:
    rows = [
        {"model": "fixture/m", "q_id": "q1", "rep": "0", "lang": "kk", "is_correct": "true"},
        {"model": "fixture/m", "q_id": "q1", "rep": "1", "lang": "kk", "is_correct": "false"},
        {"model": "fixture/m", "q_id": "q1", "rep": "0", "lang": "en", "is_correct": "false"},
        {"model": "fixture/m", "q_id": "q1", "rep": "1", "lang": "en", "is_correct": "false"},
    ]

    comparison = next(
        row for row in compute_metrics(rows, bootstrap_samples=20) if row["metric"] == "mcnemar_exact"
    )

    assert comparison["n"] == 2
    assert comparison["n_questions"] == 1
    assert comparison["effect_size"] == 0.5

"""Question-clustered accuracy intervals and paired language comparisons."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from src.common.config import LABELS_PATH, METRICS_PATH, SEED

LANGS = ("kk", "ru", "en")
BOOTSTRAP_SAMPLES = 1000
METRIC_COLUMNS = (
    "metric",
    "model",
    "lang",
    "lang_a",
    "lang_b",
    "accuracy",
    "ci_low",
    "ci_high",
    "n",
    "n_questions",
    "effect_size",
    "p_value",
    "p_value_holm",
    "preliminary",
    "data_source",
)
Label = Mapping[str, str]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the input order."""
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def _is_correct(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"Invalid is_correct value {value!r}.")
    return value == "true"


def _bootstrap_accuracy(
    per_question: Mapping[str, Sequence[bool]],
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    q_ids = sorted(per_question)
    draws: list[float] = []
    for _ in range(samples):
        selected = rng.integers(0, len(q_ids), size=len(q_ids))
        values = [
            value
            for index in selected
            for value in per_question[q_ids[int(index)]]
        ]
        draws.append(float(np.mean(values)))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def compute_metrics(
    labels: Sequence[Label],
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> list[dict[str, str | int | float]]:
    """Compute accuracy and exact paired comparisons of question-level correct shares.

    Each shared question contributes one direction to the binomial test.
    Tied shares contribute to the pair count and effect size only.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    seen: set[tuple[str, str, str, int]] = set()
    per_question: dict[tuple[str, str], dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in labels:
        model = row["model"]
        lang = row["lang"]
        if lang not in LANGS:
            raise ValueError(f"Unsupported language {lang!r}.")
        group = (model, lang)
        q_id = row["q_id"]
        key = (*group, q_id, int(row["rep"]))
        if key in seen:
            raise ValueError(f"Duplicate model/language/question/repetition {key}.")
        correct = _is_correct(row["is_correct"])
        seen.add(key)
        per_question[group][q_id].append(correct)

    rng = np.random.default_rng(seed)
    metrics: list[dict[str, str | int | float]] = []
    question_accuracy = {
        group: {q_id: sum(values) / len(values) for q_id, values in questions.items()}
        for group, questions in per_question.items()
    }
    models = sorted({model for model, _ in per_question})
    for model in models:
        for lang in LANGS:
            group = (model, lang)
            question_values = per_question.get(group)
            if not question_values:
                continue
            values = [value for q_values in question_values.values() for value in q_values]
            ci_low, ci_high = _bootstrap_accuracy(
                question_values, rng, bootstrap_samples
            )
            n_questions = len(question_values)
            metrics.append(
                {
                    "metric": "accuracy",
                    "model": model,
                    "lang": lang,
                    "lang_a": "",
                    "lang_b": "",
                    "accuracy": float(np.mean(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": len(values),
                    "n_questions": n_questions,
                    "effect_size": "",
                    "p_value": "",
                    "p_value_holm": "",
                    "preliminary": "true" if n_questions < 30 else "false",
                    "data_source": "fixture" if model.startswith("fixture/") else "external",
                }
            )

    comparisons: list[int] = []
    for model in models:
        for index, lang_a in enumerate(LANGS):
            for lang_b in LANGS[index + 1 :]:
                values_a = question_accuracy.get((model, lang_a), {})
                values_b = question_accuracy.get((model, lang_b), {})
                paired_q_ids = sorted(values_a.keys() & values_b.keys())
                if not paired_q_ids:
                    continue
                differences = [values_a[q_id] - values_b[q_id] for q_id in paired_q_ids]
                a_higher = sum(diff > 0 for diff in differences)
                b_higher = sum(diff < 0 for diff in differences)
                discordant = a_higher + b_higher
                p_value = (
                    float(binomtest(min(a_higher, b_higher), discordant, 0.5).pvalue)
                    if discordant
                    else 1.0
                )
                q_count = len(paired_q_ids)
                comparisons.append(len(metrics))
                metrics.append(
                    {
                        "metric": "mcnemar_exact",
                        "model": model,
                        "lang": "",
                        "lang_a": lang_a,
                        "lang_b": lang_b,
                        "accuracy": "",
                        "ci_low": "",
                        "ci_high": "",
                        "n": q_count,
                        "n_questions": q_count,
                        "effect_size": float(np.mean(differences)),
                        "p_value": p_value,
                        "p_value_holm": "",
                        "preliminary": "true" if q_count < 30 else "false",
                        "data_source": "fixture" if model.startswith("fixture/") else "external",
                    }
                )

    adjusted = holm_adjust([float(metrics[index]["p_value"]) for index in comparisons])
    for index, value in zip(comparisons, adjusted):
        metrics[index]["p_value_holm"] = value
    return metrics


def read_labels(path: Path = LABELS_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_metrics(
    rows: Sequence[Mapping[str, str | int | float]], path: Path = METRICS_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=METRIC_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_stats(
    labels_path: Path = LABELS_PATH, metrics_path: Path = METRICS_PATH
) -> list[dict[str, str | int | float]]:
    metrics = compute_metrics(read_labels(labels_path))
    write_metrics(metrics, metrics_path)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output", type=Path, default=METRICS_PATH)
    args = parser.parse_args(argv)
    metrics = run_stats(args.labels, args.output)
    print(f"Wrote {len(metrics)} fixture-labelled metric rows.")


if __name__ == "__main__":
    main()

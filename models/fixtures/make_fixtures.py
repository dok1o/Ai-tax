"""Generate the synthetic, contract-shaped fixtures used until real data lands.

Every number in ``models/fixtures/*.csv`` comes from this script with SEED from
``src/common/config.py``. Nothing here is a measurement: the files are fixtures,
their names end in ``_fixture.csv`` and every consumer must label them.

Run from the repository root::

    python models/fixtures/make_fixtures.py

Outputs: prompts_fixture.csv (180 rows), tokens_fixture.csv (360 rows),
labels_fixture.csv (360 rows) and routes_fixture.csv (540 rows).
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import SEED  # noqa: E402
from src.features import text_features  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
N_QUESTIONS = 60
REPS = 1
MODELS = ("fixture-small-model", "fixture-strong-model")
STRATEGIES = ("direct", "via_english", "stronger_model")
TOKENIZERS = {
    "tiktoken/cl100k_base": {"kk": 0.42, "ru": 0.34, "en": 0.24},
    "hf-tokenizers/BAAI/bge-m3@a1b2c3d": {"kk": 0.33, "ru": 0.28, "en": 0.26},
}
CATEGORIES = ("arithmetic", "geography", "biology", "history", "physics", "everyday")
DIFFICULTY = {
    "arithmetic": 0.5,
    "geography": 0.1,
    "biology": -0.2,
    "history": -0.5,
    "physics": 0.0,
    "everyday": 0.3,
}
ANSWER_TYPES = {
    "arithmetic": "integer",
    "geography": "short_text",
    "biology": "short_text",
    "history": "short_text",
    "physics": "decimal",
    "everyday": "decimal",
}
STEMS = {
    "kk": {
        "arithmetic": "Егер бір қорапта 24 қалам болса, 7 қорапта барлығы қанша қалам болады?",
        "geography": "Қазақстанның ең биік шыңы қалай аталады және оның биіктігі неше метр?",
        "biology": "Адам денесіндегі ең үлкен мүше қандай және ол қандай қызмет атқарады?",
        "history": "Ғарышқа тұңғыш ұшқан адамның аты кім және бұл қашан болды?",
        "physics": "Жарықтың вакуумдағы жылдамдығы шамамен қаншаға тең?",
        "everyday": "Бір литр суды қайнату үшін шамамен қанша уақыт қажет?",
    },
    "ru": {
        "arithmetic": "Если в одной коробке 24 ручки, сколько всего ручек в семи коробках?",
        "geography": "Как называется самая высокая вершина Казахстана и какова её высота в метрах?",
        "biology": "Какой орган человеческого тела самый большой и какую функцию он выполняет?",
        "history": "Как звали первого человека, побывавшего в космосе, и когда это произошло?",
        "physics": "Чему примерно равна скорость света в вакууме?",
        "everyday": "Сколько примерно времени нужно, чтобы вскипятить один литр воды?",
    },
    "en": {
        "arithmetic": "If one box holds 24 pens, how many pens are there in seven boxes?",
        "geography": "What is the highest peak of Kazakhstan and how tall is it in metres?",
        "biology": "Which organ of the human body is the largest and what does it do?",
        "history": "Who was the first human to travel into space and when did it happen?",
        "physics": "What is the approximate speed of light in a vacuum?",
        "everyday": "About how long does it take to boil one litre of water?",
    },
}
CLARIFIERS = {
    "kk": (
        "Жауапты бір сөйлеммен беріңіз.",
        "Нақты санды және өлшем бірлігін көрсетіңіз.",
        "Қысқа мысал келтіріп, түсіндіріңіз.",
    ),
    "ru": (
        "Ответьте одним предложением.",
        "Укажите точное число и единицу измерения.",
        "Приведите короткий пример и поясните.",
    ),
    "en": (
        "Answer in one sentence.",
        "State the exact number and its unit.",
        "Give a short example and explain.",
    ),
}
VARIANT = {"kk": "нұсқа", "ru": "вариант", "en": "variant"}
GOLD = {"kk": "қысқа жауап", "ru": "короткий ответ", "en": "short answer"}


def prompt_text(q_index: int, lang: str, category: str) -> str:
    """Build one localized prompt with deterministic length variation."""
    parts = [STEMS[lang][category]]
    for clarifier in CLARIFIERS[lang][: q_index % 3]:
        parts.append(clarifier)
    parts.append(f"({VARIANT[lang]} {q_index % 10})")
    return " ".join(parts)


def gold_answer(lang: str, category: str, q_index: int) -> str:
    """Return a contract-typed gold answer for a fixture question."""
    if ANSWER_TYPES[category] == "short_text":
        return GOLD[lang]
    return f"{q_index + 1}.{q_index % 7}" if ANSWER_TYPES[category] == "decimal" else str(q_index * 3)


def tokens_for(text: str, tokenizer: str, lang: str, rng: np.random.Generator) -> int:
    """Simulate a tokenizer count with a per-language rate and mild nonlinearity."""
    words = len(text.split())
    base = len(text) * TOKENIZERS[tokenizer][lang]
    base *= 1.0 + 0.04 * (words % 4)
    base += 0.6 * words
    return max(1, int(round(base + rng.normal(0.0, 0.6))))


def correctness_prob(
    lang: str, model: str, category: str, kk_share: float, bonus: float,
    rng: np.random.Generator,
) -> float:
    """Logistic correctness chance; language and script effects are the signal."""
    z = 1.1 + {"kk": -1.2, "ru": -0.2, "en": 0.3}[lang]
    z += 0.9 if model.endswith("strong-model") else 0.0
    z += DIFFICULTY[category] - 1.5 * kk_share + bonus
    z += rng.normal(0.0, 0.25)
    return 1.0 / (1.0 + math.exp(-z))


def draw(prob: float, rng: np.random.Generator) -> bool:
    """Sample one graded observation."""
    return bool(rng.random() < prob)


def build_prompts() -> list[dict[str, str]]:
    """Return 180 contract-shaped prompt rows."""
    rows = []
    for index in range(N_QUESTIONS):
        q_id = f"q{index:03d}"
        category = CATEGORIES[index % len(CATEGORIES)]
        for lang in ("kk", "ru", "en"):
            rows.append(
                {
                    "prompt_id": f"{q_id}-{lang}",
                    "q_id": q_id,
                    "lang": lang,
                    "category": category,
                    "prompt": prompt_text(index, lang, category),
                    "answer_type": ANSWER_TYPES[category],
                    "gold_answer": gold_answer(lang, category, index),
                    "verified_by_native": "",
                }
            )
    return rows


def build_tokens(prompts: list[dict[str, str]], rng: np.random.Generator) -> list[dict]:
    """Return one measured-count row per prompt and tokenizer."""
    rows = []
    for row in prompts:
        for tokenizer in TOKENIZERS:
            count = tokens_for(row["prompt"], tokenizer, row["lang"], rng)
            words = len(row["prompt"].split())
            rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "q_id": row["q_id"],
                    "lang": row["lang"],
                    "tokenizer": tokenizer,
                    "word_count": words,
                    "token_count": count,
                    "fertility": round(count / words, 6),
                }
            )
    return rows


def build_labels(prompts: list[dict[str, str]], rng: np.random.Generator) -> list[dict]:
    """Return graded observations for both fixture models."""
    rows = []
    for row in prompts:
        share = text_features(row["prompt"])["kk_share"]
        for model in MODELS:
            for rep in range(REPS):
                rows.append(
                    {
                        "model": model,
                        "prompt_id": row["prompt_id"],
                        "lang": row["lang"],
                        "rep": rep,
                        "q_id": row["q_id"],
                        "parsed_answer": row["gold_answer"],
                        "is_correct": str(
                            draw(correctness_prob(row["lang"], model, row["category"], share, 0.0, rng), rng)
                        ).lower(),
                    }
                )
    return rows


def build_routes(
    prompts: list[dict[str, str]], rng: np.random.Generator
) -> list[dict]:
    """Return cost and correctness per question, language and routing strategy."""
    rows = []
    for row in prompts:
        share = text_features(row["prompt"])["kk_share"]
        prompt_tokens = tokens_for(row["prompt"], "tiktoken/cl100k_base", row["lang"], rng)
        english_tokens = tokens_for(
            prompt_text(int(row["q_id"][1:]), "en", row["category"]),
            "tiktoken/cl100k_base",
            "en",
            rng,
        )
        for strategy in STRATEGIES:
            model = MODELS[1] if strategy == "stronger_model" else MODELS[0]
            bonus = {"direct": 0.0, "via_english": 0.7, "stronger_model": 0.0}[strategy]
            if strategy == "via_english":
                input_tokens = prompt_tokens + english_tokens
                translation_tokens = int(round(prompt_tokens * 0.8))
            else:
                input_tokens = int(round(prompt_tokens * (1.05 if model == MODELS[1] else 1.0)))
                translation_tokens = 0
            output_tokens = int(rng.integers(25, 70))
            correct = draw(
                correctness_prob(row["lang"], model, row["category"], share, bonus, rng), rng
            )
            rows.append(
                {
                    "q_id": row["q_id"],
                    "prompt_id": row["prompt_id"],
                    "lang": row["lang"],
                    "strategy": strategy,
                    "model": model,
                    "input_tokens": input_tokens,
                    "translation_tokens": translation_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + translation_tokens + output_tokens,
                    "is_correct": str(correct).lower(),
                    "source": "fixture",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write one fixture CSV in contract style."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Regenerate every fixture file deterministically."""
    rng = np.random.default_rng(SEED)
    prompts = build_prompts()
    tables = {
        "prompts_fixture.csv": prompts,
        "tokens_fixture.csv": build_tokens(prompts, rng),
        "labels_fixture.csv": build_labels(prompts, rng),
        "routes_fixture.csv": build_routes(prompts, rng),
    }
    for name, rows in tables.items():
        write_csv(OUT_DIR / name, rows)
        print(f"{name}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

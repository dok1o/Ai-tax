# CSV contract v2

Use UTF-8 CSV with a header in the order shown, comma delimiters and standard CSV quoting. All fields are required. Store identifiers and answers as text, including leading zeros. Integers are decimal, floats are finite, and booleans are lowercase `true` or `false`. Empty cells are allowed only where documented below. Native verification uses integer flags. Only Astra may change this contract and must increment its version.

## prompts.csv

One row per question and language. A question has one shared `q_id` and three distinct localized prompt IDs. Both `prompt_id` and the pair (`q_id`, `lang`) are unique.

| Column | Type | Meaning |
| --- | --- | --- |
| prompt_id | string | Stable identifier for the localized prompt. |
| q_id | string | Shared question ID for pairing, splits and resampling. |
| lang | enum | kk, ru or en. |
| category | string | Neutral category shared by the three translations. |
| prompt | string | Exact question text sent to the model. |
| answer_type | enum | mcq, number or short. |
| gold_answer | string | Identical across translations for mcq (A-D) and number (finite decimal). Short answers may be localized. |
| verified_by_native | integer | 0 = unverified, 1 = verified. Native reviewer's mark. Preserve supplied values and corrections. |

V1 migration preserves prompt_id and q_id. Integer and decimal become number, short_text becomes short. The four yes/no questions now offer A (yes) and B (no) in each language. Their gold letters match across languages. Legacy true maps to 1, false and an empty legacy mark map to 0. Only a human may import new native approvals or corrections. Use the v2 worksheet for the changed questions.

## responses.csv

One successful observation per (`model`, `prompt_id`, `lang`, `rep`). The prompt ID, question ID and language must match prompts.csv. Failed attempts belong in the call log. The append-only raw cache uses the same observation key.

| Column | Type | Meaning |
| --- | --- | --- |
| model | string | Configured model identifier. |
| prompt_id | string | Localized prompt ID. |
| lang | enum | kk, ru or en. |
| rep | integer | Zero-based repetition number. |
| q_id | string | Shared question ID. |
| model_version | string | Provider model version, or unknown when undisclosed. |
| called_at | timestamp | UTC call start, ISO 8601 with Z suffix. |
| temperature | float | Sampling temperature used for the call. |
| input_tokens | nullable integer | Nonnegative provider count, empty if unavailable. |
| output_tokens | nullable integer | Nonnegative provider count, empty if unavailable. |
| latency_ms | float | Nonnegative elapsed call time in milliseconds. |
| response | string | Model answer as returned, before grading. |
| usage_available | boolean | true exactly when both provider counts are available. |

## tokens.csv

One row per (`prompt_id`, `tokenizer`). Count the exact text in prompts.csv without added special tokens. Count nonempty whitespace-delimited words. Prompts must contain at least one word.

| Column | Type | Meaning |
| --- | --- | --- |
| prompt_id | string | Localized prompt ID from prompts.csv. |
| q_id | string | Matching shared question ID. |
| lang | enum | Matching language. |
| tokenizer | string | Encoding name or tokenizer repository and immutable revision. |
| word_count | integer | Positive number of words. |
| token_count | integer | Number of tokens produced by this tokenizer. |
| fertility | float | token_count divided by word_count. |
| tokens_vs_en | nullable float | token_count divided by the English count for the same q_id and tokenizer. Empty for unpaired input. |

## labels.csv

Exactly one grade per observation in responses.csv. Join on (`model`, `prompt_id`, `lang`, `rep`). Use the matching prompt's gold answer and answer type.

| Column | Type | Meaning |
| --- | --- | --- |
| model | string | Configured model identifier. |
| prompt_id | string | Localized prompt ID. |
| lang | enum | Matching language. |
| rep | integer | Matching repetition number. |
| q_id | string | Matching shared question ID. |
| parsed_answer | string | Normalized answer, or empty on parse failure. |
| is_correct | boolean | Whether the answer matches gold. A parse failure is false. |
| parse_status | enum | graded or unparsed. An unparsed answer is scored false. |

## metrics.csv

One accuracy row per model and language, plus a paired comparison row per model and language pair. Conditional fields are empty outside the specified row type.

| Column | Type | Meaning |
| --- | --- | --- |
| metric | enum | accuracy or mcnemar_exact. |
| model | string | Model identifier. |
| lang | nullable enum | Accuracy language. |
| lang_a | nullable enum | First comparison language. |
| lang_b | nullable enum | Second comparison language. |
| accuracy | nullable float | Accuracy estimate. |
| ci_low | nullable float | Lower accuracy confidence limit. |
| ci_high | nullable float | Upper accuracy confidence limit. |
| n | integer | Observations for accuracy or paired questions for contrasts. |
| n_questions | integer | Distinct questions. |
| effect_size | nullable float | Mean paired accuracy difference, A minus B. |
| p_value | nullable float | Unadjusted paired p-value. |
| p_value_holm | nullable float | Holm-adjusted paired p-value. |
| preliminary | boolean | Small-sample flag. |
| data_source | enum | fixture or external. |

## Integration decisions

Use data/validate_prompts.py and maintain the shared topic list in data/banned_topics.txt. The trilingual draft is the author's source and retains one row per q_id. Export prompt_id in every localized prompts.csv row.
Use `python run.py tokens --input <csv> --output <csv>` for other inputs. The ratio and grade-status sidecars remain available for older readers. New consumers use the columns in tokens.csv and labels.csv.
Old append-only call records remain intact. When exporting old cached responses, derive usage_available from the presence of both counts. A successful answer with missing usage is cached and graded. Exclude unknown costs from cost comparisons and report coverage.
STATUS.md records the actual local branches, including the shared ARENA submission. Labelled demo outputs are under data/demo/ and models/fixtures/. Production CSV paths are reserved for measured runs.

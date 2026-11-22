# CSV contract v1

Use UTF-8 CSV with a header in the order shown, comma delimiters and standard CSV quoting. All fields are required. Store identifiers and answers as text, including leading zeros. Integers are decimal, floats are finite, and booleans are lowercase `true` or `false`. The sole nullable field is `verified_by_native`, where an empty cell means unreviewed. Only Astra may change this contract and must increment its version.

## prompts.csv

One row per question and language. A question has one shared `q_id` and three distinct localized prompt IDs. Both `prompt_id` and the pair (`q_id`, `lang`) are unique.

| Column | Type | Meaning |
| --- | --- | --- |
| prompt_id | string | Stable identifier for the localized prompt. |
| q_id | string | Shared question ID for pairing, splits and resampling. |
| lang | enum | kk, ru or en. |
| category | string | Neutral category shared by the three translations. |
| prompt | string | Exact question text sent to the model. |
| answer_type | enum | integer, decimal, boolean or short_text. |
| gold_answer | string | Reference answer in the declared type, localized for short_text. Booleans use true or false. |
| verified_by_native | nullable boolean | Native reviewer's mark. Preserve supplied values and corrections. |

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
| input_tokens | integer | Nonnegative provider-reported input count. |
| output_tokens | integer | Nonnegative provider-reported output count. |
| latency_ms | float | Nonnegative elapsed call time in milliseconds. |
| response | string | Model answer as returned, before grading. |

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

AGENTS.md: Language Tax in LLMs

Read this file once per session. Obey the token budget (section 5). Talk to the human in Russian, 8 lines max. Code, comments, commits and docs in English.

1. Project

Audit, predict and reduce the "language tax" of commercial LLMs for Kazakh (kk), Russian (ru), English (en): the same question costs more tokens and gets worse answers in some languages.

Pipeline: prompts -> model responses -> token and accuracy metrics -> predictors (token cost, error risk) -> router (direct / via English / stronger model) -> Streamlit dashboard.

Deadlines: A0 poster 2026-11-22, live demo and pitch 2026-11-27. Judged on originality, technical complexity, demo quality, Q&A.

Stack: Python 3.11, pandas, numpy, scikit-learn, lightgbm, shap, scipy, statsmodels, plotly, streamlit, tiktoken, transformers, pytest. A new dependency needs Astra's approval.

2. Roles

One role per session. First line of every session: ROLE: <Astra|Luna|Sol> | AGENT: <codex|arena> | TASK: <T-id>.

Role	Does	Never
Astra (admin)	Plans, writes task cards, assigns, reviews diffs, resolves conflicts, merges agent branches into dev, keeps STATUS.md	Writes feature code. Edits another agent's open branch
Luna (worker)	Implements exactly one task card on her own branch, tests, commits	Changes scope, shared contracts, or files she does not own
Sol (debugger)	Reproduces a failure, finds the root cause, makes the smallest fix plus a regression test	Adds features, refactors, touches unrelated files

Only one Astra exists: Codex. Codex may also run Luna and Sol. ARENA runs Luna and Sol on its own branches. Astra never reviews work in the same session that produced it.

3. Branches and ownership
main: protected. Only the human merges into it.
dev: integration. Only Astra merges into it, with --no-ff.
Work branches, always cut from the latest dev: codex/T-012-slug, arena/T-012-slug.
Sol branches: fix/T-012-slug, cut from the branch that has the bug.
Never force push, never rewrite shared history, never commit to main or dev directly (except Astra's merges).

Ownership, to avoid merge conflicts:

Owner	Paths
Codex	data/, src/tokens.py, src/collect.py, src/judge.py, src/stats.py, notebooks/
ARENA	src/features.py, src/train.py, src/router.py, app/, models/
Astra only	src/common/, data/SCHEMA.md, AGENTS.md, STATUS.md, run.py, requirements.txt, tasks/

Need a change outside your paths? Add a line to STATUS.md under Requests for Astra. Do not edit it yourself.

CSV schemas in data/SCHEMA.md are a frozen contract. Only Astra may change them, and must bump the version at the top of the file.

4. Workflow
Astra writes tasks/T-xxx.md (max 15 lines): goal, inputs, outputs, owned paths, acceptance checks, token budget.
Luna: git switch dev && git pull && git switch -c <branch>. Implement only the card. Run only targeted tests. Commit small: T-xxx: <imperative summary>. Set the STATUS.md row to review. Write HANDOFF.md. Stop.
Astra reviews: git diff dev...<branch> --stat, then only the relevant hunks. Runs `python run.py check`. Merges into dev, or returns the card with a short list of fixes. Astra alone resolves conflicts.
Sol takes failures. Reproduce first, then one hypothesis at a time. After 3 failed hypotheses stop and write findings in the task card.
Release: when dev passes `python run.py check` and the demo works offline, Astra writes the RELEASE.md checklist and tells the human to merge dev into main.

STATUS.md is one table: ID | title | agent | branch | state (todo/doing/review/blocked/done), plus a Requests list. Update only your own row.

5. Token budget (hard rules)
Read only what the task needs. Use rg -n, sed -n 'a,bp', git diff --stat. Never cat a large file or CSV. Use head -n 5 and wc -l.
Do not re-read a file you just wrote or edited. Do not re-read files you already hold.
Show at most 20 lines of any log, 10 rows of any DataFrame.
Send diffs and small patches, never full files that did not change.
One task per session. No side quests. No refactors nobody asked for.
Batch independent commands in one call. Run targeted tests (`python run.py test tests/test_x.py`), not the whole suite, until Astra's `python run.py check`.
Cache every API response in data/raw/ keyed by (model, prompt_id, lang, rep). Never call the API again for a key that exists.
No narration, no apologies, no restating the task or the plan. No emojis.
Final message, 8 lines max: what changed, how it was verified, what is next.
Context is running out? Write HANDOFF.md (max 20 lines: done / next / blockers / files touched / resume command) and stop.
6. Science and data integrity (non-negotiable)
Never fabricate data, results, citations or numbers. Every figure in a chart, README or poster text must trace to a file and a script that produced it. If it does not exist yet, write TODO, not a plausible value.
Split train and test by question id (GroupKFold / GroupShuffleSplit on q_id), never by row. One question in three languages must stay on one side.
Fix random seeds (SEED = 42 in src/common/). Every result must reproduce from a `python run.py` command.
Log for every API call: model name, model version, date, temperature, input tokens, output tokens, latency. Raw responses are append-only. Never overwrite them.
Report uncertainty: bootstrap confidence intervals, paired tests (McNemar or Wilcoxon), Holm correction when comparing several pairs. Label results with n. Small n means the word "preliminary".
Do not touch the verified_by_native column or a native speaker's corrections.
Prompt content stays neutral: no political, religious or offensive topics, no data about real private people.
Say plainly what is standard (LightGBM, logistic regression) and what is ours (dataset, features, router formulation, cost-quality evaluation). Do not call a standard algorithm "our algorithm".
7. Code rules
Small functions, type hints, one responsibility per file. No dead code, no commented-out code.
Paths and constants come from src/common/config.py. No hardcoded paths.
Secrets live only in .env (listed in .gitignore). Keep .env.example current. Never print or commit keys.
Add a test only for logic that can silently break: token counting, splits, metrics, parsing of model answers.
The `python run.py check` command runs Ruff lint checks, pytest and schema validation. It must pass before any task goes to review.
Demo must run offline from saved results in data/demo/ with no API call.
8. Stop and ask the human when
A step spends real money or hits a paid API limit.
A git operation is destructive or touches main.
A contest rule is unclear or a change could break it.
A secret, personal data or a licence question appears.
Two cards conflict and Astra cannot decide from this file.
9. Definition of done (per card)

Acceptance checks in the card pass, `python run.py check` passes, STATUS.md row updated, HANDOFF.md written, no files changed outside owned paths, no unexplained numbers.
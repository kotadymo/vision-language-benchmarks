# Copilot instructions: benchmark-first development

This repository is a **scientific benchmark and results publication project**, not a general-purpose software framework. Optimize for **correct measured results, reproducibility, and short iteration time**, not for the number of tests, CI jobs, abstractions, or documents produced.

These instructions apply to future Copilot tasks unless the user explicitly requests a different scope. Preserve safety, license/access requirements, and branch protection.

## Priorities

In order of importance:

1. **Complete actual pretrained-model evaluations** on the required full datasets and report the observed accuracy, retrieval, speed, and memory results.
2. **Validate measurement correctness**: correct split, candidates, prompt protocol, normalization, metric direction, sample counts, and aggregation.
3. **Preserve reproducibility**: exact library/checkpoint versions and hashes, dataset checksums, source commit, device/precision, honest run status, and immutable prior results.
4. **Apply proportionate software-quality checks**, with the smallest useful implementation, tests, and CI.

This is a priority order, not permission to skip a necessary pre-run smoke check. Scientific correctness and lawful access always take precedence over speed.

## Keep work focused and fast

- Implement only the features needed for the requested benchmark. Reuse existing adapters, registry, dataset loaders, metrics, reporting, and cache mechanisms.
- Avoid unrelated refactoring, generalized frameworks, redundant abstractions, large documentation expansions, and speculative future-proofing.
- **Do not turn each model addition into a broad test-suite expansion or CI redesign.** Test count, coverage percentage, and number of workflows are not success metrics.
- Inspect the current repository state and published results before making changes. Do not redo already completed work.
- Do not rerun whole datasets or all historical models when only a report, schema formatting, documentation, or comparison presentation has changed. Regenerate outputs from validated existing raw JSON.
- Batch related lightweight checks rather than rerunning the full suite after every small edit.
- Resolve blockers promptly, but do not fabricate results or bypass checkpoint/dataset access restrictions to satisfy a completion checklist.

## Minimum useful verification

Add or change tests **only where they directly guard the scientific measurement or the code being changed**:

- Classification Top-1/Top-5 correctness and retrieval Recall@K correctness, including image-to-text versus text-to-image direction.
- Dataset split, class labels, prompts, retrieval-variant selection, candidate counts/order, and positive-pair mapping when changed.
- Embedding shape, normalization, model-specific preprocessing/tokenizer/task prompts, or checkpoint identity when an adapter changes.
- Result schema, run identity, clean-source provenance, lifecycle, and raw-to-CSV/Markdown/plot value consistency when reporting/persistence changes.
- One focused regression case for a specific real bug, rather than many overlapping mock cases.

Preferred verification strategy:

- **Docs-only changes**: review content, links, and formatting; skip irrelevant pytest or GPU runs.
- **Report-only changes**: regenerate from existing raw JSON and inspect affected tables/metrics/plots; add one targeted regression test only for an actual bug.
- **Adapter changes**: run a small real-model CUDA image/text smoke test when permitted and technically necessary; verify output dimensions and finiteness before a full benchmark.
- **Metric/dataset changes**: run targeted deterministic tests with known expected answers before measuring; rerun only the benchmark conditions materially affected.
- **New benchmark campaign**: after a successful smoke check, freeze a clean source commit and run the requested full-dataset matrix. Actual full evaluations are the deliverable.
- **Substantive code merge**: run the existing required CI/checks once on the final candidate; fix failures. Avoid repeatedly waiting on intermediate CI runs when not required.

Default local commands should be selected by impact, not executed mechanically after every edit. Examples: `uv run --locked pytest -q tests/<affected_test>.py`, `uv run --locked ruff check <changed_python_paths>`, `uv lock --check` after dependency changes, and `git diff --check`. Existing GitHub branch rules and required checks still apply.

## CI policy

- Keep GitHub Actions **CPU-only, offline, and lightweight**. Do not download model weights or run full GPU benchmarks in CI.
- Do not add CI matrices, extra operating systems, new coverage gates, or model-specific workflow jobs by default.
- Preserve the current working CI configuration unless there is a concrete bottleneck or correctness reason to change it. Any proposed simplification should identify its benefit and risk.
- Do not require a growing set of mocked inference tests just to make CI appear comprehensive.
- Keep expensive real-model tests opt-in and run them locally only when they validate changed model behavior; do not duplicate a verified full-dataset run merely as an integration test.
- Existing small and passing tests need not be deleted just to reduce their count; avoid adding unnecessary ones.

## Results are authoritative

- Never invent scores, timings, resource measurements, successful statuses, or checkpoint licensing claims.
- Publish genuine results with full dataset coverage; label smoke/subset checks separately. Keep blocked, skipped, and failed runs out of score tables.
- Preserve prior raw JSON and provenance. Do not rewrite old runs or falsely replace their source SHA with the publication SHA.
- Keep classification P0/P1, retrieval variants, language conditions, and precisions distinct; only compare matching conditions, and state historical hardware/prompt differences.
- Report actual performance (not model-size guesses), and distinguish vector storage arithmetic from measured GPU memory.
- Keep useful raw JSON, concise CSV/Markdown reports, and relevant plots. Avoid duplicate reports, decorative charts, and repeated audits unless they answer a specific research question.
- Do not commit checkpoints, datasets, large embeddings, secrets, or unrelated local changes.

## Definition of done and communication

For benchmark requests, prioritize: **real runs complete (or a specific documented blocker), validated metrics, reproducible raw records, concise comparisons, and publication**.

For a reporting-only request, completion means the requested corrected report is consistent with the authoritative data; do not demand GPU reruns.

Summarize what was measured, which conditions were completed, evidence/links, any remaining limitations, and only checks actually run. Keep task plans and PR descriptions concise. Do not present an increased test count or passing CI alone as scientific progress.

Do not merge PRs unless the user explicitly authorizes a merge.

# Architecture comparison

> **INCOMPLETE:** 24 of 136 planned runs finished. Numbers below cover finished runs only.

Provider `groq`, model `qwen/qwen3.8-27b`, repeats 1, cases `all`. Same cases, tools, deterministic engine and guard for every column; `rules_only` is the deterministic engine with no model (offline baseline, says nothing about model reasoning).

| Metric | single | staged | workflow | rules_only |
|---|---:|---:|---:|---:|
| Valid runs (live: model answered) | 6/7 | 6/7 | 6/7 | 6/6 |
| Runs that fell back to rules-only (excluded) | 1 | 1 | 1 | n/a |
| Public cases meeting minimum expectations | 6/6 | 6/6 | 6/6 | 6/6 |
| Extended cases passing strict checks | - | - | - | - |
| **Held-out cases passing strict checks** | - | - | - | - |
| Correct next action (held-out) | - | - | - | - |
| Correct next action (extended) | - | - | - | - |
| Exact approval set (extended) | - | - | - | - |
| Human escalation correct (extended) | - | - | - | - |
| Runs with only valid evidence citations | 6/6 | 6/6 | 6/6 | n/a |
| Guard corrections (total raw model misses) | 1 | 0 | 1 | n/a |
| Runs needing >=1 guard correction | 1/6 | 0/6 | 1/6 | n/a |
| Avg latency (ms) | 131,109 | 181,998 | 41,030 | 12 |
| Avg model time (ms, excl. pacing) | - | 7,039 | 2,898 | 0 |
| p95 latency (ms) | 181,421 | 188,278 | 60,582 | 23 |
| Avg LLM calls | 2.33 | 4.00 | 1.00 | 0.00 |
| Avg tool calls | 5.33 | 6.00 | 5.00 | 5.00 |
| Avg tokens in / out | 10,335 / 807 | 14,550 / 2,006 | 3,868 / 647 | 0 / 0 |
| Avg cost per request (USD) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## How each metric is measured

- **Valid runs**: live runs where the model actually answered (`ai_status == ok`). A live run that fell back to rules-only (quota, outage, bad output) is *excluded* from every quality metric below and counted separately - it would otherwise score the rules engine as if it were the model.
- **Public minimum checks**: the starter harness's own `evaluate()` on the 6 public cases.
- **Held-out**: 8 cases written from the policy text before any model was run on them and never used to tune prompts or rules (`evals/heldout_cases.json`). Extended cases *were* used while building, so held-out is the honest generalisation check.
- **Strict checks (extended / held-out)**: exact approval set, required + forbidden risk flags, missing-information fields and the recommended action, against hand-labelled expectations in `evals/extended_cases.json`.
- **Human escalation correct**: no required Security / Privacy / Legal / Finance / CFO approval missing and the action is not less cautious than review / clarification when those are expected.
- **Guard corrections**: times the deterministic policy guard had to add, remove, raise, correct or replace something in the model's draft (raw model policy misses). Final outputs include the corrections.
- **Valid citations**: every evidence ID the model cited exists in the tool results of that run.
- **Latency**: wall-clock time of `analyze_request`, *including* client-side waiting to stay under the free tier's tokens-per-minute cap - so on Groq it mostly measures the rate limit. **Model time** is the sum of the model calls themselves and is the fairer speed comparison. **Model / tool calls and tokens**: from the run's own telemetry (provider-reported token usage).

## Per-case results (valid runs)

| Case | single | staged | workflow | rules_only |
|---|---|---|---|---|
| PUB-01 | 1/1 | 1/1 | 1/1 | 1/1 |
| PUB-02 | 1/1 | 1/1 | 1/1 | 1/1 |
| PUB-03 | 1/1 | 1/1 | 1/1 | 1/1 |
| PUB-04 | 1/1 | 1/1 | 1/1 | 1/1 |
| PUB-05 | 1/1 | 1/1 | 1/1 | 1/1 |
| PUB-06 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-01 | not run / fell back | not run / fell back | not run / fell back | not run / fell back |

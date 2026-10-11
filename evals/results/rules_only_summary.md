# Architecture comparison

Provider `None`, model `none (rules-only)`, repeats 1, cases `all`. Same cases, tools, deterministic engine and guard for every column; `rules_only` is the deterministic engine with no model (offline baseline, says nothing about model reasoning).

| Metric | rules_only |
|---|---:|
| Valid runs (live: model answered) | 34/34 |
| Runs that fell back to rules-only (excluded) | n/a |
| Public cases meeting minimum expectations | 6/6 |
| Extended cases passing strict checks | 18/20 |
| **Held-out cases passing strict checks** | 7/8 |
| Correct next action (held-out) | 7/8 |
| Correct next action (extended) | 19/20 |
| Exact approval set (extended) | 17/17 |
| Human escalation correct (extended) | 20/20 |
| Runs with only valid evidence citations | n/a |
| Guard corrections (total raw model misses) | n/a |
| Runs needing >=1 guard correction | n/a |
| Avg latency (ms) | 71 |
| Avg model time (ms, excl. pacing) | 0 |
| p95 latency (ms) | 27 |
| Avg LLM calls | 0.00 |
| Avg tool calls | 5.00 |
| Avg tokens in / out | 0 / 0 |
| Avg cost per request (USD) | 0.0000 |

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

| Case | rules_only |
|---|---|
| PUB-01 | 1/1 |
| PUB-02 | 1/1 |
| PUB-03 | 1/1 |
| PUB-04 | 1/1 |
| PUB-05 | 1/1 |
| PUB-06 | 1/1 |
| EXT-01 | 1/1 |
| EXT-02 | 1/1 |
| EXT-03 | 1/1 |
| EXT-04 | 1/1 |
| EXT-05 | 1/1 |
| EXT-06 | 1/1 |
| EXT-07 | 1/1 |
| EXT-08 | 0/1 - action route_for_approval not in ['reuse_existing_tool'] |
| EXT-09 | 1/1 |
| EXT-10 | 1/1 |
| SYN-01 | 1/1 |
| SYN-02 | 1/1 |
| SYN-03 | 1/1 |
| SYN-04 | 1/1 |
| SYN-05 | 1/1 |
| SYN-06 | 1/1 |
| SYN-07 | 1/1 |
| SYN-08 | 1/1 |
| SYN-09 | 1/1 |
| SYN-10 | 0/1 - missing flag existing_tool_overlap |
| HO-01 | 1/1 |
| HO-02 | 1/1 |
| HO-03 | 1/1 |
| HO-04 | 1/1 |
| HO-05 | 1/1 |
| HO-06 | 1/1 |
| HO-07 | 1/1 |
| HO-08 | 0/1 - action route_for_approval not in ['reuse_existing_tool'] |

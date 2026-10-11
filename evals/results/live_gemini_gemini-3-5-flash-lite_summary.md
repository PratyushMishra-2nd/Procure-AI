# Architecture comparison

Provider `gemini`, model `gemini-3.5-flash-lite`, repeats 1, cases `all`. Same cases, tools, deterministic engine and guard for every column; `rules_only` is the deterministic engine with no model (offline baseline, says nothing about model reasoning).

| Metric | single | staged | workflow | rules_only |
|---|---:|---:|---:|---:|
| Valid runs (live: model answered) | 34/34 | 34/34 | 34/34 | 34/34 |
| Runs that fell back to rules-only (excluded) | 0 | 0 | 0 | n/a |
| Public cases meeting minimum expectations | 6/6 | 6/6 | 6/6 | 6/6 |
| Extended cases passing strict checks | 20/20 | 18/20 | 18/20 | 18/20 |
| **Held-out cases passing strict checks** | 7/8 | 7/8 | 7/8 | 7/8 |
| Correct next action (held-out) | 7/8 | 7/8 | 7/8 | 7/8 |
| Correct next action (extended) | 20/20 | 19/20 | 19/20 | 19/20 |
| Exact approval set (extended) | 17/17 | 17/17 | 17/17 | 17/17 |
| Human escalation correct (extended) | 20/20 | 20/20 | 20/20 | 20/20 |
| Runs with only valid evidence citations | 34/34 | 34/34 | 34/34 | n/a |
| Guard corrections (total raw model misses) | 5 | 4 | 4 | n/a |
| Runs needing >=1 guard correction | 4/34 | 4/34 | 4/34 | n/a |
| Avg latency (ms) | 8,499 | 12,923 | 4,273 | 72 |
| Avg model time (ms, excl. pacing) | 3,010 | 4,886 | 1,956 | 0 |
| p95 latency (ms) | 8,889 | 13,275 | 4,648 | 28 |
| Avg LLM calls | 2.03 | 3.00 | 1.00 | 0.00 |
| Avg tool calls | 5.06 | 5.91 | 5.00 | 5.00 |
| Avg tokens in / out | 8,112 / 475 | 8,785 / 955 | 3,060 / 398 | 0 / 0 |
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
| EXT-01 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-02 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-03 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-04 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-05 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-06 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-07 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-08 | 1/1 | 0/1 - action route_for_approval not in ['reuse_existing_tool'] | 0/1 - action route_for_approval not in ['reuse_existing_tool'] | 0/1 - action route_for_approval not in ['reuse_existing_tool'] |
| EXT-09 | 1/1 | 1/1 | 1/1 | 1/1 |
| EXT-10 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-01 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-02 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-03 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-04 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-05 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-06 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-07 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-08 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-09 | 1/1 | 1/1 | 1/1 | 1/1 |
| SYN-10 | 1/1 | 0/1 - missing flag existing_tool_overlap | 0/1 - missing flag existing_tool_overlap | 0/1 - missing flag existing_tool_overlap |
| HO-01 | 1/1 | 1/1 | 1/1 | 1/1 |
| HO-02 | 1/1 | 1/1 | 1/1 | 1/1 |
| HO-03 | 1/1 | 1/1 | 1/1 | 1/1 |
| HO-04 | 1/1 | 1/1 | 1/1 | 1/1 |
| HO-05 | 1/1 | 1/1 | 1/1 | 1/1 |
| HO-06 | 1/1 | 1/1 | 1/1 | 1/1 |
| HO-07 | 1/1 | 1/1 | 1/1 | 1/1 |
| HO-08 | 0/1 - action route_for_approval not in ['reuse_existing_tool'] | 0/1 - action route_for_approval not in ['reuse_existing_tool'] | 0/1 - action route_for_approval not in ['reuse_existing_tool'] | 0/1 - action route_for_approval not in ['reuse_existing_tool'] |

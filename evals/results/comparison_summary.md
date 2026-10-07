# Architecture comparison

Provider `None`, model `none (rules-only)`, repeats 1, cases `all`. Same cases, same tools, same deterministic engine and guard for every column; `rules_only` is the deterministic engine with no model.

| Metric | rules_only |
|---|---:|
| Public cases meeting minimum expectations | 6/6 |
| Extended cases passing strict checks | 18/20 |
| Correct next action (extended) | 19/20 |
| Exact approval set (extended) | 17/17 |
| Human escalation correct (extended) | 20/20 |
| Runs with only valid evidence citations | 26/26 |
| Raw model policy misses caught by guard (total) | n/a |
| Runs needing >=1 guard correction | n/a |
| AI fallbacks (model unavailable) | 0 |
| Avg latency (ms) | 95 |
| p95 latency (ms) | 33 |
| Avg LLM calls | 0.00 |
| Avg tool calls | 5.00 |
| Avg tokens in / out | 0 / 0 |
| Avg cost per request (USD) | 0.0000 |

## Per-case results

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

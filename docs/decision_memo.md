# Architecture Decision Memo

## Decision
Ship **A, the single agent**: one model that chooses its tools, then the deterministic policy guard.

## Evidence
Gemini `gemini-3.5-flash-lite`, the same 34 cases for every column: 6 public, 20 extended, and 8 held-out (written before any model run, never used for tuning). One trial each. Source: `evals/results/live_gemini_gemini-3-5-flash-lite_summary.md`.

| Metric | A single | B staged | W workflow | Rules only |
|---|---:|---:|---:|---:|
| Public minimum checks | 6/6 | 6/6 | 6/6 | 6/6 |
| Extended strict checks | **20/20** | 18/20 | 18/20 | 18/20 |
| Held-out strict checks | 7/8 | 7/8 | 7/8 | 7/8 |
| Human escalation correct | 20/20 | 20/20 | 20/20 | 20/20 |
| Valid evidence citations | 34/34 | 34/34 | 34/34 | n/a |
| Runs needing a guard correction | 4/34 | 4/34 | 4/34 | n/a |
| Model calls / request | 2.0 | 3.0 | 1.0 | 0 |
| Input tokens / request | 8.1k | 8.8k | 3.1k | 0 |
| Model time / request | 3.0 s | 4.9 s | 2.0 s | 0 |

Every architecture gets every approval and escalation right, because the rules engine and guard enforce them. The architectures differ only on judgement cases, where an existing tool already covers the need:

- **EXT-08** (TaskFlow already covers Marketing's need): only A recommends reusing it.
- **SYN-10** (the need is covered by a tool in another category): only A finds it. A made the extra use-case catalog search that W cannot make and B's analyst did not.
- **HO-08** (held-out, an existing wiki covers the need): all four miss it.

## Trade-offs
- **A vs W:** W is cheaper (1 call, 2.6× fewer tokens, 1 s faster) but scores exactly like rules-only. A fixed workflow cannot go looking for evidence, and that is where the model adds value.
- **B:** costs 50% more calls than A and is the slowest, and it lost both judgement cases A won. Splitting the work across two models lost context.
- **A's cost:** 3 s and ~8k tokens per request, which is acceptable for a procurement review.

## Risks / limitations
1. One trial on a small, fast model. Repeat with 3 trials and a stronger model before production; the gaps are 1–2 cases.
2. HO-08 shows reuse judgement still fails on unseen phrasing. It needs better overlap prompting or a semantic catalog search.
3. Policy interpretations need sign-off from the policy owners.
4. Injection detection is keyword-based, plus the model's own flag.

## Why this is the right MVP
Policy safety is code: tested (117 tests) and identical across designs. The only measured difference is judgement on existing-tool overlap. A wins it at modest cost, B adds cost without gain, and W is no better than rules alone. A is the simplest design that actually earns its model call.

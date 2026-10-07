# Architecture Decision Memo

## Decision
Ship **Architecture A, the single agent**, behind the deterministic policy guard.

> **Status:** provisional. Live-model cells are pending because no API key was available. Structural and rules-only numbers are measured. Set `GEMINI_API_KEY` (free tier, `gemini-2.5-flash`) and run `python evals/run_comparison.py` to fill them in.

## Evidence
Same 26 cases for every column: 6 public cases plus 20 extended ones (all 10 dataset requests and 10 synthetic edge cases).

| Metric | Single (A) | Staged (B) | Rules only |
|---|---:|---:|---:|
| Public minimum checks | pending | pending | 6/6 |
| Extended strict checks | pending | pending | 18/20 |
| Model calls / request | 2 | 3 | 0 |
| Input tokens / request (measured prompt size) | ~7.3k | ~7.9k | 0 |
| Tool calls / request | 5 | 5 | 5 |
| Avg latency | pending | pending | 89 ms |
| Policy misses caught by guard | pending | pending | n/a |

Code alone gets every approval set, risk flag, missing field and escalation right on all 20 extended cases. It fails only the two judgement cases: EXT-08 (TaskFlow already covers the need, so reuse it) and SYN-10 (an approved tool in another category covers the need). Policy compliance is therefore architecture-independent: the guard enforces it identically after either model. The model's job is narrow: interpret the need, judge overlap and gaps, ask the requester good questions, explain.

## Trade-offs
- **What B adds:** a second model checks the analyst's claims against evidence IDs, which could catch ungrounded interpretation.
- **What B costs:** one more serial model call (the dominant latency term), 50% more free-tier request quota, ~8% more input tokens, a second prompt and schema, and a handoff that can lose context.
- **What B cannot add:** policy safety. Thresholds, reviews and escalation are already enforced in code.

## Risks / limitations
Validate before production:
1. Live A/B runs with 3+ repeats. Watch guard corrections, the judgement cases and p95 latency.
2. The model over-flagging `insufficient_fields`, which would bounce complete requests back to the requester.
3. My policy interpretations, e.g. requiring Security when the vendor API is down. Confirm them with Procurement and Security owners.
4. Integration and injection detection use keywords. Novel phrasing relies on the model's flag alone.
5. There is no seat-utilisation data, so "unused capacity" cannot be verified.

## Why this is the right MVP
The risky decisions are code: tested at their boundaries (64 tests) and identical in both designs. What remains is one interpretation step, which one agent with five tools does in two calls with full context. Splitting it adds latency and moving parts while protecting nothing the guard does not already protect.

**Decision rule:** switch to B only if live runs show it beating A by at least 2 extended cases (strict passes or guard corrections) across repeats, at acceptable latency. Otherwise keep A.

# Architecture Decision Memo

## Decision
Ship W (workflow): code runs all five tools and the policy rules, then one model call interprets and decides. Every result passes through the deterministic guard.

## Evidence
Groq `qwen/qwen3.8-27b`. The same cases were used for every column.

| Metric | W workflow | A single | B staged | Rules only |
|---|---:|---:|---:|---:|
| Public minimum checks | 5/5 | 6/6 | 6/6 | 6/6 |
| Extended strict checks | 19/20 | 19/20 | 18/20 | 18/20 |
| Held-out strict checks | 8/8 | 8/8 | 7/8 | 7/8 |
| Valid evidence citations | 100% | 100% | 100% | n/a |
| Guard corrections per run | 0.2 | 0.17 | 0 | n/a |
| Model calls / request | 1.0 | 2.3 | 4.0 | 0 |
| Input tokens / request | 3.8k | 10.3k | 14.6k | 0 |
| Model time / request | 2.9 s | ~5 s | 7.0 s | 10 ms |

Rules alone get every approval, flag and escalation right; they miss only the judgement cases, where an existing tool already covers the need. That is the model's whole job, and one call does it.

## Trade-offs
- W vs A: same quality, 2.7× fewer tokens, one call instead of 2–3. All five tools are needed every time, so letting the model pick tools adds nothing.
- B: 4× W's calls and the slowest, with no gain. A second model cannot add safety that the guard already enforces.
- W's cost: it cannot ask for extra evidence on its own. A covers that if needed.

## Risks
1. Judgement cases (EXT-08, SYN-10, HO-08) need repeated trials.
2. Policy interpretations need owner sign-off.
3. Injection detection is keyword-based plus the model's own flag.

## Why this MVP
The risky logic is code: 117 tests, the same in all designs. One model call covers the remaining interpretation. The simplest system that performs as well is W.

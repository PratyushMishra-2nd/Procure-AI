# AI Procurement Request Copilot

An internal tool for checking software and service purchase requests. It gathers evidence with tools, applies the procurement policy in code, and recommends the next action. Approval always stays with a person.

- AI interprets the need, judges overlap and gaps, and writes the recommendation.
- Code handles thresholds, budget, review freshness, Security / Privacy / Legal triggers and required fields.
- Humans approve and handle exceptions.

All data is synthetic (FDE Assessment 3 starter pack).

---

## Quick start

Requires Python 3.11+ (tested on 3.14).

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1     macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_setup.py                     # expect: PRE-FLIGHT PASSED
cp .env.example .env                       # Windows: Copy-Item .env.example .env ; then set GEMINI_API_KEY
python run_local.py                        # one command: mock vendor API :8001 + UI http://127.0.0.1:8501
```

Get a free Gemini key at https://aistudio.google.com/apikey. Without any key, everything still runs on the rules-only path. Every result says so ("AI assessment unavailable"), and it is never more permissive than the policy.

| Task | Command |
|---|---|
| Tests (engine boundaries, guard, agent loops with a scripted LLM, starter integrity) | `python -m pytest -q` |
| Public eval (starter harness) | `python evals/run_public_evals.py --architecture single` (or `staged`) |
| Full comparison, same 34 cases on every architecture | `python evals/run_comparison.py` (options: `--arch`, `--repeats`, `--cases`, `--case-delay`; `--resume` continues after a quota stop) |
| Offline baseline only (kept in separate files) | `python evals/run_comparison.py --arch rules_only` |

The eval runners start the mock vendor API themselves if it is not running.

---

## Product workflow (UI)

1. Request: choose one from the queue or enter a new one. The justification is displayed as untrusted text.
2. Recommendation: an action card (Route for approval / Check existing tool first / Specialist review required / Clarification needed), the next step, the rationale, required approvals with named approvers and the reason for each, risk flags with details, missing information, and questions for the requester.
3. Evidence panel: every finding comes from a tool, with an ID and a source reference (`vendors.csv:V005`, `GET /vendor-risk/SignalWatch`, `Policy section 4`). ★ marks findings the AI cited. Also shown: guard corrections, latency / LLM / tool / token counts, a full agent trace, and the B handoff pack.
4. Human review: a named reviewer records accept / return / escalate / reject. Overrides require a comment. Entries are appended to `runs/human_decisions.jsonl`. The copilot never approves, buys or changes a budget.

The sidebar switches between A (single agent), B (analyst + reviewer) and rules only.

## Architecture

Full diagrams, tool table, handoff format, stop and escalation conditions, assumptions and starter fixes: [docs/architecture.md](docs/architecture.md).

```
A  request -> Procurement Agent (5 tools, decision JSON) ------------------------> policy guard -> decision -> human
B  request -> Procurement Analyst (5 tools, evidence pack) -> Policy/Risk Reviewer -> policy guard -> decision -> human
```

- Tools (all deterministic): `lookup_requester`, `check_budget`, `search_software_catalog`, `get_vendor_risk` (registry + mock API), `evaluate_policy_rules`.
- Models (chosen for the free tier):
  - Gemini: `gemini-3.7-flash` on Google AI Studio (`gemini-2.5-flash` is closed to new keys) through the official `google-genai` SDK, at temperature 0. Free tier: 20 requests/day per key, so `gemini-3.7-flash` suits the UI; the full live evaluation used `gemini-3.5-flash-lite`, whose free quota is larger.
  - Groq: `qwen/qwen3.8-27b` through Groq's OpenAI-compatible API. Free tier: about 1000 requests/day but 8000 tokens/minute, so calls are paced by a token-aware limiter. Qwen was picked over `openai/gpt-oss-120b` because it issues all five tool calls in one turn; gpt-oss calls one tool per turn, roughly tripling the tokens per case. Hit its daily quota after the public cases.
  - Also supported: `google/gemini-3.7-flash` through CloseRouter, and Anthropic.
  - The provider is picked automatically from the keys present (Gemini → Groq → CloseRouter → Anthropic), or forced with `LLM_PROVIDER`. A forced provider with a missing key is reported by name, not silently ignored. There is deliberately no automatic switching between providers mid-run, so an eval never mixes models.
  - Older Gemini models cannot combine function calling with a JSON schema in one call, so tool turns run without a schema; the final answer is parsed from the model's text, and if it is not valid JSON, one tool-free schema-enforced call is made.
  - Free-tier handling:
    - Calls are paced: `GEMINI_RPM` for Gemini; `LLM_RPM` / `LLM_TPM` for Groq and CloseRouter.
    - A per-minute 429 waits for the server's retry delay. A per-day 429, or a wait longer than `LLM_MAX_RETRY_WAIT`, puts that model into cooldown and fails fast.
    - Auth and config errors (400/401/403/404) are never retried.
    - `GEMINI_API_KEY_POOL` rotates Gemini keys.
  - History is append-only, so Gemini thought signatures are replayed unchanged, including `extra_content` on the OpenAI-compatible path.
  - Groq quirks: an undeclared `json` tool call, or a 400 `tool_use_failed`, is read as a candidate answer and checked against the schema. It is never executed.
- Policy guard: a union of model and rule requirements. Every correction is logged and counted in the eval.
  - Fact flags that no tool raised are removed.
  - Actions less cautious than the rules are raised.
  - Malformed draft fields (wrong types, unknown enum values) are normalized.
  - Approval claims in the recommendation, next step or rationale are replaced. Sentences that only report an injected claim are kept.
  - Overlap: code labels each catalog match as `existing_product_expansion`, `competing_product`, `same_vendor_other_product` or `possible_alternative`. The model's classification is corrected when it contradicts that label, and `reuse_existing_tool` is accepted only when the model judged the existing tool sufficient. An add-on to a product we already own is a purchase, not reuse.
- Untrusted data: request JSON is wrapped in `<request_data>`. A regex scanner plus the model flag injection. Injected text is removed before the "is the business purpose real?" check, so REQ-1006's "Need AI ASAP" counts as too vague.
- Failure handling: vendor 503 / unreachable becomes `vendor_risk_unavailable` and requires Security. An unknown vendor is treated as having no assessment. If the model is unavailable, refuses, truncates or loops, the run degrades to a labelled rules-only result.

| File | Role |
|---|---|
| `src/policy_engine.py` | deterministic rules (policy sections 1-10) |
| `src/tools.py` | agent tools, evidence IDs, caching, call accounting |
| `src/agents.py` | A and B orchestration, tool loop |
| `src/guard.py` | merge, corrections, final `CopilotDecision` |
| `src/prompts.py` | system prompts and output schemas |
| `src/llm.py`, `src/llm_gemini.py`, `src/llm_openai_compat.py`, `src/ratelimit.py` | provider interface + selection; Gemini, Groq / CloseRouter and Anthropic adapters; request/token pacing and quota cooldown; every failure maps to the rules-only fallback |
| `src/solution.py` | `handle_request(request_id, architecture)` adapter and `analyze_request(payload)` |
| `src/contracts.py` | `ProcurementDecision` (unchanged) plus the compatible subclass `CopilotDecision` |

## Evaluation

`evals/extended_cases.json` holds the expected outcome for every dataset request, labelled by hand from the policy, plus 10 synthetic hidden-style cases. They cover exact $1,000 / $1,000.01 / $25,000.01 boundaries, an unknown vendor, a polite injection claiming Security sign-off, a vendor API outage on a clean request, employee PII through Workday, a vague purpose, an unknown requester, and a need already covered by a tool in a different category. Strict scoring checks the exact approval set, required and forbidden flags, missing fields, the action, and escalation correctness.

Results are committed under [`evals/results/`](evals/results/):
- Live runs go to `live_<provider>_<model>_` files; the offline baseline goes to `rules_only_`. One can never overwrite the other.
- Progress is saved after every case (`.jsonl`). After a quota stop, `--resume` continues and refuses to mix in a different provider or model.
- A live run that fell back to rules-only is excluded from quality metrics, counted separately, and re-run on resume.

**Live comparison on all 34 cases**, Gemini `gemini-3.5-flash-lite`, same cases, tools, rules and guard for every architecture ([summary](evals/results/live_gemini_gemini-3-5-flash-lite_summary.md)):

| Metric | A single | B staged | W workflow | Rules only |
|---|---:|---:|---:|---:|
| Public cases meeting expectations | 6/6 | 6/6 | 6/6 | 6/6 |
| Extended strict checks | **20/20** | 18/20 | 18/20 | 18/20 |
| Held-out strict checks | 7/8 | 7/8 | 7/8 | 7/8 |
| Human escalation correct (extended) | 20/20 | 20/20 | 20/20 | 20/20 |
| Runs with only valid citations | 34/34 | 34/34 | 34/34 | n/a |
| Runs needing a guard correction | 4/34 | 4/34 | 4/34 | n/a |
| Model calls / request | 2.0 | 3.0 | 1.0 | 0 |
| Input tokens / request | 8.1k | 8.8k | 3.1k | 0 |
| Model time / request | 3.0 s | 4.9 s | 2.0 s | 0 |

All architectures get every approval and escalation right, because those are enforced in code. They differ only on the judgement cases where an existing tool already covers the need:
- **EXT-08:** only A recommends reusing TaskFlow.
- **SYN-10:** only A finds the cross-category match, by making an extra catalog search.
- **HO-08:** all four miss it.

An earlier Groq `qwen/qwen3.8-27b` run covered only the public cases before hitting the daily quota ([summary](evals/results/live_groq_qwen-qwen3-8-27b_summary.md)).

Cases: 6 public (starter) + 20 extended (all 10 dataset requests + 10 synthetic edge cases) + 8 held-out, written from the policy before any model run and never used for tuning = 34. For 3 repeats: `python evals/run_comparison.py --repeats 3`.

## Decision: ship A (single agent)

See [docs/decision_memo.md](docs/decision_memo.md) (under 500 words). Policy safety lives in code and is identical across architectures; the only measured difference is overlap judgement.
- **A** wins both extended judgement cases, at 2 calls and 3 s per request.
- **W** is cheapest but scores exactly like rules-only, because a fixed workflow can't go looking for extra evidence.
- **B** costs more and lost the cases A won.

A's lead is on the extended set, which was used while building; on held-out cases there is no difference.

## Known limitations

- The live evaluation is one trial on a small, fast model (`gemini-3.5-flash-lite`). The architectures differ by 1–2 cases, so repeats and a stronger model are the next validation step.
- HO-08 (held-out reuse case) fails in every architecture: reuse judgement on unseen phrasing still needs work.
- Integration and injection detection use keywords. They are conservative but not exhaustive, and novel phrasing depends on the model's flag.
- There is no seat-utilisation data, so "unused existing capacity" can only be raised as a question.
- Policy interpretations are listed in `docs/architecture.md` and need sign-off from the policy owners.
- Providers are detected from environment variables only. Free-tier quotas (requests per minute and per day) limit how many eval repeats one key can run.
- Single-user local MVP: no auth. The decision log is a local JSONL file.

# AI Procurement Request Copilot

An internal tool for checking software and service purchase requests. It gathers evidence with tools, applies the procurement policy in code, and recommends the next action. **Approval always stays with a person.**

- **AI** interprets the need, judges overlap and gaps, and writes the recommendation.
- **Code** handles thresholds, budget, review freshness, Security / Privacy / Legal triggers and required fields.
- **Humans** approve and handle exceptions.

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

Get a free Gemini key at https://aistudio.google.com/apikey. Without any key, everything still runs on the **rules-only path**. Every result says so ("AI assessment unavailable"), and it is never more permissive than the policy.

| Task | Command |
|---|---|
| Tests (engine boundaries, guard, agent loops with a scripted LLM, starter integrity) | `python -m pytest -q` |
| Public eval (starter harness) | `python evals/run_public_evals.py --architecture single` (or `staged`) |
| Full comparison, same 26 cases on every architecture | `python evals/run_comparison.py` (options: `--arch single staged rules_only --repeats 3 --cases all`) |

The eval runners start the mock vendor API themselves if it is not running.

---

## Product workflow (UI)

1. **Request:** choose one from the queue or enter a new one. The justification is displayed as untrusted text.
2. **Recommendation:** an action card (Route for approval / Check existing tool first / Specialist review required / Clarification needed), the next step, the rationale, required approvals with named approvers and the reason for each, risk flags with details, missing information, and questions for the requester.
3. **Evidence panel:** every finding comes from a tool, with an ID and a source reference (`vendors.csv:V005`, `GET /vendor-risk/SignalWatch`, `Policy section 4`). ★ marks findings the AI cited. Also shown: guard corrections, latency / LLM / tool / token counts, a full agent trace, and the B handoff pack.
4. **Human review:** a named reviewer records accept / return / escalate / reject. Overrides require a comment. Entries are appended to `runs/human_decisions.jsonl`. The copilot never approves, buys or changes a budget.

The sidebar switches between **A (single agent)**, **B (analyst + reviewer)** and **rules only**.

## Architecture

Full diagrams, tool table, handoff format, stop and escalation conditions, assumptions and starter fixes: **[docs/architecture.md](docs/architecture.md)**.

```
A  request -> Procurement Agent (5 tools, decision JSON) ------------------------> policy guard -> decision -> human
B  request -> Procurement Analyst (5 tools, evidence pack) -> Policy/Risk Reviewer -> policy guard -> decision -> human
```

- **Tools** (all deterministic): `lookup_requester`, `check_budget`, `search_software_catalog`, `get_vendor_risk` (registry + mock API), `evaluate_policy_rules`.
- **Models** (chosen for the free tier):
  - **Default:** `gemini-2.5-flash` on Google AI Studio through the official `google-genai` SDK, at temperature 0.
  - **Alternative:** `google/gemini-3.7-flash` through CloseRouter's OpenAI-compatible endpoint.
  - **Optional:** Anthropic.
  - Provider is picked automatically from the keys present, or forced with `LLM_PROVIDER`.
  - `gemini-2.5-flash` cannot combine function calling with a JSON schema in one call. Tool turns therefore run without a schema; the final answer is parsed from the model's text, and if it is not valid JSON, one tool-free schema-enforced call is made.
  - Free-tier handling: calls are spaced to `GEMINI_RPM`, 429/503 errors back off using the server's retry delay, and `GEMINI_API_KEY_POOL` rotates keys. History is append-only, so Gemini thought signatures are replayed unchanged.
- **Policy guard:** a union of model and rule requirements. Fact flags no tool raised are removed, and actions less cautious than the rules are raised. Every correction is logged and counted in the eval.
- **Untrusted data:** request JSON is wrapped in `<request_data>`. A regex scanner plus the model flag injection. Injected text is removed before the "is the business purpose real?" check, so REQ-1006's "Need AI ASAP" counts as too vague.
- **Failure handling:** vendor 503 / unreachable becomes `vendor_risk_unavailable` and requires Security. An unknown vendor is treated as having no assessment. If the model is unavailable, refuses, truncates or loops, the run degrades to a labelled rules-only result.

| File | Role |
|---|---|
| `src/policy_engine.py` | deterministic rules (policy sections 1-10) |
| `src/tools.py` | agent tools, evidence IDs, caching, call accounting |
| `src/agents.py` | A and B orchestration, tool loop |
| `src/guard.py` | merge, corrections, final `CopilotDecision` |
| `src/prompts.py` | system prompts and output schemas |
| `src/llm.py`, `src/llm_gemini.py`, `src/llm_openai_compat.py` | provider interface + selection; Gemini, CloseRouter and Anthropic adapters; every failure maps to the rules-only fallback |
| `src/solution.py` | `handle_request(request_id, architecture)` adapter and `analyze_request(payload)` |
| `src/contracts.py` | `ProcurementDecision` (unchanged) plus the compatible subclass `CopilotDecision` |

## Evaluation

`evals/extended_cases.json` holds the expected outcome for every dataset request, labelled by hand from the policy, plus 10 synthetic hidden-style cases. They cover exact $1,000 / $1,000.01 / $25,000.01 boundaries, an unknown vendor, a polite injection claiming Security sign-off, a vendor API outage on a clean request, employee PII through Workday, a vague purpose, an unknown requester, and a need already covered by a tool in a different category. Strict scoring checks the **exact approval set**, required and forbidden flags, missing fields, the action, and escalation correctness.

Results are committed under [`evals/results/`](evals/results/).

| Metric | Single (A) | Staged (B) | Rules only |
|---|---:|---:|---:|
| Public minimum checks | pending key | pending key | 6/6 |
| Extended strict checks | pending key | pending key | 18/20 |
| Model calls / request | 2 (structural) | 3 (structural) | 0 |
| Input tokens / request | ~7.3k | ~7.9k | 0 |
| Avg latency | pending key | pending key | ~90 ms |

Rules alone pass everything except the two judgement cases (EXT-08 reuse TaskFlow, SYN-10 cross-category overlap). Those two cases are what the model is for.

> Live-model numbers have not been run yet because no API key was available while this was built. To fill them in: set `GEMINI_API_KEY`, run `python evals/run_comparison.py` (cost $0 on the free tier; about 130 calls, so roughly 15 minutes at 10 RPM), and copy the summary table from `evals/results/comparison_summary.md`. For `--repeats 3`, use `GEMINI_API_KEY_POOL` or CloseRouter to stay within the daily free quota.

## Decision: ship A (single agent)

See **[docs/decision_memo.md](docs/decision_memo.md)** (under 500 words). Policy safety lives in code and is identical for both architectures. The model's remaining job is one interpretation step, which B splits across an extra serial call and about 8% more tokens without protecting anything the guard does not already protect. The memo states the measured rule that would flip the decision.

## Known limitations

- Live-model evaluation is pending. Model judgement quality (reuse vs. buy, gap credibility, vague purpose) and real latency are not yet measured.
- Integration and injection detection use keywords. They are conservative but not exhaustive, and novel phrasing depends on the model's flag.
- There is no seat-utilisation data, so "unused existing capacity" can only be raised as a question.
- Policy interpretations are listed in `docs/architecture.md` and need sign-off from the policy owners.
- Providers are detected from environment variables only. Free-tier quotas (requests per minute and per day) limit how many eval repeats one key can run.
- Single-user local MVP: no auth. The decision log is a local JSONL file.

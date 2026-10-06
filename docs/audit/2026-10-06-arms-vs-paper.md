# Audit: CacheCLM arms vs the CLM paper (2026-10-06)

Sources:
- the paper, *Context Language Models* (arXiv 2609.37725, HTML v1);
- its code, `facebookresearch/context-language-models` (CC BY-NC 4.0; read only, nothing copied): `clm/clm_harness/clm_agent/prompts.yaml`, `clm_agent/harness.py`, `context_env/env.py`, `context_env/edit_gate.py`, `utils/budget.py`, `configs/bcp.yaml`;
- our code at `d19ac21` and the logs of 4 DeepSeek smokes plus 2 local Qwen runs.

## 1. How the paper's harness works (what we compare against)

| Part | Paper (code default; BrowseComp-Plus config where different) |
|---|---|
| Context | The whole chat transcript (assistant turns, tool outputs, nudges) is mirrored to a file with `[[CTX_TURN i role=...]]` headers. The system and task prefix is protected. The model's own edit turns stay in the transcript and must be managed too. |
| Tool | Native tool call `bash`, exactly one command per turn, THOUGHT first. The example edit uses a `python3 - <<'PY'` heredoc. |
| Prompt advice | "**Locate text with code — never paste or retype it**." "Don't `cat` this file." Keep headers. An edit forces everything after it to be re-read: batch, mind what is below the edit, and be generous in summaries. |
| Edit feedback | Every result says whether the file changed, "matched nothing", or exited non-zero. A pure compaction (changed, silent, exit 0) is a *free* turn. |
| Edit gate | `fit` (default): an edit may grow the context only if the result still fits the limit. `shrink` (BCP): an edit must shrink the context. No cost-based gate. |
| Nudges | One-shot at 25/50/75% of the budget; a tier re-arms if the model compacts back below it. 25% is informational only, because a how-to at that tier "tends to trigger premature wholesale deletion". 50% and 75% add a note contract and a "compact settled spans without wiping" rule. An urgent nudge fires every turn in a band near the limit (adaptive; 0.9 in BCP). |
| Overflow | **No mechanical truncation.** The newest turns are rolled back and a compaction is demanded, up to 50 times (6 in BCP). Then the model gets one final turn and the run is graded as-is. |
| Sampling | Temperature 0.7, top_p 0.95; max_tokens 16,384 (4,096 in BCP). Budget = 32,768-token window − max_tokens, with a 2,048-token reserve. Token counts use tiktoken, calibrated to the server's counts. |
| Summary baseline (Codex-style) | At three quarters of the budget, everything except the system prompt and the task message becomes **one** summary message, written by the same model ("a pointer, not an inventory"). No recent turns are kept verbatim. |
| Cost metric | Prefix-reuse FLOPs: prefill of the unmatched suffix plus decode. A USD estimate uses input/output list prices **with no cache-hit discount**. The paper does not study API cache billing. That gap is our research question. |
| Benchmarks | ContextBench (synthetic: Needle Retention, Sudoku, KV Store, Log Triage; up to 24× context pressure), BrowseComp-Plus, TerminalBench, EdgeBench, Software World. **Not MemoryAgentBench.** In ContextBench the agent itself asks for the next operation with `echo READY_FOR_NEXT_OP`. |

## 2. Our arms, checked against the code

| Arm | What our code does (`arms.py`, `run.py`) | Logic check |
|---|---|---|
| summary | `summary_step`: when context + next part > 75% of budget, everything before the last `keep_recent_chunks` parts becomes one summary block. That includes any earlier summary. The model is told the word target and, since `d19ac21`, the reply cap. | Correct. **Differs from the paper:** we keep the last 2 parts verbatim (1 in the local fact tasks); Codex keeps none. |
| clm | `edit_phase` before each part runs up to 3 edits, plus 3 more while the next part will not fit. Each call is system + task + ctx.txt + note. The command runs in the sandbox, and an emptied file is rolled back. The gate verdict is logged, never enforced. `fit` then cuts the oldest lines if the context still does not fit. | Correct after the fixes. **Differs:** our edit turns are not kept in context; forced truncation is mechanical; there is no growth gate. |
| gate | Same as clm, but `decide` is enforced: allow if `deleted × turns_left × hit ≥ rebilled × (miss − hit)`, or if over the limit and the edit shrinks. | Correct and our own. It ignores the **output** cost of the edit reply, because that cost is already spent when the gate decides. The prompt has to discourage expensive replies instead. |
| skill | clm plus a recipe for the task's family (one per family since `9739609`). | Our analogue of the paper's "skills" and instruction steering. Fine. |
| none / full | Questions with no context / with the whole text, no budget. | Correct floor and ceiling. On EventQA, `none` scores 0.53 because the question lists the earlier events and offers 6 options. That squeezes the useful range to 0.53–0.93. |

## 3. Every difference, what it can cost us, and a verdict

**Biases accuracy or cost:** + means it favours that side; − means it disfavours it.

| # | Difference | What it can cost us | Verdict |
|---|---|---|---|
| D1 | **No "never retype" rule.** Our prompt adds ```` ```text ```` blocks for new text but never says to leave kept text alone. The paper's prompt forbids retyping. | **The biggest cost driver we saw.** In DeepSeek smoke 4, 11 of 17 clm edits rewrote the whole context. Those rewrites cost output tokens ($1.20/M) plus the rebill, and caused the 30% cut-offs. Our CLM cost is inflated, which goes against CLM (−CLM cost). | **Fix.** Add the paper's rule: "Locate text with code; never paste or retype text you keep. Use a text block only for new text (notes, a summary that replaces a region)." |
| D2 | **Overflow: mechanical oldest-line truncation** (`fit`) after 3 condense tries. The paper truncates nothing: it rolls back, demands compaction up to 6–50 times, then ends the run. | It hides CLM failures. In local run 1, the forced cuts scored better than Qwen's own edits, because "keep newest" suits fact lists (+CLM accuracy, unearned). A cut at the front also rebills the whole context (−CLM cost). | **Change; your decision.** Paper-faithful option: 6 condense tries, then stop streaming and answer with the current context ("graded as-is"). Keep `fit` only for the summary arm's rare overflow. |
| D3 | **Edit turns are not kept in context.** Each call is a fresh system + context + note. In the paper, the model's own edit turns accumulate and must be compacted too. | It understates CLM's context and cost (+CLM cost). It removes the transcript junk that CLM is best at deleting (−CLM accuracy). In streaming memory tasks most context is signal, so the effect is small. | **Keep and document.** It makes "CLM costs more" a conservative claim. |
| D4 | **Benchmark choice.** We use MemoryAgentBench streams; the paper uses agentic tasks plus ContextBench. | The paper's gains come from deleting stale tool output and dead ends. Our streams have little junk, so smaller CLM gains are expected (−CLM accuracy). | **Keep; it is the point of the project.** State the scope: "CLM on memory streams under API cache billing", not a replication. |
| D5 | **Summary keeps the last 2 parts verbatim.** Codex keeps none. | A stronger baseline than the paper's (+summary accuracy). | **Keep and document.** It makes the comparison conservative. Matching the paper would be fairer but is not needed. |
| D6 | **No growth gate for clm and skill.** The paper's default rejects growth that would not fit. | Qwen's edit grew 15,919 → 43,596 characters and was then cut back by `fit`. Wasted calls, and the cut lands on content (−CLM accuracy). | **Fix.** Reject an edit whose result exceeds the budget, for the edit arms, as the paper's default `fit` rule does. The gate arm already rejects most growth. |
| D7 | **Heredocs are refused.** The paper's own example uses one. | A heredoc reply is refused and wastes an edit. Python blocks (since `9739609`) give the same ability. | **Keep.** The safety check stays simple and python blocks cover the need. Optionally say in the prompt that heredocs are refused (it already says "No heredocs"). |
| D8 | **Text-parsed fenced blocks** instead of native tool calls. | Source of 4 parsing bugs so far, all fixed and tested. Some risk remains. | **Keep for now.** Native tool calling (DeepSeek and Ollama support it) would be closer to the paper, but it is a bigger change. Worth doing before the full run if the budget allows. |
| D9 | **Nudges.** Ours fire once on crossing and the over-limit warning repeats, but there is no 90% urgent band and no tiered wording. The paper's 25% tier is informational only, to avoid premature wholesale deletion. | Our 25/50/75% nudges all say "Your context is over N% full", with no guidance and no warning against wiping. Wholesale deletion is what Qwen did (−CLM accuracy). | **Fix (cheap).** Use informational wording at 25%, and at 50/75% add "compact settled spans; do not wipe whole regions". |
| D10 | **Temperature 0.3** vs 0.7. | Less variance per run. With our reply cache, a low temperature plus identical prompts gave replayed failures; fixed with the edits-left counter. | **Keep and document.** |
| D11 | **Token estimate: characters ÷ 4.** The paper uses tiktoken calibrated to the server. | Budgets are about 5–10% off in real tokens. The gate's inequality does not depend on scale; billing uses provider counts. | **Keep and document.** |
| D12 | **Our cost-based gate** is not in the paper; theirs is fit/shrink only. | This is the project's contribution. Its output-cost blind spot is covered by D1. | **Keep.** |
| D13 | **Edit cap per part:** 3, plus 3 when over. The paper has a step budget, with free compaction turns. | Fine for streaming. | **Keep.** |

## 4. Other risks found in the logs

1. **Runaway replies.** Qwen sometimes loops (a reply of 495 lines with only 97 different ones) or copies the book verbatim until the cap. D1 should reduce the copying. A repetition penalty would cut loops but hurts copying, so leave it. The cap bounds the cost.
2. **Destructive python edits.** Qwen's dedupe scripts deleted most facts (final context 1,000–1,700 of 4,000 tokens; 30–43 answers wrong because the fact was missing). That is model skill, not a harness bug. D9 (no-wiping guidance) is the paper's mitigation.
3. **Weak ceilings.** On facts with the whole text, Qwen scored 0.46: 27 of its wrong answers had the fact in front of it. On multi-hop every arm was at the floor. For DeepSeek, check the `full` arm first. If the ceiling is low, a task cannot separate the arms.
4. **Noise.** One sample per task and 15–50 questions; one EventQA question is worth 0.067. The full run (7 samples) is also small. Report confidence intervals and avoid strong claims.
5. **Cache accounting.** "Billed" hits depend on the provider: DeepSeek caches in 64-token units, best-effort. The "ideal" hits are our string-prefix model. Both are reported; keep both.

## 5. Architecture today: one sample through one arm

Everything below happens in `run.py: run_sample` → `stream` (`run.py:74`) for one sample and one arm.

```
for each part i of the text (then one last pass with no part):
    incoming   = tokens of part i (+8 for its header); 0 on the last pass
    turns_left = parts still to come + number of questions

    summary arm:  summary_step (arms.py:130)
                    if context + incoming > 75% of budget:
                        blocks before the last `keep` parts  ->  one summary block (one LLM call)
    edit arms:    edit_phase (arms.py:87), a loop:
                    over  = context + incoming > budget
                    limit = 3 edits, or 3 + 3 while over                  (arms.py:96)
                    note  = control(...): usage line; OVER LIMIT every call while over;
                            a nudge on the first call after crossing 25/50/75%;
                            the last result; edits left               (arms.py:69)
                    reply = LLM(system + task + ctx.txt + note)
                    no command (READY) -> phase ends
                    cut off with no command -> phase ends, no retry
                    else run the command in the sandbox on ctx.txt
                         an emptied file is rolled back
                         gate.decide -> enforced for gate; logged only for clm and skill (arms.py:120-121)
                         result (+ "ctx.txt did not change", + rejection reason) -> next note
    fit (arms.py:149), ALL arms:
        while context + incoming > budget: cut the oldest whole lines (forced truncation)
    append part i to ctx.txt

then answer_all: each question in its own call on the frozen context
```

**Where overflow is handled today.** The edit arms get up to 6 calls to make room (3 normal, 3 condense). Then `fit` cuts the oldest lines until the part fits, and the run continues to the next part. The summary arm relies on `summary_step`, with `fit` as a fallback.

## 6. Overflow: options

The question: an edit arm has used all its condense tries and the next part still does not fit. What happens?

| | A. Keep the forced cut (today) | B. Paper style, 6 tries | C. Paper style, 3 tries |
|---|---|---|---|
| Condense tries while over | 3 | **6** (the paper's BrowseComp-Plus setting; their default is 50) | 3 |
| Still over after the tries | `fit` cuts the oldest lines; streaming continues | **Streaming stops.** The remaining parts are never read, and the questions are answered on the current context ("graded as-is", as in the paper's final turn) | Same as B |
| Harness truncation of edit arms | Yes | **No** (as in the paper) | No |
| Summary arm | `fit` fallback | Keeps the `fit` fallback (Codex-style compaction has no model failure to hide; `fit` only fires if the summary itself is too long) | Same as B |
| What it measures | The model's edits plus a "keep newest" safety net | The model's own context management only | Same, with less chance to recover |
| Bias | Hides edit failures. "Keep newest" suits fact lists, so it inflates fact accuracy for weak editors. A front cut rebills the whole context, charged to the arm | A failing model loses every remaining part: harsh, but that is the paper's semantics | Harsher than B; cheaper per failure |
| Extra cost per failure | none | up to 3 more calls | none |
| Code change | none | `edit_phase` returns an "overflowed" flag (or raises a small exception); `stream` stops, logs `{"event": "overflow_stop", "part": i}`, and skips `fit` for edit arms; `max_condense_tries: 6` in the configs; report counts `overflow_stop` next to forced truncations | Same as B, without the config change |

**My recommendation: B.** It is the paper's semantics, so a CLM failure shows up as a CLM failure. The 3 extra tries cost little on DeepSeek. Keeping `fit` for the summary arm keeps the baseline as in the paper (Codex compaction never fails silently). If B leaves too many runs stopped early on a weak model, that is a real result about the model, not about the harness.

## 7. Fixes I propose to make (code and tests, no paid calls)

| # | Fix | Exact change | Test | Why |
|---|---|---|---|---|
| F1 | **No-retype rule** (D1) | In `EDITING` (`arms.py:18`), replace the text-block line with: "Locate the text you change with code (python3 or sed on the `[[CTX_TURN ...]]` headers); never paste or retype text you keep. Use a ```` ```text ```` block only for new text, such as notes or a short summary that replaces a region." | The prompt contains the rule; a whole-context rewrite still runs (the rule is advice, not enforced) | The paper's own rule. Rewrites drove DeepSeek's cost and its cut-offs |
| F2 | **Nudge wording** (D9) | In `control` (`arms.py:78`): at 25%, "Context is at ~25% of your budget." (informational, no how-to). At 50/75%, "Context is over N% full. Compact settled regions with a short, specific summary; do not wipe whole regions you may need later." | Wording per tier; still once per crossing | The paper found how-to advice at 25% triggers wholesale deletion; Qwen ended its fact runs with only 1,000–1,700 of 4,000 tokens used |
| F3 | **Growth gate** (D6) | In `edit_phase`, before the gate: if the edit grows the context and the result exceeds the budget, roll it back with "edit rejected: it grew the context past the N-token limit", for all edit arms (the paper's default `fit`) | A growing edit that fits is kept; one that overflows is rolled back and reported | Matches the paper's default; stops edits like Qwen's 15,919 → 43,596 characters |
| F4 | **Overflow, option B** (§6) | As in the table above | The stream stops after 6 failed condense tries; questions are still answered; no `fit` for edit arms; the summary arm is unchanged | Paper semantics; removes the "keep newest" confound |
| F5 | **Report** | Count `overflow_stop` per arm, and show the `full` ceiling next to each task | Report test | So a low ceiling (Qwen facts: 0.46) is visible before arms are compared |

**Not proposed now:**
- **D8, native tool calls.** It is closer to the paper, but it is a larger change to prompts and parsing. Every parsing bug found so far is fixed and tested. I would do it before a full run if the budget allows.
- **D3, D4, D5, D10, D11, D12, D13.** They stay as they are. They are scope decisions or conservative choices, documented in §3.

**Effect on runs already made.** F1–F4 change prompts and behaviour, so the DeepSeek smokes and the local runs are no longer comparable with runs after the fixes. The small DeepSeek run (`configs/deepseek_small.yaml`, about $0.5–1.0) should be rerun from scratch after the fixes, with `runs_ds_small` moved aside.

## 8. For the reviewer

Please check in particular:
1. **F1 wording.** Does "never retype text you keep" still let the skill arm write its event logs? Those are new text that replaces a part, so they should be allowed.
2. **Option B fairness.** The edit arms lose `fit` but the summary arm keeps it. Is that fair? The argument: Codex compaction always produces a summary, so `fit` there only trims an over-long summary. It does not replace a missing decision.
3. **F3.** Should the growth gate also apply to the `gate` arm? It rejects most growth (no saving, positive cost), but a **pure append** costs nothing, so `gate.decide` allows it even while over the limit (`overflow=True`, `len(new) > len(old)`: the overflow branch does not apply, and 0 ≥ 0 passes). So the gate arm can also grow past the budget.
4. **Section 3.** Is any difference between our code and the paper's harness missing?

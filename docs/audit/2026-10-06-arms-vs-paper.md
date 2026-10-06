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

## 5. Recommended before the next paid run

| Priority | Change | Size |
|---|---|---|
| 1 | D1: the paper's "never retype text you keep" rule | Prompt, 2 lines plus a test |
| 2 | D9: paper-style nudge wording (informational at 25%; no wiping at 50/75%) | Prompt, small |
| 3 | D6: the paper's `fit` gate for the edit arms (reject edits that end over the budget) | Small code change plus a test |
| 4 | D2: overflow semantics (needs your decision) | Small to medium code change plus tests |
| Optional | D8: native tool calls | Medium |

D3, D4, D5, D10, D11, D12 and D13 stay as they are and are documented as scope or conservative choices.

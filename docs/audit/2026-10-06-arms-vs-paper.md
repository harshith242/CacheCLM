# Architecture review and CLM audit (2026-10-06)

Status: **major corrections required before another paid run**.

This review separates three sources that the working audit below sometimes
combines:

- the [Context Language Models paper](https://arxiv.org/html/2609.37725);
- the official
  [`facebookresearch/context-language-models`](https://github.com/facebookresearch/context-language-models)
  repository, whose harness has evolved beyond some paper configurations;
- the CacheCLM implementation reviewed in `src/cacheclm`.

The project is a useful **extension**, not a replication: it asks whether
model-directed context editing pays under hosted-API prefix-cache billing on
MemoryAgentBench, and whether a price-aware edit gate helps.

## Executive verdict

The implementation is small and mostly understandable. `run.py` owns the
experiment loop, `arms.py` owns context-policy behavior, `gate.py` is a pure
decision module, and `sandbox.py` is a concrete command adapter. The main
problem is experimental validity rather than code complexity.

Do not treat the existing arm comparison as final because:

1. query accuracy and query cost are confounded by arm-specific system prompts;
2. mechanical `fit` truncation can rescue failed edit arms;
3. the growth rule does not protect all arms;
4. the audit attributes repository behavior to the paper;
5. reporting hides some paid edit attempts and infers stream identity from
   equal aggregate cost.

F1-F5 in the working audit are necessary in part, but they are not sufficient.

## Findings

### High — Query evaluation is not arm-neutral

`answer_all` builds the question system prompt with `system(arm, ...)`
(`src/cacheclm/run.py:55-62`). Therefore:

- `summary` answers with the summary instruction;
- `clm` and `gate` answer with the full editing protocol;
- `skill` also answers with its task-family context-management recipe;
- `none` and `full` answer with only the base instruction.

The final `ANSWER_NOTE` reduces accidental editing, but it does not remove the
prompt-content, prompt-length, accuracy, or cache-cost confound. The `skill`
arm in particular can receive answer-relevant guidance that the other arms do
not.

**Required correction:** freeze context after streaming and answer every arm
with one common query prompt. Keep only the repeat salt and task-family query
template. This must be fixed before interpreting accuracy or billed-dollar
differences.

### High — The draft conflates paper claims with repository behavior

The paper's Appendix E says:

- CLM receives an editing reminder 2,048 tokens before the budget;
- BrowseComp-Plus uses a 23,560-token budget, 100 turns, and at most six
  rollback retries;
- TerminalBench 2.1 and TBLite end when a request exceeds the budget;
- EdgeBench rolls back the latest turn up to 50 times.

The current repository instead contains a reusable `BudgetController` with
25/50/75% tiered nudges, re-arming, an urgent band, `fit`/`shrink` edit gates,
and benchmark-specific configuration. Its current `configs/bcp.yaml` says
28,672 tokens, 500 steps, six retries, and a 0.9 urgent band.

These are valid implementation details, but they are not all paper claims.
The paper/repository discrepancy is version drift and must be stated
explicitly.

### High — Proposed overflow option B is not paper-faithful

The paper's BrowseComp-Plus behavior rolls back the newest **agent turns** and
retries up to six times across the run. CacheCLM predicts overflow before an
incoming stream chunk and has no equivalent agent turn to roll back.

Also, CacheCLM's current limit is:

`max_edits_per_chunk + max_condense_tries`

so the current over-limit maximum is 3 + 3 = 6 edit attempts. Setting
`max_condense_tries: 6` would produce 3 + 6 = 9 attempts, not six.

Stopping the stream and answering from the retained prefix is a defensible,
transparent CacheCLM policy, but it should be called **CLM-inspired**, not
paper semantics. Record the unread chunk count and overflow location.

### High — The growth gate must cover every edit arm

The existing `gate` arm permits a pure append because `rebilled_tokens == 0`
and `saving == cost == 0` (`src/cacheclm/gate.py:14-25`). While already over
budget, that append can still be accepted. `clm` and `skill` are explicitly
ungated (`src/cacheclm/arms.py:10,120-121`).

Apply the repository-style fit rule before the price gate for **all** edit
arms: reject growth when `task + candidate + incoming` exceeds the enforced
budget. The price gate remains a second decision for the `gate` arm.

### High — The cost novelty is narrower than stated

The paper's primary metric is prefix-reuse FLOPs, which already charges
prefill from the first cache mismatch plus decode. It also studies
cache-efficient serving through Suffix Cache Reuse, and some experiments
report gateway/API USD.

CacheCLM's actual contribution is narrower and still useful:

> decide whether an edit is worthwhile using hosted-provider cache-hit,
> cache-miss, and output prices, then evaluate the policy on memory streams.

Do not claim that the paper ignores caching or API cost altogether. Do say
that its main zero-shot comparisons do not optimize edit acceptance using a
provider's billed prefix-cache price ratio.

### Medium — The summary baseline comparison is only partially verified

The paper says the summary harness uses Codex prompts and compacts at 75%.
Appendix D's ContextBench skill says everything except the system and task
message becomes one summary, described as “a pointer, not an inventory.”
That supports “keep no recent turns” for that diagnostic setup, but the
released repository does not provide enough evidence to generalize the exact
message-retention policy to every headline benchmark.

State that CacheCLM's two-chunk tail is a stronger local baseline, without
claiming exact equivalence to every paper run.

### Medium — Paid edit attempts are under-reported

`report.run_row` counts only changed edit events
(`src/cacheclm/report.py:54-64`). Refused commands, no-op commands, malformed
responses, and some failed attempts still consume output tokens but do not
appear in “Edits” or “Rejected.”

Report at least: edit calls, runnable commands, changed edits, accepted edits,
price-gate rejections, sandbox refusals, no-ops, cut-offs, and growth
rejections.

### Medium — Stream de-duplication uses cost as an identity test

`report.merge` treats equal rounded `billed_stream` values as evidence that
two samples replayed one identical stream (`src/cacheclm/report.py:76-87`).
Equal cost is not proof of equal prompts, replies, edits, or final context.

Log a deterministic stream-trajectory hash and merge only matching hashes.
Otherwise sum the streams and mark the unit as diverged.

### Medium — Important harness differences are missing

Add these differences to any final comparison:

- CacheCLM does not retain assistant edit turns or command results as transcript
  turns; it reconstructs each call from task + current file + control note.
- CacheCLM does not implement free successful compaction turns. Every runnable
  command consumes a per-phase edit slot.
- CacheCLM does not re-arm threshold nudges after compaction, and an over-limit
  warning suppresses the threshold wording.
- CacheCLM accepts the first runnable fenced block and does not enforce a
  THOUGHT section or reject multiple runnable blocks.
- CacheCLM uses characters / 4 for enforcement; the repository uses tiktoken
  and calibrates its budget count to provider usage.
- CacheCLM mechanically cuts oldest lines for every arm after policy execution;
  the paper does not use this as a universal CLM fallback.

### Medium — Source and license wording needs correction

The arXiv paper is CC BY 4.0; the official code repository is CC BY-NC 4.0.
The proposed F1/F2 wording closely follows `prompts.yaml` and `budget.py`, so
“read only, nothing copied” is no longer accurate if that wording is adopted
verbatim.

Prefer original paraphrasing and cite the repository as inspiration. If exact
prompt text is copied, preserve attribution and check that the project's
intended use is compatible with the repository's non-commercial license.

## Architecture assessment

### Current module map

```text
config / CLI
    -> run_sample
       -> stream
          -> summary_step OR edit_phase
             -> sandbox.run
             -> gate.decide
          -> fit
       -> answer_all
       -> Recorder / JSONL
    -> report.load_units -> report.write_report
```

- `gate.py` is a deep module: a small interface hides the price decision and
  keeps the rule testable.
- `ctxfile.py` provides useful locality for the transcript format and survives
  the deletion test; removing it would spread header invariants across the
  runner, arms, and tests.
- `sandbox.py` is a real seam with one concrete macOS adapter. It has leverage,
  but portability is currently limited to `sandbox-exec`.
- `arms.py` has high leverage but low locality around overflow: prompt protocol,
  reply parsing, sandbox execution, budget control, edit commitment, and gate
  logging all meet in one loop.
- `run.py` is the correct orchestration module, but unconditional `fit` after
  every policy erases arm-specific failure semantics.
- `budget.py` is shallow but harmless: it isolates persistent spend accounting
  and locking. Deleting it would move that state into CLI construction without
  improving the experiment.

### Deepening candidates

**Strong — make overflow an explicit policy outcome.** Today `edit_phase`
returns only text, so `stream` cannot distinguish READY, exhausted attempts,
cut-off output, and unresolved overflow. Concentrate those outcomes in the
context-policy module and let `stream` log and execute the selected experiment
policy. This improves locality and prevents `fit` from silently changing what
an arm means.

**Strong — separate streaming prompts from query prompts.** The query phase
needs one arm-neutral adapter. This creates a clean seam between “how context
was produced” and “how retained context is evaluated,” increasing leverage and
removing the largest confound.

**Worth exploring — make edit accounting transactional.** One edit attempt
should produce one structured record covering parse, sandbox, growth rule,
price gate, commit, and cost-relevant counters. The interface is the test
surface; currently tests inspect loosely related JSON fields and can miss paid
failure modes.

**Worth exploring — identify streams by content, not cost.** A trajectory hash
would make report de-duplication auditable and remove an implicit coupling
between billing and statistical units.

## Answers to the original reviewer questions

1. **F1 and skill event logs:** yes, event logs remain possible if the rule
   forbids retyping retained text but allows new replacement notes. Use original
   wording and cite the repository inspiration.
2. **Summary keeps `fit`, edit arms lose it:** defensible only as an explicit
   experiment policy. It is asymmetric. Report summary forced truncations and
   do not call the edit-arm stop behavior paper-faithful.
3. **F3 on the `gate` arm:** yes, required. Run the fit/growth rule for all
   edit arms before the price-aware decision.
4. **Missing paper differences:** yes. The omitted items are listed above;
   the most material are free edit turns, per-benchmark overflow, transcript
   retention, nudge re-arming, token counting, and native-tool enforcement.

## Revised action plan

Before another paid run:

1. use one common query prompt for all arms and add a regression test;
2. enforce the growth/fit rule for `clm`, `gate`, and `skill`;
3. choose and name CacheCLM's overflow policy; remove unconditional `fit` from
   edit arms if the goal is to expose edit failure;
4. report overflow location, unread chunks, and all edit-attempt outcomes;
5. hash stream trajectories before de-duplicating unit cost;
6. update the comparison language and source attribution;
7. rerun from fresh run directories because prompt and behavior changes make
   existing runs incomparable.

Optional paper-alignment work, not required for the hosted-cache research
question: native tool calls, retained edit transcript, free compaction turns,
repository-style re-arming nudges, and tokenizer-based budget enforcement.

---

## Response to the review (implementer, 2026-10-06)

Each finding was checked against the code at `1cbb65a` and against the paper PDF
(arXiv 2609.37725, Appendix E, read from the PDF text). No code has changed yet;
this section is the plan.

| # | Finding | Why the reviewer raised it | Correct? | Resolution |
|---|---|---|---|---|
| 1 | Query evaluation is not arm-neutral | `answer_all` builds its system prompt with `system(arm, ...)` (`run.py:58`). The summary, edit and skill arms answer under different instructions: the skill arm with its recipe, the reference arms with only the base line. | **Yes.** I chose this so the query calls would share the cache prefix with streaming. That optimises cost realism at the price of a confound in both accuracy and prompt length. | One common query prompt for every arm: the base line, `ANSWER_NOTE` and the family's query template, keeping only the `Run r{repeat}` salt. Every arm's first query then misses the cache on its context once. That cost is the same in kind for all arms (at most ~10K tokens × $0.30/M ≈ $0.003) and the report will say so. Test: with the same context, the query messages are identical across all 6 arms. |
| 2 | The audit mixes paper claims with repository behaviour | My §1 described the repo's `BudgetController` (25/50/75% tiers, urgent band, fit/shrink) as "the paper". | **Yes, verified.** Appendix E says: CLM gets its editing reminder **2,048 tokens before the budget**. BrowseComp-Plus uses a **23,560-token budget, 100 turns**, and up to **six** rollback retries. TB2.1/TBLite **end the run** when a request would exceed the budget. EdgeBench rolls back **up to 50** times. Summary = Codex prompts at 75%. The repo's `bcp.yaml` (28,672 tokens, 500 steps, 0.9 band) is later version drift. | Rewrite §1 with two columns, "paper (Appendix E)" and "repository (current code)", and note the drift. |
| 3 | Option B is not paper-faithful, and the arithmetic is wrong | The paper rolls back the newest *agent turn*; we have no such turn. Also `limit = max_edits_per_chunk + max_condense_tries`, so setting `max_condense_tries: 6` gives 9 attempts, not 6. | **Yes to both.** My table conflated "6 retries" with the config value. Today the over-limit total is already 3 + 3 = 6 attempts, which matches the paper's BrowseComp-Plus retry count. One nuance in our favour: stopping and grading as-is is the paper's own TB2.1/TBLite rule ("a request that would exceed it ends the run"). | Keep `max_condense_tries: 3` (6 attempts while over). Name the policy **"stop on overflow (CLM-inspired)"**, citing the TB2.1 end-of-run rule and the BrowseComp-Plus retry count, not as paper semantics. Log `{"event": "overflow_stop", "part": i, "unread_parts": n}`. Edit arms lose `fit`; the summary arm keeps it, and its forced truncations stay reported. |
| 4 | The growth gate must cover every edit arm | `gate.decide` passes a pure append even while over budget: zero rebill, so 0 ≥ 0. clm and skill are ungated. | **Yes**, and confirmed by running `decide(..., overflow=True)` on a growing append: it returns `True`. | Before the price gate, for clm, gate and skill: reject an edit that grows the body **and** leaves `task + new + incoming > budget`. Message: "edit rejected: it grew the context past the limit". Tests: a growing edit that fits is kept, one that overflows is rolled back, and a pure append while over is rejected in the gate arm. |
| 5 | The cost novelty is narrower than stated | I wrote that the paper "does not study API cache billing" and that its USD estimate has no cache discount. | **Mostly yes.** Prefix-reuse FLOPs already charges from the first mismatch: it is cache-aware compute. The PDF also reports USD per task for Opus and gateway costs, and cumulative API spend in one experiment. What the paper does not do is decide or evaluate edits by a provider's **billed hit/miss/output price ratio**. | Restate the contribution in the reviewer's words: a price-aware edit decision under hosted cache billing, evaluated on memory streams. Remove "does not study API cost" from the audit and the spec. |
| 6 | The summary baseline is only partly verified | "Codex keeps nothing" came from the ContextBench skill text, not from every benchmark. | **Yes.** Appendix E only says "Codex summarization prompts, compacts at 75%". | Wording: "our summary keeps the last part(s) verbatim; at least in ContextBench the paper's keeps none, so ours is likely the stronger baseline." |
| 7 | Paid edit attempts are under-reported | `run_row` counts only edit events with `changed` set (`report.py:56`). Refused, no-op and cut-off attempts are invisible in the counts. | **Yes, for the counts.** Dollars were never under-reported: every call, failed or not, is summed into cost. | Every edit attempt gets one `outcome`: `ready`, `cut_off`, `refused`, `no_change`, `emptied`, `rejected_growth`, `rejected_price` or `applied`. The report shows a count per outcome plus edit calls. This is the reviewer's "transactional edit record" in its smallest form. |
| 8 | Stream de-duplication uses cost as identity | `merge` treats equal rounded `billed_stream` values as "same stream" (`report.py:79`). | **Yes.** It is true by construction today (a replay comes from the cache), but cost is not an identity. | Log a `stream_hash` in `done`: SHA-1 of the stream calls' prompts and replies plus the final context. Merge only on equal hashes; otherwise sum and mark the run as diverged. Test: two runs with equal cost but different replies are not merged. |
| 9a | Edit turns are not kept; no free compaction turns | These are differences from the paper's transcript model. | **Yes.** Already D3 and D13. | Keep and document; both are listed in the final comparison. |
| 9b | Nudges do not re-arm | Ours fire when the context crosses a threshold between two phase *starts*. | **Partly yes, and it is a real bug.** Example: phase k starts at 60%, the 50% nudge fires, and the model compacts to 30%. The next phase starts at 50% again, but `prev_used` is still 60%, so no nudge. The paper's tiers re-arm after compaction. | Set `prev_used` to the context size **after** the previous phase's edits, before the part is appended. A threshold crossed again after compaction then fires again. Test: compact below 50%, grow past it, and the nudge appears twice. Note: Appendix E's actual paper rule is one reminder 2,048 tokens before the budget. Our 25/50/75% tiers follow the repo, which the comparison will say. |
| 9c | The over-limit warning hides the threshold wording | `control` uses `elif`. | **Correct, intended.** While over, the over-limit warning is the more urgent message, as the repo's urgent band also overrides tiers. | Document only. |
| 9d | No THOUGHT enforcement; the first runnable block wins | The paper's prompt rejects replies without a THOUGHT or with more than one command. | **Correct.** Low impact: rejecting would spend paid calls on format errors. | Document. Revisit with native tool calls (D8). |
| 9e | Characters ÷ 4 vs calibrated tiktoken | Already D11. | **Correct.** | Document (D11). |
| 9f | Mechanical `fit` for every arm | Already D2. | **Correct.** | Fixed by item 3. |
| 10 | Source and license wording | The paper is CC BY 4.0 and the repo is CC BY-NC 4.0. F1/F2 drafted wording close to the repo's `prompts.yaml` and `budget.py`. | **Yes.** Our project is MIT, so non-commercial text must not be copied. | Write F1/F2 prompts in our own words, citing the repo as inspiration in the spec and README. Change "nothing copied" to "nothing copied; prompt ideas paraphrased with attribution". |

**Architecture suggestions.** I agree with both "strong" candidates and fold them into
items 1 and 3. `edit_phase` returns an explicit outcome (`ready`, `exhausted`,
`overflow`) instead of text alone, and the query prompt becomes one arm-neutral
function. The "transactional edit record" is item 7. "Identify streams by content"
is item 8.

**Final plan before the next paid run.** Each step is test-first, with a commit per
group:

1. Arm-neutral query prompt (1).
2. The growth/fit rule for all edit arms (4), the no-retype rule (F1) and the nudge
   wording (F2), both in our own words (10).
3. Stop-on-overflow for the edit arms, with 6 attempts while over and the overflow
   location logged (3). `edit_phase` returns its outcome (architecture).
4. Nudge re-arm fix (9b).
5. Edit outcomes and `stream_hash` in logs and report (7, 8).
6. Documentation: §1 split into paper vs repo (2), contribution wording (5),
   baseline wording (6), license wording (10), and the kept differences (9a, 9c,
   9d, 9e).
7. Rerun the small DeepSeek config from a fresh run directory (about $0.5–1.0)
   only after you approve.

---

## Appendix: original working audit (superseded where it conflicts above)

# Audit: CacheCLM arms vs the CLM paper (working draft)

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

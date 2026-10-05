# CacheCLM: design

Date: 2026-10-05. Project 5 of `llm_systems_techniques/projects_revised.md`, designed in brainstorming.

## Question

Context Language Models (CLMs; Shao et al., arXiv 2609.37725, Sep 2026) let the model edit its own context, which is mirrored to a file it changes with bash. The paper reports higher accuracy and fewer FLOPs than harness-scheduled summarization, but it measures cost on a self-hosted SGLang server. There, a mid-context edit can reuse cached work through the paper's Suffix Cache Reuse patch.

Hosted APIs work differently. The prompt cache reuses only the prefix that matches the previous request. An edit in the middle of the context therefore makes every later token a cache miss, and DeepSeek bills a miss at 50x a hit. No provider offers suffix reuse (as of Oct 2026).

**Headline question:** on a hosted API with prompt caching, does model-edited context still pay off once billed dollars are counted? Does a cache-aware gate keep the accuracy gain while cutting the cost?

Hypotheses, stated before any live run:
- **H1.** CLM is more accurate than the summary baseline but costs more in billed dollars on DeepSeek, because mid-context edits break the prompt cache.
- **H2.** CLM + gate keeps most of CLM's accuracy gain at a billed cost close to or below the summary baseline.
- **H3.** Repriced under OpenAI and Anthropic price sheets, where a miss costs about 10x a hit, unrestricted editing becomes relatively cheaper.

Out of scope: RL training, Suffix Cache Reuse itself, multi-agent swarms, tail-only editing, append-only notes, a separate context-manager ("hypervisor") model, and injection-persistence tests. These are possible follow-ups if the main result is clear.

## Setting

- **Benchmark:** MemoryAgentBench (Hu et al., ICLR 2026; Hugging Face `ai-hyz/MemoryAgentBench`, MIT, 76.6 MB). Each sample is one long text followed by 60-100 questions. The text is fed to the agent in chunks, one per message, so the agent must manage its context while reading.
- **Subsets:** Accurate Retrieval (exact facts in the streamed text) and Conflict Resolution (FactConsolidation, where later facts override earlier ones). Long-Range Understanding (LLM-judged summaries) and Test-Time Learning are skipped.
- **Data pin:** commit `7ea0669` of the Hugging Face repo. Only the two parquet files are downloaded: `Accurate_Retrieval` (SHA-256 `56c3cd80…55c1`, 20.0 MB) and `Conflict_Resolution` (SHA-256 `24d5c3f0…ff45`, 1.5 MB). Both hashes match Hugging Face's own LFS hashes. The files contain data only, with no loader scripts or pickles. Their last data change (2025-10-07) predates Hugging Face's July 2026 security incident.
- **Samples:** 7, chosen by length before any run. All 5 EventQA books at about 140K tokens each (Accurate_Retrieval rows 12-16, `eventqa_131072`), plus FactConsolidation single-hop and multi-hop at about 68K tokens (Conflict_Resolution rows 6 and 2, `*_64k`). That puts 2-4.4x the context budget through the agent. No sample at 150K tokens or less reaches 5x. **Fallback:** if the smoke run projects the full run above $4.5, the EventQA rows switch to the 65K versions (rows 7-11). That switch is decided and recorded before the full run.
- **Prompts:** adapted from the benchmark's `utils/templates.py`:
  - chunks are wrapped as "The following context is the book excerpt / the facts I have learned";
  - the query templates are kept, including FactConsolidation's rule that the fact with the larger serial number is the newest.
  - The pinned task block is ours: it says what will be asked, without the questions.
- **Chunks:** 4K tokens per message, in the benchmark's order. Order matters for Conflict Resolution, so it is never shuffled.
- **Context budget:** 32K tokens, as in the paper.
- **Model:** DeepSeek Flash, thinking off, temperature 0.3, for every call in every arm.
- **Repeats:** 2 per arm and sample. Chunk order is fixed by the task, so repeats replace question orders as the noise measure.
- **Scoring:** MemoryAgentBench's own `substring_exact_match` for both subsets: the normalized gold answer must appear in the normalized prediction. It is copied from `utils/eval_other_utils.py`.
- **Budget:** hard cap of $5 for live runs, smoke included. Estimate: about $4 for the full run (about $0.11 per 140K-token sample, arm and repeat), plus about $0.10 for the smoke run. The smoke run checks this estimate (see the fallback under Samples).
- **Data download:** approved and done on 2026-10-05.

## Arms

All three arms share the model, the chunks, the pinned task block, the questions and the 32K budget.
- The CLM and gate arms use the identical system prompt. The gate is invisible until it rejects an edit.
- The summary arm's system prompt leaves out the editing instructions.
- Every system prompt starts with `Run r{repeat}.` (added after the final code review). Without it, repeat 1 would send byte-identical requests to repeat 0 and get provider cache hits that repeat 0 paid for, biasing billed dollars by arm. CLM and gate stay identical within a repeat.

1. **Summary (baseline).** When the context passes 75% of the budget, one LLM call compacts everything except the pinned blocks and the last 2 chunks into a single summary block. This is Codex-style harness-scheduled compaction, the paper's main baseline.
2. **CLM.** The model has one bash tool over `ctx.txt`.
   - Nudges are sent at 25%, 50% and 75% of the budget.
   - Before each chunk, the model may run up to 3 edit commands. Edit turns are free and do not count as steps.
   - If the next chunk would overflow the budget, the harness asks the model to condense first, as the paper's rollback-and-condense does.
3. **CLM + gate.** The same as CLM, plus the cache-aware gate (below) checking every edit.

## The cache-aware gate

The prompt cache reuses the request prefix up to the first changed token. An edit therefore has:
- a one-time cost: every token after the edit point loses its cache once;
- a recurring saving: the deleted tokens are no longer sent on each remaining call, but those tokens would have been cheap cache hits.

```text
allow if   deleted_tokens × turns_left × hit_price   >=   tokens_after_edit × (miss_price − hit_price)
```

- `tokens_after_edit` counts the **unchanged text after the first change**: the longest common suffix of the old and new context after that point. These tokens were cached, survive the edit, and must be re-billed at the miss price once. Text the edit inserts or rewrites is not counted, because new text is billed as a miss with or without the gate. So:
  - deleting a block deep in the context re-bills everything after it;
  - deleting the newest block re-bills nothing.
- `deleted_tokens` is the net shrink, old length minus new length, floored at 0. An edit that does not shrink the context has no saving, so it passes only when it invalidates nothing (a pure append). Any rewrite that keeps the length is rejected.
- `turns_left` = chunks remaining + number of questions. The harness knows both; the model never sees the questions in advance.
- **Override:** if the context would otherwise overflow the budget, an edit is allowed regardless.
- **A rejected edit** is reverted. The model gets one line, for example: `edit rejected: breaks 16,000 cached tokens to save 6,000; prefer deleting near the end`.

**Worked example** (DeepSeek, hit $0.006/M, miss $0.30/M; 30K context; 10 turns left):

| Edit | Deleted | Tokens after edit | Cost | Saving | Decision |
|---|---|---|---|---|---|
| Middle search result | 6K | 16K | $0.0047 | $0.00036 | reject |
| Latest tool output | 6K | 0.5K | $0.00015 | $0.00036 | allow |

At Anthropic prices (read $0.30/M, write $3.75/M), the middle edit costs $0.055. It is still rejected at 10 turns left (saving $0.018) and becomes roughly break-even at 30 turns left ($0.054).

The gate estimates tokens as characters / 4, because it must decide before the next call. Both sides of the inequality scale with the same factor, so only the relative density of the text matters. Billed usage fields are the source of truth for cost, and the gap between estimate and billing is logged.

## How a sample runs

1. **Streaming phase.** For each chunk:
   - The harness writes `ctx.txt`.
   - The request is the arm's fixed system prompt, then one user message: the whole context followed by a short control note (budget use, nudge, the last command's result). The context therefore sits in the cacheable prefix, and the control note sits after it.
   - The model replies with one bash command in a fenced block, or `READY` for the next chunk. At most 3 commands per chunk, plus up to 3 condense attempts when the next chunk would overflow.
   - The sandbox runs the command, and the harness reads `ctx.txt` back.
   - Gate arm only: `decide(old, new)` allows the edit or reverts it.
   - If a pinned block is damaged, the edit is rolled back.
   - The harness then appends the next chunk.
   - The summary arm skips the edit turns and compacts at the 75% threshold instead.
   - After the last chunk, the CLM arms get one more edit phase before the questions.
2. **Query phase.** The final context is frozen. Each question is asked in its own call, with no edits allowed. Questions go one at a time until the provider reports a cache hit (at most 3), then in parallel, so the parallel calls do not arrive before the cache entry exists. Separate calls keep answers independent, and because the frozen context is cached, they cost about $0.02 per sample.

## Components

Repo `new-techniques/cacheclm/`: its own git repo, MIT, Python 3.13 with uv. Each module has one job:

- **`ctxfile.py`:** the context as one text in the paper's `[[CTX_TURN i role=…]]` block format. The task block is pinned at the top; the system prompt is fixed per arm and is not in the file. Helpers: append a block, drop the oldest block, split for summary, check the pinned block.
- **`sandbox.py`:** runs one bash command in a temp folder that contains only `ctx.txt`.
  - Allow-list: `sed`, `awk`, `grep`, `head`, `tail`, `cat`, `wc`, `echo`, `printf`, `mv`, `cp`, `python3`, plus redirects. One line only; no heredocs and no subshells.
  - Enforced by macOS `sandbox-exec`: no network, no writes outside the temp folder, no reads of the home folder except the Python install.
  - No network, a 10 s timeout, and output capped at 2K characters.
- **`gate.py`:** pure functions.
  - `first_change(old, new)` returns the token position of the first difference.
  - `decide(old, new, turns_left, prices)` returns allow or reject with a one-line reason.
- **`arms.py`:** the three context policies behind one interface, `before_chunk(ctx) -> ctx`.
- **`run.py`:** the streaming and query loop. Every call is logged with the model id, date, prompt, cache-hit and cache-miss tokens, output tokens and wall time.
- **`llm.py`:** an OpenAI-compatible client.
  - Responses are cached by request hash, so re-running the analysis is free.
  - A budget guard stops at the cap, and runs resume after a stop.
- **`reprice.py`:** recomputes each call's cost under the three price sheets in `prices.yaml`.
- **`report.py`:** writes `results/summary.md` and `summary.html`.

Data is pinned by hash, like EvoSQL's data scripts.

## Error handling

- **Malformed `ctx.txt` after an edit:** roll back and tell the model "parse failed".
- **Banned command:** rejected without running, with the allow-list repeated.
- **No room made before an overflowing chunk:** the harness drops the oldest unpinned blocks and logs a forced truncation, which is counted per arm.
- **API error:** 3 retries for connection, timeout, rate-limit and server errors. A rejected request (any other 4xx) fails that sample at once. A bad key or no balance (401, 402) stops the run. Failed samples are skipped, and the report leaves out every (sample, repeat) that is not finished in all three arms.

## Metrics and analysis

**Primary endpoints** (fixed before any live run):
1. Accuracy: CLM − Summary (H1).
2. Billed dollars on DeepSeek: CLM / Summary (H1).
3. Gate: the share of CLM's accuracy gain kept, (Gate − Summary) / (CLM − Summary), and billed dollars Gate / Summary (H2).

**Secondary:**
- accuracy by subset;
- cost per correct answer;
- edits proposed, accepted and gate-rejected;
- tokens invalidated per edit;
- forced truncations;
- observed vs ideal cache-hit rate, where ideal is the prefix shared with the previous request;
- wall time.

**Repricing (H3), with no extra spend.** For each logged call, the ideal cached prefix and the new tokens are priced under three sheets:
- **DeepSeek:** hit $0.006/M, miss $0.30/M.
- **OpenAI:** cached input at 10% of the input price.
- **Anthropic:** reads at 10% of the input price; new tokens billed as cache writes at 125%, assuming a breakpoint at the end of every call. That is the best case for Anthropic; its 5-minute TTL is ignored because runs are fast.

Prices are stored with their date.

**Statistics:**
- Results are paired by sample and repeat.
- Bootstrap 95% CIs over samples, plus a sign-flip test on per-sample differences.
- With about 10 samples, a per-sample table is shown next to every aggregate, and results are framed as direction plus effect size.

**Report:**
- The headline chart: accuracy vs dollars per sample, in three panels (DeepSeek, OpenAI and Anthropic pricing), one point per arm with CIs.
- 3-4 real edit examples, allowed vs rejected, with the gate's numbers.
- A caveats section.

## Testing

Unit tests only, no network:
- **`ctxfile`:** round-trip; damaged pinned blocks are detected.
- **`first_change`:** edits at the start, middle and end; a no-op; an append-only edit.
- **`gate.decide`:**
  - the worked example (middle edit rejected and tail edit allowed on DeepSeek);
  - the Anthropic middle edit becomes roughly break-even at 30 turns left;
  - the overflow override;
  - a non-shrinking mid-context edit is rejected.
- **`sandbox`:** a banned command is rejected, the timeout fires, nothing outside the temp folder is touched.
- **`reprice`:** a hand-computed call under each sheet.
- **Budget guard:** stops at the cap.
- **Replay:** a replay from cache returns identical logs.

## Runs

1. **Smoke:** 1 short sample, 5 questions, all 3 arms, about $0.10.
2. **Full:** about 10 samples × 3 arms × 2 repeats, hard cap $5.

The user runs both in their own terminal, from a command handed over. Live runs are not extended after seeing results; a follow-up would be a separate, newly stated run.

## Risks

- **The CLM arm may edit rarely ("inert").** Edit counts are reported, as EvoSQL reports injection rates.
- **DeepSeek caching is best-effort and works in 64-token units.** The observed vs ideal gap is measured and reported.
- **Small n (about 10 samples).** Effect sizes and per-sample tables carry the result.
- **The gate's `turns_left` is large in the query phase.** Edits during streaming are judged by both remaining chunks and questions, which favours shrinking before the questions. This is intended, because the frozen context is re-read once per question.
- **The tokenizer estimate differs from billed counts.** The gap is logged.

## Credits

- Context Language Models (arXiv 2609.37725) for the method and the context-file format. Its repository is CC BY-NC 4.0, so no code is copied.
- MemoryAgentBench (arXiv 2507.05257) for the data.

## Changes after the first smoke run (2026-10-05)

The first smoke run (1 sample, 5 questions, $0.30) exposed two bugs and one design problem. All were fixed before any full run.

- **CLM edits were refused.** The model writes multi-line `python3 -c "..."` scripts, and the "one line only" rule refused 129 of 140 commands, so both CLM arms were truncated, not edited. Newlines inside quotes are now allowed; a newline outside quotes still starts a new command that must be on the allow-list.
- **CLM answered questions with "READY".** Every question message now starts with "Editing is over. Answer the question below in plain text…", the same text for all arms.
- **Summaries were cut off.** Each compaction hit the 8,000-token output cap (identical 33,576-character summaries) and ran 12 times in 17 chunks. The summary prompt now asks for at most 3,000 words, with a 5,000-token output cap.
- **Scale.** The first smoke run projected about $9 for the original plan, so the samples switch to the EventQA ~70K versions (rows 7-11, the fallback above) plus both FactConsolidation 64K sets, with **1 repeat** (about $2.7), to stay under the $5 cap. With one repeat the noise measure is the per-sample spread; repeats were not used to set any threshold.

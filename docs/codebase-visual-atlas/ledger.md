# CacheCLM visual atlas: evidence ledger

Every element drawn in the atlas maps to a row here. Line numbers are from commit `831f3f1`, except rows L27–L31, which are from `8bbb760`. All rows were traced in the
source. One row (L26) is marked `inferred`: it describes the provider's behaviour, not code in this repo.

| ID | Visual element | File:symbol (lines) | What the code does | observed / inferred |
|---|---|---|---|---|
| L1 | Job list | `src/cacheclm/__main__.py: jobs` (25–32) | Smoke: every arm (4 primary + 2 references) per smoke sample. Full run: primary arms, then the extra repeat, then references last. | observed |
| L2 | Main loop | `src/cacheclm/__main__.py: main` (53–82) | Loads config, prices, `Budget` and `LLM`. For each job and arm it calls `run_sample`; it stops cleanly on `BudgetExceeded`. | observed |
| L3 | Sample loading | `src/cacheclm/__main__.py: load, run_config` (35–50); `src/cacheclm/mab.py: load_sample, check_file` (49–61) | Reads one MemoryAgentBench parquet row, checked against a pinned SHA-256. It can cut the text to `[start:end]`, start from a given question, and override any setting. | observed |
| L4 | The book cutter | `src/cacheclm/mab.py: chunk_text` (64) | Cuts the text into parts of `chunk_tokens × 4` characters. | observed |
| L5 | One sample, one arm | `src/cacheclm/run.py: run_sample` (95–125) | Runs `stream` for the primary arms, nothing for `none`, or every part for `full`. Then it runs `answer_all` and logs each answer and a `done` record. | observed |
| L6 | Part-by-part reading | `src/cacheclm/run.py: stream` (74–92) | For each part, plus one final pass: the arm's policy, then `fit`, then the part is appended. It tracks `turns_left` and `prev_used`. | observed |
| L7 | Frozen context, questions | `src/cacheclm/run.py: answer_all` (55–71) | Asks each question in its own call on the frozen context. Calls run one at a time until a cache hit, then in parallel. | observed |
| L8 | Call logger | `src/cacheclm/run.py: Recorder.chat` (37–52) | Logs each call's billed cache hits and its ideal hits (the prefix shared with the previous request). | observed |
| L9 | Reply cache, spend guard | `src/cacheclm/llm.py: LLM.chat` (36–47); `src/cacheclm/budget.py: Budget.spend` (19–25) | Replies are cached on disk by request hash, so replays are free. Each real call's cost is added to a persisted total, and the run stops past the cap. | observed |
| L10 | Context box | `src/cacheclm/ctxfile.py: new_context, append, normalize` (18–45) | The pinned task block is sent first but never written to `ctx.txt`. The body is `[[CTX_TURN i role=...]]` blocks, and new parts are appended at the end. | observed |
| L11 | Control note | `src/cacheclm/arms.py: control` (58–71) | Shows budget use. OVER LIMIT appears on every call while the next part won't fit. A nudge appears only on the call where a 25/50/75% threshold was first crossed (computed at `edit_phase` line 79). It also shows the last command's result. | observed |
| L12 | Edit loop | `src/cacheclm/arms.py: edit_phase` (74–110) | Up to `max_edits` commands, more while over the limit. The steps: chat, parse, run in the sandbox, normalize, roll back an emptied file, ask the gate, then log the edit and feed back its result. | observed |
| L13 | Cut-off ends phase | `src/cacheclm/arms.py: edit_phase` (89–91) | A reply cut off by the length limit, with no command in it, ends the phase with no retry. | observed |
| L14 | Command parser | `src/cacheclm/arms.py: parse_command` (51–55), `TEXT` (40) | Strips ```` ```text ```` blocks, then reads the ```` ```bash ```` block. The text block is saved as `new.txt` (line 96–97). | observed |
| L15 | Sandbox door check | `src/cacheclm/sandbox.py: check` (25–43) | Only allow-listed programs, and no heredocs, `$(` or backticks. | observed |
| L16 | Sandbox booth | `src/cacheclm/sandbox.py: run` (46–73) | Runs in a temp dir under `sandbox-exec`: no network, no writes outside the temp dir, no home reads except Python. It has a 10 s timeout with a process-group kill, and reads back only a regular file. | observed |
| L17 | Emptied rollback | `src/cacheclm/arms.py: edit_phase` (99–100) | An edit that leaves only header lines is rolled back. | observed |
| L18 | The scale | `src/cacheclm/gate.py: decide` (13–27) | `first_change`. A pure append is free. Cost is the tokens after the first change × (miss − hit). Saving is deleted tokens × `turns_left` × hit. While over the limit, any shrinking edit passes. | observed |
| L19 | Gate only enforced for `gate` | `src/cacheclm/arms.py: UNGATED` (12), `edit_phase` (104) | `clm` and `skill` log the gate's verdict but are never stopped by it. | observed |
| L20 | Summary press | `src/cacheclm/arms.py: summary_step` (113–128); `ctxfile.split_for_summary` (58–64) | Once the context plus the next part passes 75%, the older blocks become one summary block, and the last `keep` parts stay. | observed |
| L21 | Forced cut | `src/cacheclm/arms.py: fit` (131–139); `ctxfile.cut_oldest_lines` (48–55) | If the next part still won't fit, the oldest whole lines are cut. | observed |
| L22 | Recipe card | `src/cacheclm/arms.py: SKILL` (25–33), `system` (43–48) | The skill arm's system prompt adds a recipe. For books: an event log of the newest part each time. For facts: dedupe, then delete the oldest. | observed |
| L23 | Reference arms | `src/cacheclm/arms.py: REFERENCES` (11); `run.py: run_sample` (109–113) | `none`: no context (floor). `full`: the whole text (ceiling). | observed |
| L24 | Report | `src/cacheclm/report.py: run_row, merge, write_report` (52–88, 169–208) | Billed cost plus the ideal-cache cost repriced under 3 price sheets. Accuracy, paired bootstrap, and the fact error split (retention vs resolution). | observed |
| L25 | Price tags | `configs/prices.yaml` | DeepSeek: hit $0.006/M, miss $0.30/M, output $1.20/M. OpenAI and Anthropic are used for repricing. | observed |
| L26 | Provider prefix cache | (provider side) | The provider bills an unchanged prompt prefix at the hit price. The gate and `Recorder` model this; the repo does not implement it. | inferred |
| L27 | Stop sign | `src/cacheclm/run.py: stream` (93–98) | An edit arm that cannot make room for the next part stops reading; logs `overflow_stop` with the unread part count (CacheCLM policy, not the paper's) | observed |
| L28 | One question prompt | `src/cacheclm/run.py: answer_all` (63) | Every arm answers under the same system prompt (`Run r{repeat}.` + `BASE`) | observed |
| L29 | Growth check | `src/cacheclm/arms.py: edit_phase` (127–133) | An edit that grows the context past the budget is rolled back, for every edit arm | observed |
| L30 | Reply parser | `src/cacheclm/arms.py: parse_reply` (56–67) | First bash block, or a python block run as `edit.py`; text blocks become `new.txt`; `echo READY` and `true` mean READY | observed |
| L31 | Toll and trains | `src/cacheclm/run.py: Recorder.chat` (37–55); `configs/prices.yaml` | Logs billed hits and ideal hits (the prefix shared with the previous request); prices hit $0.006/M, miss $0.30/M | observed (the cache itself is L26, inferred) |

## Shot list

| # | Idea | Anchor rows | Metaphor and Xiaohei's action | Takeaway |
|---|---|---|---|---|
| 01 | One sample through one arm, end to end | L4, L6, L10, L12/L20, L21, L7, L8, L24 | Reading room. Pages are cut from a book and dropped one by one into a fixed-size crate. Xiaohei leans into the crate with scissors between pages. Later the crate is lidded and question cards are held up to it, while a receipt tape spools into a ledger. | Every arm reads the same text one part at a time into a fixed-size box. Its policy shrinks the box between parts. Then questions are asked on the frozen box, and every call is logged for the cost vs accuracy report. |
| 02 | The CLM edit loop | L11, L14, L15, L16, L17, L18, L13, L12 | Glass booth. Xiaohei writes a slip (text block plus bash command) and posts it into a sealed glass booth, where an arm edits the `ctx.txt` scroll. The receipt is strung back to Xiaohei's wall note. An over-long slip gets cut off. | The model never touches the context directly. It posts one command into a jail, the result is checked (emptied? gate?) and fed back in the next note. |
| 03 | The cache-aware gate | L18, L25, L26, L19 | Long paper strip. The left part is stamped cached; Xiaohei marks the first change and everything after it turns red. A balance scale weighs coins: saving per turn on one pan, the one-time rebill on the other. | An edit pays only if what it frees, times the turns left, outweighs re-billing everything after its first change. Appends are free. Over the limit, any shrink passes. |
| 04 | Four arms, two references | L20, L12, L19, L22, L23 | A 4-scene comic, one crate each. Summary: a press, with Xiaohei carving the brick. clm: free scissors. gate: the scale at the crate's mouth. skill: a recipe card. Footer: an empty crate and an overflowing crate. | The arms differ only in who edits the box and what may stop them. The references bracket the result. |

| 00 | Why an edit costs money on a hosted API | L31, L26, L18 | Two trains pass a toll. An appended wagon is the only one to pay full price; when Xiaohei swaps a wagon in the middle, every wagon after it pays full price. | An edit in the middle turns every later token into a cache miss at 50× the hit price; an append does not. |

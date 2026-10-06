"""Context policies. summary: the harness compacts at 75%. clm: the model edits ctx.txt with bash (prompt A). gate: clm
plus the cache-aware gate. skill: clm plus a context-management skill. chat(messages, max_tokens, phase) -> reply."""
import re

from cacheclm import sandbox
from cacheclm.ctxfile import (CHARS_PER_TOKEN, HEADER, block, cut_oldest_lines, next_index, normalize,
                              split_for_summary, tokens)
from cacheclm.gate import decide

ARMS = ("summary", "clm", "gate", "skill")
REFERENCES = ("none", "full")  # floor (no context) and ceiling (whole text), outside the primary endpoints
UNGATED = ("clm", "skill")
BASE = "You are a helpful assistant that can read the context and memorize it for future retrieval."
EDITING = """
Your working context is shown in each message: first the task block ([[CTX_TURN 0 role=task pinned]]), which is fixed and is not in any file, then the file ctx.txt, which holds every block after it. Together they hold at most {budget} tokens. Parts of a long text arrive one at a time and are appended to the end of ctx.txt. Between parts you may edit ctx.txt to keep what will matter later: delete or shorten text, or keep notes in a block of your own such as [[CTX_TURN 99 role=notes]].
How to edit:
- Start your reply with a short THOUGHT about what to keep and why, then give exactly one shell command in a ```bash block.
- To put new text into ctx.txt (notes, a shortened part), write it in a ```text block before the command. It is saved as new.txt beside ctx.txt, so the command can read it (for example python3 -c "...open('new.txt').read()...") with no quoting problems.
- An edit makes everything after the edited point be re-read, so prefer one large edit over several small ones, and be generous in what you keep.
- The whole file is already shown to you: do not run commands that only look at it.
- Keep the [[CTX_TURN ...]] header line of every block you keep.
- Allowed programs: sed, awk, grep, head, tail, cat, wc, echo, printf, mv, cp, python3. No heredocs or $(...); for multi-step edits use python3 -c "..." (the script may span several lines; on this system, in-place sed is sed -i '').
- Your reply is cut off after about 12,000 words, so plan each reply so the THOUGHT, any text block and the command fit.
- Reply READY when you are done editing."""
SKILL = """

# Skill: managing your context
## Book excerpts (questions ask which event comes next, anywhere in the story)
- Right after a new part arrives, replace it with an event log: one line per event, in story order. Do this for the newest part only, every time a part arrives; never rewrite several parts at once.
- Each line keeps who did, said, felt or wore what, to or with whom, and where, with exact names and specific details (objects, colours, family relations).
- Keep every older event log.
## Numbered fact lists (newer facts override older ones)
- When the context is nearly full, in one python3 edit: delete every fact for which a later fact has the same subject and relation; then, only if still needed, delete the oldest facts."""
SUMMARY_NOTE = "\nParts of a long text are appended to your working context. When it gets full you will be asked to compact older parts into a summary."
SUMMARIZE = ("The working context is nearly full. Write one summary that replaces every block after the pinned task block "
             "and before the last {keep} parts, in at most {words:,} words. Keep the details most likely to be needed "
             "for later questions: names, events in story order, and facts with their serial numbers. Reply with only "
             "the summary text.")
FENCE = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.S)
TEXT = re.compile(r"```text\n(.*?)```", re.S)


def system(arm, budget, repeat=0):
    """The arm's fixed system prompt; the first line keeps repeats from sharing the provider's prompt cache."""
    editing = EDITING.format(budget=budget)
    extra = {"summary": SUMMARY_NOTE, "clm": editing, "gate": editing, "skill": editing + SKILL}.get(arm, "")
    return f"Run r{repeat}.\n" + BASE + extra


def parse_command(reply):
    """The command in the reply's bash block, or None (READY, no complete block, or an empty one).
    Text blocks are removed first, so a text block's closing fence is never read as the start of a command."""
    m = FENCE.search(TEXT.sub("", reply or ""))
    return (m.group(1).strip() or None) if m else None


def control(ctx, budget, incoming, nudges, last):
    """The note after the context: budget use, a nudge or the overflow warning, the last command's result."""
    used = tokens(ctx)
    lines = [f"Context: {used:,} of {budget:,} tokens ({used / budget:.0%}). Next part: {incoming:,} tokens."]
    over = used + incoming - budget
    crossed = [n for n in nudges if used >= n * budget]
    if over > 0:
        lines.append(f"OVER LIMIT: free at least {over:,} tokens before the next part arrives.")
    elif crossed:
        lines.append(f"Your context is over {max(crossed):.0%} full.")
    if last:
        lines.append(f"Result of your last command:\n{last}")
    lines.append("Reply with one shell command in a ```bash block to edit ctx.txt, or READY.")
    return "\n".join(lines)


def edit_phase(task, body, arm, chat, cfg, incoming, turns_left, price, log, repeat=0):
    """The body after up to max_edits commands, plus up to max_condense more while the next part would not fit."""
    budget, last, edits = cfg["context_budget"], "", 0
    while True:
        ctx = task + body
        over = tokens(ctx) + incoming > budget
        if edits >= cfg["max_edits_per_chunk"] + (cfg["max_condense_tries"] if over else 0):
            return body
        note = control(ctx, budget, incoming, cfg["nudges"], last)
        messages = [{"role": "system", "content": system(arm, budget, repeat)},
                    {"role": "user", "content": ctx + "\n\n" + note}]
        reply = chat(messages, cfg["edit_max_tokens"], "edit")
        command = parse_command(reply["content"])
        if command is None and reply.get("finish_reason") == "length":
            log({"event": "cut_off"})  # a retry repeats the same rewrite, so the phase ends here
            return body
        if command is None:
            return body
        edits += 1
        text = TEXT.search(reply["content"])
        new, output = sandbox.run(command, body, files={"new.txt": text.group(1)} if text else None)
        new = normalize(new)
        emptied = bool(body.strip()) and not HEADER.sub("", new).strip()  # headers alone hold nothing
        if emptied:
            allow, reason, numbers = False, "edit rolled back: it emptied ctx.txt", {}
        else:
            allow, reason, numbers = decide(ctx, task + new, turns_left, price, overflow=over)
            allow = allow or arm in UNGATED  # clm and skill log the gate's verdict but are never stopped by it
        log({"event": "edit", "command": command, "allowed": allow, "reason": reason, "over": over,
             "changed": new != body, "refused": output.startswith("REFUSED"), "emptied": emptied,
             "turns_left": turns_left, "chars_before": len(ctx), "chars_after": len(task + new), **numbers})
        if allow:
            body = new
        last = output if allow else f"{output}\n{reason}".strip()  # the gate explains itself only when it rejects


def summary_step(task, body, chat, cfg, incoming, log, repeat=0):
    """The body with older blocks compacted into one summary once the context plus the next part passes summary_at."""
    budget = cfg["context_budget"]
    if tokens(task + body) + incoming <= cfg["summary_at"] * budget:
        return body
    parts = split_for_summary(body, cfg["keep_recent_chunks"])
    if parts is None:
        return body
    older, tail = parts
    prompt = SUMMARIZE.format(keep=cfg["keep_recent_chunks"], words=cfg["summary_words"])
    messages = [{"role": "system", "content": system("summary", budget, repeat)},
                {"role": "user", "content": task + body + "\n\n" + prompt}]
    reply = chat(messages, cfg["summary_max_tokens"], "summary")
    log({"event": "summary", "chars_compacted": len(older), "summary_chars": len(reply["content"]),
         "cut_off": reply.get("finish_reason") == "length"})
    return block(next_index(body), "summary", reply["content"]) + tail


def fit(task, body, incoming, budget, log):
    """Cut the oldest whole lines of the body until the task, the body and the next part fit (forced truncation)."""
    over = (tokens(task + body) + incoming - budget) * CHARS_PER_TOKEN
    while over > 0 and body:
        smaller = cut_oldest_lines(body, over)
        log({"event": "forced_truncation", "chars_dropped": len(body) - len(smaller)})
        body = smaller
        over = (tokens(task + body) + incoming - budget) * CHARS_PER_TOKEN
    return body

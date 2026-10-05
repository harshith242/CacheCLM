"""The three context policies: summary (harness compacts at 75%), clm (model edits ctx.txt with bash),
gate (clm plus the cache-aware gate). chat(messages, max_tokens, phase) -> reply; log(record) stores an event."""
import re

from cacheclm import sandbox
from cacheclm.ctxfile import block, drop_oldest, next_index, split_for_summary, split_pinned, tokens
from cacheclm.gate import decide

ARMS = ("summary", "clm", "gate")
BASE = "You are a helpful assistant that can read the context and memorize it for future retrieval."
EDITING = """
Your working context is shown in each message: first the task block ([[CTX_TURN 0 role=task pinned]]), which is fixed and is not in any file, then the file ctx.txt, which holds every block after it. Together they hold at most {budget} tokens. Parts of a long text arrive one at a time and are appended to the end of ctx.txt. Between parts you may edit ctx.txt to keep what will matter later: delete irrelevant passages, shorten text, or keep notes in a block of your own such as [[CTX_TURN 99 role=notes]].
To edit, reply with exactly one shell command in a ```bash block. Allowed programs: sed, awk, grep, head, tail, cat, wc, echo, printf, mv, cp, python3. No heredocs or $(...); for multi-step edits use python3 -c "..." (the script may span several lines; on this system, in-place sed is sed -i ''). You will see the command's output. Reply READY when you are done editing."""
SUMMARY_NOTE = "\nParts of a long text are appended to your working context. When it gets full you will be asked to compact older parts into a summary."
SUMMARIZE = ("The working context is nearly full. Write one summary that replaces every block after the pinned task block "
             "and before the last {keep} parts, in at most {words:,} words. Keep the details most likely to be needed "
             "for later questions: names, events in story order, and facts with their serial numbers. Reply with only "
             "the summary text.")
FENCE = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.S)


def system(arm, budget, repeat=0):
    """The arm's fixed system prompt; the first line keeps repeats from sharing the provider's prompt cache."""
    return f"Run r{repeat}.\n" + BASE + (SUMMARY_NOTE if arm == "summary" else EDITING.format(budget=budget))


def parse_command(reply):
    """The command in the reply's fenced block, or None (READY, or no block)."""
    m = FENCE.search(reply or "")
    return m.group(1).strip() if m else None


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


def edit_phase(ctx, arm, chat, cfg, incoming, turns_left, price, log, repeat=0):
    """ctx after up to max_edits commands, plus up to max_condense more while the next part would not fit."""
    budget, last, edits = cfg["context_budget"], "", 0
    while True:
        over = tokens(ctx) + incoming > budget
        if edits >= cfg["max_edits_per_chunk"] + (cfg["max_condense_tries"] if over else 0):
            return ctx
        note = control(ctx, budget, incoming, cfg["nudges"], last)
        messages = [{"role": "system", "content": system(arm, budget, repeat)},
                    {"role": "user", "content": ctx + "\n\n" + note}]
        command = parse_command(chat(messages, 1024, "edit")["content"])
        if command is None:
            return ctx
        edits += 1
        pinned, editable = split_pinned(ctx)
        new_editable, output = sandbox.run(command, editable)
        new = pinned + new_editable
        allow, reason, numbers = decide(ctx, new, turns_left, price, overflow=over)
        if arm == "clm":
            allow, reason = True, "applied"
        log({"event": "edit", "command": command, "allowed": allow, "reason": reason, "over": over,
             "changed": new != ctx, "refused": output.startswith("REFUSED"), "turns_left": turns_left,
             "chars_before": len(ctx), "chars_after": len(new), **numbers})
        if allow:
            ctx = new
        last = f"{output}\n{reason}".strip()


def summary_step(ctx, chat, cfg, incoming, log, repeat=0):
    """ctx with older blocks compacted into one summary once ctx plus the next part passes summary_at of the budget."""
    budget = cfg["context_budget"]
    if tokens(ctx) + incoming <= cfg["summary_at"] * budget:
        return ctx
    parts = split_for_summary(ctx, cfg["keep_recent_chunks"])
    if parts is None:
        return ctx
    head, middle, tail = parts
    prompt = SUMMARIZE.format(keep=cfg["keep_recent_chunks"], words=cfg["summary_words"])
    messages = [{"role": "system", "content": system("summary", budget, repeat)},
                {"role": "user", "content": ctx + "\n\n" + prompt}]
    summary = chat(messages, cfg["summary_max_tokens"], "summary")["content"]
    log({"event": "summary", "chars_before": len(ctx), "chars_compacted": len(middle), "summary_chars": len(summary)})
    return head + block(next_index(ctx), "summary", summary) + tail


def fit(ctx, incoming, budget, log):
    """Drop the oldest unpinned blocks until ctx plus the next part fits (forced truncation)."""
    while tokens(ctx) + incoming > budget:
        smaller = drop_oldest(ctx)
        if smaller == ctx:
            break
        log({"event": "forced_truncation", "chars_dropped": len(ctx) - len(smaller)})
        ctx = smaller
    return ctx

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
- Start your reply with a short THOUGHT about what to keep and why, then give exactly one shell command in a ```bash block, or one Python script in a ```python block (it runs as python3 edit.py beside ctx.txt).
- Change text where it is: find it with code (python3 or sed, by its [[CTX_TURN ...]] header or a short unique line) and never paste or retype text you keep. Retyping costs output tokens and invites copying mistakes.
- Write only new text (notes, or a short summary that replaces a region) in a ```text block before the command. It is saved as new.txt beside ctx.txt, so the command can read it (for example python3 -c "...open('new.txt').read()...") with no quoting problems.
- Only ctx.txt is kept: the command must write its result to ctx.txt. new.txt is discarded after the command.
- An edit makes everything after the edited point be re-read, so prefer one large edit over several small ones, and be generous in what you keep.
- The whole file is already shown to you: do not run commands that only look at it.
- Keep the [[CTX_TURN ...]] header line of every block you keep.
- Allowed programs: sed, awk, grep, head, tail, cat, wc, echo, printf, mv, cp, python3. No heredocs or $(...); for multi-step edits use python3 -c "..." (the script may span several lines; on this system, in-place sed is sed -i '').
- Your reply is cut off after about {words} words, so plan each reply so the THOUGHT, any text block and the command fit.
- Reply READY when you are done editing."""
SKILL = {  # the skill arm gets the recipe for its task's family only (Qwen applied the book recipe to fact lists)
    "head": "\n\n# Skill: managing your context",
    "eventqa": """
## Book excerpts (questions ask which event comes next, anywhere in the story)
- Right after a new part arrives, replace it with an event log: one line per event, in story order. Do this for the newest part only, every time a part arrives; never rewrite several parts at once.
- Each line keeps who did, said, felt or wore what, to or with whom, and where, with exact names and specific details (objects, colours, family relations).
- Keep every older event log.""",
    "factconsolidation": """
## Numbered fact lists (newer facts override older ones)
- When the context is nearly full, in one python3 edit: delete every fact for which a later fact has the same subject and relation; then, only if still needed, delete the oldest facts.""",
}
SUMMARY_NOTE = "\nParts of a long text are appended to your working context. When it gets full you will be asked to compact older parts into a summary."
SUMMARIZE = ("The working context is nearly full. Write one summary that replaces every block after the pinned task block "
             "and before the last {keep} parts, in at most {words:,} words. Keep the details most likely to be needed "
             "for later questions: names, events in story order, and facts with their serial numbers. Reply with only "
             "the summary text; it is cut off after about {cap:,} words.")
BLOCK = re.compile(r"```(\w*)\n(.*?)```", re.S)  # one fenced block at a time, so text between blocks is never a block
SAYS_READY = re.compile(r"""(echo|printf)\s+["']?READY["']?""")


def system(arm, budget, repeat=0, reply_tokens=16384, fam=None):
    """The arm's fixed system prompt; the first line keeps repeats from sharing the provider's prompt cache.
    The reply cap is stated in words (0.75 per token, rounded down to a thousand). fam picks the skill's recipe."""
    editing = EDITING.format(budget=budget, words=f"{int(reply_tokens * 0.75) // 1000 * 1000:,}")
    skill = SKILL["head"] + (SKILL[fam] if fam else SKILL["eventqa"] + SKILL["factconsolidation"])
    extra = {"summary": SUMMARY_NOTE, "clm": editing, "gate": editing, "skill": editing + skill}.get(arm, "")
    return f"Run r{repeat}.\n" + BASE + extra


def parse_reply(reply):
    """(command, files to put beside ctx.txt). The command is the first bash block, or a python block run as edit.py;
    None means READY (no runnable block, an empty one, or one that only prints READY). A text block becomes new.txt."""
    blocks = BLOCK.findall(reply or "")
    files = {"new.txt": body for lang, body in blocks if lang == "text"}
    for lang, body in blocks:
        if lang in ("bash", "sh", "") and body.strip():
            command = body.strip()
            return (None if SAYS_READY.fullmatch(command) else command), files
        if lang == "python" and body.strip():
            return "python3 edit.py", {**files, "edit.py": body}
    return None, files


def control(ctx, budget, incoming, crossed, last, edits_left=None):
    """The note after the context: budget use, the overflow warning (every call) or a nudge for a threshold the context
    has just crossed (once per crossing), the last command's result, and the edits left in this phase."""
    used = tokens(ctx)
    lines = [f"Context: {used:,} of {budget:,} tokens ({used / budget:.0%}). Next part: {incoming:,} tokens."]
    over = used + incoming - budget
    if over > 0:
        lines.append(f"OVER LIMIT: free at least {over:,} tokens before the next part arrives.")
    elif crossed and max(crossed) <= 0.25:  # informational only: early how-to advice invites wholesale deletion
        lines.append(f"Context is at about {max(crossed):.0%} of your budget.")
    elif crossed:
        lines.append(f"Context is over {max(crossed):.0%} full. Shorten parts you are done with into short, specific "
                     "notes; do not delete whole regions you may still need.")
    if last:
        lines.append(f"Result of your last command:\n{last}")
    if edits_left is not None:
        lines.append(f"Edits left before the next part: {edits_left}.")
    lines.append("Reply with one ```bash command or ```python script that edits ctx.txt, or READY.")
    return "\n".join(lines)


def edit_phase(task, body, arm, chat, cfg, incoming, turns_left, price, log, repeat=0, prev_used=0, fam=None):
    """The body after up to max_edits commands, plus up to max_condense more while the next part would not fit.
    prev_used is the context size after the previous phase's edits: a nudge is shown only for thresholds crossed since,
    so a threshold crossed again after a compaction fires again. Every reply is logged with one outcome."""
    budget, last, edits = cfg["context_budget"], "", 0
    used = tokens(task + body)
    crossed = [n for n in cfg["nudges"] if prev_used < n * budget <= used]
    while True:
        ctx = task + body
        over = tokens(ctx) + incoming > budget
        limit = cfg["max_edits_per_chunk"] + (cfg["max_condense_tries"] if over else 0)
        if edits >= limit:
            return body
        # A nudge opens the phase only. The count of edits left also keeps each prompt distinct, so a failed edit is
        # retried by sampling again rather than by replaying the same cached reply.
        note = control(ctx, budget, incoming, crossed if edits == 0 else [], last, limit - edits)
        messages = [{"role": "system", "content": system(arm, budget, repeat, cfg["edit_max_tokens"], fam)},
                    {"role": "user", "content": ctx + "\n\n" + note}]
        reply = chat(messages, cfg["edit_max_tokens"], "edit")
        command, files = parse_reply(reply["content"])
        if command is None and reply.get("finish_reason") == "length":
            log({"event": "cut_off", "over": over})  # a retry repeats the same rewrite, so the phase ends here
            return body
        if command is None:
            log({"event": "ready", "over": over})
            if not over:
                return body
            edits += 1  # READY while the next part cannot fit uses up a try, as a rolled-back turn does in the paper
            last = "The next part still does not fit: free space in ctx.txt first."
            continue
        edits += 1
        new, output = sandbox.run(command, body, files=files)
        new = normalize(new)
        if new == body and not output.startswith("REFUSED"):
            output = f"{output}\nctx.txt did not change.".strip()  # e.g. the result went to new.txt, which is discarded
        emptied = bool(body.strip()) and not HEADER.sub("", new).strip()  # headers alone hold nothing
        grew_over = len(new) > len(body) and tokens(task + new) + incoming > budget  # the paper's default fit rule
        if emptied:
            allow, reason, numbers, outcome = False, "edit rolled back: it emptied ctx.txt", {}, "emptied"
        elif grew_over:
            allow, reason, numbers = False, f"edit rejected: it grew the context past the {budget:,}-token limit", {}
            outcome = "rejected_growth"
        else:
            allow, reason, numbers = decide(ctx, task + new, turns_left, price, overflow=over)
            allow = allow or arm in UNGATED  # clm and skill log the gate's verdict but are never stopped by it
            outcome = "applied" if allow else "rejected_price"
        refused = output.startswith("REFUSED")
        outcome = "refused" if refused else "no_change" if new == body else outcome
        log({"event": "edit", "command": command, "allowed": allow, "reason": reason, "over": over, "outcome": outcome,
             "changed": new != body, "refused": refused, "emptied": emptied,
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
    cap = int(cfg["summary_max_tokens"] * 0.75) // 10 * 10  # stated like the edit prompt's cap: the model is warned
    prompt = SUMMARIZE.format(keep=cfg["keep_recent_chunks"], words=cfg["summary_words"], cap=cap)
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

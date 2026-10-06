"""Cache-aware edit gate: an edit passes only if its per-turn saving outweighs what it costs once, namely every token after
its first change (the surviving tail and any text it inserts) billed as a cache miss on the next call."""
import os

from cacheclm.ctxfile import CHARS_PER_TOKEN


def first_change(old, new):
    """Character position where old and new first differ."""
    return len(os.path.commonprefix([old, new]))


def decide(old, new, turns_left, price, overflow=False):
    """(allow, reason, numbers); price has cache_read and cache_write in USD per 1M tokens."""
    at = first_change(old, new)
    appended_only = at == len(old)  # old is a prefix of new: no cached token is touched
    rebilled = 0 if appended_only else (len(new) - at) // CHARS_PER_TOKEN
    deleted = max(0, len(old) - len(new)) // CHARS_PER_TOKEN
    cost = rebilled * (price["cache_write"] - price["cache_read"])
    saving = deleted * turns_left * price["cache_read"]
    numbers = {"first_change": at, "rebilled_tokens": rebilled, "deleted_tokens": deleted}
    if overflow and len(new) < len(old):
        return True, f"allowed (over the limit): breaks {rebilled:,} cached tokens to free {deleted:,}", numbers
    if saving >= cost:
        return True, f"allowed: frees {deleted:,} tokens for {turns_left} turns, breaks {rebilled:,} cached tokens", numbers
    return False, (f"edit rejected: breaks {rebilled:,} cached tokens to save {deleted:,}; "
                   "prefer deleting near the end"), numbers

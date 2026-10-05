"""Cache-aware edit gate: an edit passes only if its per-turn saving outweighs the cached tokens it forces to be re-billed."""
import os

from cacheclm.ctxfile import CHARS_PER_TOKEN


def first_change(old, new):
    """Character position where old and new first differ."""
    return len(os.path.commonprefix([old, new]))


def survivors(old, new, at):
    """Characters after the first change that both versions end with: cached text the provider must re-bill."""
    return len(os.path.commonprefix([old[at:][::-1], new[at:][::-1]]))


def decide(old, new, turns_left, price, overflow=False):
    """(allow, reason, numbers); price has cache_read and cache_write in USD per 1M tokens."""
    at = first_change(old, new)
    rebilled = survivors(old, new, at) // CHARS_PER_TOKEN
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

"""Runs one model-written shell command on ctx.txt in a temp folder: allow-listed programs, macOS sandbox-exec jail
(no network, no writes outside the folder, no home-folder reads except Python), 10 s timeout, 2K characters of output."""
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED = {"sed", "awk", "grep", "head", "tail", "cat", "wc", "echo", "printf", "mv", "cp", "python3"}
OPERATORS = {"|", "||", "&&", ";", "&"}
MAX_OUTPUT = 2000
PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write* (require-not (subpath "{tmp}")))
(allow file-write* (literal "/dev/null"))
(deny file-read* (subpath "{home}"))
(allow file-read* (subpath "{tmp}") (subpath "{py}"))
"""


def check(command):
    """Why the command may not run, or None."""
    if "<<" in command or "$(" in command or "`" in command:
        return "no heredocs or subshells; use python3 -c '...' instead"
    if "\n" in command:
        return "one line only; use python3 -c '...' for multi-step edits"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError as e:
        return f"cannot parse the command: {e}"
    start = True
    for word in words:
        if word in OPERATORS:
            start = True
        elif start:
            if word not in ALLOWED:
                return f"{word!r} is not allowed; use one of: {', '.join(sorted(ALLOWED))}"
            start = False
    return None


def run(command, ctx, timeout=10):
    """(new ctx, output) after running the command on ctx.txt; ctx is unchanged when the command is refused."""
    why = check(command)
    if why:
        return ctx, f"REFUSED: {why}"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d).resolve()
        (tmp / "ctx.txt").write_text(ctx)
        py = Path(sys.base_prefix).resolve()
        profile = PROFILE.format(tmp=tmp, home=Path.home().resolve(), py=py)
        env = {"PATH": f"{py / 'bin'}:/usr/bin:/bin", "HOME": str(tmp), "LC_ALL": "en_US.UTF-8"}
        try:
            p = subprocess.run(["sandbox-exec", "-p", profile, "/bin/bash", "-c", command], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=timeout)
            output = p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            output = f"ERROR: timed out after {timeout} s"
        path = tmp / "ctx.txt"
        new = path.read_text(errors="replace") if path.exists() else ""
    return new, output[:MAX_OUTPUT]

"""Runs one model-written shell command on ctx.txt in a temp folder: allow-listed programs, macOS sandbox-exec jail
(no network, no writes outside the folder, no home-folder reads except Python), 10 s timeout, 2K characters of output."""
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED = {"sed", "awk", "grep", "head", "tail", "cat", "wc", "echo", "printf", "mv", "cp", "python3"}
SEPARATORS = set(";&|\n")  # operators, and newlines outside quotes, start a new command
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
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
        lexer.whitespace, lexer.whitespace_split = " \t\r", True
        words = list(lexer)
    except ValueError as e:
        return f"cannot parse the command: {e}"
    start = True
    for word in words:
        if word and set(word) <= SEPARATORS:
            start = True
        elif start:
            if word not in ALLOWED:
                return f"{word!r} is not allowed; use one of: {', '.join(sorted(ALLOWED))}"
            start = False
    return None


def run(command, ctx, timeout=10, files=None):
    """(new ctx, output) after running the command on ctx.txt (with any extra files beside it, e.g. new.txt);
    ctx is unchanged when the command is refused."""
    why = check(command)
    if why:
        return ctx, f"REFUSED: {why}"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d).resolve()
        (tmp / "ctx.txt").write_text(ctx)
        for name, text in (files or {}).items():
            (tmp / name).write_text(text)
        py = Path(sys.base_prefix).resolve()
        profile = PROFILE.format(tmp=tmp, home=Path.home().resolve(), py=py)
        env = {"PATH": f"{py / 'bin'}:/usr/bin:/bin", "HOME": str(tmp), "LC_ALL": "en_US.UTF-8",
               "PYTHONHASHSEED": "0"}  # same edit, same result: samples sharing a text replay the same stream
        p = subprocess.Popen(["sandbox-exec", "-p", profile, "/bin/bash", "-c", command], cwd=tmp, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            out, err = p.communicate(timeout=timeout)
            output = out + err
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)  # the whole process group, so pipeline children die too
            p.communicate()
            output = f"ERROR: timed out after {timeout} s"
        path = tmp / "ctx.txt"
        new = path.read_text(errors="replace") if path.exists() else ""
    return new, output[:MAX_OUTPUT]

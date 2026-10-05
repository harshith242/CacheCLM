import subprocess
import time
from pathlib import Path

from cacheclm.sandbox import check, run


def test_allow_list_and_shape_checks():
    assert check("sed -i '' 's/a/b/' ctx.txt") is None
    assert check("grep -v x ctx.txt > t && mv t ctx.txt") is None
    assert check("python3 -c \"t=open('ctx.txt').read(); open('ctx.txt','w').write(t[:10])\"") is None
    assert "not allowed" in check("curl example.com")
    assert "not allowed" in check("cat ctx.txt | nc host 1")
    assert "heredoc" in check("python3 - <<EOF\nprint(1)\nEOF")
    assert check("python3 -c \"\nt=open('ctx.txt').read()\nopen('ctx.txt','w').write(t[:10])\n\"") is None
    assert "not allowed" in check("cat ctx.txt\nrm ctx.txt")  # a second command on the next line is still checked


def test_edit_runs_on_ctx():
    new, out = run("sed -i '' 's/apple/pear/' ctx.txt", "an apple\n")
    assert new == "an pear\n"


def test_refused_command_does_not_run():
    new, out = run("curl example.com", "keep\n")
    assert new == "keep\n" and out.startswith("REFUSED")


def test_no_network():
    new, out = run("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\"", "x\n")
    assert "Error" in out and new == "x\n"


def test_no_writes_outside_and_no_home_reads(tmp_path):
    escape = tmp_path / "escape.txt"
    run(f"echo hi > {escape}", "x\n")
    assert not escape.exists()
    _, out = run(f"python3 -c \"import os; os.listdir('{Path.home()}')\"", "x\n")
    assert "not permitted" in out.lower()


def test_timeout_kills_the_whole_pipeline():
    _, out = run("python3 -c 'import time; time.sleep(31.73)' | cat", "x\n", timeout=1)
    assert "timed out" in out
    time.sleep(0.5)
    left = subprocess.run(["pgrep", "-f", "31.73"], capture_output=True, text=True).stdout
    assert left == ""


def test_multi_line_python_edit_runs():
    script = "python3 -c \"\nt = open('ctx.txt').read()\nopen('ctx.txt', 'w').write(t.replace('apple', 'pear'))\n\""
    new, out = run(script, "an apple\n")
    assert new == "an pear\n", out


def test_python_runs_are_reproducible_so_replays_match():
    script = "python3 -c \"print(list({'apple', 'pear', 'plum', 'fig', 'kiwi', 'lime'}))\""
    assert len({run(script, "x\n")[1] for _ in range(4)}) == 1  # set order depends on PYTHONHASHSEED


def test_extra_files_are_written_next_to_ctx():
    new, out = run("cat new.txt > ctx.txt", "old\n", files={"new.txt": "It's \"quoted\" text\n"})
    assert new == "It's \"quoted\" text\n", out

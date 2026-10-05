"""Downloads the two MemoryAgentBench splits pinned to commit 7ea0669 (plain HTTPS, no dataset code) and checks SHA-256."""
import urllib.request
from pathlib import Path

from cacheclm.mab import SHA256, URL, check_file


def main(out="data/memoryagentbench"):
    Path(out).mkdir(parents=True, exist_ok=True)
    for split in SHA256:
        path = Path(out) / f"{split}.parquet"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(split=split), path)
        check_file(path, split)
        print("ok", path)


if __name__ == "__main__":
    main()

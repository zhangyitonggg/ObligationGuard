"""Download the official task definitions; task containers are pulled by the evaluation harness."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "external_sources.json").read_text(encoding="utf-8"))["susvibes"]
    url = f"https://raw.githubusercontent.com/LeiLiLab/susvibes/{config['revision']}/{config['dataset']}"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    rows = [json.loads(line) for line in content.decode("utf-8-sig").splitlines()]
    if len(rows) != 186 or len({row["instance_id"] for row in rows}) != 186:
        raise ValueError("official dataset must contain 186 unique instances")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(json.dumps({"instances": 186, "sha256": hashlib.sha256(content).hexdigest()}))


if __name__ == "__main__":
    main()

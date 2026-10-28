"""SWE-bench Verified-mini for the runner (README.md).

The 50 task ids of MariusHobbhahn/swe-bench-verified-mini, each with its row from
SWE-bench/SWE-bench_Verified, whose rows also name the task's image, eval script and log parser
for swebench 5. Both datasets are downloaded at pinned revisions and checked by SHA-256. Writes
data/verified-mini.jsonl for the harness and the same file as data/verified-mini/test.jsonl for
mini-swe-agent, then checks the result against the file the gold-patch baseline ran on and the
task names against images.txt.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# (dataset, revision, sha256 of its test split)
MINI = ("MariusHobbhahn/swe-bench-verified-mini", "b316c349947c29963fce3f4a65967c9807a4b673",
        "f9ba19dea78884f1081355d2d8afb671899981f24180aa0c4c1aa14d2c23e855")
FULL = ("SWE-bench/SWE-bench_Verified", "78f471bf655a3137b2e8a75af1501690ec009ec3",
        "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25")
SPLIT_FILE = "data/test-00000-of-00001.parquet"
# The task file of the gold-patch baseline in runs/cpu/baseline/sweverified.
OUT_SHA256 = "911f0fe15b153b366dd82de3c2a018f1f54cf834198e1d2004decf838d8dac54"
COMPARED = ("base_commit", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "version")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(dataset: tuple[str, str, str]) -> list[dict]:
    repo, revision, want = dataset
    path = Path(hf_hub_download(repo, SPLIT_FILE, repo_type="dataset", revision=revision))
    if sha256(path) != want:
        sys.exit(f"{repo} at {revision[:8]}: {SPLIT_FILE} does not match its pinned SHA-256. Stop and check the download.")
    return pq.read_table(path).to_pylist()


def as_list(value):
    return json.loads(value) if isinstance(value, str) else value


def main():
    data = Path("data")
    mini = rows(MINI)
    full = {r["instance_id"]: r for r in rows(FULL)}
    out, changed = [], []
    for m in mini:
        r = full[m["instance_id"]]
        for k in COMPARED:
            a, b = (as_list(m[k]), as_list(r[k])) if k.endswith("PASS") else (m[k], r[k])
            if a != b:
                changed.append(f"{m['instance_id']}.{k}")
        out.append(r)
    out.sort(key=lambda r: r["instance_id"])
    (data / "verified-mini").mkdir(parents=True, exist_ok=True)
    target = data / "verified-mini.jsonl"
    with open(target, "w") as f:
        for r in out:
            f.write(json.dumps(r, default=str) + "\n")
    shutil.copyfile(target, data / "verified-mini" / "test.jsonl")

    print(f"{len(out)} tasks; fields SWE-bench_Verified changed since the mini set was cut: {', '.join(changed) or 'none'}")
    if sha256(target) != OUT_SHA256:
        sys.exit(f"{target} differs from the file the gold-patch baseline ran on (sha256 {OUT_SHA256[:16]}...).")
    images = sorted(line.split("@")[0] for line in Path("images.txt").read_text().split())
    if sorted("docker.io/" + r["image"] for r in out) != images:
        sys.exit("images.txt does not list exactly the dataset's task images.")
    print(f"{target} matches the baseline's task file, and images.txt lists its {len(images)} images")


if __name__ == "__main__":
    main()

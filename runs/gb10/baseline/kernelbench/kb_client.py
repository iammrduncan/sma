#!/usr/bin/env python3
"""Send kernels to the KernelBench service on the GB10 and wait for the results (standard library only).

    set -a; . runs/.env; set +a                       # KERNELBENCH_URL
    python3 kb_client.py submit --level 1 --kernels kernels.jsonl --out results.json --name run-42
    python3 kb_client.py status <job_id>
    python3 kb_client.py environment
    python3 kb_client.py baseline --level 1           # the stored eager-PyTorch times for this environment

kernels.jsonl holds one JSON object per line: {"problem_id": 19, "kernel_src": "<python module>", "label": "..."}.
kernel_src is what the model wrote: a Python module that defines `class ModelNew(nn.Module)` with the
problem's forward signature, usually with CUDA code built through torch.utils.cpp_extension.load_inline.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("KERNELBENCH_URL", "").rstrip("/")


def call(method: str, path: str, body: dict | None = None, timeout: int = 60):
    if not BASE:
        sys.exit("KERNELBENCH_URL is not set: set -a; . runs/.env; set +a")
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:      # 400 names the bad field; 429 means the queue is full
        sys.exit(f"{method} {path}: HTTP {e.code} {e.read().decode(errors='replace')[:500]}")


def options(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k] = int(v) if v.isdigit() else v
    return out


def submit(a):
    kernels = [json.loads(l) for l in open(a.kernels) if l.strip()]
    body = {"level": a.level, "kernels": kernels, "options": options(a.option)}
    if a.name:
        body["name"] = a.name
    job = call("POST", "/v1/jobs", body)
    print(f"job {job['job_id']}: {job['status']}, {job['total']} kernels, position {job['position']}", flush=True)
    while True:
        j = call("GET", f"/v1/jobs/{job['job_id']}")
        if j["status"] in ("done", "failed", "cancelled"):
            break
        print(f"  {j['status']} {j['progress']['done']}/{j['progress']['total']}", flush=True)
        time.sleep(a.poll)
    with open(a.out, "w") as f:
        json.dump(j, f, indent=1)
    rs = j.get("results", [])
    ok = sum(1 for r in rs if r["error_kind"] is None)
    print(f"{j['status']}: {ok}/{len(rs)} compiled, correct and timed; fingerprint "
          f"{(j.get('environment') or {}).get('fingerprint')}; written to {a.out}")
    if j.get("error"):
        print("job error:", j["error"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--level", type=int, required=True)
    s.add_argument("--kernels", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--name")
    s.add_argument("--option", action="append", default=[], help="e.g. timeout_s=600, precision=bf16")
    s.add_argument("--poll", type=int, default=15)
    st = sub.add_parser("status")
    st.add_argument("job_id")
    sub.add_parser("environment")
    b = sub.add_parser("baseline")
    b.add_argument("--level", type=int, default=1)
    b.add_argument("--fingerprint")
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a)
    elif a.cmd == "status":
        print(json.dumps(call("GET", f"/v1/jobs/{a.job_id}"), indent=1))
    elif a.cmd == "environment":
        print(json.dumps(call("GET", "/v1/environment"), indent=1))
    else:
        q = f"?fingerprint={a.fingerprint}" if a.fingerprint else ""
        print(json.dumps(call("GET", f"/v1/baselines/level{a.level}-fp32-cuda_event-100.json{q}"), indent=1))


if __name__ == "__main__":
    main()

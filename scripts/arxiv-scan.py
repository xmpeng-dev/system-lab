#!/usr/bin/env python3
"""Incremental arXiv scan for training-systems papers.

Produces the raw corpus behind knowledge/systems/arxiv-digest-<YYYY-MM>.md.
The 2026-08 digest lost its query terms and scoring weights to the chat
session it was run in; this script exists so later scans are reproducible
and comparable.

Funnel: full sweep of cs.DC / cs.AR / cs.PF over the window, plus
keyword-targeted queries into cs.LG / cs.CL / cs.NE (too large to sweep),
then keyword-weighted scoring, then dedup against arXiv IDs already cited
anywhere in the repo.

Usage:
    scripts/arxiv-scan.py --start 2026-08-07 --end 2026-09-03
    scripts/arxiv-scan.py --start 2026-09-04 --end 2026-09-30 --outdir /tmp/scan

Outputs <outdir>/arxiv_scored.json, sorted by score descending, each record
carrying id / title / abstract / published / authors / cats / score / hits /
known. Read that file to write the digest; the script deliberately does not
try to summarise anything itself.

Two things that will bite you:

- arXiv announcement lag. Papers submitted in the last few days before
  --end are not indexed yet. The summary prints the actual indexed range;
  state that, not the requested window, in the digest.
- Rate limiting. A full month-wide scan is ~50 requests over ~8 minutes at
  the default delay. Dropping --delay below 3 earns HTTP 429s that then
  persist for tens of minutes and poison any follow-up run. Do not lower it,
  and do not re-run back to back.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

FULL_CATS = ["cs.DC", "cs.AR", "cs.PF"]

# cs.LG / cs.CL are far too large to sweep, so go in with terms instead.
TERMS = [
    "mixture-of-experts", "expert parallelism", "megakernel", "mega-kernel",
    "persistent kernel", "communication overlap", "computation-communication",
    "pipeline parallelism", "tensor parallelism", "sequence parallelism",
    "context parallelism", "FP8 training", "FP4 training", "low-precision training",
    "quantized training", "gradient compression", "all-to-all", "all-reduce",
    "reduce-scatter", "kernel fusion", "GPU kernel", "training throughput",
    "large-scale training", "LLM training", "distributed training",
    "ROCm", "AMD GPU", "MI300X", "Instinct", "attention kernel",
    "flash attention", "activation checkpointing", "recomputation",
    "optimizer state", "ZeRO", "FSDP", "Megatron", "model FLOPs utilization",
    "load balancing experts", "expert routing", "block scaling", "microscaling",
]

# Weights are tuned so that a paper needs either a strong topical hit plus
# corroborating context, or several weak hits, to clear the threshold of 12.
# Negatives exist because domains like federated learning and molecular
# simulation share almost all of our systems vocabulary.
SCORE = [
    (r"\bmixture[- ]of[- ]experts\b|\bMoE\b", 6),
    (r"\bmega-?kernel", 6),
    (r"\bpersistent kernel", 5),
    (r"overlap", 5),
    (r"\bROCm\b|\bAMD\b|\bMI300|\bMI325|\bMI355|\bCDNA\b|\bInstinct\b|gfx9", 6),
    (r"\bexpert parallel|\bEP\b|all[- ]to[- ]all", 5),
    (r"\bFP8\b|\bFP4\b|\bFP6\b|MXFP|NVFP|low[- ]precision|microscaling|block[- ]scal", 5),
    (r"pipeline parallel|tensor parallel|sequence parallel|context parallel|\bFSDP\b|\bZeRO\b", 5),
    (r"\bMegatron\b|TorchTitan|DeepSpeed|\bJAX\b|MaxText", 4),
    (r"\bkernel\b", 3),
    (r"\bGEMM\b|matrix multipl", 4),
    (r"gradient compress|gradient quant|communication compress", 5),
    (r"\bDSL\b|compiler|codegen|code generation|autotun", 3),
    (r"\btraining\b", 3),
    (r"\bLLM\b|large language model|transformer", 2),
    (r"\bGPU\b|\bNPU\b|\bTPU\b|accelerator", 2),
    (r"roofline|performance model|microbenchmark|profil", 3),
    (r"checkpoint|fault toleran|failure|elastic|straggler", 3),
    (r"scal(e|ing|able) to|thousand|large[- ]scale|cluster", 2),
    (r"attention", 2),
    (r"federated", -7),
    (r"\bmedical\b|clinical|healthcare|diagnos(is|tic)|radiolog|protein|genomic|molecul", -8),
    (r"blockchain|consensus protocol|smart contract|cryptocurrenc", -9),
    (r"recommend(er|ation) system", -3),
    (r"\bIoT\b|edge device|smartphone|microcontroller|wireless|\b6G\b|\b5G\b|vehicular|UAV|satellite", -6),
    (r"survey|position paper", -2),
    (r"image generation|diffusion model|text-to-image|video generation|speech|\bASR\b", -4),
    (r"graph neural network|\bGNN\b", -4),
    (r"privacy|differential privacy|adversarial attack|watermark|jailbreak|hallucinat", -6),
    (r"quantum", -9),
    (r"serverless|microservice|web service|database quer|\bSQL\b", -5),
]
COMPILED = [(re.compile(p, re.I), w) for p, w in SCORE]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def known_ids():
    """arXiv IDs already cited as links anywhere in the repo."""
    try:
        out = subprocess.run(
            ["grep", "-rhoE", r"arxiv\.org/abs/[0-9]{4}\.[0-9]{4,5}",
             f"{REPO}/papers", f"{REPO}/knowledge", f"{REPO}/notes"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:  # noqa: BLE001
        return set()
    return set(re.findall(r"([0-9]{4}\.[0-9]{4,5})", out))


def fetch(search_query, start, max_results, tries=5):
    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(f"{API}?{params}", timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            # 429 means we have already annoyed arXiv; short retries make it
            # worse. Back off by minutes, not seconds.
            wait = 60 * (attempt + 1) if e.code == 429 else 5 * (attempt + 1)
            retry_after = e.headers.get("Retry-After") if e.headers else None
            if retry_after and retry_after.isdigit():
                wait = max(wait, int(retry_after))
            sys.stderr.write(f"  HTTP {e.code}, sleeping {wait}s "
                             f"(attempt {attempt + 1}/{tries})\n")
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  retry {attempt + 1}/{tries} ({e})\n")
            time.sleep(5 * (attempt + 1))
    return None


def parse(blob, out):
    root = ET.fromstring(blob)
    entries = root.findall("a:entry", NS)
    for e in entries:
        m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", e.find("a:id", NS).text)
        if not m or m.group(1) in out:
            continue
        out[m.group(1)] = {
            "id": m.group(1),
            "title": " ".join(e.find("a:title", NS).text.split()),
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
            "published": e.find("a:published", NS).text,
            "updated": e.find("a:updated", NS).text,
            "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
            "cats": [c.attrib.get("term") for c in e.findall("a:category", NS)],
        }
    return len(entries)


def sweep(query, label, out, page, cap, delay, failures):
    """Page through one query. Records, rather than swallows, dead pages.

    A literature scan that silently under-collects is worse than one that
    fails, so unreachable pages are surfaced and main() refuses to write a
    corpus built on them.
    """
    start = 0
    while start < cap:
        blob = fetch(query, start, page)
        if blob is None:
            failures.append(f"{label}@{start}")
            sys.stderr.write(f"[{label}] start={start} FAILED, giving up on query\n")
            return
        n = parse(blob, out)
        sys.stderr.write(f"[{label}] start={start} entries={n} uniq={len(out)}\n")
        if n < page:
            return
        start += page
        time.sleep(delay)
    sys.stderr.write(f"[{label}] hit cap={cap}, results may be truncated\n")
    failures.append(f"{label}@cap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="submission window start, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="submission window end, YYYY-MM-DD")
    ap.add_argument("--outdir", default="/tmp")
    ap.add_argument("--threshold", type=int, default=12)
    ap.add_argument("--delay", type=float, default=3.2,
                    help="seconds between API calls; arXiv asks for >=3")
    args = ap.parse_args()

    lo = args.start.replace("-", "") + "0000"
    hi = args.end.replace("-", "") + "2359"
    window = f"submittedDate:[{lo} TO {hi}]"

    all_recs = {}
    failures = []
    for cat in FULL_CATS:
        sweep(f"cat:{cat} AND {window}", cat, all_recs, 100, 4000, args.delay, failures)
    for term in TERMS:
        q = f'(cat:cs.LG OR cat:cs.CL OR cat:cs.NE) AND abs:"{term}" AND {window}'
        sweep(q, f"kw:{term}", all_recs, 100, 300, args.delay, failures)

    known = known_ids()
    scored = []
    for rec in all_recs.values():
        text = rec["title"] + " " + rec["abstract"]
        rec["hits"] = [(rx.pattern[:28], w) for rx, w in COMPILED if rx.search(text)]
        rec["score"] = sum(w for _, w in rec["hits"])
        if any(c in FULL_CATS for c in rec["cats"]):
            rec["score"] += 4
        rec["known"] = rec["id"] in known
        scored.append(rec)
    scored.sort(key=lambda r: -r["score"])

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "arxiv_scored.json")
    with open(path, "w") as f:
        json.dump(scored, f)

    over = [r for r in scored if r["score"] >= args.threshold]
    print(f"window            : {args.start} .. {args.end}")
    if scored:
        pub = [r["published"][:10] for r in scored]
        print(f"indexed submissions: {min(pub)} .. {max(pub)}")
    print(f"fetched unique    : {len(scored)}")
    print(f"score >= {args.threshold:<8}: {len(over)}")
    print(f"  of which new    : {sum(1 for r in over if not r['known'])}")
    print(f"already in slab   : {sum(1 for r in scored if r['known'])}")
    print(f"written           : {path}")
    if failures:
        print(f"\nINCOMPLETE — {len(failures)} queries did not finish: "
              f"{', '.join(failures[:10])}"
              f"{' ...' if len(failures) > 10 else ''}")
        print("Re-run before writing a digest; the corpus has holes.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

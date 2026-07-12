"""
compare_claims.py  —  Claim Comparison with Multi-Dimensional Annotation Scheme

For each of the ~51,946 preprint–publication pairs with extracted claims,
this script:
  1. Presents the full list of preprint claims and publication claims to the LLM.
  2. Asks the LLM to semantically align them (no fuzzy string matching).
  3. For each aligned pair, produces three independent annotations:
       - semantic   : Unchanged | Clarified | Revised | Removed | Added
       - scope      : Unchanged | Narrowed  | Broadened  (N/A for Removed/Added)
       - confidence : Unchanged | Tempered  | Amplified  (N/A for Removed/Added)
  4. Returns a matching_confidence (0.0–1.0) for each alignment.

Model: gpt-4o
Resumes from existing output — safe to interrupt and restart.

Reads from:
    computations/data/data_sources/claims/claims_extracted.jsonl

Writes to:
    computations/data/data_sources/claims/claim_changes.jsonl
    computations/data/data_sources/claims/claim_changes_flat.csv
    computations/analysis/outputs/comparison_report.txt

Usage:
    cd <project_root>
    export OPENAI_API_KEY="sk-..."          # or set in config/openai_key.txt
    python3 computations/analysis/scripts/analysis/compare_claims.py

    # Pilot mode (first 50 pairs only — run this before the full run):
    python3 computations/analysis/scripts/analysis/compare_claims.py --pilot

Estimated cost:  ~$30–50 USD for 51,946 pairs using gpt-4o
Estimated time:  3–5 hours (async, 20 concurrent requests)
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/moses/WorkPlaces/Sharebox3/WorkingProjects/The Evolution of Scientific Claims from Preprint to Publication in AI and Machine Learning")

CLAIMS_FILE  = PROJECT_ROOT / "computations" / "data" / "data_sources" / "claims" / "claims_extracted.jsonl"
KEY_FILE     = PROJECT_ROOT / "computations" / "data" / "config" / "openai_key.txt"
OUT_DIR      = PROJECT_ROOT / "computations" / "data" / "data_sources" / "claims"
OUT_CHANGES  = OUT_DIR / "claim_changes.jsonl"
OUT_FLAT     = OUT_DIR / "claim_changes_flat.csv"
OUT_REPORT   = PROJECT_ROOT / "computations" / "analysis" / "outputs" / "comparison_report.txt"

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL        = "gpt-4o"
CONCURRENCY  = 20          # async semaphore limit
MAX_RETRIES  = 8           # total attempts per pair
RETRY_WAIT   = 5           # seconds for generic errors (multiplied by attempt)
RETRY_429    = 62          # seconds to wait on rate-limit (one full TPM window)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert annotator of scientific claims. You will be given two lists of claims: one from a preprint abstract and one from the corresponding published version of the same paper.

Your task has two parts:

PART 1 — ALIGNMENT
Semantically align each preprint claim to the most similar publication claim. Base alignment on meaning, not wording. A preprint claim with no meaningful counterpart in the publication is "Removed". A publication claim with no meaningful counterpart in the preprint is "Added".

PART 2 — ANNOTATION
For each aligned pair (excluding Removed and Added), assign three independent labels:

1. semantic — What happened to the claim overall?
   - Unchanged  : the claim conveys the same meaning in both versions
   - Clarified  : the claim is reworded for precision or clarity without changing its scope or confidence
   - Revised    : the claim's substance, framing, or emphasis changed materially

2. scope — Did the domain of applicability change?
   - Unchanged  : the claim applies to the same set of conditions, datasets, or populations
   - Narrowed   : the published claim applies to a more restricted set (e.g., "across all tasks" → "on benchmark X")
   - Broadened  : the published claim applies to a wider set

3. confidence — Did the expressed certainty change?
   - Unchanged  : the level of hedging or assertion is the same
   - Tempered   : the published claim is more hedged, cautious, or uncertain (e.g., "proves" → "suggests")
   - Amplified  : the published claim is more assertive or certain

Also assign:
   - matching_confidence (float 0.0–1.0): your confidence that this is the correct alignment. Use 1.0 for obvious matches, lower values when the alignment is uncertain.

For Removed claims: set scope and confidence to "N/A".
For Added claims: set scope and confidence to "N/A".

Return a JSON object with this exact structure:
{
  "alignments": [
    {
      "preprint_claim": "<text of preprint claim, or null if Added>",
      "publication_claim": "<text of publication claim, or null if Removed>",
      "semantic": "Unchanged|Clarified|Revised|Removed|Added",
      "scope": "Unchanged|Narrowed|Broadened|N/A",
      "confidence": "Unchanged|Tempered|Amplified|N/A",
      "matching_confidence": 0.95,
      "rationale": "<one sentence explaining the annotation>"
    }
  ],
  "pair_summary": {
    "n_unchanged": 0,
    "n_clarified": 0,
    "n_revised": 0,
    "n_removed": 0,
    "n_added": 0,
    "n_scope_narrowed": 0,
    "n_scope_broadened": 0,
    "n_confidence_tempered": 0,
    "n_confidence_amplified": 0,
    "dominant_semantic": "Unchanged|Clarified|Revised|Removed|Added|Mixed"
  }
}

Rules:
- Every preprint claim must appear exactly once across all alignments.
- Every publication claim must appear exactly once across all alignments.
- dominant_semantic is the most frequent semantic label; use "Mixed" if no single label exceeds 50%.
- Return only valid JSON, no other text.
"""


def build_user_message(preprint_claims: list, pub_claims: list) -> str:
    def fmt(i, c):
        if isinstance(c, dict):
            text = c.get("claim", str(c))
        else:
            text = str(c)
        return f"{i + 1}. {text}"

    pre_text = "\n".join(fmt(i, c) for i, c in enumerate(preprint_claims))
    pub_text = "\n".join(fmt(i, c) for i, c in enumerate(pub_claims))
    return (
        f"PREPRINT CLAIMS:\n{pre_text}\n\n"
        f"PUBLICATION CLAIMS:\n{pub_text}\n\n"
        "Align and annotate these claims."
    )


# ── Resume support ─────────────────────────────────────────────────────────────

def load_done_ids() -> set:
    done = set()
    if OUT_CHANGES.exists():
        with open(OUT_CHANGES, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line).get("pair_id", ""))
                    except json.JSONDecodeError:
                        pass
    return done


# ── Single comparison call ─────────────────────────────────────────────────────

async def compare_pair(client: AsyncOpenAI, pair: dict, sem: asyncio.Semaphore) -> dict:
    pair_id         = pair["pair_id"]
    preprint_claims = pair.get("preprint_claims", [])
    pub_claims      = pair.get("publication_claims", [])

    meta = {k: pair.get(k, "") for k in
            ["source", "preprint_year", "pub_year", "venue_type",
             "linkage_method", "preprint_doi", "pub_doi"]}

    if not preprint_claims or not pub_claims:
        return {"pair_id": pair_id, "error": "missing claims", **meta}

    user_msg = build_user_message(preprint_claims, pub_claims)

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=1500,
                )
                raw    = response.choices[0].message.content
                parsed = json.loads(raw)
                summary = parsed.get("pair_summary", {})

                return {
                    "pair_id":                   pair_id,
                    **meta,
                    "alignments":                parsed.get("alignments", []),
                    "n_unchanged":               summary.get("n_unchanged", 0),
                    "n_clarified":               summary.get("n_clarified", 0),
                    "n_revised":                 summary.get("n_revised", 0),
                    "n_removed":                 summary.get("n_removed", 0),
                    "n_added":                   summary.get("n_added", 0),
                    "n_scope_narrowed":          summary.get("n_scope_narrowed", 0),
                    "n_scope_broadened":         summary.get("n_scope_broadened", 0),
                    "n_confidence_tempered":     summary.get("n_confidence_tempered", 0),
                    "n_confidence_amplified":    summary.get("n_confidence_amplified", 0),
                    "dominant_semantic":         summary.get("dominant_semantic", ""),
                    "tokens_used":               response.usage.total_tokens if response.usage else None,
                }
            except Exception as e:
                err_str = str(e)
                is_429  = "429" in err_str or "rate_limit" in err_str.lower()
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_429 if is_429 else RETRY_WAIT * (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    return {"pair_id": pair_id, "error": err_str, **meta}


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(pilot: bool = False, force: bool = False, reset: bool = False):
    print("=== Claim Comparison: Multi-Dimensional Annotation ===\n")

    # Resolve API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and KEY_FILE.exists():
        api_key = KEY_FILE.read_text().strip()
    if not api_key:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY environment variable "
            f"or place your key in {KEY_FILE}"
        )

    # Archive old output if --reset requested
    if reset:
        if OUT_CHANGES.exists():
            archive = OUT_CHANGES.with_suffix(".jsonl.bak")
            OUT_CHANGES.rename(archive)
            print(f"  Archived old output → {archive.name}")
        if OUT_FLAT.exists():
            OUT_FLAT.unlink()
        print()

    # Load extracted claims
    print("Loading extracted claims...")
    pairs = []
    with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"  Loaded {len(pairs):,} pairs")

    if pilot:
        pairs = pairs[:50]
        print(f"  PILOT MODE: processing first 50 pairs only")
        if force:
            print(f"  FORCE MODE: re-processing even if already done\n")
            todo = [p for p in pairs if not p.get("error")]
        else:
            done_ids = load_done_ids()
            if done_ids:
                print(f"  Already processed: {len(done_ids):,} — resuming...")
            todo = [p for p in pairs if p.get("pair_id", "") not in done_ids and not p.get("error")]
            print(f"  To process: {len(todo):,}\n")
            if not todo:
                print("All pilot pairs already processed. Use --force to re-run them.")
                return
    else:
        print()
        done_ids = load_done_ids()
        if done_ids:
            print(f"  Already processed: {len(done_ids):,} — resuming...")
        todo = [p for p in pairs if p.get("pair_id", "") not in done_ids and not p.get("error")]

    print(f"  To process: {len(todo):,}\n")

    if not todo:
        print("Nothing to process.")
    else:
        client = AsyncOpenAI(api_key=api_key)
        sem    = asyncio.Semaphore(CONCURRENCY)

        n_done  = 0
        n_error = 0
        start   = time.time()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

        with open(OUT_CHANGES, "a", encoding="utf-8") as out_f:
            pbar = tqdm(total=len(todo), desc="  Comparing", unit="pair")
            # Process in chunks to keep memory manageable
            chunk_size = 200
            for i in range(0, len(todo), chunk_size):
                chunk   = todo[i:i + chunk_size]
                results = await asyncio.gather(*[compare_pair(client, p, sem) for p in chunk])
                for rec in results:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    if rec.get("error"):
                        n_error += 1
                        if n_error <= 3:
                            print(f"  ERROR [{rec.get('pair_id','')}]: {rec['error']}")
                    else:
                        n_done += 1
                pbar.update(len(chunk))
            pbar.close()

        elapsed = time.time() - start
        print(f"\n  Processed {n_done:,} pairs in {elapsed / 60:.1f} minutes")
        print(f"  Errors: {n_error:,}")

    # ── Build flat CSV ─────────────────────────────────────────────────────────
    print("\nBuilding flat CSV...")
    flat_rows   = []
    all_records = []
    with open(OUT_CHANGES, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                all_records.append(rec)
                for aln in rec.get("alignments", []):
                    flat_rows.append({
                        "pair_id":             rec["pair_id"],
                        "source":              rec.get("source", ""),
                        "preprint_year":       rec.get("preprint_year", ""),
                        "pub_year":            rec.get("pub_year", ""),
                        "venue_type":          rec.get("venue_type", ""),
                        "preprint_doi":        rec.get("preprint_doi", ""),
                        "pub_doi":             rec.get("pub_doi", ""),
                        "preprint_claim":      aln.get("preprint_claim", ""),
                        "publication_claim":   aln.get("publication_claim", ""),
                        "semantic":            aln.get("semantic", ""),
                        "scope":               aln.get("scope", ""),
                        "confidence":          aln.get("confidence", ""),
                        "matching_confidence": aln.get("matching_confidence", ""),
                        "rationale":           aln.get("rationale", ""),
                    })
            except json.JSONDecodeError:
                pass

    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_csv(OUT_FLAT, index=False, encoding="utf-8")
    print(f"  Saved {len(flat_df):,} claim alignments → {OUT_FLAT.name}")

    # ── Summary report ─────────────────────────────────────────────────────────
    if flat_df.empty or "semantic" not in flat_df.columns:
        print("No alignments to report.")
        return

    total = len(flat_df)
    sem_counts  = Counter(flat_df["semantic"].tolist())
    sco_counts  = Counter(flat_df.loc[~flat_df["scope"].isin(["N/A", ""]), "scope"].tolist())
    con_counts  = Counter(flat_df.loc[~flat_df["confidence"].isin(["N/A", ""]), "confidence"].tolist())
    dom_counts  = Counter(rec.get("dominant_semantic", "") for rec in all_records)

    mc_vals = pd.to_numeric(flat_df["matching_confidence"], errors="coerce").dropna()

    lines = [
        "=== Claim Comparison Report ===\n",
        f"Total pairs processed        : {len(all_records):,}",
        f"Total claim alignments       : {total:,}",
        f"\nSemantic evolution (claim level):",
    ]
    for label, n in sorted(sem_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {label:<12}: {n:>7,}  ({n/total*100:.1f}%)" if total else f"  {label}")

    lines.append(f"\nScope evolution (aligned pairs only, excl. Removed/Added):")
    sco_total = sum(sco_counts.values())
    for label, n in sorted(sco_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {label:<12}: {n:>7,}  ({n/sco_total*100:.1f}%)" if sco_total else f"  {label}")

    lines.append(f"\nConfidence evolution (aligned pairs only, excl. Removed/Added):")
    con_total = sum(con_counts.values())
    for label, n in sorted(con_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {label:<12}: {n:>7,}  ({n/con_total*100:.1f}%)" if con_total else f"  {label}")

    lines += [
        f"\nDominant semantic (pair level):",
    ]
    for label, n in sorted(dom_counts.items(), key=lambda x: -x[1]):
        pct = n / len(all_records) * 100 if all_records else 0
        lines.append(f"  {label:<12}: {n:>7,}  ({pct:.1f}%)")

    if len(mc_vals) > 0:
        lines += [
            f"\nMatching confidence distribution:",
            f"  Mean   : {mc_vals.mean():.3f}",
            f"  Median : {mc_vals.median():.3f}",
            f"  < 0.70 : {(mc_vals < 0.70).sum():,}  ({(mc_vals < 0.70).mean()*100:.1f}%) ← review these",
        ]

    report = "\n".join(lines)
    print("\n" + report)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n  Saved report → {OUT_REPORT.name}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="Process only the first 50 pairs (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process pairs even if already in output (use with --pilot)")
    parser.add_argument("--reset", action="store_true",
                        help="Archive existing output and start the full run from scratch")
    args = parser.parse_args()
    asyncio.run(main(pilot=args.pilot, force=args.force, reset=args.reset))

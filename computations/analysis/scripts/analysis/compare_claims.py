"""
compare_claims.py  —  Step 9 (Claim Comparison and Change Classification)

For each of the 51,946 preprint–publication pairs with extracted claims,
compares the preprint claims against the publication claims and classifies
each change using an LLM judge.

Change categories:
  - strengthened : hedged → more assertive (e.g., "suggests" → "demonstrates")
  - weakened     : assertive → more hedged (e.g., "proves" → "indicates")
  - unchanged    : same claim, same certainty level
  - added        : new claim present in publication but not in preprint
  - removed      : claim present in preprint but dropped from publication

Also computes a pair-level summary:
  - dominant_change: the most common change type for the pair
  - n_strengthened, n_weakened, n_unchanged, n_added, n_removed

Reads from:
    data/claims/claims_extracted.jsonl

Writes to:
    data/claims/claim_changes.jsonl      ← one record per pair with full change details
    data/claims/claim_changes_flat.csv   ← flat CSV with one row per claim comparison
    data/claims/comparison_report.txt    ← summary statistics

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    python3 code/analysis/compare_claims.py

Estimated cost:  ~$8 USD for 51,946 pairs using gpt-4o-mini
Estimated time:  2–3 hours
"""

import asyncio
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

CLAIMS_FILE  = PROJECT_ROOT / "data" / "claims" / "claims_extracted.jsonl"
KEY_FILE     = PROJECT_ROOT / "config" / "openai_key.txt"
OUT_DIR      = PROJECT_ROOT / "data" / "claims"
OUT_CHANGES  = OUT_DIR / "claim_changes.jsonl"
OUT_FLAT     = OUT_DIR / "claim_changes_flat.csv"
OUT_REPORT   = OUT_DIR / "comparison_report.txt"

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL       = "gpt-4o-mini"
BATCH_SIZE  = 50
MAX_RETRIES = 3
RETRY_WAIT  = 5

# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a scientific claim comparator. You will be given a list of claims from a preprint abstract and a list of claims from its published journal version.

Your task is to compare the two sets of claims and classify each change.

For each preprint claim, find the most semantically similar publication claim and classify the change as:
- "strengthened": the publication claim is more assertive, confident, or certain than the preprint claim
- "weakened": the publication claim is more hedged, cautious, or uncertain than the preprint claim  
- "unchanged": the claim is essentially the same in both versions
- "removed": the preprint claim has no corresponding claim in the publication

For each publication claim with no corresponding preprint claim, classify it as:
- "added": new claim present only in the publication

Return a JSON object with this exact structure:
{
  "comparisons": [
    {
      "preprint_claim": "the original preprint claim text, or null if added",
      "publication_claim": "the publication claim text, or null if removed",
      "change_type": "strengthened|weakened|unchanged|removed|added",
      "reasoning": "one sentence explaining why this change type was assigned"
    }
  ],
  "pair_summary": {
    "n_strengthened": 0,
    "n_weakened": 0,
    "n_unchanged": 0,
    "n_added": 0,
    "n_removed": 0,
    "dominant_change": "strengthened|weakened|unchanged|mixed"
  }
}

Rules:
- Every preprint claim must appear exactly once (as strengthened, weakened, unchanged, or removed).
- Every publication claim must appear exactly once (as strengthened, weakened, unchanged, or added).
- Be precise: only classify as "strengthened" or "weakened" if there is a clear difference in certainty language.
- dominant_change should be "mixed" if no single change type accounts for more than 50% of comparisons.
- Return only valid JSON, no other text."""

def build_user_message(preprint_claims: list, pub_claims: list) -> str:
    def fmt_claim(i, c):
        if isinstance(c, dict):
            return f"{i+1}. [{c.get('certainty','?')}] {c.get('claim','')}"
        return f"{i+1}. {str(c)}"
    pre_text = "\n".join(fmt_claim(i, c) for i, c in enumerate(preprint_claims))
    pub_text = "\n".join(fmt_claim(i, c) for i, c in enumerate(pub_claims))
    return (
        f"PREPRINT CLAIMS:\n{pre_text}\n\n"
        f"PUBLICATION CLAIMS:\n{pub_text}\n\n"
        "Compare these two sets of claims and classify each change."
    )

# ── Load already-processed pair IDs ───────────────────────────────────────────

def load_done_ids() -> set:
    done = set()
    if OUT_CHANGES.exists():
        with open(OUT_CHANGES, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        done.add(obj.get("pair_id", ""))
                    except json.JSONDecodeError:
                        pass
    return done

# ── Single comparison call ─────────────────────────────────────────────────────

async def compare_pair_claims(client: AsyncOpenAI, pair: dict) -> dict:
    pair_id        = pair["pair_id"]
    preprint_claims = pair.get("preprint_claims", [])
    pub_claims      = pair.get("publication_claims", [])

    if not preprint_claims or not pub_claims:
        return {
            "pair_id": pair_id,
            "error": "missing claims",
            **{k: pair.get(k, "") for k in
               ["source", "preprint_year", "pub_year", "venue_type",
                "linkage_method", "preprint_doi", "pub_doi"]}
        }

    user_msg = build_user_message(preprint_claims, pub_claims)

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
                max_tokens=1200,
            )
            raw    = response.choices[0].message.content
            parsed = json.loads(raw)
            summary = parsed.get("pair_summary", {})

            return {
                "pair_id":          pair_id,
                "source":           pair.get("source", ""),
                "preprint_year":    pair.get("preprint_year", ""),
                "pub_year":         pair.get("pub_year", ""),
                "venue_type":       pair.get("venue_type", ""),
                "linkage_method":   pair.get("linkage_method", ""),
                "preprint_doi":     pair.get("preprint_doi", ""),
                "pub_doi":          pair.get("pub_doi", ""),
                "comparisons":      parsed.get("comparisons", []),
                "n_strengthened":   summary.get("n_strengthened", 0),
                "n_weakened":       summary.get("n_weakened", 0),
                "n_unchanged":      summary.get("n_unchanged", 0),
                "n_added":          summary.get("n_added", 0),
                "n_removed":        summary.get("n_removed", 0),
                "dominant_change":  summary.get("dominant_change", ""),
                "tokens_used":      response.usage.total_tokens if response.usage else None,
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_WAIT * (attempt + 1))
            else:
                return {"pair_id": pair_id, "error": str(e),
                        **{k: pair.get(k, "") for k in
                           ["source", "preprint_year", "pub_year", "venue_type",
                            "linkage_method", "preprint_doi", "pub_doi"]}}

# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    print("=== Step 9: Claim Comparison and Change Classification ===\n")

    api_key = KEY_FILE.read_text().strip()
    if not api_key:
        raise ValueError(f"OpenAI API key not found in {KEY_FILE}")

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
    print(f"  Loaded {len(pairs):,} pairs\n")

    # Skip already-processed
    done_ids  = load_done_ids()
    if done_ids:
        print(f"  Already processed: {len(done_ids):,} — resuming...")
    todo = [p for p in pairs if p.get("pair_id", "") not in done_ids
            and not p.get("error")]
    print(f"  To process: {len(todo):,}\n")

    if not todo:
        print("All pairs already processed.")
    else:
        client  = AsyncOpenAI(api_key=api_key)
        batches = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

        n_done  = 0
        n_error = 0
        start   = time.time()

        with open(OUT_CHANGES, "a", encoding="utf-8") as out_f:
            pbar = tqdm(total=len(todo), desc="  Comparing", unit="pair")
            for batch in batches:
                results = await asyncio.gather(*[compare_pair_claims(client, p) for p in batch])
                for rec in results:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    if rec.get("error"):
                        n_error += 1
                    else:
                        n_done += 1
                pbar.update(len(batch))
            pbar.close()

        elapsed = time.time() - start
        print(f"\n  Processed {n_done:,} pairs in {elapsed/60:.1f} minutes")
        print(f"  Errors: {n_error:,}")

    # ── Build flat CSV ─────────────────────────────────────────────────────────
    print("\nBuilding flat CSV...")
    flat_rows = []
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
                for comp in rec.get("comparisons", []):
                    flat_rows.append({
                        "pair_id":          rec["pair_id"],
                        "source":           rec.get("source", ""),
                        "preprint_year":    rec.get("preprint_year", ""),
                        "pub_year":         rec.get("pub_year", ""),
                        "venue_type":       rec.get("venue_type", ""),
                        "linkage_method":   rec.get("linkage_method", ""),
                        "preprint_doi":     rec.get("preprint_doi", ""),
                        "pub_doi":          rec.get("pub_doi", ""),
                        "preprint_claim":   comp.get("preprint_claim", ""),
                        "publication_claim":comp.get("publication_claim", ""),
                        "change_type":      comp.get("change_type", ""),
                        "reasoning":        comp.get("reasoning", ""),
                    })
            except json.JSONDecodeError:
                pass

    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_csv(OUT_FLAT, index=False, encoding="utf-8")
    print(f"  Saved {len(flat_df):,} claim comparisons → {OUT_FLAT.name}")

    # ── Summary statistics ─────────────────────────────────────────────────────
    change_counts = Counter(flat_df["change_type"].tolist())
    total_claims  = len(flat_df)

    dominant_counts = Counter(
        rec.get("dominant_change", "") for rec in all_records if not rec.get("error")
    )

    lines = [
        "=== Step 9: Claim Comparison Report ===\n",
        f"Total pairs compared         : {len(all_records):,}",
        f"Total claim comparisons      : {total_claims:,}",
        f"\nChange type breakdown (claim level):",
    ]
    for ct, n in sorted(change_counts.items(), key=lambda x: -x[1]):
        pct = n / total_claims * 100 if total_claims else 0
        lines.append(f"  {ct:<15}: {n:>7,}  ({pct:.1f}%)")

    lines += [
        f"\nDominant change type (pair level):",
    ]
    for ct, n in sorted(dominant_counts.items(), key=lambda x: -x[1]):
        pct = n / len(all_records) * 100 if all_records else 0
        lines.append(f"  {ct:<15}: {n:>7,}  ({pct:.1f}%)")

    lines += [
        f"\nBreakdown by source (dominant change):",
        pd.DataFrame(all_records)[["source", "dominant_change"]]
          .value_counts().to_string(),
        f"\nBreakdown by venue type (dominant change):",
        pd.DataFrame(all_records)[["venue_type", "dominant_change"]]
          .value_counts().to_string(),
    ]

    report = "\n".join(lines)
    print("\n" + report)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n  Saved report → {OUT_REPORT.name}")
    print("\nStep 9 complete.")


if __name__ == "__main__":
    asyncio.run(main())

"""
extract_claims.py  —  Step 8 (LLM Claim Extraction)

For each of the 51,952 preprint–publication pairs, extracts 3–5 key
scientific claims from:
  (a) the preprint abstract
  (b) the publication abstract

Uses OpenAI GPT-4o mini with structured JSON output.
Processes in batches of 50 concurrent requests with live progress bar.
Fully resumable — already-processed pairs are skipped on re-run.

Reads from:
    data/final/analysis_corpus.csv
    config/openai_key.txt

Writes to:
    data/claims/claims_extracted.jsonl   ← one JSON object per pair (appended live)
    data/claims/extraction_errors.jsonl  ← failed pairs for retry

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    pip install openai tqdm
    python3 code/analysis/extract_claims.py

Estimated cost:  ~$10 USD for 51,952 pairs using gpt-4o-mini
Estimated time:  2–3 hours
"""

import asyncio
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

CORPUS_FILE  = PROJECT_ROOT / "data" / "final" / "analysis_corpus.csv"
KEY_FILE     = PROJECT_ROOT / "config" / "openai_key.txt"
OUT_DIR      = PROJECT_ROOT / "data" / "claims"
OUT_CLAIMS   = OUT_DIR / "claims_extracted.jsonl"
OUT_ERRORS   = OUT_DIR / "extraction_errors.jsonl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL          = "gpt-4o-mini"
BATCH_SIZE     = 50      # pairs processed per batch (2 API calls each = 100 RPM)
MAX_RETRIES    = 3
RETRY_WAIT     = 5       # seconds between retries

# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a scientific claim extractor. Given an abstract from a research paper, extract the 3 to 5 most important scientific claims made in the text.

A scientific claim is a specific, testable assertion about findings, results, or conclusions — NOT a description of methods or background.

Return a JSON object with this exact structure:
{
  "claims": [
    {
      "claim": "A concise one-sentence statement of the claim",
      "certainty": "high|medium|low",
      "type": "result|conclusion|contribution|limitation"
    }
  ]
}

Rules:
- Extract exactly 3 to 5 claims. Never fewer than 3, never more than 5.
- Each claim must be a complete, standalone sentence.
- Preserve the original certainty language (e.g., if the abstract says "suggests", use certainty=low; "demonstrates" → high; "indicates" → medium).
- Do not invent claims not present in the abstract.
- Return only valid JSON, no other text."""

USER_TEMPLATE = "Extract the key scientific claims from this abstract:\n\n{abstract}"

# ── Load already-processed pair IDs ───────────────────────────────────────────

def load_done_ids() -> set:
    done = set()
    if OUT_CLAIMS.exists():
        with open(OUT_CLAIMS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        done.add(obj.get("pair_id", ""))
                    except json.JSONDecodeError:
                        pass
    return done

# ── Single extraction call ─────────────────────────────────────────────────────

async def extract_claims(client: AsyncOpenAI, abstract: str,
                         pair_id: str, version: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_TEMPLATE.format(abstract=abstract)},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=800,
            )
            raw    = response.choices[0].message.content
            parsed = json.loads(raw)
            claims = parsed.get("claims", [])
            return {
                "pair_id":     pair_id,
                "version":     version,
                "claims":      claims,
                "n_claims":    len(claims),
                "model":       MODEL,
                "tokens_used": response.usage.total_tokens if response.usage else None,
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_WAIT * (attempt + 1))
            else:
                return {"pair_id": pair_id, "version": version, "error": str(e)}

# ── Process one pair (two API calls) ──────────────────────────────────────────

async def process_pair(client: AsyncOpenAI, row: dict) -> dict:
    pair_id = row["pair_id"]

    pre_result, pub_result = await asyncio.gather(
        extract_claims(client, row["preprint_abstract"], pair_id, "preprint"),
        extract_claims(client, row["pub_abstract"],      pair_id, "publication"),
    )

    return {
        "pair_id":            pair_id,
        "source":             row.get("source", ""),
        "preprint_year":      row.get("preprint_year", ""),
        "pub_year":           row.get("pub_year", ""),
        "venue_type":         row.get("venue_type", ""),
        "linkage_method":     row.get("linkage_method", ""),
        "preprint_doi":       row.get("doi", ""),
        "pub_doi":            row.get("pub_doi", ""),
        "preprint_claims":    pre_result.get("claims", []),
        "publication_claims": pub_result.get("claims", []),
        "n_preprint_claims":  pre_result.get("n_claims", 0),
        "n_pub_claims":       pub_result.get("n_claims", 0),
        "tokens_preprint":    pre_result.get("tokens_used"),
        "tokens_pub":         pub_result.get("tokens_used"),
        "error":              pre_result.get("error") or pub_result.get("error"),
    }

# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    print("=== Step 8: LLM Claim Extraction ===\n")

    # Load API key
    api_key = KEY_FILE.read_text().strip()
    if not api_key:
        raise ValueError(f"OpenAI API key not found in {KEY_FILE}")

    # Load corpus
    print("Loading analysis corpus...")
    df = pd.read_csv(CORPUS_FILE, dtype=str, low_memory=False)
    df["pair_id"] = df.apply(
        lambda r: r.get("arxiv_id") or r.get("doi") or r.get("pub_doi") or str(r.name),
        axis=1
    )
    print(f"  Total pairs: {len(df):,}")

    # Skip already-processed pairs
    done_ids = load_done_ids()
    if done_ids:
        print(f"  Already processed: {len(done_ids):,} — resuming...")
    df_todo = df[~df["pair_id"].isin(done_ids)].copy()

    # Filter to pairs with both abstracts
    df_todo = df_todo[
        (df_todo["preprint_abstract"].fillna("").str.strip().str.len() > 20) &
        (df_todo["pub_abstract"].fillna("").str.strip().str.len() > 20)
    ].copy()
    print(f"  To process: {len(df_todo):,}\n")

    if df_todo.empty:
        print("All pairs already processed.")
        return

    client = AsyncOpenAI(api_key=api_key)
    rows   = df_todo.to_dict("records")

    # Split into batches
    batches = [rows[i:i+BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    n_done_total  = 0
    n_error_total = 0
    start_time    = time.time()

    with open(OUT_CLAIMS, "a", encoding="utf-8") as out_f, \
         open(OUT_ERRORS, "a", encoding="utf-8") as err_f:

        pbar = tqdm(total=len(rows), desc="  Extracting", unit="pair")

        for batch in batches:
            # Process all pairs in this batch concurrently
            results = await asyncio.gather(*[process_pair(client, row) for row in batch])

            for record in results:
                if record.get("error"):
                    err_f.write(json.dumps(record) + "\n")
                    err_f.flush()
                    n_error_total += 1
                else:
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    n_done_total += 1

            pbar.update(len(batch))

        pbar.close()

    elapsed = time.time() - start_time
    print(f"\n── Summary ──")
    print(f"  Pairs processed successfully : {n_done_total:,}")
    print(f"  Pairs with errors            : {n_error_total:,}")
    print(f"  Total time                   : {elapsed/60:.1f} minutes")
    print(f"\n  Output → {OUT_CLAIMS}")
    print("\nStep 8 complete.")


if __name__ == "__main__":
    asyncio.run(main())

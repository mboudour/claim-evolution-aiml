#!/usr/bin/env python3
"""
run_batch.py  — Annotate a specific slice of pairs and write to a batch output file.

Usage:
  python3 run_batch.py --batch 1 --total-batches 10
  python3 run_batch.py --batch 2 --total-batches 10
  ...

Each batch writes to: /home/ubuntu/upload/batch_N_of_10.jsonl
Resume-safe: already-processed pair_ids are skipped.
"""

import argparse
import json
import logging
import math
import os
import time
import concurrent.futures
from pathlib import Path
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
IN_JSONL     = "/home/ubuntu/upload/claims_extracted.jsonl"
OUT_DIR      = "/home/ubuntu/upload"
MODEL        = "gpt-5-mini"
BATCH_SIZE   = 5     # pairs per LLM call (smaller = more reliable)
MAX_WORKERS  = 2     # minimal concurrency to avoid proxy throttling
MAX_RETRIES  = 5
RETRY_DELAY  = 5
INTER_BATCH_SLEEP = 1.0  # seconds between submitting batches
LOG_EVERY    = 100

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── JSON Schema ───────────────────────────────────────────────────────────────
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "batch_claim_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_index": {"type": "integer"},
                            "alignments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "preprint_claim_index": {"type": "integer"},
                                        "pub_claim_index":      {"type": "integer"},
                                        "preprint_claim_text":  {"type": "string"},
                                        "pub_claim_text":       {"type": "string"},
                                        "semantic":   {"type": "string", "enum": ["Unchanged","Clarified","Revised","Removed","Added"]},
                                        "scope":      {"type": "string", "enum": ["Unchanged","Narrowed","Broadened","N/A"]},
                                        "confidence": {"type": "string", "enum": ["Unchanged","Tempered","Amplified","N/A"]},
                                        "matching_confidence": {"type": "number"},
                                        "rationale":  {"type": "string"}
                                    },
                                    "required": ["preprint_claim_index","pub_claim_index",
                                                 "preprint_claim_text","pub_claim_text",
                                                 "semantic","scope","confidence",
                                                 "matching_confidence","rationale"],
                                    "additionalProperties": False
                                }
                            },
                            "added_pub_claim_indices": {"type": "array", "items": {"type": "integer"}}
                        },
                        "required": ["pair_index","alignments","added_pub_claim_indices"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }
    }
}

SYSTEM_PROMPT = """You are a scientific claim analyst. For each preprint-publication pair:
1. Match each preprint claim (P_i) to the most similar publication claim (Q_j).
2. Classify the change on three axes:
   - Semantic: Unchanged | Clarified | Revised | Removed | Added
   - Scope: Unchanged | Narrowed | Broadened | N/A
   - Confidence: Unchanged | Tempered | Amplified | N/A
3. For Removed claims: set pub_claim_index = -1, pub_claim_text = "".
4. For Added pub claims (no preprint match): set preprint_claim_index = -1, semantic = Added.
5. matching_confidence: 0.0-1.0 (alignment certainty).
6. rationale: 1-2 sentences explaining the classification.
Return one result object per pair, with pair_index matching the input."""


def build_prompt(batch):
    parts = []
    for local_idx, (_, record) in enumerate(batch):
        pre = [c for c in (record.get("preprint_claims") or []) if isinstance(c, dict)]
        pub = [c for c in (record.get("publication_claims") or []) if isinstance(c, dict)]
        pre_str = "\n".join(
            f"  P{i}: {c['claim']} (certainty={c.get('certainty','?')}, type={c.get('type','?')})"
            for i, c in enumerate(pre)
        ) or "  (none)"
        pub_str = "\n".join(
            f"  Q{i}: {c['claim']} (certainty={c.get('certainty','?')}, type={c.get('type','?')})"
            for i, c in enumerate(pub)
        ) or "  (none)"
        parts.append(f"PAIR {local_idx} (pair_id={record.get('pair_id','?')}):\nPREPRINT CLAIMS:\n{pre_str}\nPUBLICATION CLAIMS:\n{pub_str}")
    return "\n\n".join(parts)


def annotate_batch(batch, client):
    """Annotate a batch of (global_idx, record) tuples. Returns list of result dicts."""
    # Filter out empty pairs
    llm_batch = []
    results = []
    for gi, record in batch:
        pre = [c for c in (record.get("preprint_claims") or []) if isinstance(c, dict)]
        pub = [c for c in (record.get("publication_claims") or []) if isinstance(c, dict)]
        if not pre and not pub:
            results.append({
                "pair_id": record.get("pair_id", "unknown"),
                "error": "empty_claims",
                "alignments": [],
                "n_preprint_claims": 0,
                "n_pub_claims": 0,
            })
        else:
            llm_batch.append((gi, record))

    if not llm_batch:
        return results

    prompt = build_prompt(llm_batch)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                response_format=RESPONSE_FORMAT,
                max_completion_tokens=8192,
                temperature=0,
                timeout=120,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError(f"API returned null content (finish_reason={resp.choices[0].finish_reason})")

            batch_result = json.loads(content)
            raw_results  = batch_result.get("results", [])
            pair_results = {}
            for r in raw_results:
                if isinstance(r, dict) and "pair_index" in r:
                    pair_results[r["pair_index"]] = r
            break

        except Exception as e:
            log.warning(f"Batch attempt {attempt}/{MAX_RETRIES} failed: {e}. Retry in {RETRY_DELAY * attempt}s...")
            if attempt == MAX_RETRIES:
                log.error(f"Batch failed after {MAX_RETRIES} attempts. Marking as errors.")
                pair_results = {}
            else:
                time.sleep(RETRY_DELAY * attempt)

    for local_idx, (gi, record) in enumerate(llm_batch):
        pair_id    = record.get("pair_id", "unknown")
        pre_claims = [c for c in (record.get("preprint_claims") or []) if isinstance(c, dict)]
        pub_claims = [c for c in (record.get("publication_claims") or []) if isinstance(c, dict)]
        pr = pair_results.get(local_idx)

        if pr and isinstance(pr, dict):
            alignments = pr.get("alignments", [])
            # Enrich with claim texts if missing
            for aln in alignments:
                if isinstance(aln, dict):
                    pi = aln.get("preprint_claim_index", -1)
                    qi = aln.get("pub_claim_index", -1)
                    if not aln.get("preprint_claim_text") and pi >= 0 and pi < len(pre_claims):
                        aln["preprint_claim_text"] = pre_claims[pi].get("claim", "")
                    if not aln.get("pub_claim_text") and qi >= 0 and qi < len(pub_claims):
                        aln["pub_claim_text"] = pub_claims[qi].get("claim", "")
            results.append({
                "pair_id":          pair_id,
                "n_preprint_claims": len(pre_claims),
                "n_pub_claims":     len(pub_claims),
                "alignments":       alignments,
                "added_pub_claim_indices": pr.get("added_pub_claim_indices", []),
            })
        else:
            results.append({
                "pair_id":          pair_id,
                "error":            "llm_failed",
                "n_preprint_claims": len(pre_claims),
                "n_pub_claims":     len(pub_claims),
                "alignments":       [],
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, help="Batch number (1-based)")
    parser.add_argument("--total-batches", type=int, default=10)
    args = parser.parse_args()

    batch_num    = args.batch
    total_batches = args.total_batches
    out_file     = os.path.join(OUT_DIR, f"batch_{batch_num:02d}_of_{total_batches:02d}.jsonl")

    log.info(f"=== Batch {batch_num}/{total_batches} ===")
    log.info(f"Output: {out_file}")

    # Load all records
    records = []
    with open(IN_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    total = len(records)
    log.info(f"Total records: {total:,}")

    # Compute slice for this batch
    batch_size_records = math.ceil(total / total_batches)
    start = (batch_num - 1) * batch_size_records
    end   = min(start + batch_size_records, total)
    slice_records = records[start:end]
    log.info(f"This batch: records {start+1}–{end} ({len(slice_records):,} pairs)")

    # Resume: load already-done pair_ids
    done_ids = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["pair_id"])
                    except Exception:
                        pass
        log.info(f"Resuming: {len(done_ids):,} pairs already done")

    todo = [(i, r) for i, r in enumerate(slice_records) if r.get("pair_id") not in done_ids]
    log.info(f"To process: {len(todo):,} pairs")

    if not todo:
        log.info("Nothing to do — batch already complete.")
        return

    # Build LLM batches
    llm_batches = []
    for i in range(0, len(todo), BATCH_SIZE):
        llm_batches.append(todo[i:i+BATCH_SIZE])
    log.info(f"LLM batches: {len(llm_batches):,}")

    client = OpenAI()

    processed = 0
    errors = 0
    t0 = time.time()

    with open(out_file, "a") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(annotate_batch, b, client): b
                for b in llm_batches
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    for result in batch_results:
                        fout.write(json.dumps(result) + "\n")
                        fout.flush()
                        if result.get("error"):
                            errors += 1
                        processed += 1
                except Exception as e:
                    log.error(f"Batch exception: {e}")
                    errors += 1

                if processed % LOG_EVERY == 0 or processed == len(todo):
                    elapsed = time.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = len(todo) - processed
                    eta = remaining / rate / 60 if rate > 0 else 999
                    log.info(f"Progress: {processed:,}/{len(todo):,} ({100*processed/len(todo):.1f}%) | errors={errors} | rate={rate:.1f}/s | ETA={eta:.1f}min")

    elapsed = time.time() - t0
    log.info(f"=== Batch {batch_num} COMPLETE: {processed:,} pairs in {elapsed/60:.1f}min | errors={errors} ===")
    log.info(f"Output file: {out_file}")


if __name__ == "__main__":
    main()

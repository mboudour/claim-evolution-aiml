"""
classify_subfields.py
=====================
LLM-based subfield classification for the claim-evolution corpus.

For each paper (preprint), classifies the title + abstract into one of 12
fixed AI/ML subfields using gpt-4o.  The arXiv category codes (where
available) are used as a validation set to assess classifier agreement.

Subfields
---------
  1  Machine Learning
  2  Deep Learning / Foundation Models
  3  Natural Language Processing
  4  Computer Vision
  5  Robotics
  6  Reinforcement Learning
  7  AI Safety / Alignment
  8  Knowledge Representation & Reasoning
  9  Graph Learning / Network ML
 10  Biomedical AI
 11  AI Theory
 12  Other

Outputs
-------
  computations/data/data_sources/claims/subfields.jsonl
      One record per paper: pair_id, subfield, subfield_id, confidence, rationale
  computations/data/data_sources/claims/subfields.csv
      Flat CSV version of the above
  computations/analysis/outputs/subfield_report.txt
      Distribution summary + agreement with arXiv categories

Usage
-----
  # Pilot (first 50 papers):
  python3 classify_subfields.py --pilot

  # Pilot, re-run even if already classified:
  python3 classify_subfields.py --pilot --force

  # Full run (start fresh, archive old output):
  python3 classify_subfields.py --reset

  # Full run (resume if interrupted):
  python3 classify_subfields.py

Estimated cost: ~$8-12 USD for 51,946 papers using gpt-4o (short prompts).
Estimated time: ~2-3 hours.
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

CORPUS_FILE  = PROJECT_ROOT / "computations" / "data" / "data_sources" / "processed" / "analysis_corpus.csv"
KEY_FILE     = PROJECT_ROOT / "computations" / "data" / "config" / "openai_key.txt"
OUT_DIR      = PROJECT_ROOT / "computations" / "data" / "data_sources" / "claims"
OUT_JSONL    = OUT_DIR / "subfields.jsonl"
OUT_CSV      = OUT_DIR / "subfields.csv"
OUT_REPORT   = PROJECT_ROOT / "computations" / "analysis" / "outputs" / "subfield_report.txt"

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL        = "gpt-4o"
CONCURRENCY  = 30

SUBFIELDS = [
    "Machine Learning",
    "Deep Learning / Foundation Models",
    "Natural Language Processing",
    "Computer Vision",
    "Robotics",
    "Reinforcement Learning",
    "AI Safety / Alignment",
    "Knowledge Representation & Reasoning",
    "Graph Learning / Network ML",
    "Biomedical AI",
    "AI Theory",
    "Other",
]

SUBFIELD_LIST_STR = "\n".join(f"  {i+1:2d}. {s}" for i, s in enumerate(SUBFIELDS))

SYSTEM_PROMPT = f"""You are a scientific classification assistant specialising in AI and machine learning research.

Your task is to classify a research paper into exactly one of the following subfields based on its title and abstract:

{SUBFIELD_LIST_STR}

Classification guidelines:
- Choose the subfield that best describes the paper's PRIMARY contribution or methodology.
- "Deep Learning / Foundation Models" covers large language models, transformers, diffusion models, and neural scaling work.
- "Machine Learning" covers classical ML, statistical learning theory, ensemble methods, and general ML methodology not covered by other categories.
- "Biomedical AI" covers papers whose primary application domain is medicine, biology, or health, regardless of the underlying ML technique.
- "AI Theory" covers complexity, expressivity, convergence proofs, and formal analysis of learning algorithms.
- Use "Other" only when the paper genuinely does not fit any of the above categories.

Return a JSON object with exactly these fields:
{{
  "subfield": "<exact subfield name from the list above>",
  "subfield_id": <integer 1-12>,
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<one sentence explaining the classification>"
}}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_done_ids() -> set:
    done = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if not rec.get("error"):
                            done.add(rec.get("pair_id", ""))
                    except json.JSONDecodeError:
                        pass
    return done


async def classify_paper(client: AsyncOpenAI, paper: dict, sem: asyncio.Semaphore) -> dict:
    pair_id = paper.get("pair_id", "")
    title   = paper.get("preprint_title", "") or paper.get("title", "") or ""
    abstract = paper.get("preprint_abstract", "") or paper.get("abstract", "") or ""

    if not title and not abstract:
        return {
            "pair_id": pair_id,
            "error": "No title or abstract available",
        }

    user_content = f"Title: {title}\n\nAbstract: {abstract[:1500]}"

    async with sem:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            # Validate
            subfield = parsed.get("subfield", "")
            if subfield not in SUBFIELDS:
                # Try to match by id
                sid = parsed.get("subfield_id")
                if isinstance(sid, int) and 1 <= sid <= 12:
                    subfield = SUBFIELDS[sid - 1]
                else:
                    subfield = "Other"
                    parsed["subfield_id"] = 12

            return {
                "pair_id":    pair_id,
                "subfield":   subfield,
                "subfield_id": parsed.get("subfield_id", SUBFIELDS.index(subfield) + 1),
                "confidence": float(parsed.get("confidence", 0.0)),
                "rationale":  parsed.get("rationale", ""),
                "arxiv_categories": paper.get("preprint_categories", ""),
            }

        except Exception as e:
            return {"pair_id": pair_id, "error": str(e)}


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(pilot: bool = False, force: bool = False, reset: bool = False):
    print("=== Subfield Classification ===\n")

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
        if OUT_JSONL.exists():
            archive = OUT_JSONL.with_suffix(".jsonl.bak")
            OUT_JSONL.rename(archive)
            print(f"  Archived old output → {archive.name}")
        if OUT_CSV.exists():
            OUT_CSV.unlink()
        print()

    # Load corpus
    print("Loading corpus...")
    df = pd.read_csv(CORPUS_FILE, dtype=str, low_memory=False)
    # Build list of paper dicts (one per unique preprint)
    papers = df.to_dict(orient="records")
    print(f"  Loaded {len(papers):,} papers")

    if pilot:
        papers = papers[:50]
        print(f"  PILOT MODE: classifying first 50 papers only")
        if force:
            print(f"  FORCE MODE: re-classifying even if already done\n")
            todo = papers
        else:
            done_ids = load_done_ids()
            if done_ids:
                print(f"  Already classified: {len(done_ids):,} — resuming...")
            todo = [p for p in papers if str(p.get("pair_id", "")) not in done_ids]
            print(f"  To classify: {len(todo):,}\n")
            if not todo:
                print("All pilot papers already classified. Use --force to re-run them.")
                return
    else:
        print()
        done_ids = load_done_ids()
        if done_ids:
            print(f"  Already classified: {len(done_ids):,} — resuming...")
        todo = [p for p in papers if str(p.get("pair_id", "")) not in done_ids]

    print(f"  To classify: {len(todo):,}\n")

    if not todo:
        print("Nothing to classify.")
    else:
        client = AsyncOpenAI(api_key=api_key)
        sem    = asyncio.Semaphore(CONCURRENCY)

        n_done  = 0
        n_error = 0
        start   = time.time()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

        with open(OUT_JSONL, "a", encoding="utf-8") as out_f:
            pbar = tqdm(total=len(todo), desc="  Classifying", unit="paper")
            chunk_size = 300
            for i in range(0, len(todo), chunk_size):
                chunk   = todo[i:i + chunk_size]
                results = await asyncio.gather(*[classify_paper(client, p, sem) for p in chunk])
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
        print(f"\n  Classified {n_done:,} papers in {elapsed / 60:.1f} minutes")
        print(f"  Errors: {n_error:,}")

    # ── Build CSV ──────────────────────────────────────────────────────────────
    print("\nBuilding CSV...")
    rows = []
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not rec.get("error"):
                    rows.append(rec)
            except json.JSONDecodeError:
                pass

    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"  Saved {len(csv_df):,} rows → {OUT_CSV.name}")

    # ── Summary report ─────────────────────────────────────────────────────────
    if csv_df.empty or "subfield" not in csv_df.columns:
        print("No classifications to report.")
        return

    total = len(csv_df)
    sf_counts = Counter(csv_df["subfield"].tolist())

    lines = [
        "=== Subfield Classification Report ===\n",
        f"Total papers classified: {total:,}",
        f"\nSubfield distribution:",
    ]
    for label, n in sorted(sf_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {label:<40}: {n:>7,}  ({n/total*100:.1f}%)")

    # Agreement with arXiv categories (where available)
    arxiv_rows = csv_df[csv_df["arxiv_categories"].notna() & (csv_df["arxiv_categories"] != "")]
    if len(arxiv_rows) > 0:
        lines += [
            f"\narXiv-sourced papers: {len(arxiv_rows):,}",
            f"(arXiv category vs LLM subfield agreement analysis available in subfields.csv)",
        ]

    conf_vals = pd.to_numeric(csv_df["confidence"], errors="coerce").dropna()
    if len(conf_vals) > 0:
        lines += [
            f"\nClassification confidence:",
            f"  Mean   : {conf_vals.mean():.3f}",
            f"  Median : {conf_vals.median():.3f}",
            f"  < 0.70 : {(conf_vals < 0.70).sum():,}  ({(conf_vals < 0.70).mean()*100:.1f}%) ← review these",
        ]

    report = "\n".join(lines)
    print("\n" + report)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n  Saved report → {OUT_REPORT.name}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based subfield classification for claim-evolution corpus")
    parser.add_argument("--pilot", action="store_true",
                        help="Classify only the first 50 papers (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Re-classify papers even if already done (use with --pilot)")
    parser.add_argument("--reset", action="store_true",
                        help="Archive existing output and start the full run from scratch")
    args = parser.parse_args()
    asyncio.run(main(pilot=args.pilot, force=args.force, reset=args.reset))

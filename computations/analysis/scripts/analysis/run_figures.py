"""Runner: override paths and generate figures from upload/ directory."""
import sys
sys.path.insert(0, '/home/ubuntu/code/analysis')

import make_figures as mf
from pathlib import Path

mf.CORPUS_FILE = Path("/home/ubuntu/upload/analysis_corpus.csv")
mf.COMP_FILE   = Path("/home/ubuntu/upload/claim_changes.jsonl")
mf.EXTR_FILE   = Path("/home/ubuntu/upload/claims_extracted.jsonl")
mf.OUT_DIR     = Path("/home/ubuntu/upload/figures/panels")
mf.OUT_DIR.mkdir(parents=True, exist_ok=True)

mf.main()

# The Evolution of Scientific Claims Across Publication Versions in AI and Machine Learning, 2015–2024

**Author:** Moses Boudourides, School of Professional Studies, Northwestern University

## Abstract
*To be written after results are complete. Will summarize the versioning framing, the 52,000-pair dataset, the LLM-assisted multi-dimensional annotation scheme, and the key findings regarding how claims change (Semantic, Scope, Confidence) between preprint and published versions.*

## 1. Introduction
* **The Versioned Scholarly Record:** Scholarly articles increasingly circulate as linked versions rather than as a single stable object (preprints, submitted manuscripts, accepted manuscripts, versions of record, corrected versions, etc.).
* **The Gap:** Versioning infrastructures (like Crossref and the NISO JAV framework) can identify that these objects are connected, but they do not specify how the scientific claims themselves change across versions.
* **The Context:** AI and Machine Learning (2015–2024) provide a massive, fast-moving ecosystem where preprints (arXiv) are heavily utilized alongside formal conference and journal publications.
* **The Contribution:** This study addresses the gap by tracking claim-level semantic, scope, and confidence evolution from preprint to published version in AI/ML.

## 2. Related Work
* **Publication Versioning:** Establish the theoretical framework using the NISO/ALPSP Journal Article Versions (JAV) recommended practice [1] and Crossref's version control guidance [2]. This shifts the focus from "does peer review change claims?" to "how do claims evolve across linked publication versions?"
* **From Textual Similarity to Epistemic Change:** Review prior preprint-to-publication comparisons. Note that while studies like Klein et al. [3] and Nicholson et al. [4] found high textual and linguistic similarity between preprints and published versions, textual similarity can mask significant epistemic shifts (e.g., a small text change that drastically narrows a claim's scope).
* **Claim Calibration and Tempering:** Discuss the literature on peer review as a tempering mechanism. Cite Keserlioglu et al. [5] on limitation-acknowledgment and Yin & Rust [6] on hedging shifts in biomedicine. Position our study as testing whether this "tempering" dynamic holds in the unique, conference-driven AI/ML publication ecology.
* **Scientific Exaggeration in AI:** Connect to broader concerns about hype and overstatement in AI research, citing recent work like RIGOURATE [7], which quantifies scientific exaggeration in AI conference papers. Our *Scope* and *Confidence* axes directly measure whether the publication process reins in this exaggeration.

## 3. Data and Methods
* **Corpus Construction:** Describe the dataset of ~52,000 preprint-publication pairs linked via arXiv IDs and DOIs.
* **Version Proxies:** Acknowledge that the linked publication is typically the publisher's Version of Record (VoR) or a Green OA copy (which may be the accepted manuscript).
* **Multi-dimensional Annotation Scheme:** Explain the LLM-assisted extraction and the three axes of evolution:
  * *Semantic:* Unchanged / Clarified / Revised / Removed / Added
  * *Scope:* Unchanged / Narrowed / Broadened
  * *Confidence:* Unchanged / Tempered / Amplified
* **Human Validation:** Report the results of the human rater validation (Cohen's kappa) to establish the reliability of the LLM annotations.
* **Covariates:** Subfield classification, venue prestige (CORE/SJR), and publication version proxies (OA status).

## 4. Results
* **RQ1: How often do claims evolve across publication versions?**
  * Overall stability vs. revision rates.
* **RQ2: In what ways do claims evolve?**
  * Breakdown of scope narrowing/broadening and confidence tempering/amplification within revised claims.
* **RQ3: Where do claims evolve?**
  * Differences across venue types (journal vs. conference), prestige tiers, and AI/ML subfields.
* **RQ4: What predicts claim evolution?**
  * Results of the multinomial and ordinal logistic regression models.
* **RQ5: What are the consequences of claim evolution?**
  * Results of the citation impact model: do heavily revised versions accrue more citations?

## 5. Discussion
* **Interpreting the Changes:** Avoid attributing all changes solely to peer review. Discuss how the observed evolution reflects a combination of author self-revision, reviewer feedback, editorial intervention, and venue-specific formatting constraints.
* **The Value of Versioning:** What the findings mean for the scholarly communication ecosystem and the consumption of preprints vs. versions of record.

## 6. Limitations
* **Version Imprecision:** The linkage relies on DOIs, which primarily point to the VoR, but Green OA deposits mean the exact version (Accepted Manuscript vs. VoR) is sometimes ambiguous.
* **LLM Annotation:** Acknowledge the inherent limitations of using LLMs for semantic alignment, despite human validation.

## 7. Conclusion
* Summary of the threefold contribution: empirical (large-scale AI/ML analysis), methodological (multi-dimensional annotation), and scholarly-communication (claim evolution across linked versions).

## References
[1] NISO/ALPSP. (2008). *Journal Article Versions (JAV): Recommendations of the NISO/ALPSP JAV Technical Working Group*.
[2] Crossref. *Version control, corrections, and retractions*.
[3] Klein, M., Broadwell, P., Farb, S. E., & Grappone, T. (2019). Comparing published scientific journal articles to their pre-print versions. *Journal of Informetrics*.
[4] Nicholson, D. N., et al. (2022). Examining linguistic shifts between preprints and publications. *PLOS Biology*.
[5] Keserlioglu, K., Kilicoglu, H., & ter Riet, G. (2019). Impact of peer review on discussion of study limitations and strength of claims in randomized trial reports: a before and after study. *Research Integrity and Peer Review*.
[6] Yin & Rust. (2024). Tracking claim changes from preprint to publication across 72,644 biomedical studies using large language models.
[7] RIGOURATE: Quantifying Scientific Exaggeration with Evidence-Aligned Claim Evaluation. (2025/2026). *arXiv*.

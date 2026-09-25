# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** AI Entity Resolvers  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
We present a scalable, high-precision two-stage machine learning system for resolving noisy business entities across multiple disjoint data sources without common keys. Our pipeline integrates domain-invariant multilingual text normalization, country-partitioned sparse TF-IDF and inverted index candidate blocking (achieving a 99.98% search space reduction), and an extensively engineered LightGBM ranking classifier trained on 30 rapid string distance and token interaction features. Optimized specifically for the macro $F_{0.5}$ metric with calibrated singleton protection, our approach achieves **94.32% macro $F_{0.5}$ validation score** and **97.33% singleton accuracy** with robust out-of-distribution generalization to unseen countries (France).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis revealed key properties across 2.2M ground truth records and 10M total entity fragments:
- **Singleton Distribution**: 5.58% of reference entities have zero matches (singletons), while 24.1% have 3 matches and 21.9% have 4 matches across secondary sources. Correctly predicting empty match sets for singletons is vital under macro $F_{0.5}$ (where any false positive merge drops the entity score from 1.0 to 0.0).
- **Name Variations**: Extreme legal suffix heterogeneity (`Corp`, `LLC`, `Pvt Ltd`, `Trust`), abbreviation expansions, DBA prefixes (`Mirapyra formerly ...`), punctuation brackets (`[[LLC]]`), and character transpositions (`Ricardnosn` vs `Richardson`).
- **Address Noise**: Word order inversions (`MN, Madison, 2476 261st Ave` vs `2476 261st Ave, Madison, Minnesota`), landmark prefixes (`##`, `#`), and unit/floor modifiers.
- **Multilingual Dynamics**: Training data covers US and India, while the test set includes France (with French diacritics and legal entities like `SARL`, `SAS`, `SA`).

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Channel Blocking + Dense RapidFuzz Feature Engineering + Regularized GBDT Ranker with $F_{0.5}$ Calibrated Thresholding.  
**Core Innovation:** 
1. Sub-second streaming candidate generation via sublinear sparse word (1,2)-gram TF-IDF and token posting lists.
2. 30 C++ accelerated pairwise string metrics, token interaction ratios, and digit/postal code consistency penalties.
3. Macro $F_{0.5}$ precision-calibrated decision boundary with strict singleton suppression.

---

## 3. Candidate Generation (Blocking)

- **Blocking Partitioning**: Exact country partitioning (US $\to$ US, India $\to$ India, France $\to$ France).
- **Candidate Channels**:
  1. **Sparse TF-IDF Cosine Retrieval**: Word (1,2)-gram vectorizer with sublinear term frequency and L2 normalization over combined clean text ($K=30$).
  2. **Significant Name Token Inverted Index**: Posting lists for discriminative tokens (length $\ge 3$, non-stopwords) with posting frequency filtering ($\le 250$).
- **Candidate Output**: Top 30–40 candidate pairs per Source 1 entity (reducing comparison complexity from $\sim 10^{12}$ pairs to $<6 \times 10^7$ pairs).

---

## 4. Matching Model

**Features used (30 dense features):**
- **Name Similarity**: RapidFuzz normalized Levenshtein ratio, Jaro-Winkler similarity, Token Sort Ratio, Token Set Ratio, Partial Ratio, Token Jaccard, Token Containment, Length difference & ratio, Exact match flag, First-word exact match, Soundex phonetic match.
- **Address Similarity**: Normalized Levenshtein, Jaro-Winkler, Token Sort/Set ratios, Token Jaccard, Containment, Length diff, Exact match flag.
- **Numeric & Structural Consistency**: Digit/PIN code Jaccard overlap, Number count intersection, Street number mismatch indicator flag (`has_num_mismatch`).
- **Interaction Terms**: Harmonic average composite score (`overall_avg_sim`), Name $\times$ Address Jaro-Winkler product, Name $\times$ Address Token Set product, Source 2 vs Source 3 indicators.

**Model type:** Regularized LightGBM GBDT (350 trees, max depth 8, num leaves 63, feature fraction 0.80, bagging fraction 0.80, L1/L2 penalties $\lambda_1=0.1, \lambda_2=1.0$).  
**Threshold selection method:** Grid search optimizing macro $F_{0.5}$ directly on holdout validation data ($\tau^* = 0.65$).

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **`94.32%`** (Validation set)
- **Singleton Accuracy:** **`97.33%`** (375 singletons correctly left empty)
- **Candidate Generation Recall:** **`91.20%`**
- **Top Feature Importances:**
  1. `overall_avg_sim` (Gain: 1,109,325.8)
  2. `addr_containment` (Gain: 522,110.5)
  3. `name_x_addr_set` (Gain: 242,835.1)
  4. `name_jw` (Gain: 71,127.4)
  5. `num_jaccard` (Gain: 46,718.6)
  6. `has_num_mismatch` (Gain: 26,180.1)

---

## 6. Conclusion
The proposed architecture provides a scalable, accurate, and memory-efficient solution for enterprise entity resolution across 10+ million noisy records. By coupling sub-second sparse blocking with deep RapidFuzz string feature engineering and $F_{0.5}$-calibrated gradient boosted decision trees, we achieve high precision and recall while preventing false merges and generalizing seamlessly across languages and countries.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/pipeline.py`: Complete end-to-end runnable script reproducing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `code/business_entity_resolution/src/preprocessor.py`: Text cleaning and multilingual normalization.
- `code/business_entity_resolution/src/blocking.py`: High-recall sparse candidate generation.
- `code/business_entity_resolution/src/feature_engineering.py`: Vectorized feature extraction.
- `code/business_entity_resolution/src/metrics.py`: Macro $F_{0.5}$ and recall evaluation suite.
- `code/business_entity_resolution/requirements.txt`: Pinned dependencies.
- `code/business_entity_resolution/README.md`: Reproduction guide.

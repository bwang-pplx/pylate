# Why Does Late Interaction Generalize? Decomposing OOD Robustness in Multi-Vector Retrieval

## Research Plan — v4 (lean)

---

## 1. Research Question

**RQ:** What mechanism(s) drive the out-of-domain (OOD) generalization advantage of ColBERT-style late interaction models?

We decompose this into four testable hypotheses:

| Hypothesis | Claim |
|------------|-------|
| **H1: Information Bottleneck** | Multi-vector representations preserve token-level information, alleviating the information bottleneck inherent in single-vector compression |
| **H2: Compositional Generalization** | MaxSim's sum-of-max aggregation enables each query token to independently find its best match, yielding robustness to novel query–document token combinations |
| **H3: Implicit Lexical Anchoring** | Token-level alignment implicitly preserves exact-match signals akin to BM25, providing a fallback when domain-specific terminology shifts |
| **H4: Representation Redundancy** | Multi-vector representations carry natural redundancy (multiple tokens covering the same semantics), making the model robust to drift in any single token's representation |

These factors may exhibit super-additive interaction (H1×H2 synergy).

---

## 2. Experimental Setup

### 2.1 Models

```
Backbone: ModernBERT-base (answerdotai/ModernBERT-base, 150M), shared initialization

Baselines:
  BM25                            Lexical OOD floor (via MTEB BM25-S)
  Cross-Encoder (rerank BM25      BM25-recall-conditioned OOD ceiling
    top-1000, same backbone)      (also report per-dataset BM25 R@1000)
                                  Use existing HF checkpoint if available for
                                  ModernBERT-base; otherwise train on MS MARCO

Core spectrum:
  M2: Dense (Mean-pool)           1 vec/doc, dot product [sentence-transformers]
  M4: Multi-K (K=4,8,16)          K vecs/doc, MaxSim, stride mean-pool [PyLate]
  M5: ColBERT-Full                N vecs/doc, MaxSim [PyLate]
  M6: ColBERT-Pruned-50%          N/2 vecs/doc, post-hoc IDF pruning on M5
  M7: ColBERT-Pruned-75%          N/4 vecs/doc, post-hoc IDF pruning on M5
```

### 2.2 Training

```yaml
backbone: answerdotai/ModernBERT-base
dataset: sentence-transformers/msmarco-bm25 (triplet subset)
projection_dim: 128
max_query_len: 32
max_doc_len: 256
batch_size: 128
lr: 3e-6
warmup: 1000 steps
total_steps: 100k
loss: InfoNCE + in-batch negatives + 1 hard negative (BM25 mined)
teacher: None
seeds: 1 (add more if needed for statistical significance)
```

### 2.3 Evaluation

**In-domain:** MS MARCO dev MRR@10; TREC DL 2019+2020 nDCG@10 (average = ID score)

**OOD:** Full BEIR suite nDCG@10, grouped by shift type (domain / query style / document length / lexical overlap)

**BM25:** Evaluated via MTEB BM25-S (no training needed)

**Statistics:** 1 seed initially; add more seeds and significance tests if needed

---

## 3. Experiments

### 3.1 The Spectrum (Paper §2)

Evaluate the full core spectrum: BM25, M2, M4 (K=4,8,16), M5, M6, M7, CE.

**Output:** A single table showing how OOD retention (ood_ratio = OOD_avg / ID_score) evolves across the 1-vector to N-vector continuum. This establishes the factual basis that multi-vector models are indeed more robust OOD.

### 3.2 Group A — Decomposing H1 vs H2 (Paper §3.1)

| ID | Train Aggregation | Inference Aggregation | Notes |
|----|-------------------|----------------------|-------|
| A1 | MaxSim | MaxSim | = M5, ColBERT baseline |
| A2-zs | MaxSim | MeanSim | Swap aggregation at inference on the M5 checkpoint; one line of code |
| A2-nat | MeanSim | MeanSim | Natively trained with MeanSim |
| A4 | — | dot product | = M2, Dense mean-pool baseline |

**How to read the results:**

```
A2-nat vs A4  → H1 contribution (multi-vector capacity, with MaxSim removed)
A1 vs A2-nat  → H2 contribution (MaxSim vs MeanSim, both natively trained)
A1 vs A4      → Total OOD gap

Synergy test:
  Δ_total = OOD(A1) - OOD(A4)
  Δ_H1    = OOD(A2-nat) - OOD(A4)
  Δ_H2    = OOD(A1) - OOD(A2-nat)
  Interaction = Δ_total - (Δ_H1 + Δ_H2)
  > 0 → super-additive (MaxSim needs many tokens; many tokens need a sharp aggregation)

A2-zs vs A2-nat → How much the training aggregation shapes the representation space
  ≈  → representations are general-purpose
  << → MaxSim specializes the representation space
```

**A2-nat training safeguards:**
- Report A2-nat's in-domain MRR@10 and training loss curves
- If A2-nat ID score < 85% of A1 ID score → add caveat to OOD comparison
- If A2-nat ID score < 70% of A1 ID score → A2-nat is unusable for H1/H2 separation; rely on A2-zs only
- A2-nat may use independently tuned lr/batch size (goal: best possible performance under MeanSim, not hyperparameter uniformity)
- Fallback: even if A2-nat fails to train well, A2-zs >> A4 still demonstrates that H1 contributes

**Supplementary: effect of dimensionality on OOD**

| ID | Token Count | Dimension | Storage | Protocol |
|----|-------------|-----------|---------|----------|
| D2 | N | 64 | 0.5x | Retrain projection head only (backbone frozen), lr=1e-4 |

D2 shows how reducing per-token dimensionality (while keeping all tokens) affects OOD robustness. If D2 retains most of the OOD advantage over A4, then per-token representation quality is less important than having multiple tokens.

### 3.3 Group B — Testing H3 (Paper §3.2)

**Pre-experiment:** On MS MARCO dev, compute the distribution of max_j cos(q_i, d_j) across query tokens. Select threshold θ such that ~15–30% of query tokens have at least one near-exact match above θ.

| ID | Method |
|----|--------|
| B1 | ColBERT-Full (= A1) |
| B2 | MaxSim with all matches where cos(q_i, d_j) > θ masked to −∞ |
| B4 | Dense + BM25 linear combination (α tuned on dev set) |

If B2 ≈ B1 → H3 is not a primary factor. If B4 approaches B1 → the Dense model's OOD gap can be partially closed by supplementing lexical signals.

### 3.4 Group C — H4 Sanity Check (Paper §3.3)

| ID | Method |
|----|--------|
| C2 | ColBERT with 30% of document token vectors randomly dropped at inference |

If C2 ≈ M5 → multi-vector representations have intrinsic redundancy, supporting H4. If C2 drops substantially → redundancy is limited.

Single experiment, inference-only, near-zero cost.

### 3.5 Hard Negative Confound Check

| Config | Hard Negatives per Query |
|--------|--------------------------|
| M2 (Dense) × {1, 7} |
| M5 (ColBERT) × {1, 7} |

If the OOD gap remains stable → the architectural effect is robust. If the gap closes with more hard negatives → conclusions must be qualified as conditional on training signal richness.

---

## 4. Deliverables

### Hypothesis Verification Matrix

| Result Pattern | Interpretation |
|---------------|----------------|
| A2-nat >> A4 | H1 dominant: multi-vector capacity alone provides OOD advantage |
| A1 >> A2-nat ≈ A4 | H2 dominant: OOD advantage comes from MaxSim's compositional generalization |
| Interaction > 0 | H1×H2 synergy: the factors are super-additive |
| B2 ≈ B1 | H3 ruled out as a primary factor |
| C2 ≈ M5 | H4 supported: intrinsic redundancy provides robustness |
| D2 retains OOD advantage over A4 | Per-token dim less important than having multiple tokens |
| HN gap stable | Architectural effect is robust to training signal strength |

---

## 5. Timeline

```
Week 1–2: Infrastructure + Spectrum
  Train M2 (sentence-transformers), M4 (K=4,8,16), M5 (PyLate)
  BM25 baseline (MTEB BM25-S)
  CE baseline (check for existing HF checkpoint, otherwise train)
  B2 threshold calibration (pre-experiment)
  M6/M7 post-hoc pruning (inference-only)

Week 3–4: Ablations
  A2-nat (with convergence monitoring)
  A2-zs (inference-only on M5 checkpoint)
  D2 retrain projection head
  HN check: M2 + M5 × {1, 7}

Week 5: Targeted Checks + Analysis
  B2, B4 (inference-only; B4 requires α tuning on dev)
  C2 (inference-only)
  Per-shift-type breakdown
  Synergy quantification

Week 6–7: Writing
```

---

## 6. Paper Structure

```
§1 Introduction: "multi-vector helps OOD" is established; "why" is not
§2 The Spectrum: BM25 → Dense → Multi-K → ColBERT → CE (establish the fact)
§3 Decomposition:
   §3.1 H1 vs H2 (Group A + synergy test + D2 dimensionality evidence)
   §3.2 H3 check (Group B)
   §3.3 H4 check (Group C)
§4 Discussion (practical implications: D2 dimensionality tradeoff,
   HN sensitivity, limitations)
§5 Conclusion
Appendix: A2-nat convergence details, per-dataset results, HN sensitivity
```

---

## 7. Implementation Details

### Training framework

| Model | Framework | Script |
|-------|-----------|--------|
| M5 (ColBERT-Full) | PyLate | `scripts/ood_study/train_m5_colbert.py` |
| A2-nat (MeanSim) | PyLate | `scripts/ood_study/train_a2nat_meansim.py` |
| M5 HN-7 | PyLate | `scripts/ood_study/train_m5_hn7.py` |
| D2 (dim=64) | PyLate | `scripts/ood_study/train_d2_dim64.py` |
| M4 (Multi-K) | PyLate | `scripts/ood_study/train_m4_multik.py` (needs stride-pool impl) |
| M2 (Dense) | sentence-transformers | `scripts/ood_study/train_m2_dense.py` |
| M2 HN-7 | sentence-transformers | `scripts/ood_study/train_m2_dense_hn7.py` |
| CE (Cross-Encoder) | sentence-transformers | `scripts/ood_study/train_ce.py` (or use existing HF checkpoint) |
| BM25 | MTEB BM25-S | No training needed |

### Inference-only configs (no training scripts needed)

| Config | Derived from | Modification |
|--------|-------------|--------------|
| A2-zs | M5 checkpoint | Eval with `aggregation="mean"` |
| M6 (pruned 50%) | M5 checkpoint | Post-hoc token pruning |
| M7 (pruned 75%) | M5 checkpoint | Post-hoc token pruning |
| B2 (mask exact) | M5 checkpoint | Mask cos > θ matches at inference |
| B4 (Dense+BM25) | M2 + BM25 scores | Linear combo, tune α on dev |
| C2 (token dropout) | M5 checkpoint | Random 30% doc token dropout |

### Code changes needed in PyLate

| Feature | Status | Location |
|---------|--------|----------|
| MeanSim aggregation | Done | `pylate/scores/scores.py` — `aggregation` param |
| MeanSim in reranking | Done | `pylate/rank/rank.py` — `aggregation` param |
| MeanSim in evaluation | Done | `pylate/evaluation/colbert_triplet.py` — `aggregation` param |
| M4 stride-pool | TODO | `pylate/models/colbert.py` — pool N tokens to K in forward pass |
| Post-hoc IDF pruning | TODO | New utility — drop low-IDF token vectors from encoded docs |

### Experiment Count

```
Trained models:  M2, M4×3, M5, A2-nat, D2, HN-7×2, CE = 10 configs
Inference-only:  A2-zs, M6, M7, B2, B4, C2 = 6 configs (no training)
BM25:            MTEB BM25-S, ~0 cost

Total training runs: 10
Total eval configs:  ~17
```

Manageable. Each training run ≈ MS MARCO 100k steps on ModernBERT-base ≈ a few hours on a single GPU.

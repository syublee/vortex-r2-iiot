# Vortex-R2 IIoT: AI-Scientist-v2 Pipeline Design

## Goal

Run automated experiments to validate and extend Vortex-R2 — a novel tabular-to-image
transformation for IIoT sensor data — targeting an IEEE Sensors Letters submission.

---

## Background

**Vortex-R2 (Relationship-Regularized)** inverts conventional tabular-to-image wisdom:
instead of placing similar sensors near each other (DeepInsight, REFINED), it maps
**low-correlation (independent) sensors to the image center** and redundant sensors to the
periphery via a force-directed vortex layout.

**Algorithm (3 stages):**
1. Build anti-correlation graph: `w_ij = 1 - |r_ij|` (Pearson)
2. Force-directed placement (Fruchterman-Reingold) with weighted attractive forces
3. Archimedean spiral reordering to enforce explicit center-periphery structure
4. Rasterize sensor values onto k×k grid

**Key claim:** This layout induces well-conditioned Hessian blocks and flat-minima
convergence, improving generalization especially under label scarcity.

---

## Architecture

```
ai_scientist/
├── ideas/
│   ├── vortex_r2_iiot.md        # Topic: goal, eval, background
│   └── vortex_r2_iiot.py        # Runfile: Vortex-R2 + baselines + metrics
├── blank_ieee_lsens_latex/
│   ├── template.tex             # IEEE Sensors Letters LaTeX template
│   └── IEEE_lsens.cls           # IEEE class file (user-provided)
└── perform_ieee_sensors_writeup.py  # Writeup module for IEEE format
```

**Data flow:**
1. BFTS executes `vortex_r2_iiot.py` per search node
2. Results saved as `results.json` (accuracy, F1, ECE per method/label-ratio/dataset)
3. Grad-CAM heatmaps and accuracy curves saved as PNG
4. `perform_ieee_sensors_writeup.py` reads results → fills LaTeX template → compiles PDF

---

## Datasets

All publicly downloadable, no registration required.

| Dataset | Features | Samples | Classes | Domain |
|---------|----------|---------|---------|--------|
| SECOM | 590 | 1,567 | 2 (pass/fail) | Semiconductor mfg (UCI) |
| Epileptic Seizure | 178 | 11,500 | 5 | EEG / biomedical (UCI) |
| CWRU Bearing Fault | 64 | ~10,000 | 4 | Vibration / rotating machinery |
| UCI HAR | 561 | 10,299 | 6 | Smartphone accelerometer + gyroscope |

CWRU: 64 time/frequency-domain statistical features extracted per sample
(mean, std, RMS, kurtosis, crest factor, FFT peak bins × drive-end/fan-end channels).
4-class: normal, inner race fault, ball fault, outer race fault.

SECOM and Epileptic Seizure are already referenced in the draft paper.
CWRU is the canonical IIoT vibration benchmark aligned with IEEE Sensors Letters scope.
UCI HAR provides smartphone sensor data (accelerometer + gyroscope) — directly sensor-focused
and well-suited for IEEE Sensors audience. All four datasets have ≥64 features, ensuring
k×k images large enough for ResNet-18 fine-tuning.

---

## Experiment Design

**Baselines:**
- XGBoost, CatBoost (tree ensembles)
- IGTD, DeepInsight, REFINED (tabular-to-image prior art)
- Raw-CNN (fully connected, no conversion)

**Label regimes:** 1%, 5%, 10%, 100%

**Evaluation metrics:**
- Accuracy (overall classification)
- Macro-F1 (robust to class imbalance)
- ECE — Expected Calibration Error (confidence reliability)

**Repetitions:** 5 random seeds, report mean ± std

**CNN architecture:** ResNet-18 pretrained on ImageNet, fine-tuned.
Images resized to 224×224. Adam lr=1e-4, batch 32, 50 epochs, early stopping patience 10.

**BFTS search space (parameters AI-Scientist explores):**
- Force-directed iterations T: {50, 100, 200, 500}
- Correlation threshold τ for center/periphery boundary: {0.3, 0.5, 0.7}
- Image grid size k: {16, 32, 64}
- CNN backbone: {ResNet-18, EfficientNet-B0}

---

## IEEE Sensors Letters Template Integration

**`blank_ieee_lsens_latex/template.tex`:**
- `\documentclass{IEEE_lsens}`
- Abstract + keywords inside `\IEEEtitleabstractindextext{}` (before `\maketitle`)
- 2-column, 9pt, `newtxmath` for math fonts
- `\usepackage[noadjust]{cite}` for citation compression

**`perform_ieee_sensors_writeup.py`:**
- Adapted from `perform_mdpi_writeup.py`
- Sections: Introduction → Related Work → Methodology → Experiments → Conclusion
- Auto-generates Table 1 (accuracy), Table 2 (Macro-F1), Table 3 (ECE) from results JSON
- Auto-inserts Grad-CAM and accuracy-vs-label-ratio figures
- 4-page limit enforced via LLM prompt instruction
- Max ~15 references (IEEE Sensors Letters guideline)
- Calls `compile_latex()` with pdflatex + bibtex (same as MDPI module)

---

## Topic File (`vortex_r2_iiot.md`) Key Fields

```yaml
goal: >
  Implement and validate Vortex-R2, a force-directed tabular-to-image transformation
  that places anti-correlated (independent) sensors at the image center. Compare against
  IGTD, DeepInsight, REFINED, XGBoost, and CatBoost on SECOM, Epileptic Seizure, CWRU,
  and PHM08 under label-scarce regimes (1%, 5%, 10%, 100%). Report Accuracy, Macro-F1,
  and ECE. Include ablation (no spiral, random layout, inverted correlation weights).

eval: >
  Primary: mean Accuracy across all datasets at 1% label regime.
  Secondary: Macro-F1 at 1% labels, ECE at 10% labels.
  Success threshold: Vortex-R2 outperforms best baseline by ≥3% accuracy at 1% labels
  on at least 3 of 4 datasets.
```

---

## Out of Scope

- Training from scratch (always use ImageNet pretrained ResNet-18)
- Temporal/sequential extension (noted as future work in paper)
- Nonlinear correlation measures (Mutual Information variant — future work)
- Custom novel IIoT dataset collection (use public datasets only)

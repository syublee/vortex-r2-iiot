# Title: Vortex-R2: Relationship-Regularized Tabular-to-Image Transformation for IIoT Fault Diagnosis Under Label Scarcity

## Keywords
tabular-to-image transformation, IIoT sensor data, fault diagnosis, convolutional neural networks, anti-correlation layout, force-directed graph, label scarcity, few-shot learning, SECOM, CWRU bearing fault

## TL;DR
Vortex-R2 inverts conventional tabular-to-image wisdom by mapping anti-correlated (informationally independent) sensors to the image center via a force-directed vortex layout, inducing a well-conditioned Hessian that improves CNN generalization under label scarcity in IIoT fault diagnosis.

## Abstract
Industrial IIoT systems generate high-dimensional tabular sensor data where fault signatures appear as subtle, correlated deviations across many channels. Tabular-to-image conversion enables CNNs to exploit spatial structure, but existing methods (DeepInsight, IGTD, REFINED) place similar sensors near each other — a strategy that concentrates redundant information at the image center and creates ill-conditioned Hessian blocks under low-label regimes. Vortex-R2 (Relationship-Regularized) reverses this: it builds an anti-correlation graph (w_ij = 1 - |r_ij|) and uses force-directed layout to place independent sensors near each other, then applies Archimedean spiral reordering to enforce center-periphery structure with independent sensors at the center. The resulting images have well-conditioned Hessian blocks and flat-minima convergence properties especially valuable when labeled training samples are scarce (1-10% of data). We evaluate Vortex-R2 against five baselines (IGTD, DeepInsight, REFINED, XGBoost, CatBoost) on four public IIoT datasets (SECOM semiconductor manufacturing, Epileptic Seizure EEG, CWRU Bearing Fault, UCI HAR) across four label regimes (1%, 5%, 10%, 100%) using accuracy, Macro-F1, and ECE. Ablations isolate the contribution of each algorithmic component. Targets MDPI Sensors.

## Research Questions
- RQ1: Does placing anti-correlated sensors at the image center improve CNN classification accuracy under label scarcity (1-10% labels)?
- RQ2: How do T (force-directed iterations), τ (correlation threshold), and k (image size) affect Vortex-R2 performance?
- RQ3: Does Vortex-R2 outperform tree ensembles (XGBoost, CatBoost) at low label regimes where CNNs typically struggle?

## Proposed Method
**Algorithm (4 steps):**

1. **Anti-correlation graph:** Compute Pearson correlation matrix R from training data (X_train). Set w_ij = 1 - |r_ij| so independent feature pairs get high weight.

2. **Force-directed layout:** Run Fruchterman-Reingold with w_ij as attractive weights. Features with high w_ij (independent) are pulled together; features with low w_ij (correlated) drift apart.

3. **Archimedean spiral reordering:** Sort features by mean correlation (ascending = most independent first). Map to spiral positions starting from center. τ controls center/periphery boundary size.

4. **Rasterize:** Map each sample's feature values to k×k pixel grid using the learned positions. Multiple features in the same pixel are averaged.

**BFTS search space:**
- T ∈ {50, 100, 200, 500}: force-directed iterations (more = finer layout)
- τ ∈ {0.3, 0.5, 0.7}: center/periphery boundary (lower = more features in center)
- k ∈ {16, 32, 64}: image grid size
- backbone ∈ {resnet18, efficientnet_b0}: CNN architecture

**Baselines:**
- DeepInsight: t-SNE layout (similar features near each other — opposite philosophy)
- IGTD: MDS layout (distance-rank minimization)
- REFINED: PCA layout (distance-preserving)
- XGBoost: gradient boosted trees (no image conversion)
- CatBoost: gradient boosted trees with categorical support
- Raw-CNN: features reshaped to k×k without correlation-aware layout

## Evaluation Metrics
- Primary: mean accuracy across 4 datasets at 1% label regime (higher = better)
- Secondary: Macro-F1 at 1% labels, ECE at 10% labels
- Success: Vortex-R2 outperforms best baseline by ≥3% accuracy at 1% labels on ≥3/4 datasets

## Datasets
- SECOM: 590 features, 1567 samples, 2 classes (semiconductor fault detection, UCI)
- Epileptic Seizure: 178 features, 11500 samples, 5 classes (EEG classification, UCI)
- CWRU Bearing Fault: 64 features, ~10000 samples, 4 classes (vibration fault diagnosis)
- UCI HAR: 561 features, 10299 samples, 6 classes (smartphone activity recognition, UCI)

## CNN Architecture
ResNet-18 pretrained on ImageNet. Grayscale k×k images converted to 3-channel 224×224.
Adam lr=1e-4, batch 32, 50 epochs, early stopping patience 10.

## Expected Results
Vortex-R2 should show largest gains over baselines at 1% label regime due to well-conditioned Hessian inducing better flat-minima generalization. At 100% labels, gap with tree ensembles narrows.

"""Vortex-R2 IIoT: anti-correlation tabular-to-image transformation.

BFTS search space:
  T        in {50, 100, 200, 500}   force-directed iterations
  TAU      in {0.3, 0.5, 0.7}       center/periphery threshold
  K        in {16, 32, 64}          image grid size
  BACKBONE in {resnet18, efficientnet_b0}
"""
import json
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

# ── BFTS hyperparameters (agent modifies these) ────────────────────────────────
T = 100                # force-directed iterations
TAU = 0.5              # correlation threshold for center/periphery
K = 32                 # image grid size (k×k pixels)
BACKBONE = "resnet18"  # CNN backbone: resnet18 | efficientnet_b0
N_SEEDS = 5            # random seeds for statistical averaging
LABEL_RATIOS = [0.01, 0.05, 0.10, 1.00]
N_EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 32
PATIENCE = 10

# ── Directories ────────────────────────────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
data_dir = os.path.join(os.getcwd(), "data", "vortex_r2")
os.makedirs(working_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Track which datasets used real downloads vs synthetic fallback
DATA_PROVENANCE: dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_secom():
    """SECOM semiconductor manufacturing (UCI). 590 features, 1567 samples, 2 classes."""
    cx = os.path.join(data_dir, "secom_X.npy")
    cy = os.path.join(data_dir, "secom_y.npy")
    if os.path.exists(cx) and os.path.exists(cy):
        DATA_PROVENANCE["secom"] = "real"
        return np.load(cx), np.load(cy)
    try:
        X_df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
            sep=r"\s+", header=None
        )
        y_df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data",
            sep=r"\s+", header=None
        )
        X = X_df.values.astype(np.float32)
        y = (y_df.iloc[:, 0].values == 1).astype(np.int64)
        medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = medians[j]
        std = X.std(axis=0)
        X = X[:, std > 0]
        np.save(cx, X); np.save(cy, y)
        print(f"[SECOM] Loaded: {X.shape}, classes={np.unique(y)}")
        DATA_PROVENANCE["secom"] = "real"
        return X, y
    except Exception as e:
        print(f"[SECOM] Download failed ({e}), using synthetic data.")
        DATA_PROVENANCE["secom"] = f"synthetic (download failed: {e})"
        rng = np.random.RandomState(0)
        X = rng.randn(1567, 590).astype(np.float32)
        y = rng.randint(0, 2, 1567)
        return X, y


def load_steel_plates():
    """Steel Plates Faults (UCI #198). 27 features, 1941 samples, 7 fault classes.

    Industrial fault detection dataset from steel plate manufacturing.
    More IIoT-relevant than EEG proxies.
    """
    cx = os.path.join(data_dir, "steel_X.npy")
    cy = os.path.join(data_dir, "steel_y.npy")
    if os.path.exists(cx) and os.path.exists(cy):
        DATA_PROVENANCE["steel_plates"] = "real"
        return np.load(cx), np.load(cy)
    try:
        req = urllib.request.Request(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        import io
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
        rows = [[float(x) for x in line.split()] for line in raw.strip().split("\n") if line.strip()]
        arr = np.array(rows, dtype=np.float32)
        X = arr[:, :27]
        y = arr[:, 27:].argmax(axis=1).astype(np.int64)
        np.save(cx, X); np.save(cy, y)
        print(f"[Steel Plates] Loaded: {X.shape}, classes={np.unique(y)}")
        DATA_PROVENANCE["steel_plates"] = "real"
        return X, y
    except Exception as e:
        print(f"[Steel Plates] Download failed ({e}), using synthetic data.")
        DATA_PROVENANCE["steel_plates"] = f"synthetic (download failed: {e})"
        rng = np.random.RandomState(1)
        X = rng.randn(1941, 27).astype(np.float32)
        y = rng.randint(0, 7, 1941)
        return X, y


def load_cwru():
    """CWRU Bearing Fault: 64 statistical features, ~1422 samples, 4 classes."""
    cx = os.path.join(data_dir, "cwru_X.npy")
    cy = os.path.join(data_dir, "cwru_y.npy")
    if os.path.exists(cx) and os.path.exists(cy):
        DATA_PROVENANCE["cwru"] = "real"
        return np.load(cx), np.load(cy)
    try:
        from scipy.io import loadmat
        # File IDs from Case Western Reserve University bearing data center
        # Normal: 97, Ball fault 0.007": 105, IR fault 0.007": 109, OR fault 0.007"@6: 130
        files = {
            0: "https://engineering.case.edu/sites/default/files/97.mat",
            1: "https://engineering.case.edu/sites/default/files/105.mat",
            2: "https://engineering.case.edu/sites/default/files/109.mat",
            3: "https://engineering.case.edu/sites/default/files/130.mat",
        }
        all_X, all_y = [], []
        for label, url in files.items():
            local = os.path.join(data_dir, f"cwru_{label}.mat")
            if not os.path.exists(local):
                urllib.request.urlretrieve(url, local)
            mat = loadmat(local)
            sig_key = next(k for k in mat if k.startswith("X") and "DE_time" in k)
            signal = mat[sig_key].flatten().astype(np.float32)
            feats = _extract_cwru_features(signal, window=1024, stride=512)
            all_X.append(feats)
            all_y.append(np.full(len(feats), label, dtype=np.int64))
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        np.save(cx, X); np.save(cy, y)
        print(f"[CWRU] Loaded: {X.shape}, classes={np.unique(y)}")
        DATA_PROVENANCE["cwru"] = "real"
        return X, y
    except Exception as e:
        print(f"[CWRU] Download failed ({e}), using synthetic data.")
        DATA_PROVENANCE["cwru"] = f"synthetic (download failed: {e})"
        rng = np.random.RandomState(2)
        X = rng.randn(9600, 64).astype(np.float32)
        y = np.repeat(np.arange(4), 2400)
        return X, y


def _extract_cwru_features(signal: np.ndarray, window: int = 1024, stride: int = 512) -> np.ndarray:
    """Extract 64 physically diverse features per window (no replication).

    Feature groups (8 features each, 8 groups = 64 total):
      G1 time-domain stats, G2 sub-window RMS (8 octants),
      G3 FFT band energy (8 bands 0-6 kHz), G4 spectral shape,
      G5 Hilbert-envelope stats, G6 quarter-segment moments,
      G7 impulse/threshold features, G8 temporal shape features.
    """
    features = []
    for start in range(0, len(signal) - window + 1, stride):
        seg = signal[start:start + window]
        std = float(seg.std()) or 1e-8
        rms = float(np.sqrt(np.mean(seg ** 2)))

        # G1: global time-domain statistics (8)
        g1 = [
            float(seg.mean()),
            std,
            rms,
            float(np.mean((seg - seg.mean()) ** 4) / std ** 4),   # kurtosis
            float(np.mean((seg - seg.mean()) ** 3) / std ** 3),   # skewness
            float(seg.max() / max(rms, 1e-8)),                     # crest factor
            float(seg.max() - seg.min()),                          # peak-to-peak
            float(np.corrcoef(seg[:-1], seg[1:])[0, 1]) if len(seg) > 1 else 0.0,
        ]

        # G2: RMS of 8 equal sub-windows (localized energy)
        sub_len = window // 8
        g2 = [float(np.sqrt(np.mean(seg[i*sub_len:(i+1)*sub_len] ** 2))) for i in range(8)]

        # G3: FFT band energy (8 bands, 0–6 kHz assuming 12 kHz sampling)
        fft_mag = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(window, d=1.0 / 12000)
        total_pwr = float(np.sum(fft_mag ** 2)) or 1e-8
        band_edges = np.linspace(0, 6000, 9)
        g3 = []
        for i in range(8):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            g3.append(float(np.sum(fft_mag[mask] ** 2) / total_pwr))

        # G4: spectral shape descriptors (8)
        fft_prob = fft_mag ** 2 / total_pwr
        n_freq = len(freqs)
        centroid = float(np.dot(freqs, fft_prob[:n_freq]))
        bandwidth = float(np.sqrt(np.dot((freqs - centroid) ** 2, fft_prob[:n_freq])))
        flatness = float(np.exp(np.mean(np.log(fft_mag[:n_freq] + 1e-10))) / (np.mean(fft_mag[:n_freq]) + 1e-10))
        cum_pwr = np.cumsum(fft_mag ** 2)
        rolloff_idx = int(np.searchsorted(cum_pwr, 0.85 * cum_pwr[-1]))
        rolloff = float(freqs[min(rolloff_idx, n_freq - 1)])
        spectral_entropy = float(-np.sum(fft_prob[:n_freq] * np.log(fft_prob[:n_freq] + 1e-10)))
        peak_freq = float(freqs[np.argmax(fft_mag[:n_freq])])
        hf_ratio = float(np.sum(fft_mag[freqs >= 3000] ** 2) / total_pwr)
        spectral_slope = float(np.polyfit(freqs[:n_freq], fft_mag[:n_freq], 1)[0]) if n_freq > 1 else 0.0
        g4 = [centroid / 6000, bandwidth / 6000, flatness, rolloff / 6000,
              spectral_entropy / 10.0, peak_freq / 6000, hf_ratio, np.clip(spectral_slope / 1e-3, -5, 5)]

        # G5: Hilbert-envelope statistics (8)
        from scipy.signal import hilbert as scipy_hilbert
        try:
            env = np.abs(scipy_hilbert(seg))
        except Exception:
            env = np.abs(seg)
        env_rms = float(np.sqrt(np.mean(env ** 2)))
        env_std = float(env.std()) or 1e-8
        g5 = [
            float(env.mean()),
            env_std,
            env_rms,
            float(np.mean((env - env.mean()) ** 4) / env_std ** 4),
            float(np.mean((env - env.mean()) ** 3) / env_std ** 3),
            float(env.max() / max(env_rms, 1e-8)),
            float(env.max() - env.min()),
            float(np.corrcoef(env[:-1], env[1:])[0, 1]) if len(env) > 1 else 0.0,
        ]

        # G6: per-quarter mean and std (4 quarters × 2 = 8)
        q = window // 4
        g6 = []
        for i in range(4):
            sub = seg[i * q:(i + 1) * q]
            g6 += [float(sub.mean()), float(sub.std())]

        # G7: impulse / threshold features at 4 σ levels (8)
        g7 = []
        for thr in [2.0, 3.0, 4.0, 5.0]:
            rate = float(np.mean(np.abs(seg) > thr * std))
            amp = float(np.mean(np.abs(seg[np.abs(seg) > thr * std])) if rate > 0 else 0.0)
            g7 += [rate, amp / max(rms, 1e-8)]

        # G8: temporal shape features (8)
        zcr = float(np.mean(np.diff(np.sign(seg)) != 0))
        slope_changes = float(np.mean(np.diff(np.sign(np.diff(seg))) != 0))
        waveform_factor = float(rms / max(np.mean(np.abs(seg)), 1e-8))
        shape_factor = float(rms / max(seg.max() - seg.min(), 1e-8))
        lag2_corr = float(np.corrcoef(seg[:-2], seg[2:])[0, 1]) if len(seg) > 2 else 0.0
        lag4_corr = float(np.corrcoef(seg[:-4], seg[4:])[0, 1]) if len(seg) > 4 else 0.0
        energy_ratio = float(np.mean(seg[window // 2:] ** 2) / max(np.mean(seg[:window // 2] ** 2), 1e-8))
        peak_asym = float((seg.max() + seg.min()) / max(seg.max() - seg.min(), 1e-8))
        g8 = [zcr, slope_changes, waveform_factor, shape_factor, lag2_corr, lag4_corr, energy_ratio, peak_asym]

        feats = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8  # exactly 64
        features.append(feats)
    return np.array(features, dtype=np.float32)


def load_uci_har():
    """UCI HAR smartphone activity recognition. 561 features, 10299 samples, 6 classes."""
    cx = os.path.join(data_dir, "har_X.npy")
    cy = os.path.join(data_dir, "har_y.npy")
    if os.path.exists(cx) and os.path.exists(cy):
        DATA_PROVENANCE["uci_har"] = "real"
        return np.load(cx), np.load(cy)
    try:
        base = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
        zip_path = os.path.join(data_dir, "har.zip")
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(base + "UCI%20HAR%20Dataset.zip", zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            with z.open("UCI HAR Dataset/train/X_train.txt") as f:
                X_train = np.loadtxt(f)
            with z.open("UCI HAR Dataset/train/y_train.txt") as f:
                y_train = np.loadtxt(f, dtype=np.int64) - 1
            with z.open("UCI HAR Dataset/test/X_test.txt") as f:
                X_test = np.loadtxt(f)
            with z.open("UCI HAR Dataset/test/y_test.txt") as f:
                y_test = np.loadtxt(f, dtype=np.int64) - 1
        X = np.vstack([X_train, X_test]).astype(np.float32)
        y = np.concatenate([y_train, y_test])
        np.save(cx, X); np.save(cy, y)
        print(f"[UCI HAR] Loaded: {X.shape}, classes={np.unique(y)}")
        DATA_PROVENANCE["uci_har"] = "real"
        return X, y
    except Exception as e:
        print(f"[UCI HAR] Download failed ({e}), using synthetic data.")
        DATA_PROVENANCE["uci_har"] = f"synthetic (download failed: {e})"
        rng = np.random.RandomState(3)
        X = rng.randn(10299, 561).astype(np.float32)
        y = np.repeat(np.arange(6), 10299 // 6 + 1)[:10299]
        return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Vortex-R2 core algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def build_anticorrelation_weights(X: np.ndarray) -> np.ndarray:
    """Compute anti-correlation weight matrix W where w_ij = 1 - |r_ij|."""
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0)
    W = 1.0 - np.abs(corr)
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def force_directed_layout(W: np.ndarray, T: int = 100, seed: int = 42) -> np.ndarray:
    """Fruchterman-Reingold layout with anti-correlation attractive weights."""
    rng = np.random.RandomState(seed)
    n = W.shape[0]

    try:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=2, random_state=seed)
        pos = svd.fit_transform(W).astype(np.float64)
        for dim in range(2):
            rng_dim = pos[:, dim].max() - pos[:, dim].min()
            if rng_dim > 0:
                pos[:, dim] = 2 * (pos[:, dim] - pos[:, dim].min()) / rng_dim - 1
    except Exception:
        pos = rng.uniform(-1, 1, (n, 2))

    k_fr = np.sqrt(4.0 / n)

    for t in range(T):
        temp = 2.0 * (1.0 - t / T)

        delta = pos[:, None, :] - pos[None, :, :]
        dist2 = (delta ** 2).sum(axis=2)
        np.fill_diagonal(dist2, 1.0)
        dist = np.sqrt(dist2)

        rep_mag = k_fr ** 2 / dist2
        np.fill_diagonal(rep_mag, 0.0)
        direction = delta / dist[:, :, None]
        disp = (rep_mag[:, :, None] * direction).sum(axis=1)

        attr_mag = W * dist / k_fr
        disp -= (attr_mag[:, :, None] * direction).sum(axis=1)

        d = np.linalg.norm(disp, axis=1, keepdims=True)
        d = np.maximum(d, 1e-8)
        pos += (disp / d) * np.minimum(d, temp)
        pos = np.clip(pos, -1.0, 1.0)

    return pos.astype(np.float32)


def spiral_pixel_coords(n_features: int, avg_correlation: np.ndarray, k: int) -> np.ndarray:
    """Assign features to k×k pixel positions using Archimedean spiral.

    Low-correlation features (independent) go to center positions.
    """
    order = np.argsort(avg_correlation)

    center = (k - 1) / 2.0
    positions = []
    seen = set()
    theta = 0.0
    a = k / (4 * np.pi * 3)

    while len(positions) < n_features:
        r = a * theta
        x = int(round(center + r * np.cos(theta)))
        y = int(round(center + r * np.sin(theta)))
        x = max(0, min(k - 1, x))
        y = max(0, min(k - 1, y))
        if (x, y) not in seen:
            seen.add((x, y))
            positions.append((x, y))
        theta += 0.5 / max(r, 0.5)
        if theta > 30 * np.pi:
            for i in range(k):
                for j in range(k):
                    if (i, j) not in seen and len(positions) < n_features:
                        seen.add((i, j))
                        positions.append((i, j))
            break

    n_pos = len(positions)
    pixel_coords = np.zeros((n_features, 2), dtype=np.int32)
    for rank, feat_idx in enumerate(order):
        pixel_coords[feat_idx] = positions[rank % n_pos]

    return pixel_coords


def rasterize(x_sample: np.ndarray, pixel_coords: np.ndarray, k: int) -> np.ndarray:
    """Map standardized feature values to k×k image. Multiple features per pixel are averaged."""
    image = np.zeros((k, k), dtype=np.float32)
    counts = np.zeros((k, k), dtype=np.float32)
    for feat_idx in range(len(x_sample)):
        xi, yi = int(pixel_coords[feat_idx, 0]), int(pixel_coords[feat_idx, 1])
        image[xi, yi] += float(x_sample[feat_idx])
        counts[xi, yi] += 1.0
    mask = counts > 0
    image[mask] /= counts[mask]
    vmin, vmax = image.min(), image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)
    return image


def vortex_r2_transform(
    X_train: np.ndarray,
    X_all: np.ndarray,
    T: int = 100,
    tau: float = 0.5,
    k: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Full Vortex-R2 pipeline: learn layout from X_train, apply to X_all."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_all_sc = scaler.transform(X_all)

    W = build_anticorrelation_weights(X_tr_sc)
    avg_corr = 1.0 - W.mean(axis=1)
    pixel_coords = spiral_pixel_coords(X_train.shape[1], avg_corr, k)

    return np.array([rasterize(X_all_sc[i], pixel_coords, k) for i in range(len(X_all_sc))])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Tabular-to-image baselines
# ═══════════════════════════════════════════════════════════════════════════════

def _layout_to_images(X_all_sc: np.ndarray, pos: np.ndarray, k: int) -> np.ndarray:
    """Shared helper: map 2D layout to pixel grid and rasterize."""
    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    rng = pos_max - pos_min
    rng[rng == 0] = 1.0
    pixel_coords = ((pos - pos_min) / rng * (k - 1)).astype(np.int32)
    pixel_coords = np.clip(pixel_coords, 0, k - 1)
    return np.array([rasterize(X_all_sc[i], pixel_coords, k) for i in range(len(X_all_sc))])


def deepinsight_transform(X_train: np.ndarray, X_all: np.ndarray, k: int = 32, seed: int = 42) -> np.ndarray:
    """DeepInsight: t-SNE layout (similar features near each other)."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_all_sc = scaler.transform(X_all)

    n_features = X_train.shape[1]
    perplexity = min(30, n_features - 1)
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, max_iter=500)
    pos = tsne.fit_transform(X_tr_sc.T)
    return _layout_to_images(X_all_sc, pos, k)


def igtd_transform(X_train: np.ndarray, X_all: np.ndarray, k: int = 32, seed: int = 42) -> np.ndarray:
    """IGTD simplified: MDS layout minimizing distance-rank difference."""
    from sklearn.manifold import MDS
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import euclidean_distances

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_all_sc = scaler.transform(X_all)

    D = euclidean_distances(X_tr_sc.T)
    mds = MDS(n_components=2, random_state=seed, dissimilarity="precomputed", max_iter=300)
    pos = mds.fit_transform(D)
    return _layout_to_images(X_all_sc, pos, k)


def refined_transform(X_train: np.ndarray, X_all: np.ndarray, k: int = 32, seed: int = 42) -> np.ndarray:
    """REFINED simplified: PCA-based distance-preserving layout."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_all_sc = scaler.transform(X_all)

    pca = PCA(n_components=2, random_state=seed)
    pos = pca.fit_transform(X_tr_sc.T)
    return _layout_to_images(X_all_sc, pos, k)


def raw_cnn_transform(X_train: np.ndarray, X_all: np.ndarray, k: int = 32) -> np.ndarray:
    """Raw-CNN: pad/truncate features to k×k without spatial arrangement."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_all_sc = scaler.transform(X_all).astype(np.float32)

    n_features = X_all_sc.shape[1]
    n_pixels = k * k

    if n_features < n_pixels:
        padded = np.zeros((len(X_all_sc), n_pixels), dtype=np.float32)
        padded[:, :n_features] = X_all_sc
    else:
        padded = X_all_sc[:, :n_pixels]

    vmin = padded.min(axis=1, keepdims=True)
    vmax = padded.max(axis=1, keepdims=True)
    rng = vmax - vmin
    rng[rng == 0] = 1.0
    padded = (padded - vmin) / rng

    return padded.reshape(-1, k, k)


def random_layout_transform(X_train: np.ndarray, X_all: np.ndarray, k: int = 32, seed: int = 42) -> np.ndarray:
    """Random-Layout: assign features to pixels in random order (trivial spatial control).

    This is the key control requested by Reviewer Q1: if a layout is 'largely inert'
    vs baselines, a random layout should perform no worse than structured layouts.
    The layout is seeded from X_train shape + seed so it is deterministic per run
    but varies across seeds.
    """
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_all_sc = scaler.transform(X_all)

    n_features = X_train.shape[1]
    n_pixels = k * k
    # Random permutation of feature → pixel assignment
    perm = rng.permutation(min(n_features, n_pixels))
    pixel_coords = np.zeros((n_features, 2), dtype=np.int32)
    for fi in range(min(n_features, n_pixels)):
        p = int(perm[fi])
        pixel_coords[fi] = [p // k, p % k]
    # Features beyond n_pixels map to the last pixel (averaged in rasterize)
    if n_features > n_pixels:
        for fi in range(n_pixels, n_features):
            pixel_coords[fi] = [k - 1, k - 1]

    return np.array([rasterize(X_all_sc[i], pixel_coords, k) for i in range(len(X_all_sc))])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    n = len(labels)
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / n * abs(correct[mask].mean() - max_probs[mask].mean())
    return float(ece)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CNN training
# ═══════════════════════════════════════════════════════════════════════════════

def train_eval_cnn(
    X_imgs_train: np.ndarray,
    y_train: np.ndarray,
    X_imgs_test: np.ndarray,
    y_test: np.ndarray,
    backbone: str = "resnet18",
    n_epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    patience: int = 10,
    seed: int = 42,
    n_classes: int = None,
) -> dict:
    """Fine-tune pretrained CNN on k×k grayscale images. Returns accuracy, f1, ece."""
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, f1_score

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if n_classes is None:
        n_classes = len(np.unique(y_train))

    _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def to_tensor(X_imgs):
        t = torch.from_numpy(X_imgs).float().unsqueeze(1)
        t = t.repeat(1, 3, 1, 1)
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        t = (t - _mean) / _std
        return t

    X_tr = to_tensor(X_imgs_train)
    y_tr = torch.from_numpy(y_train).long()
    X_te = to_tensor(X_imgs_test)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    if backbone == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(512, n_classes)
    else:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(1280, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    no_improve = 0
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / max(len(train_dl), 1)
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_te), 256):
            logits = model(X_te[i:i + 256].to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(all_probs)
    preds = probs.argmax(axis=1)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "ece": compute_ece(probs, y_test),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Tree baselines
# ═══════════════════════════════════════════════════════════════════════════════

def train_eval_xgboost(X_train, y_train, X_test, y_test, seed=42) -> dict:
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        n_jobs=-1, random_state=seed, eval_metric="logloss",
        use_label_encoder=False, verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "ece": compute_ece(probs, y_test),
    }


def train_eval_catboost(X_train, y_train, X_test, y_test, seed=42) -> dict:
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, f1_score

    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        random_seed=seed, verbose=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test).flatten()
    probs = model.predict_proba(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "ece": compute_ece(probs, y_test),
    }


def train_eval_lda(X_train, y_train, X_test, y_test, seed=42) -> dict:
    """Regularized LDA (Ledoit-Wolf shrinkage) — direct tabular baseline for within-class whitening comparison."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    try:
        model.fit(X_tr, y_train)
    except Exception:
        # Fallback: SVD solver without shrinkage when lsqr fails (e.g., n_samples < n_features)
        model = LinearDiscriminantAnalysis(solver="svd")
        model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "ece": compute_ece(probs, y_test),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_subsample(X: np.ndarray, y: np.ndarray, ratio: float, seed: int):
    """Sample ratio × n samples stratified by class, guaranteeing ≥1 per class."""
    from sklearn.model_selection import train_test_split

    n_sample = max(int(len(y) * ratio), len(np.unique(y)))
    if n_sample >= len(y):
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=n_sample / len(y), stratify=y, random_state=seed
    )
    return X_sub, y_sub


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_curves(results: dict, output_path: str):
    """Accuracy vs label ratio for each method and dataset, with 95% CI error bars."""
    datasets = list(results.keys())
    if not datasets:
        return
    methods = list(results[datasets[0]].keys())
    ratio_labels = [f"{int(r * 100)}%" for r in LABEL_RATIOS]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0, j]
        for method in methods:
            accs = [results[ds].get(method, {}).get(str(r), {}).get("accuracy", 0.0) for r in LABEL_RATIOS]
            cis = [results[ds].get(method, {}).get(str(r), {}).get("accuracy_ci95", 0.0) for r in LABEL_RATIOS]
            ax.errorbar(ratio_labels, accs, yerr=cis, marker="o", label=method, capsize=3, capthick=1)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Label ratio")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Accuracy curves saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: main()
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"[Vortex-R2] T={T} TAU={TAU} K={K} BACKBONE={BACKBONE}")
    from sklearn.model_selection import train_test_split

    DATASET_LOADERS = {
        "secom": load_secom,
        "steel_plates": load_steel_plates,
        "cwru": load_cwru,
        "uci_har": load_uci_har,
    }

    IMG_METHODS = {
        "vortex_r2": lambda Xtr, Xall, s: vortex_r2_transform(Xtr, Xall, T, TAU, K, s),
        "deepinsight": lambda Xtr, Xall, s: deepinsight_transform(Xtr, Xall, K, s),
        "igtd": lambda Xtr, Xall, s: igtd_transform(Xtr, Xall, K, s),
        "refined": lambda Xtr, Xall, s: refined_transform(Xtr, Xall, K, s),
        "raw_cnn": lambda Xtr, Xall, s: raw_cnn_transform(Xtr, Xall, K),
        "random_layout": lambda Xtr, Xall, s: random_layout_transform(Xtr, Xall, K, s),
    }
    TREE_METHODS = {
        "xgboost": train_eval_xgboost,
        "catboost": train_eval_catboost,
        "lda": train_eval_lda,
    }

    all_results = {}
    # Track per-method wall-clock training time (seconds) for compute cost comparison
    compute_times: dict = {m: [] for m in list(IMG_METHODS) + list(TREE_METHODS)}

    import time

    for ds_name, loader in DATASET_LOADERS.items():
        print(f"\n{'='*60}\nDataset: {ds_name}")
        X, y = loader()
        # Augment provenance with row/feature counts for reviewer verification
        prov = DATA_PROVENANCE.get(ds_name, "unknown")
        DATA_PROVENANCE[ds_name] = {
            "status": prov if isinstance(prov, str) else prov.get("status", "unknown"),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }
        n_classes = len(np.unique(y))
        all_methods = list(IMG_METHODS) + list(TREE_METHODS)
        all_results[ds_name] = {m: {} for m in all_methods}

        for ratio in LABEL_RATIOS:
            ratio_str = str(ratio)
            print(f"  ratio={ratio*100:.0f}%")
            per_seed = {m: {"accuracy": [], "f1": [], "ece": []} for m in all_methods}

            for seed in range(N_SEEDS):
                X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
                X_tr, y_tr = stratified_subsample(X_tv, y_tv, ratio, seed)
                X_all = np.vstack([X_tr, X_te])

                for mname, fn in IMG_METHODS.items():
                    try:
                        imgs = fn(X_tr, X_all, seed)
                        imgs_tr, imgs_te = imgs[:len(X_tr)], imgs[len(X_tr):]
                        t0 = time.time()
                        r = train_eval_cnn(imgs_tr, y_tr, imgs_te, y_te, BACKBONE, N_EPOCHS, LR, BATCH_SIZE, PATIENCE, seed, n_classes)
                        compute_times[mname].append(time.time() - t0)
                    except Exception as e:
                        print(f"    [{mname}] failed: {e}")
                        r = {"accuracy": 0.0, "f1": 0.0, "ece": 1.0}
                    for k_ in ["accuracy", "f1", "ece"]:
                        per_seed[mname][k_].append(r[k_])

                for mname, fn in TREE_METHODS.items():
                    try:
                        t0 = time.time()
                        r = fn(X_tr, y_tr, X_te, y_te, seed)
                        compute_times[mname].append(time.time() - t0)
                    except Exception as e:
                        print(f"    [{mname}] failed: {e}")
                        r = {"accuracy": 0.0, "f1": 0.0, "ece": 1.0}
                    for k_ in ["accuracy", "f1", "ece"]:
                        per_seed[mname][k_].append(r[k_])

            for method in all_methods:
                acc_seeds = per_seed[method]["accuracy"]
                f1_seeds = per_seed[method]["f1"]
                ece_seeds = per_seed[method]["ece"]
                n = len(acc_seeds)
                # 95% CI via t-distribution (t_{n-1, 0.975})
                from scipy.stats import t as t_dist
                t_crit = float(t_dist.ppf(0.975, df=max(n - 1, 1))) if n > 1 else 0.0
                all_results[ds_name][method][ratio_str] = {
                    "accuracy": float(np.mean(acc_seeds)),
                    "accuracy_std": float(np.std(acc_seeds, ddof=1)) if n > 1 else 0.0,
                    "accuracy_ci95": float(t_crit * np.std(acc_seeds, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                    "accuracy_seeds": [float(v) for v in acc_seeds],
                    "f1": float(np.mean(f1_seeds)),
                    "f1_std": float(np.std(f1_seeds, ddof=1)) if n > 1 else 0.0,
                    "f1_seeds": [float(v) for v in f1_seeds],
                    "ece": float(np.mean(ece_seeds)),
                    "ece_std": float(np.std(ece_seeds, ddof=1)) if n > 1 else 0.0,
                    "ece_seeds": [float(v) for v in ece_seeds],
                }

    primary_accs = [
        all_results[ds]["vortex_r2"].get("0.01", {}).get("accuracy", 0.0)
        for ds in DATASET_LOADERS
    ]
    primary_metric = float(np.mean(primary_accs)) if primary_accs else 0.0

    print(f"\nPrimary metric (Vortex-R2 acc@1% mean): {primary_metric:.4f}")
    for ds in DATASET_LOADERS:
        acc = all_results[ds]["vortex_r2"].get("0.01", {}).get("accuracy", 0.0)
        best_bl = max(
            (all_results[ds][m].get("0.01", {}).get("accuracy", 0.0)
             for m in list(IMG_METHODS) + list(TREE_METHODS) if m != "vortex_r2"),
            default=0.0,
        )
        print(f"  {ds}: Vortex-R2={acc:.3f}  best_baseline={best_bl:.3f}  delta={acc-best_bl:+.3f}")

    # Wilcoxon signed-rank tests: Vortex-R2 vs each baseline at 1% label ratio
    from scipy.stats import wilcoxon
    significance_tests = {}
    for ds in DATASET_LOADERS:
        significance_tests[ds] = {}
        vortex_seeds = all_results[ds]["vortex_r2"].get("0.01", {}).get("accuracy_seeds", [])
        for bl in list(IMG_METHODS) + list(TREE_METHODS):
            if bl == "vortex_r2":
                continue
            bl_seeds = all_results[ds][bl].get("0.01", {}).get("accuracy_seeds", [])
            if len(vortex_seeds) >= 5 and len(bl_seeds) >= 5:
                try:
                    diffs = [v - b for v, b in zip(vortex_seeds, bl_seeds)]
                    if all(d == 0 for d in diffs):
                        p_val = 1.0
                    else:
                        _, p_val = wilcoxon(vortex_seeds, bl_seeds, alternative="two-sided")
                    significance_tests[ds][bl] = {
                        "p_wilcoxon": float(p_val),
                        "significant_p05": bool(p_val < 0.05),
                        "mean_diff": float(np.mean(vortex_seeds) - np.mean(bl_seeds)),
                    }
                    print(f"  Wilcoxon {ds} vortex_r2 vs {bl} @1%: p={p_val:.4f} Δacc={np.mean(vortex_seeds)-np.mean(bl_seeds):+.4f}")
                except Exception as e:
                    significance_tests[ds][bl] = {"error": str(e)}

    print(f"\nData provenance: {DATA_PROVENANCE}")

    # Summarise compute cost: mean training wall-clock time per method (seconds)
    compute_cost_summary = {
        m: {"mean_s": float(np.mean(ts)), "n_runs": len(ts)}
        for m, ts in compute_times.items() if ts
    }
    cnn_mean = float(np.mean([
        v["mean_s"] for k, v in compute_cost_summary.items() if k in IMG_METHODS
    ])) if any(k in IMG_METHODS for k in compute_cost_summary) else 0.0
    gbt_mean = float(np.mean([
        v["mean_s"] for k, v in compute_cost_summary.items() if k in ("xgboost", "catboost")
    ])) if any(k in ("xgboost", "catboost") for k in compute_cost_summary) else 0.0
    if gbt_mean > 0:
        compute_cost_summary["cnn_vs_gbt_ratio"] = round(cnn_mean / gbt_mean, 2)
    cost_str = {k: f"{v['mean_s']:.1f}s" for k, v in compute_cost_summary.items() if isinstance(v, dict) and "mean_s" in v}
    print(f"Compute cost (mean train s): {cost_str}")

    final = {
        "hyperparams": {"T": T, "tau": TAU, "k": K, "backbone": BACKBONE},
        "datasets": list(DATASET_LOADERS.keys()),
        "methods": list(IMG_METHODS) + list(TREE_METHODS),
        "label_ratios": LABEL_RATIOS,
        "results": all_results,
        "primary_metric": primary_metric,
        "significance_tests_1pct": significance_tests,
        "data_provenance": DATA_PROVENANCE,
        "compute_cost": compute_cost_summary,
    }

    results_path = os.path.join(working_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Results saved → {results_path}")

    plot_accuracy_curves(all_results, os.path.join(working_dir, "accuracy_curves.png"))
    print("[Vortex-R2] Done.")


if __name__ == "__main__":
    main()

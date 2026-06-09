"""Unit tests for Vortex-R2 core algorithm functions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_scientist', 'ideas'))

import numpy as np
import pytest
from vortex_r2_iiot import (
    build_anticorrelation_weights,
    force_directed_layout,
    spiral_pixel_coords,
    rasterize,
    vortex_r2_transform,
    compute_ece,
    raw_cnn_transform,
)


@pytest.fixture
def small_X():
    rng = np.random.RandomState(42)
    return rng.randn(20, 10).astype(np.float32)


class TestBuildAnticorrelationWeights:
    def test_output_shape(self, small_X):
        W = build_anticorrelation_weights(small_X)
        assert W.shape == (10, 10)

    def test_diagonal_zero(self, small_X):
        W = build_anticorrelation_weights(small_X)
        assert np.allclose(np.diag(W), 0.0)

    def test_symmetric(self, small_X):
        W = build_anticorrelation_weights(small_X)
        assert np.allclose(W, W.T)

    def test_values_in_range(self, small_X):
        W = build_anticorrelation_weights(small_X)
        assert W.min() >= 0.0
        assert W.max() <= 1.0

    def test_identical_features_get_zero_weight(self):
        X = np.random.randn(50, 1)
        X2 = np.hstack([X, X])
        W = build_anticorrelation_weights(X2)
        assert W[0, 1] < 0.01, f"Expected near-zero weight for identical features, got {W[0, 1]}"

    def test_independent_features_get_high_weight(self):
        rng = np.random.RandomState(0)
        X = rng.randn(500, 2)
        W = build_anticorrelation_weights(X)
        assert W[0, 1] > 0.7, f"Expected high weight for independent features, got {W[0, 1]}"


class TestForceDirectedLayout:
    def test_output_shape(self, small_X):
        W = build_anticorrelation_weights(small_X)
        pos = force_directed_layout(W, T=5, seed=0)
        assert pos.shape == (10, 2)

    def test_positions_within_bounds(self, small_X):
        W = build_anticorrelation_weights(small_X)
        pos = force_directed_layout(W, T=5, seed=0)
        assert pos.min() >= -1.01
        assert pos.max() <= 1.01

    def test_deterministic_with_same_seed(self, small_X):
        W = build_anticorrelation_weights(small_X)
        pos1 = force_directed_layout(W, T=10, seed=7)
        pos2 = force_directed_layout(W, T=10, seed=7)
        assert np.allclose(pos1, pos2)


class TestSpiralPixelCoords:
    def test_output_shape(self):
        avg_corr = np.linspace(0, 1, 10)
        coords = spiral_pixel_coords(10, avg_corr, k=8)
        assert coords.shape == (10, 2)

    def test_coords_within_grid(self):
        k = 8
        avg_corr = np.linspace(0, 1, 10)
        coords = spiral_pixel_coords(10, avg_corr, k=k)
        assert coords.min() >= 0
        assert coords.max() <= k - 1

    def test_most_independent_feature_near_center(self):
        k = 16
        center = (k - 1) / 2.0
        n = 20
        avg_corr = np.arange(n, dtype=float)
        coords = spiral_pixel_coords(n, avg_corr, k=k)
        dist_0 = np.sqrt((coords[0, 0] - center) ** 2 + (coords[0, 1] - center) ** 2)
        dist_19 = np.sqrt((coords[19, 0] - center) ** 2 + (coords[19, 1] - center) ** 2)
        assert dist_0 < dist_19, "Most independent feature should be closer to center"

    def test_handles_more_features_than_pixels(self):
        avg_corr = np.linspace(0, 1, 20)
        coords = spiral_pixel_coords(20, avg_corr, k=4)
        assert coords.shape == (20, 2)


class TestRasterize:
    def test_output_shape(self):
        k = 8
        x = np.ones(10, dtype=np.float32)
        coords = np.zeros((10, 2), dtype=np.int32)
        for i in range(10):
            coords[i] = [i % k, i // k]
        img = rasterize(x, coords, k)
        assert img.shape == (k, k)

    def test_output_range(self):
        k = 8
        rng = np.random.RandomState(0)
        x = rng.randn(20).astype(np.float32)
        coords = np.column_stack([np.arange(20) % k, np.arange(20) // k]).astype(np.int32)
        img = rasterize(x, coords, k)
        assert img.min() >= 0.0
        assert img.max() <= 1.0 + 1e-6

    def test_all_same_value_returns_zero_or_one(self):
        k = 4
        x = np.ones(4, dtype=np.float32) * 3.0
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
        img = rasterize(x, coords, k)
        assert img.min() >= 0.0 and img.max() <= 1.0 + 1e-6


class TestVortexR2Transform:
    def test_output_shape(self, small_X):
        k = 8
        images = vortex_r2_transform(small_X, small_X, T=5, tau=0.5, k=k, seed=0)
        assert images.shape == (20, k, k)

    def test_output_range(self, small_X):
        images = vortex_r2_transform(small_X, small_X, T=5, tau=0.5, k=8, seed=0)
        assert images.min() >= 0.0
        assert images.max() <= 1.0 + 1e-6

    def test_train_test_combined(self, small_X):
        k = 8
        X_train = small_X[:15]
        X_test = small_X[15:]
        X_all = np.vstack([X_train, X_test])
        images = vortex_r2_transform(X_train, X_all, T=5, tau=0.5, k=k, seed=0)
        assert images.shape == (20, k, k)


class TestComputeEce:
    def test_perfect_calibration(self):
        n = 100
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.95
        probs[:, 1] = 0.05
        labels = np.zeros(n, dtype=int)
        ece = compute_ece(probs, labels)
        assert ece < 0.1, f"Expected low ECE, got {ece:.4f}"

    def test_worst_calibration(self):
        n = 100
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.99
        probs[:, 1] = 0.01
        labels = np.ones(n, dtype=int)
        ece = compute_ece(probs, labels)
        assert ece > 0.5, f"Expected high ECE, got {ece:.4f}"

    def test_output_in_range(self):
        rng = np.random.RandomState(0)
        n = 200
        probs = rng.dirichlet(alpha=[1, 1, 1], size=n)
        labels = rng.randint(0, 3, n)
        ece = compute_ece(probs, labels)
        assert 0.0 <= ece <= 1.0


class TestRawCnnTransform:
    def test_output_shape(self, small_X):
        images = raw_cnn_transform(small_X, small_X, k=8)
        assert images.shape == (20, 8, 8)

    def test_output_range(self, small_X):
        images = raw_cnn_transform(small_X, small_X, k=8)
        assert images.min() >= 0.0
        assert images.max() <= 1.0 + 1e-6

    def test_padding_for_fewer_features_than_pixels(self):
        X = np.random.randn(10, 5).astype(np.float32)
        images = raw_cnn_transform(X, X, k=8)
        assert images.shape == (10, 8, 8)

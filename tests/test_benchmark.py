import numpy as np

from eikonal_pinn.benchmark import exact_T, exact_V, make_grid, ridge_distance, sample_data, T_MAX


def test_eikonal_satisfied_away_from_ridge():
    n = 200
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    T = exact_T(xy).reshape(n, n)
    V = exact_V(xy).reshape(n, n)
    dTdy, dTdx = np.gradient(T, x, x)  # rows = y, cols = x
    speed = V * np.sqrt(dTdx**2 + dTdy**2)
    far = ridge_distance(xy).reshape(n, n) > 0.05
    # exclude the two wave sources (cone tips of T) where grad T is not defined
    far &= np.sqrt(X**2 + Y**2) > 0.03
    far &= np.sqrt((X - 1) ** 2 + (Y - 1) ** 2) > 0.03
    assert far.sum() > 0.5 * n * n
    assert np.all(np.abs(speed[far] - 1.0) < 1e-2)


def test_grid_and_data_shapes():
    xy, X_m, Y_m = make_grid(50)
    assert xy.shape == (2500, 2)
    assert X_m.shape == (50, 50)
    assert np.isclose(xy[:, 0].min(), 0) and np.isclose(xy[:, 1].max(), 1)
    d = sample_data(50, seed=1234)
    assert d.shape == (50, 2) and d.min() >= 0 and d.max() <= 1
    assert np.allclose(sample_data(50, seed=1234), d)  # seeded
    assert exact_T(xy).shape == (2500, 1) and exact_V(xy).shape == (2500, 1)


def test_tmax_matches_grid():
    xy, _, _ = make_grid(400)
    assert abs(exact_T(xy).max() - T_MAX) < 2e-3

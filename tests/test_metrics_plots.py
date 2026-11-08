import numpy as np

from eikonal_pinn.benchmark import exact_T, exact_V, make_grid, sample_data
from eikonal_pinn.metrics import compute_metrics
from eikonal_pinn.plots import plot_fields, plot_loss


def test_metrics_values():
    T = np.zeros((4, 1)); V = np.ones((4, 1))
    m = compute_metrics(T + 0.01, V + np.array([[0.1], [-0.3], [0], [0]]), T, V)
    assert np.isclose(m["rmse_T"], 0.01) and np.isclose(m["nrmse_T"], 1.0)
    assert np.isclose(m["mae_V"], 0.1) and np.isclose(m["nmae_V"], 10.0)
    assert np.isclose(m["maxerr_T"], 0.01) and np.isclose(m["maxerr_V"], 0.3)


def test_plots_write_files(tmp_path):
    xy, X_m, Y_m = make_grid(50)
    T, V = exact_T(xy), exact_V(xy)
    plot_fields(tmp_path / "f.png", X_m, Y_m, T, T, V, V, sample_data(50), X_rar=np.array([[0.5, 0.5]]))
    assert (tmp_path / "f.png").stat().st_size > 0
    csv = tmp_path / "loss.csv"
    csv.write_text(
        "phase,step,total,data,res_data,res_colloc,tv,l2,lambda_data,lambda_res_data,lambda_tv\n"
        "adam,0,1,0.5,0.2,0.2,0.05,0.05,1,1,1\nadam,100,0.5,0.2,0.1,0.1,0.05,0.05,1,1,1\n"
        "lbfgs,1,0.4,0.1,0.1,0.1,0.05,0.05,1,1,1\n"
    )
    plot_loss(tmp_path / "l.png", csv)
    assert (tmp_path / "l.png").stat().st_size > 0

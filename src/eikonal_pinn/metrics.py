"""Error metrics as in Table 1 of the paper: RMSE for T, MAE for V, reported in percent.

Paper p.5/6: RMSE normalised by 1 ms, MAE by 1 m/s, shown in percent (Fig. 6 axes).
So nRMSE_T = 100 * RMSE, nMAE_V = 100 * MAE over the 2500 grid points. Raw values stored too.
"""
from __future__ import annotations

import numpy as np


def compute_metrics(T_pred, V_pred, T_true, V_true) -> dict:
    T_pred = np.asarray(T_pred, float).ravel()
    V_pred = np.asarray(V_pred, float).ravel()
    T_true = np.asarray(T_true, float).ravel()
    V_true = np.asarray(V_true, float).ravel()
    eT = T_pred - T_true
    eV = V_pred - V_true
    rmse_T = float(np.sqrt(np.mean(eT**2)))
    mae_V = float(np.mean(np.abs(eV)))
    return {
        "nrmse_T": 100.0 * rmse_T,
        "nmae_V": 100.0 * mae_V,
        "rmse_T": rmse_T,
        "mae_V": mae_V,
        "maxerr_T": float(np.max(np.abs(eT))),
        "maxerr_V": float(np.max(np.abs(eV))),
    }

import numpy as np

def count_metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    out = {}
    names = ["home", "away"]
    for i, name in enumerate(names):
        err = p[:, i] - y[:, i]
        out[f"{name}_MAE"] = float(np.mean(np.abs(err)))
        out[f"{name}_RMSE"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{name}_WAPE_pct"] = float(100*np.sum(np.abs(err))/max(np.sum(np.abs(y[:, i])), 1e-8))
        out[f"{name}_actual_mean"] = float(np.mean(y[:, i]))
        out[f"{name}_pred_mean"] = float(np.mean(p[:, i]))
    out["joint_MAE"] = float(np.mean(np.abs(p-y)))
    out["joint_RMSE"] = float(np.sqrt(np.mean((p-y)**2)))
    return out

# src/features.py
"""
Feature utilities for the AirTempTS project.

Provides:
 - make_feature_matrix(df, ...)
 - save_features(features_df, out_path)
 - window_cluster_dtw_pipeline(...)  # PCA+KMeans + automatic k selection + LB_Keogh-pruned DTW
 - plot_k_selection_force(...)       # robust plotting helper to visualize k-selection

Notes
-----
- window_cluster_dtw_pipeline uses optional tslearn for DTW. Install with: pip install tslearn
- Required: numpy, pandas, scikit-learn, scipy, joblib
"""
from typing import Optional, Tuple, Dict, Any
import os
import math
import numpy as np
import pandas as pd

# -------------------------
# Basic feature helpers
# -------------------------
def _ensure_month_datetime(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    if month_col not in df.columns:
        raise KeyError(f"Expected a '{month_col}' column (datetime-like) in the DataFrame.")
    if not np.issubdtype(df[month_col].dtype, np.datetime64):
        df = df.copy()
        df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    df = df.sort_values(by=month_col).reset_index(drop=True)
    return df


def _detect_target_col(df: pd.DataFrame, month_col: str = "month") -> str:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in DataFrame to use as target.")
    prefs = ["temperature", "temp", "value", "air_temp", "anomaly", "mean"]
    for p in prefs:
        for c in numeric_cols:
            if p in c.lower():
                return c
    return numeric_cols[0]


def make_feature_matrix(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    month_col: str = "month",
    n_lags: int = 12,
    rolling_windows=(3, 6, 12),
) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_month_datetime(df, month_col=month_col)
    if target_col is None:
        target_col = _detect_target_col(df, month_col=month_col)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    df = df[[month_col, target_col]].rename(columns={target_col: "y"})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(window=w, min_periods=1).mean()
    months = df[month_col].dt.month
    df["month"] = months
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    features_df = df.dropna().reset_index(drop=True)
    cols = ["month", "y"] + [c for c in features_df.columns if c not in ("month", "y")]
    features_df = features_df[cols]
    return features_df


def save_features(features_df: pd.DataFrame, out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    features_df.to_csv(out_path, index=False)
    return out_path


# -------------------------
# New: Combined PCA+KMeans + LB_Keogh-pruned DTW pipeline + helpers
# -------------------------
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except Exception:
    PCA = KMeans = StandardScaler = silhouette_score = None

from scipy.spatial.distance import cdist

# Optional tslearn import for DTW
try:
    from tslearn.metrics import cdist_dtw, dtw
    _TSLEARN_AVAILABLE = True
except Exception:
    try:
        from tslearn.metrics import dtw
        cdist_dtw = None
        _TSLEARN_AVAILABLE = True
    except Exception:
        dtw = None
        cdist_dtw = None
        _TSLEARN_AVAILABLE = False

# joblib for model saving
try:
    import joblib
except Exception:
    joblib = None


def _z_norm_rows(A: np.ndarray) -> np.ndarray:
    B = np.empty_like(A, dtype=float)
    for i in range(A.shape[0]):
        a = A[i].astype(float)
        mu = np.nanmean(a); sd = np.nanstd(a)
        if sd == 0 or np.isnan(sd):
            B[i] = a - mu
        else:
            B[i] = (a - mu) / sd
    return B


def _elbow_k_by_max_distance(k_vals: np.ndarray, inertias: np.ndarray) -> int:
    xs = k_vals.astype(float)
    ys = inertias.astype(float)
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    denom = math.hypot(x2 - x1, y2 - y1)
    if denom == 0:
        return int(k_vals[0])
    distances = []
    for xi, yi in zip(xs, ys):
        num = abs((y2 - y1) * xi - (x2 - x1) * yi + x2*y1 - y2*x1)
        distances.append(num / denom)
    distances = np.array(distances)
    if len(distances) <= 2:
        return int(k_vals[0])
    idx = np.argmax(distances[1:-1]) + 1
    return int(k_vals[idx])


def _auto_value_col(df: pd.DataFrame) -> str:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if c.lower() not in ("idx", "index")]
    if not num:
        raise ValueError("No numeric columns to use as series values.")
    for pref in ("mean_temp", "temperature", "temp", "value", "y", "air_temp"):
        for c in num:
            if pref in c.lower():
                return c
    return num[0]


def _lb_keogh(query: np.ndarray, candidate: np.ndarray, r: int) -> float:
    n = len(candidate)
    lower = np.empty(n)
    upper = np.empty(n)
    for i in range(n):
        start = max(0, i - r)
        end = min(n, i + r + 1)
        window = candidate[start:end]
        lower[i] = np.min(window)
        upper[i] = np.max(window)
    diff = 0.0
    for i, q in enumerate(query):
        if q > upper[i]:
            diff += (q - upper[i]) ** 2
        elif q < lower[i]:
            diff += (q - lower[i]) ** 2
    return math.sqrt(diff)


def window_cluster_dtw_pipeline(
    df: pd.DataFrame,
    value_col: Optional[str] = None,
    month_col: str = "month",
    window: int = 12,
    stride: int = 1,
    normalize_windows: bool = True,
    n_pca_components: int = 6,
    n_clusters: Optional[int] = None,
    k_range: Tuple[int, int] = (2, 8),
    k_select_method: str = "both",          # "silhouette" | "elbow" | "both"
    k_select_sample: Optional[int] = 2000,
    prune_percentile: float = 0.95,
    lb_radius: Optional[int] = None,
    compute_dtw_to_centroids: bool = True,
    compute_dtw_diagnostics: bool = True,
    save_pairwise_matrix: bool = False,
    output_prefix: str = "combined_pca_kmeans_dtw",
    results_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    pca_random_state: int = 42,
    kmeans_random_state: int = 42,
) -> Dict[str, Any]:
    """
    Runs PCA+KMeans on sliding windows, optional auto-selection of k (silhouette/elbow/both),
    then LB_Keogh-pruned DTW distances to cluster centroids and optional DTW diagnostics.

    Saves PCA and KMeans models using joblib to models_dir (default set to the path you requested).

    Returns dict containing:
      - 'clusters_df', 'dtw_df' (or None), 'diag_df' (or None), 'combined_df', 'chosen_k', 'artifacts'
    """
    # runtime check for sklearn availability
    if PCA is None or KMeans is None or StandardScaler is None:
        raise RuntimeError("scikit-learn is required for window_cluster_dtw_pipeline. Install scikit-learn.")

    # default results_dir and models_dir (use exact path user requested)
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # explicit default models dir requested by user:
    default_models_dir = r"E:\AI-Engineering-Capstone-Projects\AirTempTs\PRAICP-1003-AirTempTS\results\models"
    if models_dir is None:
        models_dir = default_models_dir
    os.makedirs(models_dir, exist_ok=True)

    if value_col is None:
        value_col = _auto_value_col(df)
    df_local = df.copy().reset_index(drop=True)
    if month_col in df_local.columns:
        df_local[month_col] = pd.to_datetime(df_local[month_col], errors="coerce")

    series = df_local[value_col].astype(float).values
    n = len(series)
    W = int(window)
    if n < W:
        raise ValueError(f"Series length {n} is shorter than window {W}.")

    # build windows
    starts, ends, windows = [], [], []
    for s in range(0, n - W + 1, stride):
        w = series[s:s + W]
        if np.isnan(w).any():
            continue
        starts.append(s)
        ends.append(s + W - 1)
        windows.append(w.astype(float))
    if len(windows) == 0:
        raise RuntimeError("No valid windows after NaN exclusion.")
    X = np.vstack(windows)
    m = X.shape[0]

    # normalize windows if requested
    X_proc = _z_norm_rows(X) if normalize_windows else X.astype(float)

    # PCA
    scaler = StandardScaler()
    Z = scaler.fit_transform(X_proc)
    nc = min(int(n_pca_components), Z.shape[1])
    pca = PCA(n_components=nc, random_state=int(pca_random_state))
    PC = pca.fit_transform(Z)

    # automatic k selection if needed
    chosen_k = n_clusters
    inertias = []
    sil_scores = []
    k_vals = []
    k_selection_meta = {}
    if chosen_k is None:
        kmin, kmax = k_range
        kmin = max(2, int(kmin))
        kmax = max(kmin, int(kmax))
        k_vals = np.arange(kmin, kmax + 1)
        compute_sil = k_select_method in ("silhouette", "both")
        if compute_sil and k_select_sample is not None and m > k_select_sample:
            rng = np.random.RandomState(int(pca_random_state))
            sample_idx = rng.choice(np.arange(m), size=int(k_select_sample), replace=False)
            PC_sample = PC[sample_idx]
        else:
            PC_sample = PC

        for k_try in k_vals:
            km_try = KMeans(n_clusters=int(k_try), random_state=int(kmeans_random_state), n_init=10)
            km_try.fit(PC)
            inertias.append(km_try.inertia_)
            if compute_sil:
                try:
                    sc = silhouette_score(PC_sample, km_try.predict(PC_sample))
                except Exception:
                    sc = float("nan")
                sil_scores.append(sc)
            else:
                sil_scores.append(float("nan"))

        inertias = np.array(inertias)
        sil_scores = np.array(sil_scores)
        elbow_k = _elbow_k_by_max_distance(k_vals, inertias)
        sil_k = None
        if compute_sil and not np.all(np.isnan(sil_scores)):
            sil_idx = int(np.nanargmax(sil_scores))
            sil_k = int(k_vals[sil_idx])

        if k_select_method == "silhouette":
            chosen_k = sil_k if sil_k is not None else int(elbow_k)
        elif k_select_method == "elbow":
            chosen_k = int(elbow_k)
        else:  # both: prefer silhouette where available
            chosen_k = sil_k if sil_k is not None else int(elbow_k)

        k_selection_meta = {
            "k_vals": k_vals.tolist(),
            "inertias": inertias.tolist(),
            "sil_scores": (np.round(sil_scores, 6).tolist() if len(sil_scores) > 0 else []),
            "elbow_k": int(elbow_k),
            "sil_k": int(sil_k) if sil_k is not None else None,
            "chosen_k": int(chosen_k),
        }
    else:
        k_selection_meta = {"chosen_k": int(chosen_k)}

    # fit final kmeans
    k_final = max(1, int(chosen_k))
    kmeans = KMeans(n_clusters=k_final, random_state=int(kmeans_random_state), n_init=10)
    labels = kmeans.fit_predict(PC)
    centers_pca = kmeans.cluster_centers_
    dist_to_centers_pca = cdist(PC, centers_pca, metric="euclidean")

    # compute final silhouette score for reporting (only valid if k>1)
    final_silhouette = float("nan")
    try:
        if k_final > 1:
            final_silhouette = float(silhouette_score(PC, labels))
    except Exception:
        final_silhouette = float("nan")

    # final inertia
    final_inertia = float(kmeans.inertia_)

    # Print diagnostics to terminal
    print(f"[window_cluster_dtw_pipeline] chosen_k = {k_final}")
    print(f"[window_cluster_dtw_pipeline] final inertia = {final_inertia:.6f}")
    print(f"[window_cluster_dtw_pipeline] silhouette score = {final_silhouette if not np.isnan(final_silhouette) else 'NaN'}")

    clusters_df = pd.DataFrame({
        "start_idx": starts,
        "end_idx": ends,
        "cluster_id": labels,
    })
    if month_col in df_local.columns:
        clusters_df["start_month"] = df_local.loc[clusters_df["start_idx"], month_col].values
    for j in range(k_final):
        clusters_df[f"dist_to_centroid_pca_{j}"] = dist_to_centers_pca[:, j]
    clusters_out = os.path.join(results_dir, f"{output_prefix}_clusters.csv")
    clusters_df.to_csv(clusters_out, index=False)

    # save PCA and KMeans models to models_dir using joblib
    pca_model_path = os.path.join(models_dir, f"pca_{output_prefix}.joblib")
    kmeans_model_path = os.path.join(models_dir, f"kmeans_{output_prefix}.joblib")
    if joblib is None:
        print("[warn] joblib not available — skipping model save. Install joblib to enable model saving.")
        pca_model_path = None
        kmeans_model_path = None
    else:
        try:
            joblib.dump(pca, pca_model_path)
            joblib.dump(kmeans, kmeans_model_path)
            print(f"[saved] PCA model -> {pca_model_path}")
            print(f"[saved] KMeans model -> {kmeans_model_path}")
        except Exception as e:
            print("[warn] Failed to save models via joblib:", e)
            pca_model_path = None
            kmeans_model_path = None

    # reconstruct centroids in original window space
    centers_scaled = pca.inverse_transform(centers_pca)
    centers_orig = scaler.inverse_transform(centers_scaled)
    if normalize_windows:
        centroids_z = np.empty_like(centers_orig, dtype=float)
        for j in range(centers_orig.shape[0]):
            a = centers_orig[j]
            mu = a.mean(); sd = a.std()
            centroids_z[j] = (a - mu) / (sd if sd > 0 else 1.0)
    else:
        centroids_z = centers_orig.astype(float)

    # LB radius
    r = lb_radius if lb_radius is not None else max(1, W // 4)

    # DTW to centroids with LB pruning
    dtw_df = None
    if compute_dtw_to_centroids:
        dtw_to_centroids = np.full((m, centroids_z.shape[0]), np.nan, dtype=float)
        lb_to_centroids = np.full((m, centroids_z.shape[0]), np.nan, dtype=float)
        exact_flag = np.zeros((m, centroids_z.shape[0]), dtype=bool)

        for j in range(centroids_z.shape[0]):
            cent = centroids_z[j]
            lbs = np.array([_lb_keogh(X_proc[i], cent, r) for i in range(m)], dtype=float)
            lb_to_centroids[:, j] = lbs
            thr = np.percentile(lbs, prune_percentile * 100.0) if prune_percentile < 1.0 else np.inf
            idxs = np.where(lbs <= thr)[0]

            if len(idxs) > 0:
                if _TSLEARN_AVAILABLE and cdist_dtw is not None:
                    try:
                        sel = X_proc[idxs]
                        dists = cdist_dtw(sel, cent.reshape(1, -1), sakoe_chiba_radius=r).ravel()
                        dtw_to_centroids[idxs, j] = dists
                        exact_flag[idxs, j] = True
                    except Exception:
                        for ii in idxs:
                            dtw_to_centroids[ii, j] = dtw(X_proc[ii], cent)
                            exact_flag[ii, j] = True
                else:
                    if dtw is None:
                        raise RuntimeError("tslearn required for DTW distance computations. Install tslearn.")
                    for ii in idxs:
                        dtw_to_centroids[ii, j] = dtw(X_proc[ii], cent)
                        exact_flag[ii, j] = True

            pruned_mask = ~exact_flag[:, j]
            dtw_to_centroids[pruned_mask, j] = lb_to_centroids[pruned_mask, j]

        # optional pairwise (careful)
        if save_pairwise_matrix and _TSLEARN_AVAILABLE and cdist_dtw is not None:
            try:
                pairwise = cdist_dtw(X_proc, X_proc, sakoe_chiba_radius=r)
                np.save(os.path.join(results_dir, f"{output_prefix}_pairwise.npy"), pairwise)
            except Exception:
                pass

        # build dtw_df
        rows = []
        for i, s in enumerate(starts):
            row = {"start_idx": int(s), "end_idx": int(ends[i])}
            if month_col in df_local.columns:
                row["start_month"] = df_local.loc[s, month_col]
            for j in range(centroids_z.shape[0]):
                row[f"lb_centroid_{j}"] = float(lb_to_centroids[i, j])
                row[f"dtw_centroid_{j}"] = float(dtw_to_centroids[i, j])
                row[f"dtw_exact_flag_centroid_{j}"] = bool(exact_flag[i, j])
            rows.append(row)
        dtw_df = pd.DataFrame(rows)
        dtw_out = os.path.join(results_dir, f"{output_prefix}_dtw_centroids.csv")
        dtw_df.to_csv(dtw_out, index=False)
    else:
        dtw_out = None

    # optional dtw diagnostics
    diag_df = None
    if compute_dtw_diagnostics:
        if dtw is None and not _TSLEARN_AVAILABLE:
            diag_df = None
        else:
            mean_template = np.nanmean(X_proc, axis=0)
            median_template = np.nanmedian(X_proc, axis=0)
            dtw_to_mean = np.full(m, np.nan, dtype=float)
            dtw_to_median = np.full(m, np.nan, dtype=float)
            dtw_to_prev = np.full(m, np.nan, dtype=float)
            if _TSLEARN_AVAILABLE and cdist_dtw is not None:
                try:
                    dtw_to_mean = cdist_dtw(X_proc, mean_template.reshape(1, -1), sakoe_chiba_radius=r).ravel()
                    dtw_to_median = cdist_dtw(X_proc, median_template.reshape(1, -1), sakoe_chiba_radius=r).ravel()
                except Exception:
                    for i in range(m):
                        dtw_to_mean[i] = dtw(X_proc[i], mean_template)
                        dtw_to_median[i] = dtw(X_proc[i], median_template)
            else:
                for i in range(m):
                    dtw_to_mean[i] = dtw(X_proc[i], mean_template)
                    dtw_to_median[i] = dtw(X_proc[i], median_template)
            if dtw is not None:
                dtw_to_prev[0] = np.nan
                for i in range(1, m):
                    dtw_to_prev[i] = dtw(X_proc[i], X_proc[i-1])
            diag_df = pd.DataFrame({
                "start_idx": starts,
                "end_idx": ends,
                "start_month": [df_local.loc[s, month_col] if month_col in df_local.columns else s for s in starts],
                "dtw_to_mean": dtw_to_mean,
                "dtw_to_median": dtw_to_median,
                "dtw_to_prev": dtw_to_prev
            })
            diag_out = os.path.join(results_dir, f"{output_prefix}_dtw_diagnostics.csv")
            diag_df.to_csv(diag_out, index=False)

    # combine outputs
    combined_df = clusters_df.copy().reset_index(drop=True)
    if dtw_df is not None:
        cols_to_merge = [c for c in dtw_df.columns if c not in ("start_idx", "end_idx", "start_month")]
        combined_df = combined_df.merge(dtw_df[["start_idx"] + cols_to_merge], on="start_idx", how="left")
    if diag_df is not None:
        diag_cols = [c for c in diag_df.columns if c not in ("start_idx", "end_idx", "start_month")]
        combined_df = combined_df.merge(diag_df[["start_idx"] + diag_cols], on="start_idx", how="left")

    combined_out = os.path.join(results_dir, f"{output_prefix}_combined_features.csv")
    combined_df.to_csv(combined_out, index=False)

    artifacts = {
        "clusters_csv": clusters_out,
        "dtw_csv": dtw_out,
        "diagnostics_csv": (diag_out if compute_dtw_diagnostics and diag_df is not None else None),
        "combined_csv": combined_out,
        "pca_model": pca_model_path,
        "kmeans_model": kmeans_model_path,
    }
    if 'k_selection_meta' in locals():
        artifacts['k_selection_meta'] = k_selection_meta

    return {
        "clusters_df": clusters_df,
        "dtw_df": dtw_df,
        "diag_df": diag_df,
        "combined_df": combined_df,
        "chosen_k": int(k_final),
        "artifacts": artifacts,
    }


# -------------------------
# Robust plotting helper (force show across environments)
# -------------------------
def plot_k_selection_force(k_vals, inertias, sil_scores=None, title="K selection (inertia + silhouette)"):
    try:
        import matplotlib.pyplot as plt
        import tempfile
        from IPython.display import display, Image
    except Exception:
        raise RuntimeError("matplotlib and IPython required for plot_k_selection_force")

    k_vals = np.array(list(k_vals))
    inertias = np.array(list(inertias), dtype=float)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(k_vals, inertias, marker='o', linestyle='-', label='Inertia (WSS)')
    ax1.set_xlabel('k (clusters)')
    ax1.set_ylabel('Inertia (WSS)')
    ax1.grid(True, linestyle=':', linewidth=0.5)
    ax1.tick_params(axis='y')

    ax2 = None
    if sil_scores is not None:
        sil_scores = np.array(list(sil_scores), dtype=float)
        ax2 = ax1.twinx()
        ax2.plot(k_vals, sil_scores, marker='s', linestyle='--', label='Silhouette')
        ax2.set_ylabel('Silhouette score')
        ax2.tick_params(axis='y')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
    else:
        ax1.legend(loc='best')

    ax1.set_title(title)
    plt.tight_layout()

    try:
        plt.show()
    except Exception:
        pass

    try:
        tmpdir = tempfile.gettempdir()
        fname = os.path.join(tmpdir, f"k_selection_plot_{np.random.randint(1e9)}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        display(Image(filename=fname))
        try:
            os.remove(fname)
        except Exception:
            pass
    except Exception:
        pass

    return fig, ax1, ax2


# quick demonstration when run directly
if __name__ == "__main__":
    rng = pd.date_range("2000-01-01", periods=60, freq="MS")
    vals = np.sin(np.linspace(0, 6 * np.pi, len(rng))) + np.linspace(0, 1, len(rng)) + np.random.normal(0, 0.1, len(rng))
    demo_df = pd.DataFrame({"month": rng, "temperature": vals})

    print("Running make_feature_matrix demo:")
    feats = make_feature_matrix(demo_df, target_col="temperature", n_lags=6)
    print(feats.head())

    try:
        print("\nRunning window_cluster_dtw_pipeline demo:")
        res = window_cluster_dtw_pipeline(demo_df.rename(columns={"temperature": "temp"}),
                                          value_col="temp",
                                          window=12,
                                          stride=1,
                                          n_pca_components=4,
                                          n_clusters=None,
                                          k_range=(2,5),
                                          k_select_method="both",
                                          prune_percentile=0.9,
                                          compute_dtw_to_centroids=False,  # skip DTW in quick demo
                                          results_dir=os.path.join(os.getcwd(), "results_demo"))
        print("Chosen k:", res["chosen_k"])
        print("Cluster head:")
        print(res["clusters_df"].head())
    except Exception as demo_e:
        print("Pipeline demo skipped (sklearn/tslearn/joblib may be missing):", demo_e)

"""Fixed-effects OLS with clustered SEs, and a one-line CI summary."""
import numpy as np
import pandas as pd


def fe_ols(d, y, xs, fe="player", cluster="player", controls=None):
    """OLS with one set of fixed effects (by demeaning), SEs clustered by `cluster`.

    Returns a DataFrame: coef, se, t, n. Controls are categorical columns turned into dummies."""
    use = [y] + xs + ([fe] if fe else []) + ([cluster] if cluster and cluster != fe else []) + (controls or [])
    z = d[list(dict.fromkeys(use))].replace([np.inf, -np.inf], np.nan).dropna()
    X = z[xs].astype("float64")
    if controls:
        dm = pd.get_dummies(z[controls].astype(str), drop_first=True, dtype="float64")
        if dm.shape[1]:
            X = pd.concat([X, dm], axis=1)
    Y = z[y].astype("float64")
    if fe:
        g = z[fe]
        X = X - X.groupby(g, observed=True).transform("mean")
        Y = Y - Y.groupby(g, observed=True).transform("mean")
    X = X.loc[:, X.std() > 0]
    Xv, Yv = X.values, Y.values
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    b = XtX_inv @ Xv.T @ Yv
    e = Yv - Xv @ b
    cl = z[cluster].cat.codes.values if hasattr(z[cluster], "cat") else pd.factorize(z[cluster])[0]
    G = cl.max() + 1
    S = np.zeros((G, Xv.shape[1]))
    np.add.at(S, cl, Xv * e[:, None])
    k = Xv.shape[1]
    n = len(Yv)
    adj = G / (G - 1) * (n - 1) / max(n - k, 1)
    V = adj * XtX_inv @ (S.T @ S) @ XtX_inv
    se = np.sqrt(np.diag(V))
    out = pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=X.columns)
    out["n"] = n
    return out.loc[[c for c in xs if c in out.index]]


def ci(r, name, scale=100, unit="pts"):
    """One-line summary: coefficient and 95% CI, scaled (default: proportions -> percentage points)."""
    c, s = r.loc[name, "coef"] * scale, r.loc[name, "se"] * scale
    return f"{c:+.2f} {unit} [{c - 1.96 * s:+.2f}, {c + 1.96 * s:+.2f}] (t={r.loc[name, 't']:.1f}, n={int(r.loc[name, 'n']):,})"

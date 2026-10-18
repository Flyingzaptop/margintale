"""Learn when the signal buys, then test that rule on years it never traded.

Positives: first entries of the signal's buy baskets. Negatives: random minutes of
its period. Features are scale-free (ATR units, range positions, hour). Model:
gradient boosting, validated on the last 30% of its period. Then every minute of
2017-2024 is scored; the top-scored minutes become buy entries (one position at a
time) with ATR exits (TP 2 ATR, SL 4 ATR, 1 day max, ~ its TP $10 / SL $20), and
are compared with random buys and with buys at the lowest-scored minutes.

    python3 signal_model.py
"""
import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from engine import load_m1

COST = 0.37


def feature_frame(m):
    c, h, l, a = m.close, m.high, m.low, m.atr
    f = {}
    for n in (5, 15, 60, 240, 1440, 7200):
        f[f"ход_{n}"] = (c - c.shift(n)) / a
    for n in (15, 60, 240, 1440):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"место_{n}"] = (c - lo) / (hi - lo).clip(lower=1e-9)
    for n in (240, 1440, 7200):
        f[f"от_EMA_{n}"] = (c - c.ewm(span=n).mean()) / a
    path = c.diff().abs()
    for n in (60, 240, 1440):
        f[f"ER_{n}"] = (c - c.shift(n)).abs() / path.rolling(n).sum()
    f["волат"] = a / a.rolling(28800, min_periods=1440).mean()
    f["час"] = m.index.hour
    f["день_нед"] = m.index.dayofweek
    return pd.DataFrame(f, index=m.index).shift(1)  # only what is known before the entry bar


@njit(cache=True)
def exits_atr(h, l, c, atr, i0, tp_k, sl_k, hold, cost):
    """Long trades from bars i0, one at a time (skip entries while a trade is open)."""
    out = np.full(len(i0), np.nan)
    busy = -1
    for k in range(len(i0)):
        if i0[k] <= busy:
            continue
        e = c[i0[k]]
        tp, sl = tp_k * atr[i0[k]], sl_k * atr[i0[k]]
        r = 0.0
        last = min(i0[k] + hold, len(c) - 1)
        j = last
        for i in range(i0[k] + 1, last + 1):
            if l[i] <= e - sl:
                r, j = -sl, i
                break
            if h[i] >= e + tp:
                r, j = tp, i
                break
            r = c[i] - e
        out[k] = r - cost
        busy = j
    return out


if __name__ == "__main__":
    m = load_m1()
    F = feature_frame(m)
    t = pd.read_csv("../results/signal_trades.csv", parse_dates=["open_utc", "t_close"])
    t["key"] = t.t_close.astype(str) + t.type
    first = t.sort_values("open_utc").groupby("key").head(1)
    buys = first[first.side > 0]
    pos = m.index.get_indexer(pd.DatetimeIndex(buys.open_utc).as_unit("ms"), method="bfill")
    pos = pos[pos > 0]
    a, b = pos.min(), pos.max()
    rng = np.random.default_rng(0)
    neg = rng.integers(a, b, 30000)
    X = pd.concat([F.iloc[pos], F.iloc[neg]])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    tpos = np.r_[pos, neg]
    ok = X.notna().all(axis=1).to_numpy()
    X, y, tpos = X[ok], y[ok], tpos[ok]
    cut = a + 0.7 * (b - a)
    tr, te = tpos < cut, tpos >= cut
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=0)
    clf.fit(X[tr], y[tr])
    auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
    print(f"Покупок-входов: {len(pos)}, случайных минут: {len(neg)}. AUC на последних 30% его периода: {auc:.3f}")
    imp = permutation_importance(clf, X[te], y[te], n_repeats=5, random_state=0, scoring="roc_auc")
    order = np.argsort(-imp.importances_mean)
    print("\nВажность признаков (падение AUC при перемешивании):")
    for i in order[:10]:
        col = X.columns[i]
        print(f"  {col:<14} {imp.importances_mean[i]:.3f}   его медиана {X[col][y == 1].median():+.2f}, "
              f"случайно {X[col][y == 0].median():+.2f}")
    clf.fit(X, y)

    h, l, c, at = (m[x].to_numpy() for x in ("high", "low", "close", "atr"))
    print("\nПравило на годах без сигнала: покупка в минуты с высокой оценкой модели, TP 2 ATR / SL 4 ATR / 1 день,"
          " одна позиция за раз, издержки $0.37; $ на унцию за сделку")
    rows = []
    for y0 in range(2017, 2025):
        i_a, i_b = m.index.searchsorted(pd.Timestamp(f"{y0}-01-01", tz="UTC")), \
            m.index.searchsorted(pd.Timestamp(f"{y0 + 1}-01-01", tz="UTC"))
        Fy = F.iloc[i_a:i_b]
        okm = Fy.notna().all(axis=1).to_numpy()
        idx = np.arange(i_a, i_b)[okm]
        s = clf.predict_proba(Fy[okm])[:, 1]
        row = {"год": y0}
        for name, sel in (("топ 0.5%", s >= np.quantile(s, 0.995)), ("топ 2%", s >= np.quantile(s, 0.98)),
                          ("случайно", rng.random(len(s)) < 0.02), ("низ 2%", s <= np.quantile(s, 0.02))):
            r = exits_atr(h, l, c, at, idx[sel].astype(np.int64), 2.0, 4.0, 1440, COST)
            r = r[~np.isnan(r)]
            row[f"{name}: ср"] = r.mean()
            row[f"{name}: n"] = len(r)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("год")
    pd.set_option("display.width", 250)
    print(df.round(2).to_string())
    print("\nВ среднем по годам:", df[[c for c in df.columns if c.endswith(': ср')]].mean().round(2).to_dict())

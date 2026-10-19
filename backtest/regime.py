"""When to launch a session entity: market-state features at the start vs P(x2 within a year).

For each candidate (evo3_check.CANDIDATES) every daily start 2017-01..2025-09 is run;
features use only data before the start. Then:
  * P(x2) by feature quintile;
  * leave-one-year-out launch rule: for each year, the threshold rule ("launch only if
    feature >= / <= q") is chosen on the other years (best P(x2) with at least 20% of
    days launched) and applied to the held-out year.

    python3 regime.py
"""
import numpy as np
import pandas as pd

from engine import load_m1
from engine3 import Genes, Market3, SPREAD
from evo3_check import CANDIDATES, STARTS, best_row

DAYS = 365


def daily_features(bars):
    d = bars.resample("D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    rng = d.high - d.low
    lon_ny = bars[(bars.index.hour >= 12) & (bars.index.hour < 16)]
    s = lon_ny.resample("D").agg({"high": "max", "low": "min"}).dropna()
    srng = (s.high - s.low).reindex(d.index)
    atr = bars.atr.resample("D").last().reindex(d.index)
    f = pd.DataFrame({
        "ATR_$": atr,
        "ATR_%": atr / d.close * 100,
        "спред/ATR": SPREAD / atr,
        "диапазон_20д/250д": rng.rolling(20).mean() / rng.rolling(250).mean(),
        "диапазон_дня_%": rng.rolling(20).mean() / d.close * 100,
        "тренд_20д_%": (d.close / d.close.shift(20) - 1).abs() * 100,
        "ER_20д": (d.close - d.close.shift(20)).abs() / d.close.diff().abs().rolling(20).sum(),
        "сессия_LonNY/диапазон_дня": (srng / rng).rolling(20, min_periods=10).mean(),
    }, index=d.index)
    return f.shift(1)  # known before the start day


def outcomes(m, g):
    eq, _, _ = m.batch([g], STARTS, DAYS)
    x = eq[0] / g.deposit
    hit = x >= 2 * 0.999
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, DAYS + 1)
    return pd.DataFrame({"hit": first <= DAYS, "days": first, "value": np.where(first <= DAYS, 2.0, x[:, -1])},
                        index=STARTS)


def loyo(df, feat):
    rows = []
    years = sorted(df.index.year.unique())
    for y in years:
        tr, te = df[df.index.year != y], df[df.index.year == y]
        best = None
        for q in np.linspace(0.1, 0.8, 15):
            for side in (1, -1):
                thr = tr[feat].quantile(q if side > 0 else 1 - q)
                sel = tr[feat] >= thr if side > 0 else tr[feat] <= thr
                if sel.mean() < 0.2:
                    continue
                p = tr.hit[sel].mean()
                if best is None or p > best[0]:
                    best = (p, side, thr)
        _, side, thr = best
        sel = te[feat] >= thr if side > 0 else te[feat] <= thr
        rows.append(dict(год=y, правило=f"{'≥' if side > 0 else '≤'} {thr:.3g}", запусков=sel.mean(),
                         P_с_правилом=te.hit[sel].mean() if sel.any() else np.nan, P_без=te.hit.mean(),
                         итог_с_правилом=te.value[sel].mean() if sel.any() else np.nan, итог_без=te.value.mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fin = pd.read_csv("../results/evo3_s1_final.csv")
    fields = list(Genes.__dataclass_fields__)
    m = Market3()
    feats = daily_features(m.bars)
    pd.set_option("display.width", 250)
    for name, flt in CANDIDATES.items():
        r = best_row(fin, **flt)
        g = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
        out = outcomes(m, g)
        fi = feats.copy()
        fi.index = fi.index.tz_convert("UTC").as_unit("ns")
        key = out.index.floor("D").as_unit("ns")
        df = pd.concat([out, fi.reindex(key).set_axis(out.index)], axis=1).dropna()
        print(f"\n=== {name}: стартов {len(df)}, P(x2 за год) {df.hit.mean():.0%}")
        print("P(x2) по квинтилям признака (1 = низкие значения … 5 = высокие):")
        tab = {}
        for f in feats.columns:
            qn = pd.qcut(df[f], 5, labels=False, duplicates="drop")
            tab[f] = df.hit.groupby(qn).mean()
        print(pd.DataFrame(tab).T.map(lambda v: f"{v:.0%}").to_string())
        corr = {f: df[f].corr(df.hit.astype(float), method="spearman") for f in feats.columns}
        top = sorted(corr, key=lambda f: -abs(corr[f]))[:3]
        for f in top:
            res = loyo(df, f)
            w = res.запусков
            print(f"\nПравило по «{f}» (порог подобран без проверяемого года):")
            print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
            print(f"  итого: запусков {np.average(w):.0%} дней, P(x2) с правилом "
                  f"{np.nansum(res.P_с_правилом * w) / w.sum():.0%} против {res.P_без.mean():.0%} без правила; "
                  f"средний итог ×{np.nansum(res.итог_с_правилом * w) / w.sum():.2f} против ×{res.итог_без.mean():.2f}")

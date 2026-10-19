"""evo4 winners: launch rule (shadow), daily starts 2017..2025-09. python3 evo4_shadow.py"""
import sys
import numpy as np, pandas as pd
from engine3 import Genes, Market3
from evo3_check import DAYS, STARTS
from evolve import OUT
fields = list(Genes.__dataclass_fields__)
m = Market3()
for cut in ("202310", "202410"):
    r = pd.read_csv(OUT / f"evo4_{cut}_s1.csv").sort_values(["t_p", "score"], ascending=False).iloc[0]
    g = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
    eq, _, _ = m.batch([g], STARTS, DAYS)
    x = eq[0] / 100
    got = (x >= 1.998).any(1)
    val = np.where(got, 2.0, x[:, -1])
    print("срез", cut, "без правила", f"{got.mean():.0%} ×{val.mean():.2f}")
    for k in (30, 60, 90):
        sh = np.full(len(STARTS), np.nan)
        sh[k:] = x[:-k, k - 1]       # shadow started k days earlier, its equity now
        for thr in (1.0, 1.2, 1.5, 1.998):
            s = sh >= thr
            yrs = pd.Series(got[s], index=STARTS.year[s]).groupby(level=0).mean()
            print(f"  тень {k}д назад ≥×{thr:.2f}: запусков {s.mean():.0%}, P(x2) {got[s].mean():.0%}, итог ×{val[s].mean():.2f}, при неудаче ×{val[s & ~got].mean():.2f}; "
                  + " ".join(f"{y}:{v:.0%}({(STARTS.year[s]==y).sum()})" for y, v in yrs.items()))

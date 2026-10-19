"""evo4 winners: launch rule (trendrule), daily starts 2017..2025-09. python3 evo4_trendrule.py"""
import sys
import numpy as np, pandas as pd
from engine3 import Genes, Market3
from evo3_check import DAYS, STARTS
from evolve import OUT
fields = list(Genes.__dataclass_fields__)
m = Market3()
d = m.bars.close.resample("D").last().dropna()
d.index = d.index.as_unit("ns")
for cut in ("202310", "202410"):
    r = pd.read_csv(OUT / f"evo4_{cut}_s1.csv").sort_values(["t_p", "score"], ascending=False).iloc[0]
    g = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
    eq, _, _ = m.batch([g], STARTS, DAYS)
    got = (eq[0] / 100 >= 1.998).any(1)
    val = np.where(got, 2.0, eq[0][:, -1] / 100)
    key = STARTS.floor("D").as_unit("ns")
    print("срез", cut)
    for n in (60, 120, 250):
        ret = (d / d.shift(n) - 1).shift(1).reindex(key).to_numpy()
        for thr in (0.0, 0.05, 0.10, 0.15):
            s = ret > thr
            yrs = pd.Series(got[s], index=STARTS.year[s]).groupby(level=0).mean()
            print(f"  рост за {n} дн > {thr:.0%}: запусков {s.mean():.0%}, P(x2) {got[s].mean():.0%}, итог ×{val[s].mean():.2f}; "
                  + " ".join(f"{y}:{v:.0%}({(STARTS.year[s]==y).sum()})" for y, v in yrs.items()))

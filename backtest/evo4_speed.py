"""Speed vs survival for the evo4 winners: the same entry, different money management
(risk per trade, multiplier after a win), daily starts; P(x2 before death) within 20/60/250 days,
median days to x2, deaths before x2.

    python3 evo4_speed.py
"""
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from engine3 import Genes, Market3
from evolve import OUT

STARTS = pd.date_range("2017-01-02", "2025-09-01", freq="D", tz="UTC")
LATE = pd.date_range("2025-09-02", "2026-08-01", freq="D", tz="UTC")

if __name__ == "__main__":
    fields = list(Genes.__dataclass_fields__)
    m = Market3()
    rows = []
    for ent, cut in (("A", "202410"), ("B", "202310")):
        r = pd.read_csv(OUT / f"evo4_{cut}_s1.csv").sort_values(["t_p", "score"], ascending=False).iloc[0]
        g0 = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
        risks = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
        gs = [replace(g0, risk=rk, win_mult=wm, mult_cap=cap, target=2.0)
              for rk, (wm, cap) in product(risks, ((1.0, 1.0), (1.5, 2.0), (2.0, 2.0), (2.0, 4.0)))]
        for nm, starts in (("2017-2025", STARTS), ("late", LATE)):
            eq, st, _ = m.batch(gs, starts, 365)
            for g, e, s in zip(gs, eq, st):
                x = e / 100
                hit = x >= 1.998
                f = np.where(hit.any(1), hit.argmax(1) + 1, 999)
                y = starts.year
                for grp, sel in ((nm, np.ones(len(starts), bool)),) + ((("2023-2025", y >= 2023),) if nm != "late" else ()):
                    ff, ss = f[sel], s[sel]
                    rows.append(dict(особь=ent, старты=grp, риск=g.risk, win_mult=g.win_mult, cap=g.mult_cap,
                                     p20=(ff <= 28).mean(), p60=(ff <= 85).mean(), p365=(ff <= 365).mean(),
                                     дней_мед=np.median(ff[ff <= 365]) if (ff <= 365).any() else np.nan,
                                     смерть_до_x2=((ss >= 2) & (ff > 365)).mean(),
                                     жива_ниже_x2=((ss == 0) & (ff > 365)).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "evo4_speed.csv", index=False)
    pd.set_option("display.width", 250)
    for (e, gname), d in df.groupby(["особь", "старты"], sort=False):
        print(f"\n=== {e}, старты {gname}  (p20/p60/p365 = ×2 раньше смерти за 20/60/250 торговых дней)")
        print(d.drop(columns=["особь", "старты"]).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

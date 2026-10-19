"""Daily starts 2017..2025-09 for the evo4 winners (best on the walk-forward test of each cut):
P(x2 within a year), days to x2 and what is left on failure, by start year; real and mirrored price.

    python3 evo4_check.py
"""
import numpy as np
import pandas as pd

from engine3 import Genes, Market3
from evo3_check import DAYS, STARTS
from evolve import OUT, mirrored

if __name__ == "__main__":
    fields = list(Genes.__dataclass_fields__)
    real, mirror = Market3(), Market3(mirrored())
    for cut in ("202310", "202410"):
        d = pd.read_csv(OUT / f"evo4_{cut}_s1.csv")
        r = d.sort_values(["t_p", "score"], ascending=False).iloc[0]
        g = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
        print(f"\n=== срез {cut}: {g}")
        for mname, m in (("настоящая", real), ("зеркальная", mirror)):
            eq, st, _ = m.batch([g], STARTS, DAYS)
            x = eq[0] / g.deposit
            hit = x >= 2 * 0.999
            first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, DAYS + 1)
            got = first <= DAYS
            val = np.where(got, 2.0, x[:, -1])
            t = pd.DataFrame({"y": STARTS.year, "got": got, "days": np.where(got, first, np.nan), "val": val})
            print(f"  {mname}: x2 за год {got.mean():.0%}, медиана {np.nanmedian(t.days):.0f} дн, "
                  f"средний итог ×{val.mean():.2f}, остаток при неудаче ×{val[~got].mean():.2f}")
            print("    по году старта: " + ", ".join(f"{y}: {s.got.mean():.0%}/{np.nanmedian(s.days):.0f}д/×{s.val.mean():.2f}"
                                               for y, s in t.groupby("y")))
            if mname == "настоящая":
                mo = t.assign(m=STARTS.to_period("M")).groupby("m").got.mean()
                print("    по месяцам старта 2023-2025: " + " ".join(f"{p.strftime('%y%m')}:{v:.0%}"
                                                              for p, v in mo[mo.index.year >= 2023].items()))

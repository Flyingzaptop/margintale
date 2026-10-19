"""Daily starts 2017..2025-09 for chosen evo3 genomes: P(x2 within a year) and time by start year,
on the real and the mirrored price.

    python3 evo3_check.py
"""
import numpy as np
import pandas as pd

from engine3 import Genes, Market3
from evolve import mirrored

STARTS = pd.date_range("2017-01-02", "2025-09-01", freq="D", tz="UTC")
DAYS = 365
CANDIDATES = {
    "A: весь день, пробой вчерашнего дня по тренду": dict(sess=7, entry=2, dirs=0, trend=1),
    "B: пересечение Лондон-NY, отскок от экстремумов сессии": dict(sess=3, entry=5, dirs=0, trend=0),
}


def best_row(df, **flt):
    d = df
    for k, v in flt.items():
        d = d[d[k] == v]
    return d.sort_values("score", ascending=False).iloc[0]


if __name__ == "__main__":
    fin = pd.read_csv("../results/evo3_s1_final.csv")
    fields = list(Genes.__dataclass_fields__)
    real, mirror = Market3(), Market3(mirrored())
    for name, flt in CANDIDATES.items():
        r = best_row(fin, **flt)
        g = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
        print(f"\n=== {name}\n{g}")
        for mname, m in (("настоящая", real), ("зеркальная", mirror)):
            eq, st, _ = m.batch([g], STARTS, DAYS)
            x = eq[0] / g.deposit
            hit = x >= 2 * 0.999
            first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, DAYS + 1)
            got = first <= DAYS
            val = np.where(got, 2.0, x[:, -1])
            d = pd.DataFrame({"y": STARTS.year, "got": got, "days": np.where(got, first, np.nan), "val": val})
            print(f"  {mname}: x2 за год {got.mean():.0%}, за 30 дн {(first <= 30).mean():.0%}, за 90 дн "
                  f"{(first <= 90).mean():.0%}, медиана {np.nanmedian(d.days):.0f} дн, средний итог ×{val.mean():.2f}, "
                  f"остаток при неудаче ×{val[~got].mean():.2f}")
            print("    по году старта: " + ", ".join(f"{y}: {s.got.mean():.0%}/{np.nanmedian(s.days):.0f}д/×{s.val.mean():.2f}"
                                               for y, s in d.groupby("y")))

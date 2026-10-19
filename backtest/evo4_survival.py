"""Survival of the evo4 winners, daily starts.
(1) as traded (stop at x2): reached x2 / controlled death (kill 90% from peak) / broke / alive below x2
    after a year, and the deepest end-of-day equity on the way to x2;
(2) never stopped (target 1e9): share alive after 30..730 days and the multiple of the alive ones.

    python3 evo4_survival.py
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from engine3 import Genes, Market3
from evolve import OUT

STARTS = pd.date_range("2017-01-02", "2025-09-01", freq="D", tz="UTC")
LONG = pd.date_range("2017-01-02", "2024-08-31", freq="D", tz="UTC")


def groups(starts):
    y = starts.year
    return (("2017-2025", np.ones(len(starts), bool)), ("2017-2022", y <= 2022), ("2023-2025", y >= 2023))


if __name__ == "__main__":
    fields = list(Genes.__dataclass_fields__)
    m = Market3()
    for cut in ("202310", "202410"):
        r = pd.read_csv(OUT / f"evo4_{cut}_s1.csv").sort_values(["t_p", "score"], ascending=False).iloc[0]
        g = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
        print(f"\n=== особь среза {cut} (риск {g.risk:.0%}, {'только лонг' if g.dirs == 1 else 'лонг и шорт'})")
        eq, st, _ = m.batch([g], STARTS, 365)
        x, st = eq[0] / 100, st[0]
        hit = x >= 1.998
        first = np.where(hit.any(1), hit.argmax(1), 365)
        low = np.array([x[i, :first[i] + 1].min() for i in range(len(x))])
        print("(1) как торгуем — стоп на ×2, за год:")
        for name, s in groups(STARTS):
            ok = hit.any(1)[s]
            print(f"  {name}: ×2 {ok.mean():.0%} | смерть (−90% от пика) {((st[s] == 2) & ~ok).mean():.0%} | "
                  f"не может торговать {((st[s] == 3) & ~ok).mean():.0%} | жива ниже ×2 {((st[s] == 0) & ~ok).mean():.0%}"
                  f" | у дошедших до ×2 дно по пути: медиана ×{np.median(low[s][ok]):.2f}, 10% худших ≤×{np.quantile(low[s][ok], .1):.2f}")
        eq, st, te = m.batch([replace(g, target=1e9)], LONG, 730)
        x = eq[0] / 100
        dead = st[0] >= 2
        dday = np.full(len(LONG), 10 ** 6)
        # day of death = first day after which equity stops changing
        for i in np.where(dead)[0]:
            row = x[i]
            dday[i] = int(np.argmax(np.isclose(row, row[-1]))) + 1
        print("(2) без остановки на ×2 (живёт до смерти), старты 2017-01..2024-08:")
        for name, s in groups(LONG):
            parts = []
            for d in (30, 90, 180, 365, 730):
                alive = dday[s] > d
                parts.append(f"{d}д: жива {alive.mean():.0%} (медиана ×{np.median(x[s][alive, d - 1]):.1f})" if alive.any()
                             else f"{d}д: жива 0%")
            print(f"  {name}: " + " | ".join(parts))

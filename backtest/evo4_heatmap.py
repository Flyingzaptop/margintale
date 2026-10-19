"""Data for the launch heat maps of the evo4 winners: for every daily start 2017-01..2025-09 the
entity runs without a target (lives until death or the end of data, max 730 days); recorded:
day of death (kill 90% from peak / broke) and first day the end-of-day equity reached x1.5..x20.
"Reached xk before death" = that day comes before the death day. Output: results/evo4_launches.csv
(one row per entity and start; empty = never / not yet within the available data).

    python3 evo4_heatmap.py
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from engine3 import Genes, Market3
from evolve import OUT

STARTS = pd.date_range("2017-01-02", "2026-08-01", freq="D", tz="UTC")
H = 730
KS = (1.5, 2, 3, 5, 10, 20)
NEVER = 10 ** 5


def run(m, g):
    end = m.index[-1]
    eq, st, te = m.batch([replace(g, target=1e9)], STARTS, H)
    x, st, te = eq[0] / g.deposit, st[0], te[0]
    avail = np.minimum(H, np.array([(end - t).days for t in STARTS]))
    death = np.full(len(STARTS), NEVER)
    dead = st >= 2
    death[dead] = [(m.index[min(b, len(m.index) - 1)] - STARTS[i]).days + 1 for i, b in zip(np.where(dead)[0], te[dead])]
    first = {}
    for k in KS:
        hit = x >= k * 0.999
        f = np.where(hit.any(1), hit.argmax(1) + 1, NEVER)
        f[f > avail] = NEVER
        first[k] = np.where(f < death, f, NEVER)   # reached before death
    shadow = np.full(len(STARTS), np.nan)
    shadow[60:] = x[:-60, 59]
    return dict(avail=avail, death=death, first=first, shadow_ok=shadow >= 1)


if __name__ == "__main__":
    fields = list(Genes.__dataclass_fields__)
    m = Market3()
    rows = []
    for ent, cut in (("A", "202410"), ("B", "202310")):
        r = pd.read_csv(OUT / f"evo4_{cut}_s1.csv").sort_values(["t_p", "score"], ascending=False).iloc[0]
        g = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
        d = run(m, g)
        df = pd.DataFrame({"entity": ent, "start": STARTS.strftime("%Y-%m-%d"), "avail": d["avail"],
                           "death": np.where(d["death"] < NEVER, d["death"], -1)})
        for k in KS:
            df[f"x{k:g}".replace(".", "_")] = np.where(d["first"][k] < NEVER, d["first"][k], -1)
        df["shadow"] = d["shadow_ok"].astype(int)
        rows.append(df)
        dd, f2 = d["death"], d["first"][2]
        for nm, s in (("все", np.ones(len(STARTS), bool)), ("2023+", STARTS.year >= 2023), ("тень", d["shadow_ok"])):
            ok = s & (d["avail"] >= 365)
            print(f"{cut} {nm}: P(×2 до смерти, ≤365д) {(f2[ok] <= 365).mean():.0%}, "
                  f"P(×2 до смерти, ≤730д) {(f2[s & (d['avail'] >= 730)] <= 730).mean():.0%}, "
                  f"P(смерть ≤365д) {(dd[ok] <= 365).mean():.0%}")
    out = pd.concat(rows)
    out.replace(-1, "").to_csv(OUT / "evo4_launches.csv", index=False)

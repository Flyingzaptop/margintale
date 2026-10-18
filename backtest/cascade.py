"""Cascade: the safe account feeds an aggressive account only with what it withdraws.

1. SAFE ($300, fixed 0.01 lot) withdraws once a month 50% of the balance above
   2 x deposit (docs/09_withdrawal.md) -- not to the pocket but to account B.
2. Account B trades only that money (lot grows with its balance) and pays out to
   the pocket by its own withdrawal rule. If B dies it waits for the next transfer
   and restarts; the safe account is never touched.
Value of a start at 24 months = pocket + what is left on both accounts (if alive).
Starts: 1st of every month 2017-01..2024-09.

    python3 cascade.py
"""
from dataclasses import astuple, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from engine import DEAD, Market, _run_core
from phases import AGRO
from pwm import SAFE

ROOT = Path(__file__).resolve().parent.parent
DEP = 300.0
MONTHS = 24
FEED = replace(SAFE, wd_days=21, wd_frac=0.5, wd_floor=2.0)
STARTS = pd.date_range("2017-01-01", "2024-09-01", freq="MS", tz="UTC")
NO_DEP = (np.empty(0, np.int64), np.empty(0))

PAYOUT = {  # B -> pocket
    "без вывода": dict(wd_days=0),
    "раз в мес 50% сверх 2×": dict(wd_days=21, wd_frac=0.5, wd_floor=2.0),
    "раз в нед 25% сверх 1×": dict(wd_days=5, wd_frac=0.25, wd_floor=1.0),
    "каждый день 1% (дин.)": dict(wd_days=-1, wd_frac=0.01, wd_floor=0.5, wd_vol_g=1.0, wd_tau=90.0),
    "каждый день 3% (дин.)": dict(wd_days=-1, wd_frac=0.03, wd_floor=0.5, wd_vol_g=1.0, wd_tau=90.0),
}


def run(m, p, i0, i1, deposit, dep=NO_DEP, rec=False):
    return _run_core(m.o, m.h, m.l, m.c, m.atr, m.ema(p.ema), m.filt, i0, i1, deposit,
                     np.array(astuple(p), dtype=np.float64), dep[0], dep[1], rec)


if __name__ == "__main__":
    m = Market()
    feeds = []
    for t in STARTS:
        i0, i1 = m.span(t, t + pd.Timedelta(days=30.44 * MONTHS))
        r = run(m, FEED, i0, i1, DEP, rec=True)
        wd = r[10]
        steps = np.nonzero(np.diff(wd) > 0)[0] + 1
        safe_end = r[0] if r[2] < 0 else 0.0
        feeds.append((i0, i1, i0 + steps, np.diff(wd)[steps - 1], safe_end, r[8]))
    base = np.array([f[4] + f[5] for f in feeds])
    rows = [dict(B="— (вывод SAFE сразу в карман)", вывод_B="", риск_B=np.nan, итог_мед=np.median(base),
                 итог_ср=base.mean(), итог_25=np.percentile(base, 25), итог_10=np.percentile(base, 10),
                 хуже_депозита=(base < DEP).mean(), в_кармане_мед=np.median([f[5] for f in feeds]),
                 смертей_B_ср=0.0)]

    for (bname, bp), risk, (wname, w) in product((("SAFE-логика", SAFE), ("AGRO-логика", AGRO)),
                                                  (0.0, 1.0, 2.0, 3.0, 5.0), PAYOUT.items()):
        B = replace(bp, risk=risk, **w)
        val, pocket, deaths = [], [], []
        for i0, i1, idx, amt, safe_end, _ in feeds:
            if len(idx) == 0:
                val.append(safe_end), pocket.append(0.0), deaths.append(0)
                continue
            r = run(m, B, int(idx[0]), i1, float(amt[0]), (idx[1:].astype(np.int64), amt[1:].astype(np.float64)))
            b_end = r[0] if r[0] >= 0 else 0.0
            val.append(safe_end + r[8] + b_end), pocket.append(r[8]), deaths.append(r[9])
        val = np.array(val)
        rows.append(dict(B=bname, вывод_B=wname, риск_B=risk, итог_мед=np.median(val), итог_ср=val.mean(),
                         итог_25=np.percentile(val, 25), итог_10=np.percentile(val, 10),
                         хуже_депозита=(val < DEP).mean(), в_кармане_мед=np.median(pocket),
                         смертей_B_ср=np.mean(deaths)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "cascade.csv", index=False)
    pd.set_option("display.width", 250)
    f = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.3g}")
    print(f(df.head(1)))
    print("\nЛучшие по медиане:")
    print(f(df.iloc[1:].sort_values("итог_мед", ascending=False).head(10)))
    print("\nЛучшие по среднему:")
    print(f(df.iloc[1:].sort_values("итог_ср", ascending=False).head(8)))
    print("\nЛучшие по 25%:")
    print(f(df.iloc[1:].sort_values("итог_25", ascending=False).head(8)))

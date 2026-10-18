"""Two phases with dynamic daily withdrawals: one pair of entities.

Phase A (SAFE, fixed 0.01 lot) runs T months with a light dynamic withdrawal; at T
everything is closed, and phase B (AGRO, lot grows with the balance) continues
from the balance A left, with a heavier withdrawal, up to 24 months from the
start. Withdrawal rule: see engine.Params (wd_days = -1). Value of a start =
withdrawn in A + withdrawn in B + what is left on the account at 24 months.

Starts: 1st of every month 2017-01..2024-09, $300.

    python3 phases.py
"""
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from engine import DEAD, Market, Params
from pwm import SAFE

ROOT = Path(__file__).resolve().parent.parent
DEP = 300.0
HORIZON = 24
STARTS = pd.date_range("2017-01-01", "2024-09-01", freq="MS", tz="UTC")
# lifecycle3 $1000, row 6 (docs/08_growth_search.md), with a growing lot
AGRO = Params(dirs=0, step=30.0, step_atr=1, step_grow=1.0, mult=2.0, max_orders=8, tp=10.0, tp_atr=0,
              sl=0.0, leverage=100.0, risk=0.0, ema=0, er_win=1, er_max=0.1, sess=0, vol_max=1.5)


def dyn(p, rate, g, tau, floor=0.5):
    return replace(p, wd_days=-1, wd_frac=rate, wd_floor=floor, wd_vol_g=g, wd_tau=tau)


if __name__ == "__main__":
    m = Market()
    rows = []
    # baselines: SAFE alone, no withdrawal / best monthly rule of docs/09_withdrawal.md
    for name, p in (("SAFE без вывода", SAFE),
                    ("SAFE, раз в месяц 50% сверх 2×деп", replace(SAFE, wd_days=21, wd_frac=0.5, wd_floor=2.0))):
        _, eq, _ = m.lifecycles([p], months=HORIZON, start_times=STARTS, deposit=DEP)
        end = eq[0, :, -1]
        wd = m.last_wd[0]
        value = wd + np.where(end >= DEAD * DEP, end, 0.0)
        rows.append(dict(T=name, живы_24=(end >= DEAD * DEP).mean(), выведено_мед=np.median(wd),
                         выведено_ср=wd.mean(), итог_мед=np.median(value), итог_ср=value.mean(),
                         итог_25=np.percentile(value, 25), итог_10=np.percentile(value, 10),
                         хуже_депозита=(value < DEP).mean()))
    for T, a_rate, g, tau in product((6, 9), (0.0, 0.001, 0.002), (0.0, 1.0), (0.0, 90.0, 365.0)):
        A = dyn(SAFE, a_rate, g, tau)
        _, eqA, _ = m.lifecycles([A], months=T, start_times=STARTS, deposit=DEP)
        wdA, endA = m.last_wd[0], eqA[0, :, -1]
        aliveA = endA >= DEAD * DEP
        t_b = [t + pd.Timedelta(days=30.44 * T) for t in STARTS]
        Bs = [dyn(replace(base, risk=r), b_rate, g, tau) for base in (SAFE, AGRO)
              for b_rate, r in product((0.003, 0.005, 0.01, 0.02), (1.0, 2.0, 3.0))]
        _, eqB, _ = m.lifecycles(Bs, months=HORIZON - T, start_times=t_b,
                                 deposits=np.where(aliveA, endA, 1.0))
        for j, B in enumerate(Bs):
            endB = np.where(aliveA, eqB[j, :, -1], 0.0)
            aliveB = aliveA & (endB >= DEAD * endA)
            wd = wdA + np.where(aliveA, m.last_wd[j], 0.0)
            value = wd + np.where(aliveB, endB, 0.0)
            rows.append(dict(T=T, B="SAFE" if B.step == SAFE.step else "AGRO", a_rate=a_rate,
                             b_rate=B.wd_frac, risk_B=B.risk, vol_g=g, tau=tau,
                             живы_A=aliveA.mean(), живы_24=aliveB.mean(),
                             выведено_A_мед=np.median(wdA), выведено_мед=np.median(wd), выведено_ср=wd.mean(),
                             итог_мед=np.median(value), итог_ср=value.mean(),
                             итог_25=np.percentile(value, 25), итог_10=np.percentile(value, 10),
                             хуже_депозита=(value < DEP).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "phases.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    f = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.3g}")
    print("Базовые:")
    print(f(df.head(2)))
    print("\nЛучшие по среднему итогу за 24 мес:")
    print(f(df.sort_values("итог_ср", ascending=False).head(12)))
    print("\nЛучшие по 25-му перцентилю:")
    print(f(df.sort_values("итог_25", ascending=False).head(12)))
    print("\nЛучшие по медиане:")
    print(f(df.sort_values("итог_мед", ascending=False).head(8)))

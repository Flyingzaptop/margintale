"""(1) Take-and-restart: run the $1000 safe grid until the equity first reaches
TARGET x deposit, close everything, take the whole equity to the pocket and start
again with a fresh deposit from the pocket. A death costs one deposit and the
algorithm is restarted the same way. Net = pocket - deposits put in, at 24 months.
(2) Survival of the plain $1000 account (and its withdrawing version) by month.

Starts: 1st of every month 2017-01..2024-09.

    python3 restart.py
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from engine import DEAD, Market
from pwm import SAFE

DEP = 1000.0
MONTHS = 24
A = replace(SAFE, lot_oz=3.0)  # 0.01 lot per ~$300
STARTS = pd.date_range("2017-01-01", "2024-09-01", freq="MS", tz="UTC")


def take_and_restart(m, t0, target):
    end = t0 + pd.Timedelta(days=30.44 * MONTHS)
    t, net, wins, deaths = t0, 0.0, 0, 0
    while t < end:
        r = m.run(A, t, end, deposit=DEP)
        eq = r["equity"]
        net -= DEP
        hit = np.nonzero(eq.to_numpy() >= target * DEP)[0]
        if len(hit) and (r["died"] is None or eq.index[hit[0]] < r["died"]):
            net += eq.iloc[hit[0]]
            wins += 1
            t = eq.index[hit[0]] + pd.Timedelta(minutes=1)
        elif r["died"] is not None:
            net += r["balance"]
            deaths += 1
            t = r["died"] + pd.Timedelta(minutes=1)
        else:  # alive at the horizon: count what is on the account
            net += r["balance"]
            break
    return net, wins, deaths


if __name__ == "__main__":
    m = Market()
    rows = []
    for target in (1.25, 1.5, 2.0, 3.0):
        res = np.array([take_and_restart(m, t, target) for t in STARTS])
        net, wins, deaths = res.T
        rows.append({"снимаем при": f"×{target}", "прибыль, медиана": np.median(net),
                     "среднее": net.mean(), "25% худших": np.percentile(net, 25),
                     "10% худших": np.percentile(net, 10), "в минусе": (net < 0).mean(),
                     "циклов-побед, ср.": wins.mean(), "смертей, ср.": deaths.mean()})
    print("(1) Дотянуть до ×X, снять всё, перезапустить с новыми $1000 (за 24 мес, прибыль сверх вложенного):")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\n(2) Доля живых счетов по месяцам, $1000, лот 0.03:")
    for name, p in (("без вывода", A),
                    ("раз в месяц 50% сверх 2×", replace(A, wd_days=21, wd_frac=0.5, wd_floor=2.0))):
        _, eq, _ = m.lifecycles([p], months=MONTHS, start_times=STARTS, deposit=DEP)
        alive = eq[0] >= DEAD * DEP
        print(f"  {name}: " + " ".join(f"{k}м {alive[:, k - 1].mean():.0%}" for k in (1, 3, 6, 9, 12, 18, 24)))

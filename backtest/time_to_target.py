"""Time to the target or to death for one target-gene genome: many starts, exact timing.

    python3 time_to_target.py [target] [kill_dd]
"""
import sys

import numpy as np
import pandas as pd

from engine import Market, params_from

T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
K = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
DEP = 100.0
HORIZON_D = 180

if __name__ == "__main__":
    x = pd.read_parquet("../results/aggro2_d100_s2_kill.parquet")
    g = x[(x.target == T) & (x.kill_dd == K)].sort_values("p_hit3", ascending=False).iloc[0]
    p = params_from(g)
    print(p)
    m = Market()
    starts = pd.date_range("2017-01-02", "2026-02-28", freq="W-MON", tz="UTC").append(
        pd.date_range("2017-01-05", "2026-02-28", freq="W-THU", tz="UTC")).sort_values()
    rows = []
    for t in starts:
        r = m.run(p, t, t + pd.Timedelta(days=HORIZON_D), deposit=DEP)
        eq = r["equity"]
        hit = np.nonzero(eq.to_numpy() >= T * DEP * 0.999)[0]
        if len(hit):
            rows.append(dict(start=t, outcome="цель", days=(eq.index[hit[0]] - t).total_seconds() / 86400,
                             result=eq.iloc[hit[0]] / DEP))
        elif r["died"] is not None:
            rows.append(dict(start=t, outcome="смерть", days=(r["died"] - t).total_seconds() / 86400,
                             result=r["balance"] / DEP))
        elif K > 0 and eq.iloc[-1] < eq.max() * (1 - K) + 1e-9:
            rows.append(dict(start=t, outcome="выключатель", days=np.nan, result=eq.iloc[-1] / DEP))
        else:
            rows.append(dict(start=t, outcome="ещё идёт", days=HORIZON_D, result=eq.iloc[-1] / DEP))
    d = pd.DataFrame(rows)
    print(f"\n{len(d)} запусков по ${DEP:.0f}, цель ×{T:g}, выключатель {K:.0%}, горизонт {HORIZON_D} дней")
    print(d.groupby("outcome").agg(доля=("result", lambda s: len(s) / len(d)), итог_средний=("result", "mean"),
                                   дней_медиана=("days", "median"), дней_25=("days", lambda s: s.quantile(.25)),
                                   дней_75=("days", lambda s: s.quantile(.75))).round(2).to_string())
    h = d[d.outcome == "цель"].days
    print("\nДошли до цели, доля от всех запусков к дню: " +
          " ".join(f"{k}д {(h <= k).sum() / len(d):.0%}" for k in (1, 3, 7, 14, 30, 60, 90, 180)))
    dd = d[d.outcome != "цель"]
    print("Умерли/остановились к дню: " + " ".join(f"{k}д {(dd.days <= k).sum() / len(d):.0%}" for k in (1, 3, 7, 14, 30, 60, 90, 180)))
    print(f"Средний итог запуска ×{d.result.mean():.2f}; длительность запуска (до цели или смерти), медиана "
          f"{d.days.median():.0f} дн.")
    for y, s in d.groupby(d.start.dt.year):
        print(f"  {y}: цель {(s.outcome == 'цель').mean():.0%}, средний итог ×{s.result.mean():.2f}")
    d.to_csv(f"../results/time_to_target_T{T:g}_K{K:g}.csv", index=False)

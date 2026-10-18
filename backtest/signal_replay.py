"""Replay the signal's own trades on our M1 data: same entry/exit minutes and sides,
our prices, a range of costs. Tells whether its entries carry an edge before costs
and how much cost it survives. P&L in $ per oz (= per 0.01 lot).

    python3 signal_replay.py
"""
import numpy as np
import pandas as pd

from engine import load_m1

if __name__ == "__main__":
    m = load_m1()
    t = pd.read_csv("../results/signal_trades.csv", parse_dates=["open_utc", "close_utc"]).dropna(subset=["close_utc"])
    t["close_utc"] = pd.to_datetime(t.close_utc, utc=True).dt.floor("min")
    t = t[t.close_utc <= m.index[-1]]
    i0 = m.index.get_indexer(pd.DatetimeIndex(t.open_utc).as_unit("ms"), method="bfill")
    i1 = m.index.get_indexer(pd.DatetimeIndex(t.close_utc).as_unit("ms"), method="bfill")
    ok = (i0 >= 0) & (i1 >= 0)
    t, i0, i1 = t[ok], i0[ok], i1[ok]
    c = m.close.to_numpy()
    ours = t.side.to_numpy() * (c[i1] - c[i0])          # our close-to-close, before costs
    theirs = t.dist.to_numpy()                          # their fill-to-fill (their costs inside)
    print(f"{len(t)} сделок. Корреляция нашей и его разницы цен: {np.corrcoef(ours, theirs)[0, 1]:.2f}")
    print(f"Его результат, $ на унцию за сделку: среднее {theirs.mean():+.3f}, итог {theirs.sum():+,.0f}")
    print(f"На наших ценах до издержек:           среднее {ours.mean():+.3f}, итог {ours.sum():+,.0f}")
    for cost in (0.0, 0.1, 0.2, 0.3, 0.37, 0.5):
        print(f"  издержки ${cost:.2f}/унцию на сделку: итог {(ours - cost).sum():+,.0f} $/унцию, "
              f"среднее {(ours - cost).mean():+.3f}, прибыльных {((ours - cost) > 0).mean():.0%}")
    # the same entries with random sides (is the direction informative?)
    rng = np.random.default_rng(0)
    rs = rng.choice([-1, 1], len(t))
    print(f"Те же моменты, случайное направление: среднее {(rs * (c[i1] - c[i0])).mean():+.3f}")
    # same holding times at random moments, his side rule = with the 1-day move
    n = len(c)
    j0 = rng.integers(2000, n - 5000, len(t))
    j1 = np.minimum(j0 + (i1 - i0), n - 1)
    tr = np.sign(c[j0 - 1] - c[j0 - 1441])
    print(f"Случайные моменты, направление по суточному ходу, те же длительности: среднее {(tr * (c[j1] - c[j0])).mean():+.3f}")

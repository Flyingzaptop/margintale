"""Keep the signal's entries (minute + side), replace its exits by fixed rules:
TP / SL in $ from entry, time stop. Pessimistic intrabar order (SL first).
Result in $ per oz per trade after COST.

    python3 signal_exits.py
"""
from itertools import product

import numpy as np
import pandas as pd
from numba import njit

from engine import load_m1

COST = 0.37


@njit(cache=True)
def exits(h, l, c, i0, side, tp, sl, hold):
    out = np.empty(len(i0))
    for k in range(len(i0)):
        e = c[i0[k] - 1] if False else c[i0[k]]
        r = 0.0
        for i in range(i0[k] + 1, min(i0[k] + hold, len(c) - 1)):
            if side[k] > 0:
                if l[i] <= e - sl:
                    r = -sl
                    break
                if h[i] >= e + tp:
                    r = tp
                    break
            else:
                if h[i] >= e + sl:
                    r = -sl
                    break
                if l[i] <= e - tp:
                    r = tp
                    break
            r = side[k] * (c[i] - e)
        out[k] = r
    return out


if __name__ == "__main__":
    m = load_m1()
    t = pd.read_csv("../results/signal_trades.csv", parse_dates=["open_utc"])
    i0 = m.index.get_indexer(pd.DatetimeIndex(t.open_utc).as_unit("ms"), method="bfill")
    ok = i0 > 0
    i0, side = i0[ok].astype(np.int64), t.side.to_numpy()[ok].astype(np.float64)
    h, l, c = (m[x].to_numpy() for x in ("high", "low", "close"))
    rows = []
    for tp, sl, hold in product((1.0, 1.5, 2.0, 3.0, 5.0, 10.0), (3.0, 5.0, 10.0, 20.0, 40.0), (60, 240, 1440)):
        r = exits(h, l, c, i0, side, tp, sl, hold) - COST
        rows.append(dict(TP=tp, SL=sl, удерж_мин=hold, среднее=r.mean(), итог=r.sum(), прибыльных=(r > 0).mean()))
    df = pd.DataFrame(rows).sort_values("среднее", ascending=False)
    pd.set_option("display.width", 200)
    print("Его входы + простой выход, $ на унцию за сделку после издержек $0.37:")
    print(df.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # same exits on random entries with the 1-day trend side
    rng = np.random.default_rng(0)
    j0 = rng.integers(3000, len(c) - 3000, 20000).astype(np.int64)
    sd = np.sign(c[j0 - 1] - c[j0 - 1441]).astype(np.float64)
    b = df.iloc[0]
    rr = exits(h, l, c, j0, sd, b.TP, b.SL, int(b.удерж_мин)) - COST
    print(f"\nТот же выход (TP {b.TP}, SL {b.SL}, {int(b.удерж_мин)} мин) на случайных входах по суточному тренду, "
          f"вся история 2016-2026: среднее {rr.mean():+.3f}")

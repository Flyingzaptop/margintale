"""Pyramiding ("anti-martingale") one-shot entity: ride a breakout, add on profit with
all free margin, exit the whole pyramid on a trailing stop.

Each start (every day 00:00 UTC, 2017-01..2026-08) is one shot from DEPOSIT:
  wait for a breakout of the N-bar high (long) / low (short, if allowed);
  open with `util` of the buying power (equity * leverage / price), gram units;
  every `add_k` ATR further in profit add `util` of the free buying power;
  exit all when price retraces `trail_k` ATR from the best price, or after `hold` bars;
  liquidation at margin level 30% = death.
Costs: spread $0.20 + commission $25 per $1M per side, per oz, on every fill.
Intrabar order is pessimistic: adverse extreme (stop-out, trailing exit) first.

    python3 pyramid.py
"""
from itertools import product

import numpy as np
import pandas as pd
from numba import njit, prange

from engine import load_m1

GRAM = 1 / 31.1035
DEPOSIT = 100.0
STOP_OUT = 0.30


@njit(cache=True)
def shot(o, h, l, c, atr, hh, ll, i0, end, dep, lev, util, add_k, trail_k, hold, both):
    eq_bal = dep
    side = 0
    oz = 0.0
    cost_sum = 0.0      # sum(oz * entry incl. costs)
    best = 0.0
    nxt = 0.0
    a0 = 0.0
    t_in = -1
    peak = dep
    adds = 0
    for i in range(i0, end):
        if side == 0:
            if h[i] > hh[i]:
                side = 1
                px = max(hh[i], o[i])
            elif both and l[i] < ll[i]:
                side = -1
                px = min(ll[i], o[i])
            else:
                continue
            a0 = atr[i - 1]
            size = np.floor(eq_bal * lev * util / px / GRAM) * GRAM
            if size < GRAM:
                return eq_bal / dep, peak / dep, 0, 0, 0
            fee = 0.20 + 2 * 25e-6 * px
            oz = size
            cost_sum = size * (px + side * fee)
            best = px
            nxt = px + side * add_k * a0
            t_in = i
            continue
        # adverse side first: stop-out, trailing exit
        worst = l[i] if side > 0 else h[i]
        eq_w = eq_bal + side * (oz * worst - cost_sum)
        if eq_w <= STOP_OUT * oz * worst / lev:
            return max(eq_w, 0.0) / dep, peak / dep, i - t_in, 1, adds
        stop = best - side * trail_k * a0
        if (side > 0 and l[i] <= stop) or (side < 0 and h[i] >= stop):
            px = min(stop, o[i]) if side > 0 else max(stop, o[i])
            return (eq_bal + side * (oz * px - cost_sum)) / dep, peak / dep, i - t_in, 0, adds
        if i - t_in >= hold:
            return (eq_bal + side * (oz * c[i] - cost_sum)) / dep, peak / dep, i - t_in, 0, adds
        # favourable side: adds, new best
        fav = h[i] if side > 0 else l[i]
        while (side > 0 and fav >= nxt) or (side < 0 and fav <= nxt):
            eq_n = eq_bal + side * (oz * nxt - cost_sum)
            free = eq_n - oz * nxt / lev
            size = np.floor(free * lev * util / nxt / GRAM) * GRAM
            if size >= GRAM:
                fee = 0.20 + 2 * 25e-6 * nxt
                oz += size
                cost_sum += size * (nxt + side * fee)
                adds += 1
            nxt += side * add_k * a0
        if (side > 0 and fav > best) or (side < 0 and fav < best):
            best = fav
        eq_c = eq_bal + side * (oz * c[i] - cost_sum)
        if eq_c > peak:
            peak = eq_c
    if side == 0:
        return 1.0, 1.0, 0, 0, 0
    return (eq_bal + side * (oz * c[end - 1] - cost_sum)) / dep, peak / dep, end - t_in, 0, adds


@njit(parallel=True, cache=True)
def shots(o, h, l, c, atr, hh, ll, starts, horizon, dep, lev, util, add_k, trail_k, hold, both):
    S = len(starts)
    fin, pk, dur, liq, nad = np.empty(S), np.empty(S), np.empty(S), np.empty(S), np.empty(S)
    for s in prange(S):
        r = shot(o, h, l, c, atr, hh, ll, starts[s], min(starts[s] + horizon, len(c) - 1),
                 dep, lev, util, add_k, trail_k, hold, both)
        fin[s], pk[s], dur[s], liq[s], nad[s] = r
    return fin, pk, dur, liq, nad


if __name__ == "__main__":
    m = load_m1()
    o, h, l, c, a = (m[x].to_numpy() for x in ("open", "high", "low", "close", "atr"))
    starts = m.index.searchsorted(pd.date_range("2017-01-01", "2026-08-31", freq="D", tz="UTC"))
    hhs = {n: m.high.rolling(n).max().shift(1).bfill().to_numpy() for n in (60, 240, 1440)}
    lls = {n: m.low.rolling(n).min().shift(1).bfill().to_numpy() for n in (60, 240, 1440)}
    rows = []
    for n, add_k, trail_k, util, lev, both, hold in product((60, 240, 1440), (0.5, 1.0, 2.0, 4.0), (1.0, 2.0, 4.0, 8.0),
                                                            (0.5, 0.9), (100.0, 500.0, 1000.0), (True, False),
                                                            (1440, 4320, 10080)):
        fin, pk, dur, liq, nad = shots(o, h, l, c, a, hhs[n], lls[n], starts, 20000, DEPOSIT, lev, util,
                                       add_k, trail_k, hold, both)
        rows.append(dict(N=n, add_k=add_k, trail_k=trail_k, util=util, плечо=lev, обе=both, держать_дн=hold // 1440,
                         медиана=np.median(fin), среднее=fin.mean(), p_x2=(fin >= 2).mean(), p_x10=(fin >= 10).mean(),
                         p_x100=(fin >= 100).mean(), макс=fin.max(), ликвид=liq.mean(), p_минус=(fin < 1).mean(),
                         дней_мед=np.median(dur) / 1440, доливок=np.median(nad)))
    df = pd.DataFrame(rows)
    df.to_csv("../results/pyramid.csv", index=False)
    pd.set_option("display.width", 250)
    f = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.3g}")
    for lev in (100.0, 500.0, 1000.0):
        d = df[df.плечо == lev]
        print(f"\n=== плечо 1:{int(lev)} — лучшие по среднему итогу выстрела (EV, × депозита):")
        print(f(d.sort_values("среднее", ascending=False).head(6)))
        print(f"--- лучшие по вероятности ×10:")
        print(f(d.sort_values("p_x10", ascending=False).head(4)))
        print(f"--- лучшие по вероятности ×100:")
        print(f(d.sort_values(["p_x100", "среднее"], ascending=False).head(4)))

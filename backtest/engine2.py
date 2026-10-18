"""Single-position strategies with pending orders, SL/TP, trailing (numba, XAUUSD M1).

Broker model (Spotware cTrader demo, XAUUSD spec):
* 1 lot = 100 oz, min 0.01 lot, step 0.01 -> sizes are whole ounces, at least 1;
* spread $0.20, commission $25 per $1M of volume per side;
* swap per oz per day: long -0.020895, short -0.0192, triple on Wednesday,
  charged when the trading day changes;
* gold leverage 1:100, stop-out at 30% margin level; an entry needs free margin.

Entries when flat (and allowed by the filters), pending or market:
  entry 0 breakout: buy stop above the N-bar high + offset, sell stop below the N-bar low;
  entry 1 pullback: buy limit at EMA - offset when close > EMA, sell limit mirrored;
  entry 2 range fade: buy limit at the N-bar low - offset, sell limit at the N-bar high + offset;
  entry 3 market with the trend: buy at the open if close > EMA, else sell;
  entry 4 market reversion: buy at the open if close < EMA - offset, sell if > EMA + offset.
Pending sides are OCO; unfilled orders expire after `pend_life` bars.

Exits: SL / TP always set, in ATR from the fill; break-even after `be` ATR in profit,
trailing stop at `trail` ATR once `trail_start` ATR in profit, time stop after
`max_hold` bars, and with `exit_sig` the strategy closes at market when the close
crosses the EMA against the position.

Size: risk `risk` of equity at the stop distance, rounded down to whole ounces,
at least 1 oz. After a loss the risk is multiplied by `loss_mult` (capped at
`loss_cap` x base risk), after a win it resets.

Intrabar order is pessimistic: stop before target; no target in the fill bar.
Death: liquidation (margin level <= STOP_OUT) or equity too small to open 0.01 lot.
"""
from dataclasses import dataclass, astuple, fields
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

from engine import load_m1

SPREAD = 0.20
COMMISSION = 25e-6        # per $ of volume per side
SWAP_LONG = -0.020895     # $ per oz per day
SWAP_SHORT = -0.0192
LEVERAGE = 100.0
STOP_OUT = 0.30

LOOKBACKS = (15, 60, 240)
EMAS = (60, 240, 1440)
ER_WINDOWS = (60, 240, 1440)
SESSIONS = ((0, 24), (0, 7), (7, 12), (12, 17), (17, 24), (7, 17))


@dataclass
class Genes:
    entry: int = 0
    dirs: int = 0          # 0 both, 1 long only, 2 short only
    look: int = 1          # index into LOOKBACKS
    ema: int = 1           # index into EMAS
    offset: float = 0.2    # ATR
    sl: float = 2.0        # ATR
    tp: float = 3.0        # ATR
    trail_start: float = 0.0  # ATR, 0 = no trailing
    trail: float = 1.0     # ATR
    be: float = 0.0        # ATR, 0 = no break-even
    max_hold: int = 0      # bars, 0 = none
    pend_life: int = 60    # bars
    sess: int = 0          # index into SESSIONS
    vol_max: float = 0.0   # ATR / 20-day ATR, 0 = off
    er_win: int = 0        # 0 off, else 1..3 into ER_WINDOWS
    er_max: float = 0.0
    risk: float = 0.02     # share of equity at the stop
    loss_mult: float = 1.0
    loss_cap: float = 1.0
    deposit: float = 300.0
    exit_sig: int = 0      # 1 = close at market when close crosses the EMA against the position


NG = len(fields(Genes))


def indicators(bars):
    """Matrices the kernel reads: hh/ll per lookback, ema per length, filters."""
    c, h, l = bars.close, bars.high, bars.low
    hh = np.stack([h.rolling(n).max().shift(1).bfill().to_numpy() for n in LOOKBACKS])
    ll = np.stack([l.rolling(n).min().shift(1).bfill().to_numpy() for n in LOOKBACKS])
    ema = np.stack([c.ewm(span=n).mean().to_numpy() for n in EMAS])
    path = c.diff().abs()
    er = np.stack([((c - c.shift(w)).abs() / path.rolling(w).sum()).fillna(1.0).to_numpy()
                   for w in ER_WINDOWS])
    volr = (bars.atr / bars.atr.rolling(28800, min_periods=1440).mean()).bfill().to_numpy()
    ts = bars.index
    hour = ts.hour.to_numpy().astype(np.int64)
    day = ((ts - pd.Timestamp("2000-01-01", tz="UTC")).days).to_numpy().astype(np.int64)
    wday = ts.dayofweek.to_numpy().astype(np.int64)
    return hh, ll, ema, er, volr, hour, day, wday


@njit(cache=True)
def _run(o, h, l, c, atr, hh, ll, ema, er, volr, hour, day, wday, start, end, g, checks, out_eq):
    entry, dirs, look, ema_i = int(g[0]), int(g[1]), int(g[2]), int(g[3])
    offset, sl_k, tp_k, trail_start, trail_k, be_k = g[4], g[5], g[6], g[7], g[8], g[9]
    max_hold, pend_life, sess = int(g[10]), int(g[11]), int(g[12])
    vol_max, er_win, er_max = g[13], int(g[14]), g[15]
    risk0, loss_mult, loss_cap, deposit = g[16], g[17], g[18], g[19]
    exit_sig = int(g[20])
    liquidated = False
    s_lo = (0, 0, 7, 12, 17, 7)[sess]
    s_hi = (24, 7, 12, 17, 24, 17)[sess]

    bal = deposit
    risk = risk0
    pos = 0            # +1 long, -1 short, 0 flat
    oz = 0.0
    px = 0.0           # fill price incl. costs
    stop = 0.0
    target = 0.0
    a_fill = 0.0
    held = 0
    filled_now = False
    pend = False
    buy_lvl = 0.0
    sell_lvl = 0.0
    pend_age = 0
    peak = deposit
    maxdd = 0.0
    trades = 0
    wins = 0
    died = -1
    ck = 0
    nck = len(checks)

    for i in range(start, end):
        # swap at day change
        if pos != 0 and day[i] != day[i - 1]:
            mult = 3.0 if wday[i] == 3 else 1.0
            bal += oz * (SWAP_LONG if pos > 0 else SWAP_SHORT) * mult

        # place pending orders when flat
        if pos == 0 and not pend:
            ok = hour[i] >= s_lo and hour[i] < s_hi
            if ok and vol_max > 0 and volr[i - 1] >= vol_max:
                ok = False
            if ok and er_win > 0 and er_max > 0 and er[er_win - 1, i - 1] >= er_max:
                ok = False
            if ok:
                a = atr[i - 1]
                if entry == 0:
                    buy_lvl = hh[look, i] + offset * a
                    sell_lvl = ll[look, i] - offset * a
                elif entry == 1:
                    e = ema[ema_i, i - 1]
                    up = c[i - 1] > e
                    buy_lvl = e - offset * a if up else -1.0
                    sell_lvl = e + offset * a if not up else -1.0
                elif entry == 2:
                    buy_lvl = ll[look, i] - offset * a
                    sell_lvl = hh[look, i] + offset * a
                else:  # market entries
                    e = ema[ema_i, i - 1]
                    if entry == 3:
                        up = c[i - 1] > e
                        buy_lvl = 1.0 if up else -1.0
                        sell_lvl = 1.0 if not up else -1.0
                    else:
                        buy_lvl = 1.0 if c[i - 1] < e - offset * a else -1.0
                        sell_lvl = 1.0 if c[i - 1] > e + offset * a else -1.0
                if dirs == 2:
                    buy_lvl = -1.0
                if dirs == 1:
                    sell_lvl = -1.0
                pend = buy_lvl > 0 or sell_lvl > 0
                pend_age = 0

        # fills
        filled_now = False
        if pend and pos == 0:
            side = 0
            fill = 0.0
            if entry == 0:  # stop orders
                if buy_lvl > 0 and h[i] >= buy_lvl:
                    side, fill = 1, max(buy_lvl, o[i])
                elif sell_lvl > 0 and l[i] <= sell_lvl:
                    side, fill = -1, min(sell_lvl, o[i])
            elif entry >= 3:  # market at the open
                if buy_lvl > 0:
                    side, fill = 1, o[i]
                elif sell_lvl > 0:
                    side, fill = -1, o[i]
            else:           # limit orders
                if buy_lvl > 0 and l[i] <= buy_lvl:
                    side, fill = 1, min(buy_lvl, o[i])
                elif sell_lvl > 0 and h[i] >= sell_lvl:
                    side, fill = -1, max(sell_lvl, o[i])
            if side != 0:
                a = atr[i - 1]
                dist = sl_k * a
                eq = bal
                size = np.floor(eq * risk / dist)
                if size < 1.0:
                    size = 1.0
                cap = np.floor(eq * LEVERAGE / fill * 0.9)  # free margin
                if cap < size:
                    size = cap
                pend = False
                if size >= 1.0:
                    cost = SPREAD + 2 * COMMISSION * fill
                    pos = side
                    oz = size
                    px = fill + cost if side > 0 else fill - cost
                    stop = fill - dist if side > 0 else fill + dist
                    target = fill + tp_k * a if side > 0 else fill - tp_k * a
                    a_fill = a
                    held = 0
                    filled_now = True
            else:
                pend_age += 1
                if pend_age >= pend_life:
                    pend = False

        # manage position: adverse first
        if pos != 0:
            held += 1
            exit_px = -1.0
            if pos > 0 and l[i] <= stop:
                exit_px = min(stop, o[i]) if not filled_now else stop
            elif pos < 0 and h[i] >= stop:
                exit_px = max(stop, o[i]) if not filled_now else stop
            # stop-out on worst price
            if exit_px < 0:
                worst = l[i] if pos > 0 else h[i]
                eq_w = bal + (oz * (worst - px) if pos > 0 else oz * (px - worst))
                if eq_w <= STOP_OUT * oz * worst / LEVERAGE:
                    exit_px = worst
                    liquidated = True
            if exit_px < 0 and not filled_now:
                if pos > 0 and h[i] >= target:
                    exit_px = target
                elif pos < 0 and l[i] <= target:
                    exit_px = target
            if exit_px < 0 and max_hold > 0 and held >= max_hold:
                exit_px = c[i]
            if exit_px < 0 and exit_sig and not filled_now:
                e = ema[ema_i, i]
                if (pos > 0 and c[i] < e) or (pos < 0 and c[i] > e):
                    exit_px = c[i]
            if exit_px >= 0:
                pnl = oz * (exit_px - px) if pos > 0 else oz * (px - exit_px)
                bal += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                    risk = risk0
                else:
                    risk = min(risk * loss_mult, risk0 * loss_cap)
                pos = 0
            else:
                # trailing / break-even from this bar's favourable extreme (applies next bar)
                fav = h[i] - px if pos > 0 else px - l[i]
                if be_k > 0 and fav >= be_k * a_fill:
                    if pos > 0:
                        stop = max(stop, px)
                    else:
                        stop = min(stop, px)
                if trail_start > 0 and fav >= trail_start * a_fill:
                    if pos > 0:
                        stop = max(stop, h[i] - trail_k * a_fill)
                    else:
                        stop = min(stop, l[i] + trail_k * a_fill)

        eq_c = bal
        if pos != 0:
            eq_c += oz * (c[i] - px) if pos > 0 else oz * (px - c[i])
        if eq_c > peak:
            peak = eq_c
        dd = 1.0 - eq_c / peak
        if dd > maxdd:
            maxdd = dd
        while ck < nck and i + 1 >= checks[ck]:
            out_eq[ck] = eq_c
            ck += 1
        if liquidated or (pos == 0 and eq_c * LEVERAGE < c[i]):
            died = i
            maxdd = max(maxdd, 1.0 - eq_c / peak)
            while ck < nck:
                out_eq[ck] = eq_c
                ck += 1
            break
    while ck < nck:
        out_eq[ck] = eq_c
        ck += 1
    return eq_c, died, maxdd, trades, wins


@njit(parallel=True, cache=True)
def _lifecycles(o, h, l, c, atr, hh, ll, ema, er, volr, hour, day, wday, starts, checks, G):
    J, S, K = len(G), len(starts), checks.shape[1]
    died = np.full((J, S), -1, np.int64)
    eq = np.empty((J, S, K))
    dd = np.empty((J, S))
    tr = np.empty((J, S))
    wr = np.empty((J, S))
    for js in prange(J * S):
        j, s = js // S, js % S
        out = np.empty(K)
        r = _run(o, h, l, c, atr, hh, ll, ema, er, volr, hour, day, wday,
                 starts[s], checks[s, K - 1], G[j], checks[s], out)
        eq[j, s] = out
        died[j, s] = r[1]
        dd[j, s] = r[2]
        tr[j, s] = r[3]
        wr[j, s] = r[4] / max(r[3], 1)
    return died, eq, dd, tr, wr


class Market2:
    def __init__(self, bars=None):
        self.bars = load_m1() if bars is None else bars
        b = self.bars
        self.o, self.h, self.l, self.c = (b[x].to_numpy() for x in ("open", "high", "low", "close"))
        self.atr = b.atr.to_numpy()
        self.ind = indicators(b)
        self.index = b.index

    def pos(self, t):
        t = pd.Timestamp(t).floor("min")
        return max(int(self.index.searchsorted(t if t.tzinfo else t.tz_localize("UTC"))), 1)

    def lifecycles(self, genes, starts, months=36):
        """Each genome from each start for up to `months` (30.44-day months).

        Returns dict of arrays: died (bar or -1), eq (configs, starts, months) at month ends,
        dd (max drawdown), trades, winrate.
        """
        st = np.array([self.pos(t) for t in starts])
        ck = np.array([[self.pos(pd.Timestamp(t) + pd.Timedelta(days=30.44 * (k + 1)))
                        for k in range(months)] for t in starts])
        G = np.array([astuple(g) for g in genes], dtype=np.float64)
        died, eq, dd, tr, wr = _lifecycles(self.o, self.h, self.l, self.c, self.atr, *self.ind, st, ck, G)
        return dict(died=died, eq=eq, dd=dd, trades=tr, winrate=wr, starts=pd.DatetimeIndex(starts))

    def run(self, g, start, end):
        """One account; returns final equity, death time, max dd, trades, winrate and
        daily equity (sampled at the end of each day)."""
        i0, i1 = self.pos(start), self.pos(end)
        days = pd.date_range(pd.Timestamp(start).floor("D") + pd.Timedelta(days=1), end, freq="D")
        ck = np.array([max(self.pos(t), i0 + 1) for t in days] + [i1])
        out = np.empty(len(ck))
        eq, died, dd, tr, wins = _run(self.o, self.h, self.l, self.c, self.atr, *self.ind,
                                      i0, i1, np.array(astuple(g), dtype=np.float64), ck, out)
        return dict(balance=eq, died=self.index[died] if died >= 0 else None, maxdd=dd,
                    trades=tr, winrate=wins / max(tr, 1),
                    equity=pd.Series(out, index=list(days) + [pd.Timestamp(end)]))

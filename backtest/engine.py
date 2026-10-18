"""Generalised grid/martingale engine (numba) for XAUUSD M1.

Runs on M1 bars. One run = one account from `start` to `end` bar. Up to two baskets (long and
short) share the account. Everything the search varies is in Params.

Prices in the data are bid; every entry pays SPREAD + commission (long buys at
level + cost, short sells at level - cost), exits are at the raw price.

Intrabar order is pessimistic: for each basket the adverse extreme is processed
first (adds, basket stop, margin stop-out), the favourable one last (take
profit). A basket that added in a bar cannot take profit in that same bar:
the order of the low and the high inside it is unknown.
"""
from dataclasses import dataclass, astuple
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

DATA = Path(__file__).resolve().parent.parent / "data" / "XAUUSD_41.parquet"
SPREAD = 0.30 + 0.07  # spread + $7/lot round-turn commission
STOP_OUT = 0.5   # margin level at which the broker closes everything
DEAD = 0.1       # account counts as dead below this share of the deposit


@dataclass
class Params:
    dirs: int = 0          # 0 long, 1 short, 2 both baskets, 3 trend: one side by EMA
    step: float = 6.0      # grid step: $ (step_atr=0) or ATR multiples (step_atr=1)
    step_atr: int = 0
    step_grow: float = 1.0 # each next step = previous * step_grow
    mult: float = 1.6      # lot of order n = base * mult**n
    max_orders: int = 0    # 0 = unlimited
    tp: float = 5.0        # take profit beyond average: $ or ATR multiples
    tp_atr: int = 0
    sl: float = 0.0        # basket stop: loss as share of equity at basket open, 0 = none
    leverage: float = 0.0  # 0 = unlimited (no margin, dies at zero equity)
    risk: float = 0.0      # 0 = fixed 0.01 lot; else base oz per $1000 of balance
    ema: int = 0           # trend filter length in bars (dirs=3)


def load_m1(path=DATA):
    return add_atr(pd.read_parquet(path).set_index("timestamp"))


def add_atr(bars):
    rng5 = bars.high.rolling(5).max() - bars.low.rolling(5).min()
    bars["atr"] = rng5.rolling(1440, min_periods=60).mean().bfill()  # mean 5m range over a day
    return bars


@njit(cache=True)
def _run(o, h, l, c, atr, ema, start, end, deposit, p):
    dirs, step_v, step_atr, step_grow = p[0], p[1], p[2], p[3]
    mult, max_orders, tp_v, tp_atr = p[4], p[5], p[6], p[7]
    sl, leverage, risk = p[8], p[9], p[10]
    n_ord = np.zeros(2, np.int64)
    added = np.zeros(2, np.bool_)
    oz = np.zeros(2)
    cost = np.zeros(2)       # sum(oz * entry)
    level = np.zeros(2)      # last grid level
    step = np.zeros(2)       # next step size
    tp = np.zeros(2)
    base = np.zeros(2)
    open_eq = np.zeros(2)
    balance = deposit
    equity = np.empty(end - start)
    cycles = 0
    stops = 0
    stopouts = 0
    min_eq = deposit
    died = -1

    for i in range(start, end):
        k = i - start
        # open baskets
        for s in range(2):
            if n_ord[s] > 0:
                continue
            want = (dirs == s) or dirs == 2 or (dirs == 3 and (c[i - 1] > ema[i - 1]) == (s == 0))
            if not want:
                continue
            eq_now = balance
            for t in range(2):
                if n_ord[t] > 0:
                    eq_now += (oz[t] * o[i] - cost[t]) if t == 0 else (cost[t] - oz[t] * o[i])
            b = 1.0 if risk == 0 else max(1.0, np.floor(eq_now / 1000.0 * risk))
            base[s] = b
            step[s] = step_v * atr[i - 1] if step_atr else step_v
            tp[s] = tp_v * atr[i - 1] if tp_atr else tp_v
            level[s] = o[i]
            px = o[i] + SPREAD if s == 0 else o[i] - SPREAD
            n_ord[s] = 1
            oz[s] = b
            cost[s] = b * px
            open_eq[s] = eq_now

        # adverse side: adds
        added[:] = False
        for s in range(2):
            if n_ord[s] == 0:
                continue
            while max_orders == 0 or n_ord[s] < max_orders:
                nxt = level[s] - step[s] if s == 0 else level[s] + step[s]
                if (s == 0 and l[i] > nxt) or (s == 1 and h[i] < nxt):
                    break
                lot = max(1.0, np.round(base[s] * mult ** n_ord[s]))
                if leverage > 0:  # broker rejects the add without free margin
                    eq_w = balance
                    marg = lot * nxt / leverage
                    for t in range(2):
                        if n_ord[t] > 0:
                            eq_w += (oz[t] * nxt - cost[t]) if t == 0 else (cost[t] - oz[t] * nxt)
                            marg += oz[t] * nxt / leverage
                    if eq_w < marg:
                        break
                level[s] = nxt
                step[s] *= step_grow
                px = nxt + SPREAD if s == 0 else nxt - SPREAD
                n_ord[s] += 1
                added[s] = True
                oz[s] += lot
                cost[s] += lot * px

        # adverse side: basket stop
        if sl > 0:
            for s in range(2):
                if n_ord[s] == 0:
                    continue
                worst = l[i] if s == 0 else h[i]
                fl = (oz[s] * worst - cost[s]) if s == 0 else (cost[s] - oz[s] * worst)
                lim = -sl * open_eq[s]
                if fl <= lim:
                    balance += lim
                    n_ord[s] = 0
                    stops += 1

        # adverse side: equity / margin stop-out
        eq_w = balance
        marg = 0.0
        for s in range(2):
            if n_ord[s] > 0:
                worst = l[i] if s == 0 else h[i]
                eq_w += (oz[s] * worst - cost[s]) if s == 0 else (cost[s] - oz[s] * worst)
                if leverage > 0:
                    marg += oz[s] * worst / leverage
        min_eq = min(min_eq, eq_w)
        if eq_w <= 0 or (leverage > 0 and marg > 0 and eq_w <= STOP_OUT * marg):
            balance = max(eq_w, 0.0)
            n_ord[:] = 0
            stopouts += 1
            if balance < DEAD * deposit:
                died = i
                equity[k:] = balance
                break

        # favourable side: take profit
        for s in range(2):
            if n_ord[s] == 0 or added[s]:
                continue
            avg = cost[s] / oz[s]
            if s == 0 and h[i] >= avg + tp[s]:
                balance += oz[s] * tp[s]
                n_ord[s] = 0
                cycles += 1
            elif s == 1 and l[i] <= avg - tp[s]:
                balance += oz[s] * tp[s]
                n_ord[s] = 0
                cycles += 1

        eq_c = balance
        for s in range(2):
            if n_ord[s] > 0:
                eq_c += (oz[s] * c[i] - cost[s]) if s == 0 else (cost[s] - oz[s] * c[i])
        equity[k] = eq_c

    if died < 0:  # close what is left at the last close
        balance = equity[end - start - 1]
    return balance, min_eq, died, cycles, stops, stopouts, equity


@njit(parallel=True, cache=True)
def _run_batch(o, h, l, c, atr, emas, ema_idx, start, end, deposits, P):
    out = np.empty(len(deposits))
    for j in prange(len(deposits)):
        out[j] = _run(o, h, l, c, atr, emas[ema_idx[j]], start, end, deposits[j], P[j])[0]
    return out


EMA_LENGTHS = (0, 1440, 7200, 28800)


class Market:
    def __init__(self, bars=None):
        self.bars = load_m1() if bars is None else bars
        b = self.bars
        self.o, self.h, self.l, self.c = (b[x].to_numpy() for x in ("open", "high", "low", "close"))
        self.atr = b.atr.to_numpy()
        self.index = b.index
        self._ema = {}

    def ema(self, n):
        if n not in self._ema:
            self._ema[n] = (self.bars.close.ewm(span=n).mean().to_numpy() if n
                            else np.zeros(len(self.c)))
        return self._ema[n]

    def span(self, start, end):
        i0 = max(int(self.index.searchsorted(pd.Timestamp(start, tz="UTC"))), 1)
        i1 = int(self.index.searchsorted(pd.Timestamp(end, tz="UTC")))
        return i0, i1

    def run(self, p: Params, start, end, deposit=1000.0):
        i0, i1 = self.span(start, end)
        arr = np.array(astuple(p), dtype=np.float64)
        bal, min_eq, died, cyc, stops, so, eq = _run(
            self.o, self.h, self.l, self.c, self.atr, self.ema(p.ema), i0, i1, deposit, arr)
        return dict(balance=bal, min_eq=min_eq, died=self.index[died] if died >= 0 else None,
                    cycles=cyc, stops=stops, stopouts=so,
                    equity=pd.Series(eq, self.index[i0:i1]))

    def run_batch(self, params, deposits, start, end):
        """End balances of many accounts over the same span (parallel)."""
        i0, i1 = self.span(start, end)
        emas = np.stack([self.ema(n) for n in EMA_LENGTHS])
        idx = np.array([EMA_LENGTHS.index(p.ema) for p in params], dtype=np.int64)
        P = np.array([astuple(p) for p in params], dtype=np.float64)
        return _run_batch(self.o, self.h, self.l, self.c, self.atr, emas, idx, i0, i1,
                          np.asarray(deposits, dtype=np.float64), P)

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
STOP_OUT = 0.3   # margin level at which the broker closes everything (Spotware: 30%)
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
    # regime filters, checked when a basket opens (0 = off)
    er_win: int = 0        # efficiency-ratio window: 0 off, 1 = 1h, 2 = 4h, 3 = 1 day
    er_max: float = 0.0    # open only while ER < er_max (low ER = choppy market)
    sess: int = 0          # 0 any, 1 Asia 0-7, 2 London 7-12, 3 New York 12-17, 4 late 17-24 UTC
    vol_max: float = 0.0   # open only while ATR / 20-day mean ATR < vol_max
    # PWM of aggressiveness: with pwm_k > 0 the lot grows as risk(E) = risk * (1 - (D / E) ** pwm_k)
    # oz per $1000 of equity E (D = deposit, 0 while E <= D), never below 0.01 lot
    pwm_k: float = 0.0
    # profit withdrawal: every wd_days trading days, when no basket is open, take wd_frac of
    # the balance above wd_floor x deposit off the account (0 = never)
    wd_days: int = 0
    wd_frac: float = 0.0
    wd_floor: float = 1.0
    # dynamic daily withdrawal (wd_days = -1): each new trading day, when no basket is open,
    # withdraw wd_frac * m_vol * m_age of the balance, m_vol = clip(ATR / 20-day ATR, 0.5, 2) ** wd_vol_g,
    # m_age = 1 + age_days / wd_tau (1 if wd_tau = 0); never below wd_floor x deposit
    wd_vol_g: float = 0.0
    wd_tau: float = 0.0


INT_GENES = {"dirs", "step_atr", "max_orders", "tp_atr", "ema", "er_win", "sess"}


def params_from(row):
    """Params from a mapping of gene values (e.g. a DataFrame row)."""
    return Params(**{k: int(row[k]) if k in INT_GENES else float(row[k])
                     for k in Params.__dataclass_fields__})


def load_m1(path=DATA):
    return add_atr(pd.read_parquet(path).set_index("timestamp"))


ER_WINDOWS = (60, 240, 1440)
SESSIONS = ((0, 24), (0, 7), (7, 12), (12, 17), (17, 24))


def filters(bars):
    """Rows: hour, ATR ratio, ER for each of ER_WINDOWS."""
    c = bars.close
    path = c.diff().abs()
    rows = [bars.index.hour.to_numpy().astype(np.float64),
            (bars.atr / bars.atr.rolling(28800, min_periods=1440).mean()).bfill().to_numpy()]
    for w in ER_WINDOWS:
        er = (c - c.shift(w)).abs() / path.rolling(w).sum()
        rows.append(er.fillna(1.0).to_numpy())
    return np.stack(rows)


def add_atr(bars):
    rng5 = bars.high.rolling(5).max() - bars.low.rolling(5).min()
    bars["atr"] = rng5.rolling(1440, min_periods=60).mean().bfill()  # mean 5m range over a day
    return bars


@njit(cache=True)
def _run(o, h, l, c, atr, ema, filt, start, end, deposit, p):
    return _run_core(o, h, l, c, atr, ema, filt, start, end, deposit, p,
                     np.empty(0, np.int64), np.empty(0), False)


@njit(cache=True)
def _run_core(o, h, l, c, atr, ema, filt, start, end, deposit, p, dep_idx, dep_amt, rec_wd):
    """dep_idx / dep_amt: external deposits (bar, $), sorted. A dead account with deposits
    still to come goes dormant and restarts on the next one (deposit reference = new balance).
    rec_wd: also return cumulative withdrawn per bar."""
    dirs, step_v, step_atr, step_grow = p[0], p[1], p[2], p[3]
    mult, max_orders, tp_v, tp_atr = p[4], p[5], p[6], p[7]
    sl, leverage, risk = p[8], p[9], p[10]
    er_win, er_max, sess, vol_max = int(p[12]), p[13], int(p[14]), p[15]
    pwm_k = p[16]
    wd_days, wd_frac, wd_floor = int(p[17]), p[18], p[19]
    wd_vol_g, wd_tau = p[20], p[21]
    withdrawn = 0.0
    days = 0
    age = 0
    sess_lo = (0, 0, 7, 12, 17)[sess]
    sess_hi = (24, 7, 12, 17, 24)[sess]
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
    peak = deposit
    maxdd = 0.0              # max drawdown of worst-case equity from the closing peak
    died = -1
    deaths = 0
    dormant = False
    di = 0
    wd_arr = np.zeros(end - start if rec_wd else 1)

    for i in range(start, end):
        k = i - start
        while di < len(dep_idx) and dep_idx[di] <= i:
            balance += dep_amt[di]
            di += 1
            if dormant:
                dormant = False
                deposit = balance
                peak = balance
        if rec_wd:
            wd_arr[k] = withdrawn
        if dormant:
            equity[k] = balance
            continue
        if wd_days < 0 and filt[0, i] < filt[0, i - 1]:  # dynamic daily withdrawal
            age += 1
            if n_ord[0] == 0 and n_ord[1] == 0 and balance > wd_floor * deposit:
                v = min(max(filt[1, i - 1], 0.5), 2.0) ** wd_vol_g
                rate = wd_frac * v * ((1.0 + age / wd_tau) if wd_tau > 0 else 1.0)
                x = min(rate * balance, balance - wd_floor * deposit)
                balance -= x
                withdrawn += x
                peak -= x
        if wd_days > 0 and filt[0, i] < filt[0, i - 1]:  # new trading day
            days += 1
            if days >= wd_days and n_ord[0] == 0 and n_ord[1] == 0:
                days = 0
                excess = balance - wd_floor * deposit
                if excess > 0:
                    x = wd_frac * excess
                    balance -= x
                    withdrawn += x
                    peak -= x
        # open baskets
        for s in range(2):
            if n_ord[s] > 0:
                continue
            want = (dirs == s) or dirs == 2 or (dirs == 3 and (c[i - 1] > ema[i - 1]) == (s == 0))
            if not want:
                continue
            hr = filt[0, i]
            if hr < sess_lo or hr >= sess_hi:
                continue
            if vol_max > 0 and filt[1, i - 1] >= vol_max:
                continue
            if er_win > 0 and er_max > 0 and filt[1 + er_win, i - 1] >= er_max:
                continue
            eq_now = balance
            for t in range(2):
                if n_ord[t] > 0:
                    eq_now += (oz[t] * o[i] - cost[t]) if t == 0 else (cost[t] - oz[t] * o[i])
            r = risk
            if pwm_k > 0:
                r = risk * (1.0 - (deposit / eq_now) ** pwm_k) if eq_now > deposit else 0.0
            b = 1.0 if r <= 0 else max(1.0, np.floor(eq_now / 1000.0 * r))
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
        maxdd = max(maxdd, 1.0 - eq_w / peak)
        if eq_w <= 0 or (leverage > 0 and marg > 0 and eq_w <= STOP_OUT * marg):
            balance = max(eq_w, 0.0)
            n_ord[:] = 0
            stopouts += 1
            if balance < DEAD * deposit:
                deaths += 1
                maxdd = 1.0
                if di < len(dep_idx):
                    dormant = True
                    equity[k] = balance
                    continue
                died = i
                equity[k:] = balance
                if rec_wd:
                    wd_arr[k:] = withdrawn
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
        peak = max(peak, eq_c)

    if died < 0:  # close what is left at the last close
        balance = equity[end - start - 1]
    return balance, min_eq, died, cycles, stops, stopouts, equity, min(maxdd, 1.0), withdrawn, deaths, wd_arr


@njit(parallel=True, cache=True)
def _run_batch(o, h, l, c, atr, emas, ema_idx, filt, start, end, deposits, P):
    out = np.empty(len(deposits))
    dd = np.empty(len(deposits))
    for j in prange(len(deposits)):
        r = _run(o, h, l, c, atr, emas[ema_idx[j]], filt, start, end, deposits[j], P[j])
        out[j] = r[0]
        dd[j] = r[7]
    return out, dd


EMA_LENGTHS = (0, 1440, 7200, 28800)


class Market:
    def __init__(self, bars=None):
        self.bars = load_m1() if bars is None else bars
        b = self.bars
        self.o, self.h, self.l, self.c = (b[x].to_numpy() for x in ("open", "high", "low", "close"))
        self.atr = b.atr.to_numpy()
        self.index = b.index
        self._ema = {}
        self.filt = filters(b)

    def ema(self, n):
        if n not in self._ema:
            self._ema[n] = (self.bars.close.ewm(span=n).mean().to_numpy() if n
                            else np.zeros(len(self.c)))
        return self._ema[n]

    def span(self, start, end):
        utc = lambda t: (t if t.tzinfo else t.tz_localize("UTC")).floor("min")
        i0 = max(int(self.index.searchsorted(utc(pd.Timestamp(start)))), 1)
        i1 = int(self.index.searchsorted(utc(pd.Timestamp(end))))
        return i0, i1

    def run(self, p: Params, start, end, deposit=1000.0):
        i0, i1 = self.span(start, end)
        arr = np.array(astuple(p), dtype=np.float64)
        bal, min_eq, died, cyc, stops, so, eq, maxdd, wd, _, _ = _run(
            self.o, self.h, self.l, self.c, self.atr, self.ema(p.ema), self.filt, i0, i1, deposit, arr)
        return dict(balance=bal, min_eq=min_eq, died=self.index[died] if died >= 0 else None,
                    cycles=cyc, stops=stops, stopouts=so, maxdd=maxdd, withdrawn=wd,
                    equity=pd.Series(eq, self.index[i0:i1]))

    def _genes(self, params):
        emas = np.stack([self.ema(n) for n in EMA_LENGTHS])
        idx = np.array([EMA_LENGTHS.index(p.ema) for p in params], dtype=np.int64)
        return emas, idx, np.array([astuple(p) for p in params], dtype=np.float64)

    def lifecycles(self, params, first="2017-01", last="2024-09", months=24, deposit=1000.0,
                   start_times=None, deposits=None):
        """Start every config on the 1st of every month first..last (or at each of
        `start_times`), run up to `months` (checkpoints every 30.44 days for start_times).

        Returns death bar per (config, start) (-1 = alive at the horizon), equity at each
        monthly checkpoint (config, start, month) and the start timestamps; the max
        drawdown per (config, start) is left in self.last_dd.
        """
        if start_times is None:
            periods = pd.period_range(first, last, freq="M")
            t0 = [p.start_time for p in periods]
            tk = [[(p + k + 1).start_time for k in range(months)] for p in periods]
        else:
            t0 = list(start_times)
            tk = [[t + pd.Timedelta(days=30.44 * (k + 1)) for k in range(months)] for t in t0]
        def pos(t):
            t = pd.Timestamp(t).floor("min")
            return max(int(self.index.searchsorted(t if t.tzinfo else t.tz_localize("UTC"))), 1)

        starts = np.array([pos(t) for t in t0])
        checks = np.array([[pos(t) for t in row] for row in tk])
        emas, idx, P = self._genes(params)
        died, eq, dd, wd = _run_lifecycles(self.o, self.h, self.l, self.c, self.atr, emas, idx, self.filt,
                                       starts, checks, np.full(len(starts), float(deposit)) if deposits is None
                                       else np.asarray(deposits, dtype=np.float64), P)
        self.last_dd = dd  # max drawdown per (config, start) over the whole run
        self.last_wd = wd  # money withdrawn per (config, start) over the whole run
        return died, eq, self.index[starts]

    def run_batch(self, params, deposits, start, end):
        """End balances and max drawdowns of many accounts over the same span (parallel)."""
        i0, i1 = self.span(start, end)
        emas = np.stack([self.ema(n) for n in EMA_LENGTHS])
        idx = np.array([EMA_LENGTHS.index(p.ema) for p in params], dtype=np.int64)
        P = np.array([astuple(p) for p in params], dtype=np.float64)
        return _run_batch(self.o, self.h, self.l, self.c, self.atr, emas, idx, self.filt, i0, i1,
                          np.asarray(deposits, dtype=np.float64), P)


@njit(parallel=True, cache=True)
def _run_lifecycles(o, h, l, c, atr, emas, ema_idx, filt, starts, checks, deposits, P):
    """Each config from each start: bar of death (-1 if alive) and equity at checkpoints.

    checks[s, k] is the bar index of checkpoint k for start s (end of month k + 1).
    """
    J, S, K = len(P), len(starts), checks.shape[1]
    died = np.full((J, S), -1, np.int64)
    dd = np.empty((J, S))
    wd = np.empty((J, S))
    eq = np.empty((J, S, K))
    for js in prange(J * S):
        j, s = js // S, js % S
        i0, i1 = starts[s], checks[s, K - 1]
        r = _run(o, h, l, c, atr, emas[ema_idx[j]], filt, i0, i1, deposits[s], P[j])
        died[j, s] = r[2]
        dd[j, s] = r[7]
        wd[j, s] = r[8]
        for k in range(K):
            eq[j, s, k] = r[6][checks[s, k] - 1 - i0]
    return died, eq, dd, wd

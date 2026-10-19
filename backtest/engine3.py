"""Intraday single-position engine (numba, XAUUSD M1, gram contract) for session entities.

One account trades only inside its session window and is flat at the session end.
Broker: gram units (0.01 lot = 1 g), spread $0.20 + $25/$1M per side, leverage 1:100,
stop-out 30% (no swaps: never held overnight).

Entries (pending stop/limit orders, OCO, valid until the session end):
  0 opening-range breakout: range of the first `orb` minutes of the session, stop orders beyond it
  1 Asian-range breakout: today's 00:00-07:00 UTC range (sessions starting at 07:00 or later)
  2 previous-day high / low breakout
  3 N-bar breakout (LOOKBACKS)
  4 pullback: limit at EMA -/+ offset in the direction of the close vs EMA
  5 fade: limit orders at the session high / low so far +/- offset (after `orb` minutes)
  6 market at the session open in the direction of the higher-timeframe trend
Filters: higher-timeframe trend (daily EMA of TREND_DAYS) with / against / off, direction
mask, weekday mask, max trades per day, daily loss stop, daily profit target.
Exits: SL and TP in ATR (always set), break-even, trailing, time stop, session end.
Money: risk % of equity at the SL distance (compounding), x win_mult after a win,
x loss_mult after a loss (each capped at x mult_cap), reset otherwise.
The run stops for good when equity >= target x deposit (money taken), when the worst
equity falls below (1 - kill_dd) x its peak (controlled death, rest kept), or when it
can no longer afford 1 g. Intrabar order is pessimistic (stop before target, no target
in the fill bar).
"""
from dataclasses import dataclass, astuple

import numpy as np
import pandas as pd
from numba import njit, prange

from engine import load_m1

GRAM = 1 / 31.1035
SPREAD = 0.20
COMMISSION = 25e-6
LEVERAGE = 100.0
STOP_OUT = 0.30

# session windows in UTC hours [lo, hi): Asia, London, New York, London-NY overlap,
# Asia-London overlap, London+NY, late US, all day
SESSIONS = ((0, 7), (7, 12), (12, 17), (12, 16), (6, 8), (7, 17), (17, 21), (0, 21))
LOOKBACKS = (15, 60, 240)
EMAS = (60, 240, 1440)
TREND_DAYS = (5, 20)


@dataclass
class Genes:
    sess: int = 1
    entry: int = 0
    orb: int = 30           # opening-range minutes
    look: int = 1
    ema: int = 1
    offset: float = 0.1     # ATR beyond the level
    dirs: int = 0           # 0 both, 1 long, 2 short
    trend: int = 0          # 0 off, 1 trade with the daily trend, 2 against it
    trend_len: int = 0      # index into TREND_DAYS
    sl: float = 1.0         # ATR
    tp: float = 2.0         # ATR
    be: float = 0.0         # ATR in profit -> stop to entry
    trail_start: float = 0.0
    trail: float = 1.0
    max_hold: int = 0       # bars, 0 = until the session end
    max_trades: int = 2     # per day
    day_loss: float = 0.0   # stop for the day after losing this share of the day-start equity
    day_profit: float = 0.0 # stop for the day after making this share
    skip_mon: int = 0
    skip_fri: int = 0
    risk: float = 0.02
    win_mult: float = 1.0
    loss_mult: float = 1.0
    mult_cap: float = 1.0
    target: float = 2.0
    kill_dd: float = 0.5
    deposit: float = 100.0


def indicators(bars):
    c, h, l = bars.close, bars.high, bars.low
    hh = np.stack([h.rolling(n).max().shift(1).bfill().to_numpy() for n in LOOKBACKS])
    ll = np.stack([l.rolling(n).min().shift(1).bfill().to_numpy() for n in LOOKBACKS])
    ema = np.stack([c.ewm(span=n).mean().to_numpy() for n in EMAS])
    tr = np.stack([c.ewm(span=d * 1440).mean().to_numpy() for d in TREND_DAYS])
    day = bars.index.floor("D")
    dh, dl = h.groupby(day).max(), l.groupby(day).min()
    prev_hi = pd.Series(dh.shift(1).reindex(day).to_numpy(), index=bars.index).bfill().to_numpy()
    prev_lo = pd.Series(dl.shift(1).reindex(day).to_numpy(), index=bars.index).bfill().to_numpy()
    asia = bars.index.hour < 7
    ah = h.where(asia).groupby(day).max().reindex(day).to_numpy()
    al = l.where(asia).groupby(day).min().reindex(day).to_numpy()
    ah, al = np.nan_to_num(ah, nan=1e9), np.nan_to_num(al, nan=-1e9)
    ts = bars.index
    hour = ts.hour.to_numpy().astype(np.int64)
    dnum = ((ts - pd.Timestamp("2000-01-01", tz="UTC")).days).to_numpy().astype(np.int64)
    wday = ts.dayofweek.to_numpy().astype(np.int64)
    return hh, ll, ema, tr, prev_hi, prev_lo, ah, al, hour, dnum, wday


@njit(cache=True)
def _run(o, h, l, c, atr, hh, ll, ema, tr, phi, plo, ah, al, hour, dnum, wday, start, end, g, checks, out):
    sess, entry, orb, look, ema_i = int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4])
    offset, dirs, trend, trend_i = g[5], int(g[6]), int(g[7]), int(g[8])
    sl_k, tp_k, be_k, tr_start, tr_k, max_hold = g[9], g[10], g[11], g[12], g[13], int(g[14])
    max_tr, day_loss, day_profit, skip_mon, skip_fri = int(g[15]), g[16], g[17], int(g[18]), int(g[19])
    risk0, win_mult, loss_mult, mult_cap, target, kill_dd, dep = g[20], g[21], g[22], g[23], g[24], g[25], g[26]
    s_lo = (0, 7, 12, 12, 6, 7, 17, 0)[sess]
    s_hi = (7, 12, 17, 16, 8, 17, 21, 21)[sess]
    bal = dep
    peak = dep
    risk = risk0
    pos = 0
    oz = 0.0
    px = 0.0
    stop = 0.0
    tgt = 0.0
    a_fill = 0.0
    held = 0
    buy_lvl = -1.0
    sell_lvl = -1.0
    lim = False
    in_sess = False
    s_start = 0
    s_hi_px = 0.0
    s_lo_px = 1e18
    placed = False
    trades_day = 0
    day_eq = dep
    halted = False
    status = 0            # 0 running/horizon, 1 target hit, 2 killed, 3 broke
    t_end = -1
    ck = 0
    eq_c = dep
    for i in range(start, end):
        if dnum[i] != dnum[i - 1]:
            trades_day = 0
            day_eq = bal
            halted = False
        hr = hour[i]
        now_in = hr >= s_lo and hr < s_hi and not ((skip_mon and wday[i] == 0) or (skip_fri and wday[i] == 4))
        if now_in and not in_sess:  # session opens
            s_start = i
            s_hi_px = h[i]
            s_lo_px = l[i]
            placed = False
            buy_lvl = -1.0
            sell_lvl = -1.0
        in_sess = now_in
        if in_sess:
            s_hi_px = max(s_hi_px, h[i])
            s_lo_px = min(s_lo_px, l[i])
        last_bar = in_sess and (i + 1 >= end or hour[i + 1] >= s_hi or hour[i + 1] < s_lo or dnum[i + 1] != dnum[i])

        # place orders
        if in_sess and pos == 0 and not placed and not halted and trades_day < max_tr:
            a = atr[i - 1]
            up = c[i - 1] > tr[trend_i, i - 1]
            ready = True
            bl, sl_ = -1.0, -1.0
            lim = False
            if entry == 0 or entry == 5:
                ready = i - s_start >= orb
                if ready and entry == 0:
                    bl, sl_ = s_hi_px + offset * a, s_lo_px - offset * a
                elif ready:
                    bl, sl_ = s_lo_px - offset * a, s_hi_px + offset * a
                    lim = True
            elif entry == 1:
                if ah[i] < 1e8 and s_lo >= 7:
                    bl, sl_ = ah[i] + offset * a, al[i] - offset * a
                else:
                    ready = False
            elif entry == 2:
                bl, sl_ = phi[i] + offset * a, plo[i] - offset * a
            elif entry == 3:
                bl, sl_ = hh[look, i] + offset * a, ll[look, i] - offset * a
            elif entry == 4:
                e = ema[ema_i, i - 1]
                bl = e - offset * a if c[i - 1] > e else -1.0
                sl_ = e + offset * a if c[i - 1] <= e else -1.0
                lim = True
            else:
                bl = 1.0 if up else -1.0
                sl_ = -1.0 if up else 1.0
            if ready:
                if dirs == 2 or (trend == 1 and not up) or (trend == 2 and up):
                    bl = -1.0
                if dirs == 1 or (trend == 1 and up) or (trend == 2 and not up):
                    sl_ = -1.0
                buy_lvl, sell_lvl = bl, sl_
                placed = buy_lvl > 0 or sell_lvl > 0

        # fills
        filled = False
        if in_sess and pos == 0 and placed and not last_bar:
            side = 0
            fill = 0.0
            if entry == 6:
                if buy_lvl > 0:
                    side, fill = 1, o[i]
                elif sell_lvl > 0:
                    side, fill = -1, o[i]
            elif lim:
                if buy_lvl > 0 and l[i] <= buy_lvl:
                    side, fill = 1, min(buy_lvl, o[i])
                elif sell_lvl > 0 and h[i] >= sell_lvl:
                    side, fill = -1, max(sell_lvl, o[i])
            else:
                if buy_lvl > 0 and h[i] >= buy_lvl:
                    side, fill = 1, max(buy_lvl, o[i])
                elif sell_lvl > 0 and l[i] <= sell_lvl:
                    side, fill = -1, min(sell_lvl, o[i])
            if side != 0:
                a = atr[i - 1]
                dist = sl_k * a
                size = np.floor(bal * risk / dist / GRAM) * GRAM
                cap = np.floor(bal * LEVERAGE / fill * 0.9 / GRAM) * GRAM
                size = min(max(size, GRAM), cap)
                placed = True  # one fill per placement; re-arm only after the trade
                buy_lvl, sell_lvl = -1.0, -1.0
                if size >= GRAM:
                    cost = SPREAD + 2 * COMMISSION * fill
                    pos, oz = side, size
                    px = fill + cost if side > 0 else fill - cost
                    stop = fill - side * dist
                    tgt = fill + side * tp_k * a
                    a_fill = a
                    held = 0
                    filled = True
                    trades_day += 1

        # manage
        if pos != 0:
            held += 1
            ex = -1.0
            if pos > 0 and l[i] <= stop:
                ex = stop if filled else min(stop, o[i])
            elif pos < 0 and h[i] >= stop:
                ex = stop if filled else max(stop, o[i])
            if ex < 0:
                worst = l[i] if pos > 0 else h[i]
                eq_w = bal + pos * oz * (worst - px)
                if eq_w <= STOP_OUT * oz * worst / LEVERAGE:
                    ex = worst
            if ex < 0 and not filled:
                if (pos > 0 and h[i] >= tgt) or (pos < 0 and l[i] <= tgt):
                    ex = tgt
            if ex < 0 and ((max_hold > 0 and held >= max_hold) or last_bar):
                ex = c[i]
            if ex >= 0:
                pnl = pos * oz * (ex - px)
                bal += pnl
                if pnl > 0:
                    risk = min(risk * win_mult, risk0 * mult_cap) if win_mult > 1 else risk0
                else:
                    risk = min(risk * loss_mult, risk0 * mult_cap) if loss_mult > 1 else risk0
                pos = 0
                placed = False  # can re-arm in the same session
                if day_loss > 0 and bal <= day_eq * (1 - day_loss):
                    halted = True
                if day_profit > 0 and bal >= day_eq * (1 + day_profit):
                    halted = True
            else:
                fav = h[i] - px if pos > 0 else px - l[i]
                if be_k > 0 and fav >= be_k * a_fill:
                    stop = max(stop, px) if pos > 0 else min(stop, px)
                if tr_start > 0 and fav >= tr_start * a_fill:
                    stop = max(stop, h[i] - tr_k * a_fill) if pos > 0 else min(stop, l[i] + tr_k * a_fill)
        if last_bar and pos == 0:
            placed = True  # nothing new in the closing bar

        worst_eq = bal
        eq_c = bal
        if pos != 0:
            worst_eq += pos * oz * ((l[i] if pos > 0 else h[i]) - px)
            eq_c += pos * oz * (c[i] - px)
        if eq_c > peak:
            peak = eq_c
        while ck < len(checks) and i + 1 >= checks[ck]:
            out[ck] = eq_c
            ck += 1
        if eq_c >= target * dep:
            status, t_end = 1, i
            break
        if kill_dd > 0 and worst_eq <= (1 - kill_dd) * peak:
            eq_c = max(worst_eq, 0.0)
            status, t_end = 2, i
            break
        if pos == 0 and eq_c * LEVERAGE < c[i] * GRAM * 1.2:
            status, t_end = 3, i
            break
    while ck < len(checks):
        out[ck] = eq_c
        ck += 1
    return eq_c, status, t_end


@njit(parallel=True, cache=True)
def _batch(o, h, l, c, atr, hh, ll, ema, tr, phi, plo, ah, al, hour, dnum, wday, starts, checks, G):
    J, S, K = len(G), len(starts), checks.shape[1]
    eq = np.empty((J, S, K))
    st = np.empty((J, S), np.int64)
    te = np.empty((J, S), np.int64)
    for js in prange(J * S):
        j, s = js // S, js % S
        out = np.empty(K)
        r = _run(o, h, l, c, atr, hh, ll, ema, tr, phi, plo, ah, al, hour, dnum, wday,
                 starts[s], checks[s, K - 1], G[j], checks[s], out)
        eq[j, s] = out
        st[j, s] = r[1]
        te[j, s] = r[2]
    return eq, st, te


class Market3:
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

    def batch(self, genes, starts, days):
        """Equity at the end of each day 1..days for every genome and start; status per run
        (0 still running at the horizon, 1 target, 2 kill switch, 3 broke) and end bar."""
        st = np.array([self.pos(t) for t in starts])
        ck = np.array([[self.pos(pd.Timestamp(t) + pd.Timedelta(days=k + 1)) for k in range(days)] for t in starts])
        G = np.array([astuple(g) for g in genes], dtype=np.float64)
        return _batch(self.o, self.h, self.l, self.c, self.atr, *self.ind, st, ck, G)

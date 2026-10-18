"""PWM of two entities: a safe one and an aggressive one share the capital.

The account is split each week: the aggressive entity gets the share
    w = w_max * (1 - (D / E) ** k)      (0 while E <= D),
of the current capital E (D = starting deposit), the safe entity the rest.
Both trade the week on their own sub-account; everything is closed at the end
of the week (Friday close, so no weekend gap) and the capital is pooled and
split again. If the aggressive part blows up only its share is lost.

The account is dead when E falls below DEAD of the deposit.

    python3 pwm.py
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

from engine import DEAD, EMA_LENGTHS, Market, Params, _run, params_from

ROOT = Path(__file__).resolve().parent.parent
WEEKS = 104

# safe entity: best balanced $300 config of lifecycle3 (docs/08_growth_search.md)
SAFE = Params(dirs=0, step=2.0, step_atr=1, step_grow=1.5, mult=1.0, max_orders=0, tp=20.0, tp_atr=0,
              sl=0.0, leverage=100.0, risk=0.0, ema=0, er_win=1, er_max=0.05, sess=2, vol_max=0.7)


@njit(parallel=True, cache=True)
def _pwm(o, h, l, c, atr, emas, filt, weeks, starts_w, deposit, Ps, safe_ema, Pa, agro_ema, ks, wmaxs):
    """weeks: bar index of each week boundary; starts_w: index into weeks of each start.
    Returns capital at the end of each of WEEKS weeks, per (pair, start)."""
    J, S = len(Pa), len(starts_w)
    out = np.empty((J, S, WEEKS))
    for js in prange(J * S):
        j, s = js // S, js % S
        E = deposit
        dead = False
        for k in range(WEEKS):
            wi = starts_w[s] + k
            if dead or wi + 1 >= len(weeks):
                out[j, s, k] = E
                continue
            a, b = weeks[wi], weeks[wi + 1]
            w = 0.0
            if E > deposit:
                w = wmaxs[j] * (1.0 - (deposit / E) ** ks[j])
            ea = E * w
            es = E - ea
            if es > 0:
                es = _run(o, h, l, c, atr, emas[safe_ema], filt, a, b, es, Ps)[0]
            if ea > 0:
                ea = _run(o, h, l, c, atr, emas[agro_ema[j]], filt, a, b, ea, Pa[j])[0]
            E = max(es, 0.0) + max(ea, 0.0)
            if E < DEAD * deposit:
                dead = True
            out[j, s, k] = E
    return out


class PWM:
    def __init__(self, m: Market):
        self.m = m
        idx = m.index
        wk = pd.date_range(idx[0].normalize(), idx[-1], freq="W-MON", tz="UTC")
        self.week_t = wk
        self.weeks = np.array([m.span(t, "2100-01-01")[0] for t in wk])
        self.emas = np.stack([m.ema(n) for n in EMA_LENGTHS])

    def run(self, agros, ks, wmaxs, starts, deposit=300.0, safe=SAFE):
        sw = np.searchsorted(self.week_t, pd.DatetimeIndex(starts))
        Pa = np.array([[float(x) for x in a.__dict__.values()] for a in agros])
        aidx = np.array([EMA_LENGTHS.index(a.ema) for a in agros], dtype=np.int64)
        Ps = np.array([float(x) for x in safe.__dict__.values()])
        return _pwm(self.m.o, self.m.h, self.m.l, self.m.c, self.m.atr, self.emas, self.m.filt, self.weeks,
                    sw, float(deposit), Ps, EMA_LENGTHS.index(safe.ema), Pa, aidx,
                    np.asarray(ks, dtype=np.float64), np.asarray(wmaxs, dtype=np.float64))


def stats(E, deposit):
    x = E / deposit
    alive = x >= DEAD
    q = lambda k, p: np.percentile(x[:, :, k - 1], p, axis=1)
    return pd.DataFrame(dict(s26=alive[:, :, 25].mean(1), s52=alive[:, :, 51].mean(1),
                             s104=alive[:, :, 103].mean(1), med26=q(26, 50), med52=q(52, 50),
                             q25_52=q(52, 25), q75_52=q(52, 75), med104=q(104, 50)))


if __name__ == "__main__":
    m = Market()
    p = PWM(m)
    train = pd.date_range("2017-01-02", "2022-12-31", freq="MS", tz="UTC")
    test = pd.date_range("2023-01-01", "2024-09-01", freq="MS", tz="UTC")

    # candidate aggressive entities: compounding top configs of lifecycle3 + the safe one with a growing lot
    agro = [replace(SAFE, risk=r) for r in (1.0, 2.0, 3.0, 5.0, 8.0)]
    for f in ("lifecycle3_d300_s1.parquet", "lifecycle3_d1000_s1.parquet"):
        df = pd.read_parquet(ROOT / "results" / f)
        df = df[df.risk >= 1.0].sort_values("med6", ascending=False)
        agro += [params_from(r) for _, r in df.head(40).iterrows()]
    agro = list({tuple(a.__dict__.values()): a for a in agro}.values())
    grid = [(a, k, w) for a in agro for k in (0.5, 1.0, 2.0) for w in (0.5, 0.8, 0.95, 1.0)]
    A, K, W = zip(*grid)
    print(f"{len(agro)} aggressive candidates, {len(grid)} PWM configs")

    base = stats(p.run([SAFE], [1.0], [0.0], train), 300).iloc[0]
    print("только безопасная (с недельным закрытием), обучение:", base.round(2).to_dict())
    res = stats(p.run(A, K, W, train), 300)
    res["score"] = np.log(np.maximum(res.med52, 1e-3)) - 5 * np.maximum(0, 0.75 - res.s52)
    top = res.sort_values("score", ascending=False).head(15)
    sel = [grid[i] for i in top.index]
    tst = stats(p.run([g[0] for g in sel], [g[1] for g in sel], [g[2] for g in sel], test), 300)
    out = top.reset_index(drop=True).join(tst.add_prefix("test_"))
    out.insert(0, "k", [g[1] for g in sel])
    out.insert(1, "w_max", [g[2] for g in sel])
    out.insert(2, "agro", [str(g[0]).replace("Params", "") for g in sel])
    base_t = stats(p.run([SAFE], [1.0], [0.0], test), 300).iloc[0]
    print("только безопасная, проверка:", base_t.round(2).to_dict())
    out.to_csv(ROOT / "results" / "pwm_top.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)
    print(out.drop(columns="agro").round(2).to_string())
    for i, g in enumerate(sel[:5]):
        print(i, g[0])

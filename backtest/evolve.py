"""Evolution of grid/martingale accounts, month by month.

Each organism is an account with Params genes. Every calendar month it trades
(open baskets are closed at month end), then gets one child per CHILD_PER $ of
the month's profit (at most MAX_CHILDREN). A child starts with DEPOSIT and the
parent's genes with one mutated gene. Accounts below DEAD_BELOW die. If the
population exceeds CAP, survivors are culled at random (no fitness bias).

Selection only uses the past, so the population's P&L in each month is out of
sample. `--mirror` runs the same on the inverted price (k / price): the same
noise with the trend flipped, as a control.

    python3 evolve.py [--mirror] [--n0 2000] [--seed 1]
"""
import argparse
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Market, Params, add_atr, load_m1
from search import sample

DEPOSIT = 1000.0
CHILD_PER = 100.0
MAX_CHILDREN = 10
DEAD_BELOW = 100.0
CAP = 20000
OUT = Path(__file__).resolve().parent.parent / "results"

CHOICES = {  # neighbour values for mutation
    "dirs": [0, 1, 2, 3], "ema": [1440, 7200, 28800], "step_atr": [0, 1],
    "step_grow": [1.0, 1.2, 1.5], "mult": [1.0, 1.2, 1.4, 1.6, 2.0],
    "max_orders": [0, 3, 5, 8, 12], "tp_atr": [0, 1],
    "sl": [0.0, 0.05, 0.1, 0.2, 0.4], "leverage": [0, 100, 500], "risk": [0, 1, 2],
}
REGIME = {  # regime-filter genes, used by evolve2.py
    "er_win": [0, 1, 2, 3], "er_max": [0.0, 0.02, 0.05, 0.1, 0.2],
    "sess": [0, 1, 2, 3, 4], "vol_max": [0.0, 0.7, 1.0, 1.5],
}
STEP_VALUES = {0: [3, 5, 8, 12, 20, 30], 1: [2, 4, 8, 16, 30]}
TP_VALUES = {0: [2, 5, 10, 20], 1: [1, 3, 6, 12]}


def mutate(p, rng, extra=()):
    g = rng.choice(list(CHOICES) + ["step", "tp"] + list(extra))
    if g in REGIME:
        return replace(p, **{g: rng.choice(REGIME[g])})
    if g == "step":
        return replace(p, step=rng.choice(STEP_VALUES[p.step_atr]))
    if g == "tp":
        return replace(p, tp=rng.choice(TP_VALUES[p.tp_atr]))
    q = replace(p, **{g: rng.choice(CHOICES[g])})
    if g == "step_atr":
        q.step = rng.choice(STEP_VALUES[q.step_atr])
    if g == "tp_atr":
        q.tp = rng.choice(TP_VALUES[q.tp_atr])
    if g == "dirs":
        q.ema = rng.choice(CHOICES["ema"]) if q.dirs == 3 else 0
    return q


def mirrored():
    b = load_m1()[["open", "high", "low", "close"]]
    k = b.close.iloc[0] ** 2
    m = pd.DataFrame({"open": k / b.open, "high": k / b.low, "low": k / b.high,
                      "close": k / b.close}, index=b.index)
    return add_atr(m)


def evolve(market, n0, seed):
    rng = random.Random(seed)
    pop = [dict(id=i, genes=sample(rng), bal=DEPOSIT, born="2017-01") for i in range(n0)]
    next_id = n0
    months = pd.period_range("2017-01", "2026-09", freq="M")
    log = []
    for mo in months:
        start, end = mo.start_time.strftime("%Y-%m-%d"), (mo + 1).start_time.strftime("%Y-%m-%d")
        cap0 = sum(o["bal"] for o in pop)
        bal0 = np.array([o["bal"] for o in pop])
        children, alive, alive_before = [], [], pop
        new_bal, _ = market.run_batch([o["genes"] for o in pop], [o["bal"] for o in pop], start, end)
        for o, nb in zip(pop, new_bal):
            profit = nb - o["bal"]
            o["bal"] = nb
            if o["bal"] >= DEAD_BELOW:
                alive.append(o)
            for _ in range(min(MAX_CHILDREN, int(max(profit, 0) // CHILD_PER))):
                children.append(dict(id=next_id, genes=mutate(o["genes"], rng), bal=DEPOSIT,
                                     born=str(mo + 1)))
                next_id += 1
        cap1 = sum(o["bal"] for o in pop)  # incl. the dead, before culling
        med_ret = float(np.median(new_bal / bal0 - 1))
        pop = alive + children
        if len(pop) > CAP:
            pop = rng.sample(pop, CAP)
        dirs = Counter(o["genes"].dirs for o in pop)
        log.append(dict(month=str(mo), pop=len(pop), ret=cap1 / cap0 - 1, med_ret=med_ret,
                        long=dirs[0] / max(len(pop), 1), short=dirs[1] / max(len(pop), 1),
                        both=dirs[2] / max(len(pop), 1), trend=dirs[3] / max(len(pop), 1),
                        born=len(children), died=len(alive_before) - len(alive)))
        if not pop:
            break
    return pd.DataFrame(log), pop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--n0", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    market = Market(mirrored() if a.mirror else None)
    log, pop = evolve(market, a.n0, a.seed)
    tag = ("mirror" if a.mirror else "real") + f"_n{a.n0}_s{a.seed}"
    OUT.mkdir(exist_ok=True)
    log.to_csv(OUT / f"evolve_{tag}.csv", index=False)
    pd.DataFrame([dict(id=o["id"], bal=o["bal"], born=o["born"], **asdict(o["genes"])) for o in pop]
                 ).to_csv(OUT / f"evolve_{tag}_pop.csv", index=False)
    print(log.to_string())

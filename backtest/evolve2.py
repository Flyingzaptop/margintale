"""Evolution with regime-filter genes and a trend-neutral fitness.

Like evolve.py, but every organism trades two accounts with the same genes:
one on the real price and one on the mirrored price (k / price, trend flipped).
Offspring come from the mean monthly profit of the two, so being on the right
side of gold's trend earns nothing: a long grid gains on one and loses on the
other. Either account below DEAD_BELOW kills the organism.

New genes (see engine.Params): efficiency-ratio filter (er_win, er_max),
trading session (sess), volatility filter (vol_max).

    python3 evolve2.py [--n0 2000] [--seed 1]
"""
import argparse
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Market
from evolve import CAP, CHILD_PER, DEAD_BELOW, DEPOSIT, MAX_CHILDREN, OUT, REGIME, mirrored, mutate
from search import sample


def sample2(rng):
    p = sample(rng)
    return replace(p, **{g: rng.choice(v) for g, v in REGIME.items()})


def share(pop, f):
    return sum(1 for o in pop if f(o["genes"])) / max(len(pop), 1)


def evolve(real, mirror, n0, seed):
    rng = random.Random(seed)
    pop = [dict(id=i, genes=sample2(rng), r=DEPOSIT, m=DEPOSIT, born="2017-01") for i in range(n0)]
    next_id = n0
    log = []
    for mo in pd.period_range("2017-01", "2026-09", freq="M"):
        start, end = mo.start_time.strftime("%Y-%m-%d"), (mo + 1).start_time.strftime("%Y-%m-%d")
        genes = [o["genes"] for o in pop]
        r0 = np.array([o["r"] for o in pop])
        m0 = np.array([o["m"] for o in pop])
        r1, _ = real.run_batch(genes, r0, start, end)
        m1, _ = mirror.run_batch(genes, m0, start, end)
        fit = ((r1 - r0) + (m1 - m0)) / 2
        alive, children = [], []
        for o, nr, nm, f in zip(pop, r1, m1, fit):
            o["r"], o["m"] = nr, nm
            if nr >= DEAD_BELOW and nm >= DEAD_BELOW:
                alive.append(o)
            for _ in range(min(MAX_CHILDREN, int(max(f, 0) // CHILD_PER))):
                children.append(dict(id=next_id, genes=mutate(o["genes"], rng, REGIME), r=DEPOSIT,
                                     m=DEPOSIT, born=str(mo + 1)))
                next_id += 1
        died = len(pop) - len(alive)
        pop = alive + children
        if len(pop) > CAP:
            pop = rng.sample(pop, CAP)
        dirs = Counter(o["genes"].dirs for o in pop)
        n = max(len(pop), 1)
        log.append(dict(
            month=str(mo), pop=len(pop), born=len(children), died=died,
            ret_real=r1.sum() / r0.sum() - 1, ret_mirror=m1.sum() / m0.sum() - 1,
            med_fit=float(np.median(fit)), pos_fit=float((fit > 0).mean()),
            long=dirs[0] / n, short=dirs[1] / n, both=dirs[2] / n, trend=dirs[3] / n,
            er_on=share(pop, lambda g: g.er_win > 0 and g.er_max > 0),
            vol_on=share(pop, lambda g: g.vol_max > 0),
            **{f"sess{k}": share(pop, lambda g, k=k: g.sess == k) for k in range(5)},
            mult1=share(pop, lambda g: g.mult == 1.0)))
        if not pop:
            break
    return pd.DataFrame(log), pop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    log, pop = evolve(Market(), Market(mirrored()), a.n0, a.seed)
    tag = f"neutral_n{a.n0}_s{a.seed}"
    OUT.mkdir(exist_ok=True)
    log.to_csv(OUT / f"evolve2_{tag}.csv", index=False)
    pd.DataFrame([dict(id=o["id"], r=o["r"], m=o["m"], born=o["born"], **asdict(o["genes"])) for o in pop]
                 ).to_csv(OUT / f"evolve2_{tag}_pop.csv", index=False)
    print(log.round(3).to_string())

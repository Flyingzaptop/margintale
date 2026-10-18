"""Grid/martingale search for the highest median compounding growth over 12 months.

Like lifecycle.py, but: broker leverage 1:100 for gold, lot grows with equity
(`risk` = base oz per $1000, finer steps), a chosen starting deposit, and the
score is the log of the median 12-month equity multiple over all starts (the
dead count at their remains), with at least MIN_S12 of starts alive at 12 months.
Train starts: monthly 2017-01..2022-12; the finalists are re-run on 2023-01..2024-09.

    python3 lifecycle3.py --deposit 300 [--n 3000] [--gens 8] [--seed 1]
"""
import argparse
import random
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from engine import Market
from evolve import OUT, REGIME, mutate as mutate1
from evolve2 import sample2

MONTHS = 24
MIN_S12 = 0.6
DEAD = 0.1  # engine.DEAD: an account is dead below this share of the deposit
RISKS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
ELITE, CHILDREN = 200, 2000


def fix(p, rng=None):
    return replace(p, leverage=100.0)


def sample(rng):
    return fix(replace(sample2(rng), risk=rng.choice(RISKS)))


def mutate(p, rng):
    if rng.random() < 0.15:
        return fix(replace(p, risk=rng.choice(RISKS)))
    return fix(mutate1(p, rng, REGIME))


def metrics(eq, dep):
    x = eq / dep
    alive = x >= DEAD
    s6, s12, s24 = alive[:, :, 5].mean(1), alive[:, :, 11].mean(1), alive[:, :, 23].mean(1)
    med12 = np.median(x[:, :, 11], axis=1)
    score = np.log(np.maximum(med12, 1e-3)) - 5 * np.maximum(0, MIN_S12 - s12)
    return pd.DataFrame(dict(score=score, s6=s6, s12=s12, s24=s24, med6=np.median(x[:, :, 5], 1),
                             med12=med12, q25_12=np.percentile(x[:, :, 11], 25, axis=1),
                             q75_12=np.percentile(x[:, :, 11], 75, axis=1),
                             med24=np.median(x[:, :, 23], 1)))


def key(p):
    return tuple(asdict(p).values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", type=float, default=300)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-s12", type=float, default=MIN_S12)
    a = ap.parse_args()
    MIN_S12 = a.min_s12
    rng = random.Random(a.seed)
    m = Market()
    seen = {}

    def add(genes, chunk=500):
        for i in range(0, len(genes), chunk):
            part = genes[i:i + chunk]
            _, eq, _ = m.lifecycles(part, "2017-01", "2022-12", MONTHS, deposit=a.deposit)
            for p, r in zip(part, metrics(eq, a.deposit).to_dict("records")):
                seen[key(p)] = dict(params=p, **r)

    add(list({key(p): p for p in (sample(rng) for _ in range(a.n))}.values()))
    for gen in range(a.gens + 1):
        top = sorted(seen.values(), key=lambda v: -v["score"])[:ELITE]
        print(f"gen {gen}: {len(seen)} configs, best med12 ×{top[0]['med12']:.2f} (s12 {top[0]['s12']:.0%}), "
              f"elite median ×{np.median([v['med12'] for v in top]):.2f}", flush=True)
        if gen == a.gens:
            break
        kids = {}
        for _ in range(CHILDREN):
            p = rng.choice(top)["params"]
            for _ in range(rng.choice([1, 1, 2, 3])):
                p = mutate(p, rng)
            if key(p) not in seen:
                kids[key(p)] = p
        add(list(kids.values()))

    df = pd.DataFrame([dict(**asdict(v["params"]), **{k: x for k, x in v.items() if k != "params"})
                       for v in seen.values()]).sort_values("score", ascending=False)
    fin = [v["params"] for v in sorted(seen.values(), key=lambda v: -v["score"])[:50]]
    _, eq, _ = m.lifecycles(fin, "2023-01", "2024-09", MONTHS, deposit=a.deposit)
    test = metrics(eq, a.deposit).add_prefix("test_")
    out = pd.concat([df.head(50).reset_index(drop=True), test], axis=1)
    tag = f"d{int(a.deposit)}_s{a.seed}" + (f"_m{int(MIN_S12 * 100)}" if MIN_S12 != 0.6 else "")
    df.to_parquet(OUT / f"lifecycle3_{tag}.parquet")
    out.to_csv(OUT / f"lifecycle3_{tag}_final.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    cols = ["dirs", "step", "step_atr", "step_grow", "mult", "max_orders", "tp", "tp_atr", "sl", "risk",
            "er_win", "er_max", "sess", "vol_max", "s6", "s12", "s24", "med6", "med12", "q25_12", "med24",
            "test_s6", "test_s12", "test_med6", "test_med12", "test_q25_12"]
    print(out[cols].head(20).round(2).to_string())

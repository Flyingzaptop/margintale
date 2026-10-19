"""Aggressive gram-contract grids with a profit target: big X with a high probability.

Every genome carries a target T: once equity reaches T x deposit it closes everything
and stops (the money is taken). Score = mean, over monthly starts, of what you hold
after 3 months (T for the starts that hit it, the remains otherwise, dead ~0).
Also reported: P(hit T) by 1 / 3 / 6 months. Genes: grid genes + base size in grams
+ lot growth + T. Train starts 2017-01..2022-12, test 2023-01..2024-09.

    python3 aggro2.py --deposit 100 [--n 2000] [--gens 6] [--seed 1]
"""
import argparse
import random
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from engine import Market
from evolve import OUT, REGIME, mutate as mutate1
from evolve2 import sample2

GRAM = 1 / 31.1035
GRAMS = [1, 2, 3, 5, 8, 12, 20, 30, 50]           # fixed base size, grams
RISKS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0]
TARGETS = [2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
ELITE, CHILDREN, MONTHS = 200, 1500, 6


def fix(p):
    return replace(p, leverage=100.0, unit_oz=GRAM)


def sample(rng):
    return fix(replace(sample2(rng), risk=rng.choice(RISKS), lot_oz=rng.choice(GRAMS) * GRAM,
                       target=rng.choice(TARGETS)))


def mutate(p, rng):
    u = rng.random()
    if u < 0.12:
        return fix(replace(p, risk=rng.choice(RISKS)))
    if u < 0.24:
        return fix(replace(p, lot_oz=rng.choice(GRAMS) * GRAM))
    if u < 0.34:
        return fix(replace(p, target=rng.choice(TARGETS)))
    return fix(mutate1(p, rng, REGIME))


def metrics(eq, dep, targets):
    x = eq / dep
    T = np.asarray(targets)[:, None, None]
    hit = x >= T * 0.999
    v3 = x[:, :, 2]
    return pd.DataFrame(dict(score=v3.mean(1), ev3=v3.mean(1), med3=np.median(v3, 1),
                             p_hit1=hit[:, :, 0].mean(1), p_hit3=hit[:, :, 2].mean(1), p_hit6=hit[:, :, 5].mean(1),
                             p_lose3=(v3 < 1).mean(1), dead3=(v3 < 0.1).mean(1)))


def key(p):
    return tuple(asdict(p).values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", type=float, default=100)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    m = Market()
    seen = {}

    def add(genes, chunk=500):
        for i in range(0, len(genes), chunk):
            part = genes[i:i + chunk]
            _, eq, _ = m.lifecycles(part, "2017-01", "2022-12", MONTHS, deposit=a.deposit)
            for p, r in zip(part, metrics(eq, a.deposit, [q.target for q in part]).to_dict("records")):
                seen[key(p)] = dict(params=p, **r)

    add(list({key(p): p for p in (sample(rng) for _ in range(a.n))}.values()))
    for gen in range(a.gens + 1):
        top = sorted(seen.values(), key=lambda v: -v["score"])[:ELITE]
        print(f"gen {gen}: {len(seen)}, best EV3 ×{top[0]['ev3']:.2f} (цель ×{top[0]['params'].target:g}, "
              f"P(цель за 3 мес) {top[0]['p_hit3']:.0%}), elite median ×{np.median([v['ev3'] for v in top]):.2f}",
              flush=True)
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
    out = pd.concat([df.head(50).reset_index(drop=True),
                     metrics(eq, a.deposit, [q.target for q in fin]).add_prefix("test_")], axis=1)
    out["grams"] = (out.lot_oz / GRAM).round()
    tag = f"d{int(a.deposit)}_s{a.seed}"
    df.to_parquet(OUT / f"aggro2_{tag}.parquet")
    out.to_csv(OUT / f"aggro2_{tag}_final.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    cols = ["dirs", "step", "step_atr", "step_grow", "mult", "max_orders", "tp", "tp_atr", "sl", "risk", "grams",
            "target", "er_win", "er_max", "sess", "vol_max", "ev3", "med3", "p_hit1", "p_hit3", "p_hit6", "p_lose3",
            "test_ev3", "test_p_hit1", "test_p_hit3", "test_p_hit6", "test_p_lose3"]
    print(out[cols].head(20).round(2).to_string())

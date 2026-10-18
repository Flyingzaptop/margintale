"""The most aggressive grids on a gram contract (0.01 lot = 1 g) from a $100 deposit.

Life does not matter; the score is the median, over monthly starts, of the highest
equity reached in the first 6 months (what a perfectly timed withdrawal could take),
in multiples of the deposit. Genes: all grid genes + fixed base size (grams) + lot
growth with equity (risk). Train starts 2017-01..2022-12, test 2023-01..2024-09.

    python3 aggro.py [--deposit 100] [--n 3000] [--gens 8] [--seed 1]
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
ELITE, CHILDREN, MONTHS = 200, 2000, 12


def fix(p):
    return replace(p, leverage=100.0, unit_oz=GRAM)


def sample(rng):
    return fix(replace(sample2(rng), risk=rng.choice(RISKS), lot_oz=rng.choice(GRAMS) * GRAM))


def mutate(p, rng):
    u = rng.random()
    if u < 0.12:
        return fix(replace(p, risk=rng.choice(RISKS)))
    if u < 0.24:
        return fix(replace(p, lot_oz=rng.choice(GRAMS) * GRAM))
    return fix(mutate1(p, rng, REGIME))


def metrics(eq, dep):
    x = eq / dep
    peak6 = x[:, :, :6].max(axis=2)
    alive = x >= 0.1
    life = np.where(alive.all(axis=2), MONTHS + 1, alive.argmin(axis=2) + 1)
    return pd.DataFrame(dict(score=np.log(np.maximum(np.median(peak6, axis=1), 1e-3)),
                             peak6_med=np.median(peak6, axis=1), peak6_q25=np.percentile(peak6, 25, axis=1),
                             peak6_q75=np.percentile(peak6, 75, axis=1), peak6_q90=np.percentile(peak6, 90, axis=1),
                             life_med=np.median(life, axis=1), s3=alive[:, :, 2].mean(1), s6=alive[:, :, 5].mean(1)))


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
            for p, r in zip(part, metrics(eq, a.deposit).to_dict("records")):
                seen[key(p)] = dict(params=p, **r)

    add(list({key(p): p for p in (sample(rng) for _ in range(a.n))}.values()))
    for gen in range(a.gens + 1):
        top = sorted(seen.values(), key=lambda v: -v["score"])[:ELITE]
        print(f"gen {gen}: {len(seen)}, best peak6 ×{top[0]['peak6_med']:.2f} (life {top[0]['life_med']:.0f} мес), "
              f"elite median ×{np.median([v['peak6_med'] for v in top]):.2f}", flush=True)
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
    out = pd.concat([df.head(50).reset_index(drop=True), metrics(eq, a.deposit).add_prefix("test_")], axis=1)
    out["grams"] = (out.lot_oz / GRAM).round()
    tag = f"d{int(a.deposit)}_s{a.seed}"
    df.to_parquet(OUT / f"aggro_{tag}.parquet")
    out.to_csv(OUT / f"aggro_{tag}_final.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    cols = ["dirs", "step", "step_atr", "step_grow", "mult", "max_orders", "tp", "tp_atr", "sl", "risk", "grams",
            "er_win", "er_max", "sess", "vol_max", "peak6_med", "peak6_q25", "peak6_q90", "life_med", "s6",
            "test_peak6_med", "test_peak6_q25", "test_peak6_q90", "test_life_med"]
    print(out[cols].head(20).round(2).to_string())

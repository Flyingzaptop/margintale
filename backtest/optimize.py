"""Find the grid/martingale with the best growth per drawdown, walk-forward.

One continuous account per config (no yearly resets). Score on the train span
is MAR = CAGR / max drawdown (worst-case intrabar equity); a config that dies
or loses money scores below every survivor. Steps:

1. random search over all genes (incl. regime filters) on TRAIN;
2. genetic refinement: mutate the best, keep the best, GENERATIONS times;
3. the finalists are run on TEST (never seen while searching) and on the full span.

    python3 optimize.py [--n 40000] [--seed 1]
"""
import argparse
import random
from dataclasses import asdict

import numpy as np
import pandas as pd

from engine import Market, Params
from evolve import OUT, REGIME, mutate
from evolve2 import sample2

TRAIN = ("2017-01-01", "2023-01-01")
TEST = ("2023-01-01", "2026-10-01")
FULL = ("2017-01-01", "2026-10-01")
GENERATIONS = 15
ELITE = 300
CHILDREN = 4000


def years(span):
    return (pd.Timestamp(span[1]) - pd.Timestamp(span[0])).days / 365.25


def score(bal, dd, span, deposit=1000.0):
    cagr = (np.maximum(bal, 1e-9) / deposit) ** (1 / years(span)) - 1
    mar = cagr / np.maximum(dd, 0.02)
    return np.where((dd < 1) & (bal > deposit), mar, -1 + bal / deposit - 1), cagr


def evaluate(m, genes, span):
    bal, dd = m.run_batch(genes, np.full(len(genes), 1000.0), *span)
    s, cagr = score(bal, dd, span)
    return pd.DataFrame(dict(bal=bal, dd=dd, cagr=cagr, mar=s))


def key(p):
    return tuple(asdict(p).values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    m = Market()

    seen = {}
    genes = list({key(p): p for p in (sample2(rng) for _ in range(a.n))}.values())
    res = evaluate(m, genes, TRAIN)
    for p, (_, r) in zip(genes, res.iterrows()):
        seen[key(p)] = (p, r.mar, r.bal, r.dd)
    print(f"random: {len(genes)} configs, survivors with profit {(res.mar > 0).mean():.1%}, best MAR {res.mar.max():.2f}")

    for g in range(GENERATIONS):
        elite = sorted(seen.values(), key=lambda v: -v[1])[:ELITE]
        kids = {}
        for _ in range(CHILDREN):
            p = rng.choice(elite)[0]
            for _ in range(rng.choice([1, 1, 2, 3])):
                p = mutate(p, rng, REGIME)
            if key(p) not in seen:
                kids[key(p)] = p
        kids = list(kids.values())
        if not kids:
            break
        res = evaluate(m, kids, TRAIN)
        for p, (_, r) in zip(kids, res.iterrows()):
            seen[key(p)] = (p, r.mar, r.bal, r.dd)
        top = sorted(seen.values(), key=lambda v: -v[1])[:ELITE]
        print(f"gen {g + 1}: {len(kids)} new, best MAR {top[0][1]:.2f}, elite median {np.median([v[1] for v in top]):.2f}")

    final = sorted(seen.values(), key=lambda v: -v[1])[:50]
    genes = [v[0] for v in final]
    tr, te, fu = (evaluate(m, genes, s) for s in (TRAIN, TEST, FULL))
    rows = [dict(**asdict(p), train_mar=tr.mar[i], train_cagr=tr.cagr[i], train_dd=tr.dd[i],
                 test_mar=te.mar[i], test_cagr=te.cagr[i], test_dd=te.dd[i],
                 full_bal=fu.bal[i], full_cagr=fu.cagr[i], full_dd=fu.dd[i], full_mar=fu.mar[i])
            for i, p in enumerate(genes)]
    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / f"optimize_s{a.seed}.csv", index=False)
    pd.DataFrame([dict(**asdict(v[0]), mar=v[1], bal=v[2], dd=v[3]) for v in seen.values()]
                 ).to_parquet(OUT / f"optimize_s{a.seed}_all.parquet")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(out.round(3).head(20).to_string())

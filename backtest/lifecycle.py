"""Search for a grid/martingale with a predictable life: safe to start any month.

Every config is started on the 1st of each month 2017-01..2024-09 with $1000 and
run for up to 24 months. From that:

* survival curve S(k): share of starts still alive at the end of month k;
* safe window T: the last month with S(T) >= SAFE (death is rare before T);
* score: median equity multiple at T over all starts (the dead count as ~0),
  annualised: med_mult ** (12 / T) - 1, only if T >= MIN_T. Used as "start, run
  T months, withdraw". Median, not mean: a few compounding outliers must not
  carry the score.

Random search, then genetic refinement around the best.

    python3 lifecycle.py [--n 4000] [--gens 10] [--seed 1]
"""
import argparse
import random
from dataclasses import asdict

import numpy as np
import pandas as pd

from engine import Market
from evolve import OUT, REGIME, mutate
from evolve2 import sample2

SAFE = 0.9
MIN_T = 6
DEAD = 100.0
ELITE = 200
CHILDREN = 2000
FIRST, LAST, MONTHS = "2017-01", "2024-09", 24


def metrics(eq):
    """eq: (configs, starts, months) -> DataFrame of survival/growth metrics."""
    alive = eq >= DEAD
    surv = alive.mean(axis=1)                              # (configs, months)
    ok = surv >= SAFE
    T = np.where(ok.all(axis=1), MONTHS, ok.argmin(axis=1))  # months with S >= SAFE from the start
    T = np.where(ok[:, 0], T, 0)
    rows = []
    for j in range(len(eq)):
        t = int(T[j])
        mult = eq[j, :, t - 1].mean() / 1000 if t else 0.0
        med_mult = np.median(eq[j, :, t - 1]) / 1000 if t else 0.0
        life = np.where(alive[j].all(axis=1), MONTHS + 1, alive[j].argmin(axis=1) + 1)
        rows.append(dict(T=t, grow=mult, med_grow=med_mult,
                         score=med_mult ** (12 / t) - 1 if t >= MIN_T and med_mult > 0 else -1.0 + t / MONTHS,
                         s6=surv[j, 5], s12=surv[j, 11], s24=surv[j, 23],
                         life_med=float(np.median(life)), life_q25=float(np.percentile(life, 25)),
                         worst=eq[j, :, t - 1].min() / 1000 if t else 0.0))
    return pd.DataFrame(rows)


def key(p):
    return tuple(asdict(p).values())


def evaluate(m, genes, chunk=500):
    out = []
    for i in range(0, len(genes), chunk):
        _, eq, _ = m.lifecycles(genes[i:i + chunk], FIRST, LAST, MONTHS)
        out.append(metrics(eq))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    m = Market()
    seen = {}

    def add(genes):
        res = evaluate(m, genes)
        for p, r in zip(genes, res.to_dict("records")):
            seen[key(p)] = dict(params=p, **r)
        return res

    genes = list({key(p): p for p in (sample2(rng) for _ in range(a.n))}.values())
    res = add(genes)
    print(f"random: {len(genes)}, with a safe window {(res['T'] > 0).mean():.0%}, best score {res.score.max():.2f}",
          flush=True)
    for g in range(a.gens):
        elite = sorted(seen.values(), key=lambda v: -v["score"])[:ELITE]
        kids = {}
        for _ in range(CHILDREN):
            p = rng.choice(elite)["params"]
            for _ in range(rng.choice([1, 1, 2, 3])):
                p = mutate(p, rng, REGIME)
            if key(p) not in seen:
                kids[key(p)] = p
        if not kids:
            break
        add(list(kids.values()))
        top = sorted(seen.values(), key=lambda v: -v["score"])[:ELITE]
        print(f"gen {g + 1}: {len(kids)} new, best {top[0]['score']:.2f} (T={top[0]['T']}), "
              f"elite median {np.median([v['score'] for v in top]):.2f}", flush=True)

    df = pd.DataFrame([dict(**asdict(v["params"]), **{k: x for k, x in v.items() if k != "params"})
                       for v in seen.values()]).sort_values("score", ascending=False)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / f"lifecycle_s{a.seed}.parquet")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df.head(20).round(3).to_string())

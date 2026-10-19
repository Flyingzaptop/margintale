"""Fastest and most probable x2: target fixed at 2, judged on the most recent years.

Starts every 2 days; train 2023-01..2025-06, test 2025-07..2026-08 (never seen by the
search). Equity is checked every day for 30 days. Score = mean over starts of
hit * exp(-days / 10) (a start that doubles in 2 days counts almost fully, in 20 days
~14%, never 0). Genes as in aggro2.py (grams, lot growth, kill switch allowed).

    python3 fast2.py [--n 3000] [--gens 8] [--seed 1]
"""
import argparse
import random
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

import aggro2
from aggro2 import key, mutate, sample
from engine import Market, params_from
from evolve import OUT

DEP, DAYS = 100.0, 30
TRAIN = pd.date_range("2023-01-02", "2025-06-30", freq="2D", tz="UTC")
TEST = pd.date_range("2025-07-01", "2026-08-03", freq="2D", tz="UTC")
ELITE, CHILDREN = 200, 1500


def metrics(eq):
    x = eq / DEP
    hit = x >= 2 * 0.999
    first = np.where(hit.any(axis=2), hit.argmax(axis=2) + 1, DAYS + 1)  # day of reaching x2
    got = first <= DAYS
    score = (got * np.exp(-first / 10)).mean(1)
    lost = x[:, :, -1] < 1
    return pd.DataFrame(dict(score=score, p3=(first <= 3).mean(1), p7=(first <= 7).mean(1), p14=(first <= 14).mean(1),
                             p30=got.mean(1), days_med=np.array([np.median(f[f <= DAYS]) if (f <= DAYS).any() else np.nan
                                                                 for f in first]),
                             ev30=x[:, :, -1].mean(1), lose30=lost.mean(1),
                             left_if_lost=np.array([r[r < 1].mean() if (r < 1).any() else 1.0 for r in x[:, :, -1]])))


def run(m, genes, starts):
    _, eq, _ = m.lifecycles(genes, months=DAYS, start_times=starts, deposit=DEP, step_days=1.0)
    return metrics(eq)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    aggro2.TARGETS[:] = [2.0]
    rng = random.Random(a.seed)
    m = Market()
    seen = {}

    def add(genes, chunk=400):
        for i in range(0, len(genes), chunk):
            part = genes[i:i + chunk]
            for p, r in zip(part, run(m, part, TRAIN).to_dict("records")):
                seen[key(p)] = dict(params=p, **r)

    add(list({key(p): p for p in (sample(rng) for _ in range(a.n))}.values()))
    for gen in range(a.gens + 1):
        top = sorted(seen.values(), key=lambda v: -v["score"])[:ELITE]
        b = top[0]
        print(f"gen {gen}: {len(seen)}, best: x2 за 7 дн {b['p7']:.0%}, за 30 дн {b['p30']:.0%}, "
              f"медиана {b['days_med']:.1f} дн", flush=True)
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
    fin_rows = df.drop_duplicates(subset=["score", "p30", "p7"]).head(30)
    fin = [params_from(r) for _, r in fin_rows.iterrows()]
    out = pd.concat([fin_rows.reset_index(drop=True), run(m, fin, TEST).add_prefix("test_")], axis=1)
    out["grams"] = (out.lot_oz * 31.1035).round()
    df.to_parquet(OUT / f"fast2_s{a.seed}.parquet")
    out.to_csv(OUT / f"fast2_s{a.seed}_final.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    cols = ["dirs", "step", "step_atr", "step_grow", "mult", "max_orders", "tp", "tp_atr", "sl", "risk", "grams",
            "kill_dd", "er_win", "er_max", "sess", "vol_max", "p3", "p7", "p14", "p30", "days_med", "ev30", "lose30",
            "left_if_lost", "test_p3", "test_p7", "test_p14", "test_p30", "test_days_med", "test_ev30", "test_lose30",
            "test_left_if_lost"]
    print(out[cols].head(15).round(2).to_string())

"""Island evolution of session entities for a P(x2) that RISES with time (recent regime first).

Old years may fail; what counts is a high and growing probability in the latest years.
Fitness on real price only, starts every 7 days from 2017 up to the train cutoff:
    S_y   = mean over starts of year y of hit * exp(-days_to_x2 / 180)
    score = sum_y w_y S_y / sum_y w_y,  w_y = 2 ** (y - last_train_year)   (last year weight 1)
            + 0.5 * max(slope of P_y over the last 4 years, 0)
Honest test (walk-forward): train starts end `cut - 365 days`, so no train run sees prices after
`cut`; test = daily starts from `cut` to 2025-09 (P(x2 within a year) and by year).

    python3 evo4.py --cut 2024-10-01   # test on starts 2024-10..2025-09
    python3 evo4.py --cut 2023-10-01   # test on starts 2023-10..2025-09
"""
import argparse
import random
from dataclasses import asdict, astuple, replace

import numpy as np
import pandas as pd

from engine3 import SESSIONS, Genes, Market3
from evo3 import crossover, mutate, sample
from evolve import OUT

DAYS = 365


def per_start(m, genes, starts):
    eq, _, _ = m.batch(genes, starts, DAYS)
    x = eq / 100.0
    hit = x >= 2 * 0.999
    first = np.where(hit.any(axis=2), hit.argmax(axis=2) + 1, DAYS + 1)
    got = first <= DAYS
    return got, first, np.where(got, 2.0, x[:, :, -1])


def score(m, genes, starts):
    got, first, val = per_start(m, genes, starts)
    yrs = starts.year.to_numpy()
    ys = np.unique(yrs)
    S = np.stack([(got[:, yrs == y] * np.exp(-first[:, yrs == y] / 180)).mean(1) for y in ys], 1)
    P = np.stack([got[:, yrs == y].mean(1) for y in ys], 1)
    w = 2.0 ** (ys - ys[-1])
    k = min(4, len(ys))
    slope = np.polyfit(np.arange(k), P[:, -k:].T, 1)[0]
    sc = (S * w).sum(1) / w.sum() + 0.5 * np.maximum(slope, 0)
    out = pd.DataFrame(P, columns=[f"P{y}" for y in ys])
    out["score"], out["slope"], out["p_all"], out["value"] = sc, slope, got.mean(1), val.mean(1)
    return out


def test(m, genes, starts):
    got, first, val = per_start(m, genes, starts)
    yrs = starts.year.to_numpy()
    out = pd.DataFrame({"t_p": got.mean(1), "t_p90": (first <= 90).mean(1),
                        "t_days_med": [np.median(f[f <= DAYS]) if (f <= DAYS).any() else np.nan for f in first],
                        "t_value": val.mean(1), "t_left": [v[v < 2].mean() if (v < 2).any() else 2.0 for v in val]})
    for y in np.unique(yrs):
        out[f"t_P{y}"] = got[:, yrs == y].mean(1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2024-10-01")
    ap.add_argument("--n0", type=int, default=500)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--kids", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cut = pd.Timestamp(a.cut)
    TRAIN = pd.date_range("2017-01-02", cut - pd.Timedelta(days=DAYS), freq="7D", tz="UTC")
    TEST = pd.date_range(cut, "2025-09-01", freq="D", tz="UTC")
    tag = f"evo4_{cut:%Y%m}_s{a.seed}"
    rng = random.Random(a.seed)
    m = Market3()
    finals = []
    for sess in range(len(SESSIONS)):
        seen = {}

        def add(genes):
            genes = [g for g in {astuple(g): g for g in genes}.values() if astuple(g) not in seen]
            if genes:
                for g, r in zip(genes, score(m, genes, TRAIN).to_dict("records")):
                    seen[astuple(g)] = dict(genes=g, **r)

        add([sample(rng, sess) for _ in range(a.n0)])
        for gen in range(a.gens):
            elite = sorted(seen.values(), key=lambda v: -v["score"])[:60]
            kids = []
            for _ in range(a.kids):
                k = (crossover(rng.choice(elite)["genes"], rng.choice(elite)["genes"], rng)
                     if rng.random() < 0.3 else rng.choice(elite)["genes"])
                kids.append(replace(mutate(k, rng), sess=sess))
            add(kids)
        top = sorted(seen.values(), key=lambda v: -v["score"])[:10]
        t = test(m, [v["genes"] for v in top], TEST)
        part = pd.concat([pd.DataFrame([dict(**asdict(v["genes"]), **{k: x for k, x in v.items() if k != "genes"})
                                        for v in top]), t], axis=1)
        finals.append(part)
        b = part.iloc[0]
        last = [c for c in part.columns if c.startswith("P20")][-2:]
        print(f"сессия {SESSIONS[sess]}: {len(seen)} особей; лучшая: обучение {', '.join(f'{c}={b[c]:.0%}' for c in last)}"
              f" → ТЕСТ P(x2) {b.t_p:.0%}, медиана {b.t_days_med:.0f} дн, итог ×{b.t_value:.2f}", flush=True)
    out = pd.concat(finals, ignore_index=True)
    out.to_csv(OUT / f"{tag}.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    pc = [c for c in out.columns if c.startswith("P20")]
    tc = [c for c in out.columns if c.startswith("t_P")]
    cols = ["sess", "entry", "dirs", "trend", "sl", "tp", "risk", "kill_dd", "score", "slope"] + pc + \
           ["t_p", "t_p90", "t_days_med", "t_value", "t_left"] + tc
    print(out.sort_values("score", ascending=False)[cols].head(25).round(2).to_string(index=False))

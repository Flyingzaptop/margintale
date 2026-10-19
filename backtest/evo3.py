"""Island evolution of intraday session entities (engine3) for a fast, highly probable x2.

One island per session window (Asia, London, New York, London-NY overlap, Asia-London
overlap, London+NY, late US, all day); islands evolve separately.
Fitness on recent starts (every 5 days 2023-01..2025-09, 365-day horizon), on the real
AND on the mirrored price (same noise, trend flipped):
    S = mean over starts of hit * exp(-days_to_x2 / 180)
    score = min(S_real, S_mirror)   -- an edge that is not just "gold goes up"
The finalists of every island are also run on 2017-2022 starts (older years, never seen).

    python3 evo3.py [--n0 600] [--gens 6] [--seed 1]
"""
import argparse
import random
from dataclasses import asdict, astuple, replace

import numpy as np
import pandas as pd

from engine3 import SESSIONS, Genes, Market3
from evolve import OUT, mirrored

SPACE = {
    "entry": [0, 1, 2, 3, 4, 5, 6], "orb": [15, 30, 60], "look": [0, 1, 2], "ema": [0, 1, 2],
    "offset": [0.0, 0.1, 0.3, 0.5, 1.0], "dirs": [0, 0, 1, 2], "trend": [0, 1, 2], "trend_len": [0, 1],
    "sl": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0], "tp": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
    "be": [0.0, 0.0, 0.5, 1.0], "trail_start": [0.0, 0.0, 1.0, 2.0], "trail": [0.5, 1.0, 2.0],
    "max_hold": [0, 0, 30, 60, 240], "max_trades": [1, 2, 3, 5, 10], "day_loss": [0.0, 0.02, 0.05, 0.1, 0.2],
    "day_profit": [0.0, 0.03, 0.05, 0.1, 0.2], "skip_mon": [0, 1], "skip_fri": [0, 1],
    "risk": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2], "win_mult": [1.0, 1.5, 2.0],
    "loss_mult": [1.0, 1.5, 2.0], "mult_cap": [1.0, 2.0, 4.0, 8.0], "kill_dd": [0.3, 0.5, 0.7, 0.9],
}
RECENT = pd.date_range("2023-01-02", "2025-09-01", freq="5D", tz="UTC")
OLDER = pd.date_range("2017-01-02", "2022-09-01", freq="10D", tz="UTC")
DAYS = 365


def sample(rng, sess):
    return Genes(sess=sess, **{k: rng.choice(v) for k, v in SPACE.items()})


def mutate(g, rng):
    for _ in range(rng.choice([1, 1, 2, 3])):
        k = rng.choice(list(SPACE))
        g = replace(g, **{k: rng.choice(SPACE[k])})
    return g


def crossover(a, b, rng):
    da, db = asdict(a), asdict(b)
    return Genes(**{k: (da[k] if rng.random() < 0.5 else db[k]) for k in da})


def stats(m, genes, starts):
    eq, st, _ = m.batch(genes, starts, DAYS)
    x = eq / 100.0
    hit = x >= 2 * 0.999
    first = np.where(hit.any(axis=2), hit.argmax(axis=2) + 1, DAYS + 1)
    got = first <= DAYS
    S = (got * np.exp(-first / 180)).mean(1)
    value = np.where(got, 2.0, x[:, :, -1])
    return dict(S=S, p_hit=got.mean(1), p30=(first <= 30).mean(1), p90=(first <= 90).mean(1),
                days_med=np.array([np.median(f[f <= DAYS]) if (f <= DAYS).any() else np.nan for f in first]),
                value=value.mean(1), lose=(value < 1).mean(1),
                left=np.array([v[v < 1].mean() if (v < 1).any() else 1.0 for v in value]))


def evaluate(real, mirror, genes, starts):
    r, mm = stats(real, genes, starts), stats(mirror, genes, starts)
    df = pd.DataFrame({**{f"{k}_r": v for k, v in r.items()}, **{f"{k}_m": v for k, v in mm.items()}})
    df["score"] = np.minimum(df.S_r, df.S_m)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=600)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--kids", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    real, mirror = Market3(), Market3(mirrored())
    finals = []
    for sess in range(len(SESSIONS)):
        seen = {}

        def add(genes):
            genes = [g for g in {astuple(g): g for g in genes}.values() if astuple(g) not in seen]
            if not genes:
                return
            df = evaluate(real, mirror, genes, RECENT)
            for g, r in zip(genes, df.to_dict("records")):
                seen[astuple(g)] = dict(genes=g, **r)

        add([sample(rng, sess) for _ in range(a.n0)])
        for gen in range(a.gens):
            elite = sorted(seen.values(), key=lambda v: -v["score"])[:60]
            kids = []
            for _ in range(a.kids):
                if rng.random() < 0.3:
                    k = crossover(rng.choice(elite)["genes"], rng.choice(elite)["genes"], rng)
                else:
                    k = rng.choice(elite)["genes"]
                kids.append(replace(mutate(k, rng), sess=sess))
            add(kids)
        top = sorted(seen.values(), key=lambda v: -v["score"])
        b = top[0]
        print(f"сессия {SESSIONS[sess]}: {len(seen)} особей; лучшая: x2 за год реальная {b['p_hit_r']:.0%} "
              f"(медиана {b['days_med_r']:.0f} дн), зеркальная {b['p_hit_m']:.0%}", flush=True)
        best_real = sorted(seen.values(), key=lambda v: -v["S_r"])[:5]
        pick = {astuple(v["genes"]): v for v in top[:10] + best_real}.values()
        fin = [v["genes"] for v in pick]
        old = evaluate(real, mirror, fin, OLDER).add_prefix("old_")
        part = pd.concat([pd.DataFrame([dict(**asdict(v["genes"]), **{k: x for k, x in v.items() if k != "genes"})
                                        for v in pick]).reset_index(drop=True), old], axis=1)
        finals.append(part)
        pd.DataFrame([dict(**asdict(v["genes"]), **{k: x for k, x in v.items() if k != "genes"})
                      for v in seen.values()]).to_parquet(OUT / f"evo3_s{a.seed}_sess{sess}.parquet")
    out = pd.concat(finals, ignore_index=True)
    out.to_csv(OUT / f"evo3_s{a.seed}_final.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    cols = ["sess", "entry", "dirs", "trend", "sl", "tp", "risk", "kill_dd", "max_trades", "score",
            "p_hit_r", "p90_r", "days_med_r", "value_r", "left_r", "p_hit_m", "value_m",
            "old_p_hit_r", "old_days_med_r", "old_value_r", "old_p_hit_m"]
    print(out.sort_values("score", ascending=False)[cols].head(20).round(2).to_string(index=False))
    print("\nЛучшие только по реальной цене (свежие годы):")
    print(out.sort_values("S_r", ascending=False)[cols].head(10).round(2).to_string(index=False))

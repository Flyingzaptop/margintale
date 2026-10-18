"""Search single-position SL/TP strategies (engine2) for a predictable 1-3 year life
with exponential (compounding) growth from a small deposit.

Train starts: every 2 months 2017-01..2022-11, each run up to 36 months.
Fitness (per genome over the train starts):
  s12   share of starts alive after 12 months (want >= 0.9);
  gm    median monthly growth over the first 12 months (dead count at their equity);
  r2    median R^2 of log equity vs time over each start's life (exponential = straight);
  score = gm * (0.5 + 0.5 * r2) - 2 * max(0, 0.9 - s12) - 0.01 * log10(deposit / 100)
Random search, then genetic refinement. Test starts (monthly 2023-01..2024-09)
are only evaluated for the finalists.

    python3 search2.py [--n 4000] [--gens 8] [--seed 1]
"""
import argparse
import random
from dataclasses import asdict, astuple, replace
from pathlib import Path

import numpy as np
import pandas as pd

from engine2 import Genes, Market2

OUT = Path(__file__).resolve().parent.parent / "results"
TRAIN = pd.date_range("2017-01-01", "2022-12-01", freq="2MS", tz="UTC")
TEST = pd.date_range("2023-01-01", "2024-09-01", freq="MS", tz="UTC")
MONTHS = 36
ELITE = 200
CHILDREN = 2000

SPACE = {
    "entry": [0, 1, 2], "dirs": [0, 0, 1, 2], "look": [0, 1, 2], "ema": [0, 1, 2],
    "offset": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0], "sl": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
    "tp": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0], "trail_start": [0.0, 0.0, 0.5, 1.0, 2.0, 3.0],
    "trail": [0.3, 0.5, 1.0, 2.0], "be": [0.0, 0.0, 0.5, 1.0, 2.0],
    "max_hold": [0, 0, 30, 120, 480, 1440], "pend_life": [15, 60, 240, 1440],
    "sess": [0, 1, 2, 3, 4, 5], "vol_max": [0.0, 0.7, 1.0, 1.5], "er_win": [0, 1, 2, 3],
    "er_max": [0.02, 0.05, 0.1, 0.2], "risk": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "loss_mult": [1.0, 1.0, 1.5, 2.0], "loss_cap": [1.0, 2.0, 4.0, 8.0],
    "deposit": [100.0, 200.0, 300.0, 500.0, 1000.0],
}


def sample(rng):
    return Genes(**{k: rng.choice(v) for k, v in SPACE.items()})


def mutate(g, rng):
    k = rng.choice(list(SPACE))
    return replace(g, **{k: rng.choice(SPACE[k])})


def lives(L, index):
    """Months each (genome, start) lived; MONTHS + 1 if alive at the horizon."""
    d = L["died"]
    st = L["starts"]
    out = np.full(d.shape, MONTHS + 1.0)
    for j, s in zip(*np.nonzero(d >= 0)):
        out[j, s] = (index[d[j, s]] - st[s]).days / 30.44
    return out


def fitness(L, genes, index):
    eq, life = L["eq"], lives(L, index)
    dep = np.array([g.deposit for g in genes])[:, None]
    rows = []
    t = np.arange(1, MONTHS + 1)
    for j in range(len(genes)):
        x = np.maximum(eq[j] / dep[j], 1e-6)
        r2s = []
        for s in range(x.shape[0]):
            n = int(min(life[j, s], MONTHS))
            if n >= 3:
                y = np.log(x[s, :n])
                r = np.corrcoef(t[:n], y)[0, 1] if y.std() > 0 else 0.0
                r2s.append(r * r if r > 0 else 0.0)
            else:
                r2s.append(0.0)
        s12 = (life[j] >= 12).mean()
        gm = np.median(x[:, 11]) ** (1 / 12) - 1
        r2 = float(np.median(r2s))
        score = gm * (0.5 + 0.5 * r2) - 2 * max(0.0, 0.9 - s12) - 0.01 * np.log10(dep[j, 0] / 100)
        rows.append(dict(score=score, s12=s12, s24=(life[j] >= 24).mean(), s36=(life[j] > MONTHS).mean(),
                         life_med=float(np.median(life[j])), life_q25=float(np.percentile(life[j], 25)),
                         life_q75=float(np.percentile(life[j], 75)), gm=gm, r2=r2,
                         g12_med=float(np.median(x[:, 11])), g12_q25=float(np.percentile(x[:, 11], 25)),
                         dd_med=float(np.median(L["dd"][j])), trades_m=float(np.median(L["trades"][j])),
                         winrate=float(np.median(L["winrate"][j]))))
    return pd.DataFrame(rows)


def key(g):
    return astuple(g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    m = Market2()
    seen = {}

    def add(genes, chunk=500):
        for i in range(0, len(genes), chunk):
            part = genes[i:i + chunk]
            f = fitness(m.lifecycles(part, TRAIN, MONTHS), part, m.index)
            for g, r in zip(part, f.to_dict("records")):
                seen[key(g)] = dict(genes=g, **r)

    add(list({key(g): g for g in (sample(rng) for _ in range(a.n))}.values()))
    best = max(seen.values(), key=lambda v: v["score"])
    print(f"random: {len(seen)}, s12>=0.9: {np.mean([v['s12'] >= 0.9 for v in seen.values()]):.1%}, "
          f"best {best['score']:.3f} (gm {best['gm']:+.1%}/мес, s12 {best['s12']:.0%})", flush=True)
    for gen in range(a.gens):
        elite = sorted(seen.values(), key=lambda v: -v["score"])[:ELITE]
        kids = {}
        for _ in range(CHILDREN):
            g = rng.choice(elite)["genes"]
            for _ in range(rng.choice([1, 1, 2, 3])):
                g = mutate(g, rng)
            if key(g) not in seen:
                kids[key(g)] = g
        add(list(kids.values()))
        top = sorted(seen.values(), key=lambda v: -v["score"])[:ELITE]
        print(f"gen {gen + 1}: {len(kids)} new, best {top[0]['score']:.3f} (gm {top[0]['gm']:+.1%}/мес, "
              f"s12 {top[0]['s12']:.0%}, life_med {top[0]['life_med']:.0f}), "
              f"elite median {np.median([v['score'] for v in top]):.3f}", flush=True)

    df = pd.DataFrame([dict(**asdict(v["genes"]), **{k: x for k, x in v.items() if k != "genes"})
                       for v in seen.values()]).sort_values("score", ascending=False)
    fin = [v["genes"] for v in sorted(seen.values(), key=lambda v: -v["score"])[:50]]
    ft = fitness(m.lifecycles(fin, TEST, MONTHS), fin, m.index).add_prefix("test_")
    out = pd.concat([df.head(50).reset_index(drop=True), ft], axis=1)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / f"search2_s{a.seed}.parquet")
    out.to_csv(OUT / f"search2_s{a.seed}_final.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    cols = ["entry", "dirs", "sl", "tp", "trail_start", "be", "sess", "risk", "loss_mult", "deposit",
            "score", "s12", "s24", "life_med", "gm", "r2", "g12_med", "dd_med", "trades_m", "winrate",
            "test_s12", "test_life_med", "test_gm", "test_r2", "test_g12_med"]
    print(out[cols].head(20).round(3).to_string())

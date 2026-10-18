"""Random search over grid/martingale variants, each run per calendar year.

Every config runs 10 times: from Jan 1 of 2017..2026 to year end (2026 ends in
September), $1000 each. Writes results/search.parquet with per-year balances.
"""
import random
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from engine import Market, Params

YEARS = list(range(2017, 2027))
OUT = Path(__file__).resolve().parent.parent / "results"


def sample(rng):
    p = Params()
    p.dirs = rng.choice([0, 1, 2, 3])
    p.ema = rng.choice([1440, 7200, 28800]) if p.dirs == 3 else 0
    p.step_atr = rng.choice([0, 1])
    p.step = rng.choice([3, 5, 8, 12, 20, 30]) if not p.step_atr else rng.choice([2, 4, 8, 16, 30])
    p.step_grow = rng.choice([1.0, 1.0, 1.2, 1.5])
    p.mult = rng.choice([1.0, 1.2, 1.4, 1.6, 2.0])
    p.max_orders = rng.choice([0, 3, 5, 8, 12])
    p.tp_atr = rng.choice([0, 1])
    p.tp = rng.choice([2, 5, 10, 20]) if not p.tp_atr else rng.choice([1, 3, 6, 12])
    p.sl = rng.choice([0.0, 0.0, 0.05, 0.1, 0.2, 0.4])
    p.leverage = rng.choice([0, 100, 500])
    p.risk = rng.choice([0, 0, 1, 2])
    return p


def evaluate(seed):
    rng = random.Random(seed)
    m = MARKET
    rows = []
    for _ in range(200):
        p = sample(rng)
        row = asdict(p)
        for y in YEARS:
            r = m.run(p, f"{y}-01-01", f"{y + 1}-01-01")
            row[f"y{y}"] = r["balance"]
            row[f"d{y}"] = r["died"] is not None
        rows.append(row)
    return rows


def init():
    global MARKET
    MARKET = Market()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # batches of 200 configs
    with Pool(4, initializer=init) as pool:
        rows = [r for batch in pool.imap_unordered(evaluate, range(n)) for r in batch]
    df = pd.DataFrame(rows).drop_duplicates(subset=list(asdict(Params())))
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "search.parquet")
    print(len(df), "configs")

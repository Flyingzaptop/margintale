"""Top genomes found on detrended gold, run on detrended, real and mirrored gold.

Starts: 1st of every month 2017-01..2024-09, $100, 6 months; target genes as found.

    python3 detrend_check.py [results/aggro2_..._detrend_T2.parquet]
"""
import sys

import numpy as np
import pandas as pd

from engine import Market, detrended, params_from
from evolve import mirrored

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "../results/aggro2_d100_s3_kill_detrend_T2.parquet"
    df = pd.read_parquet(src).sort_values("score", ascending=False)
    df = df.drop_duplicates(subset=["ev3", "p_hit3"]).head(8)
    genes = [params_from(r) for _, r in df.iterrows()]
    rows = []
    for name, m in (("без тренда", Market(detrended())), ("настоящая", Market()), ("зеркальная", Market(mirrored()))):
        _, eq, _ = m.lifecycles(genes, "2017-01", "2024-09", 6, deposit=100.0)
        x = eq / 100
        for j, g in enumerate(genes):
            T = g.target
            rows.append(dict(особь=j, рынок=name, P_цель_1м=(x[j, :, 0] >= T * .999).mean(),
                             P_цель_3м=(x[j, :, 2] >= T * .999).mean(), итог_3м=x[j, :, 2].mean(),
                             в_минусе=(x[j, :, 2] < 1).mean()))
    out = pd.DataFrame(rows).pivot(index="особь", columns="рынок")
    pd.set_option("display.width", 250)
    print(out.round(2).to_string())
    for j, g in enumerate(genes):
        print(j, g)

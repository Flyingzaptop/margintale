"""Pick one trading entity from the lifecycle search: long life, high and stable growth.

Re-runs the top configs of results/lifecycle_s1.parquet over monthly starts,
merges configs with identical behaviour, and prints a comparison table.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Market, params_from as params
from lifecycle import DEAD, FIRST, LAST, MONTHS

ROOT = Path(__file__).resolve().parent.parent
TOP = 600


def describe(eq, dd, starts):
    """eq: (starts, months), dd: (starts,) max drawdown, for one config."""
    alive = eq >= DEAD
    m12 = eq[:, 11] / 1000
    prev = np.concatenate([np.full((len(eq), 1), 1000.0), eq[:, :-1]], axis=1)
    up = ((eq > prev) & alive & (prev >= DEAD)).sum() / max(((prev >= DEAD) & alive).sum(), 1)
    by_year = pd.Series(m12, index=starts.year).groupby(level=0).median()
    surv = alive[:, 23]
    return dict(s12=alive[:, 11].mean(), s24=alive[:, 23].mean(),
                dd_med=np.median(dd[surv]) if surv.any() else 1.0,
                dd_q90=np.percentile(dd[surv], 90) if surv.any() else 1.0,
                g12_med=np.median(m12), g12_q25=np.percentile(m12, 25),
                g24_med=np.median(eq[:, 23] / 1000),
                up_months=up, worst_year_g12=by_year.min(), best_year_g12=by_year.max(),
                **{f"g12_{y}": v for y, v in by_year.items()})


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "results" / "lifecycle_s1.parquet")
    cand = pd.concat([df.nlargest(TOP, "score"), df.nlargest(TOP // 2, "s24"),
                      df[df.s12 >= 0.85].nlargest(TOP // 2, "med_grow")]).drop_duplicates()
    m = Market()
    genes = [params(r) for _, r in cand.iterrows()]
    _, eq, starts = m.lifecycles(genes, FIRST, LAST, MONTHS)
    rows, seen = [], set()
    for j, p in enumerate(genes):
        sig = tuple(np.round(eq[j, :, [5, 11, 23]].ravel(), 0))
        if sig in seen:
            continue
        seen.add(sig)
        rows.append(dict(j=j, **describe(eq[j], m.last_dd[j], starts)))
    res = pd.DataFrame(rows)
    res = res[(res.s12 >= 0.8)]
    # Pareto front on life (s24), growth (g12_q25), transfer (worst_year_g12) and drawdown
    keys = ["s24", "g12_q25", "worst_year_g12", "dd_med"]
    v = res[keys].to_numpy() * np.array([1, 1, 1, -1])
    front = [i for i in range(len(v)) if not ((v >= v[i]).all(axis=1) & (v > v[i]).any(axis=1)).any()]
    out = res.iloc[front].sort_values("dd_med")
    out.insert(1, "genes", [str(genes[j]).replace("Params", "") for j in out.j])
    out.to_csv(ROOT / "results" / "entity_candidates.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 400)
    print(f"{len(cand)} candidates, {len(res)} distinct with s12 >= 80%, Pareto front {len(out)}")
    print(out.drop(columns="genes").round(2).to_string())
    for _, r in out.iterrows():
        print(r.j, r.genes)

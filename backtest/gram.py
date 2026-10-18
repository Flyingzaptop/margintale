"""XAUUSD (0.01 lot = 1 oz) vs a gram contract (GAUUSD at VT Markets: 1 lot = 100 g,
0.01 lot = 1 g). The gram price is the ounce price / 31.1035, so the same history is
used with order sizes in grams. Gram-contract spread is not published; two cost
scenarios: same cost per oz as XAUUSD, and +$0.40 per oz.

Entity: lifecycle3 $1000 best (docs/08_growth_search.md), lot growing 0.01 per $1000
(risk) and finer around the 0.01-0.02 cliff. Starts: 1st of month 2017-01..2024-09.

    python3 gram.py
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from engine import DEAD, Market
from phases import AGRO

GRAM = 1 / 31.1035

if __name__ == "__main__":
    m = Market()
    rows = []
    for name, unit, extra in (("XAUUSD (1 oz)", 1.0, 0.0), ("граммы, издержки как у XAUUSD", GRAM, 0.0),
                              ("граммы, +$0.40/oz издержек", GRAM, 0.40)):
        for dep in (100.0, 300.0, 1000.0):
            for risk in (0.0, 1.0, 1.25, 1.5, 2.0):
                p = replace(AGRO, risk=risk, unit_oz=unit, extra_cost=extra,
                            lot_oz=1.0 if unit == 1.0 else max(1.0, round(dep / 1000 / GRAM)) * GRAM)
                for per, (a, b) in {"обуч": ("2017-01", "2022-12"), "пров": ("2023-01", "2024-09")}.items():
                    _, eq, _ = m.lifecycles([p], a, b, 24, deposit=dep)
                    x = eq[0] / dep
                    al = x >= DEAD
                    rows.append(dict(инструмент=name, депозит=dep, лот=("фикс" if risk == 0 else f"{risk / 100:.4f}/$1000"),
                                     период=per, живы12=al[:, 11].mean(), мед6=np.median(x[:, 5]),
                                     мед12=np.median(x[:, 11]), q25_12=np.percentile(x[:, 11], 25),
                                     q90_12=np.percentile(x[:, 11], 90)))
    df = pd.DataFrame(rows)
    df.to_csv("../results/gram.csv", index=False)
    out = df.pivot_table(index=["инструмент", "депозит", "лот"], columns="период",
                         values=["живы12", "мед12", "q25_12"], sort=False)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string())

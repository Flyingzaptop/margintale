"""Profit-withdrawal rules for the safe $300 entity (docs/08_growth_search.md).

Every rule: every `wd_days` trading days, when no basket is open, withdraw
`wd_frac` of the balance above `wd_floor` x deposit. Each rule is started on the
1st of every month 2017-01..2024-09 with $300 and run for 12 and 24 months.
Value of a run = money withdrawn + what is left on the account at the horizon
(only the withdrawn part if the account died).

    python3 withdraw.py
"""
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from engine import DEAD, Market
from pwm import SAFE

ROOT = Path(__file__).resolve().parent.parent
DEP = 300.0


def evaluate(m, rules, months, first="2017-01", last="2024-09"):
    _, eq, _ = m.lifecycles(rules, first, last, months, deposit=DEP)
    wd = m.last_wd
    left = eq[:, :, -1]
    alive = left >= DEAD * DEP
    value = wd + np.where(alive, left, 0.0)
    return pd.DataFrame({
        f"живы_{months}": alive.mean(1),
        f"выведено_мед_{months}": np.median(wd, 1), f"выведено_ср_{months}": wd.mean(1),
        f"итог_мед_{months}": np.median(value, 1), f"итог_ср_{months}": value.mean(1),
        f"итог_25%_{months}": np.percentile(value, 25, 1), f"итог_10%_{months}": np.percentile(value, 10, 1),
    })


if __name__ == "__main__":
    m = Market()
    grid = [(0, 0.0, 1.0)] + list(product((1, 5, 21), (0.1, 0.25, 0.5, 1.0), (1.0, 1.25, 1.5, 2.0, 3.0)))
    rules = [replace(SAFE, wd_days=d, wd_frac=f, wd_floor=fl) for d, f, fl in grid]
    res = pd.concat([evaluate(m, rules, 12), evaluate(m, rules, 24)], axis=1)
    res.insert(0, "дней", [g[0] for g in grid])
    res.insert(1, "доля", [g[1] for g in grid])
    res.insert(2, "порог×деп", [g[2] for g in grid])
    res.to_csv(ROOT / "results" / "withdraw.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    show = ["дней", "доля", "порог×деп", "живы_12", "итог_мед_12", "итог_25%_12", "выведено_мед_12",
            "живы_24", "итог_мед_24", "итог_ср_24", "итог_25%_24", "итог_10%_24", "выведено_мед_24"]
    print("Без вывода:")
    print(res.iloc[[0]][show].round(0).to_string(index=False))
    print("\nЛучшие по медианному итогу за 24 мес:")
    print(res.sort_values("итог_мед_24", ascending=False)[show].head(12).round(2).to_string(index=False))
    print("\nЛучшие по 25-му перцентилю итога за 24 мес (защита от худших стартов):")
    print(res.sort_values("итог_25%_24", ascending=False)[show].head(12).round(2).to_string(index=False))

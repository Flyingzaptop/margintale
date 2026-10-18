"""Report on the best lifecycle configs: survival, growth, when and why they die.

    python3 lifecycle_report.py [results/lifecycle_s1.parquet]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import Market, params_from as params
from lifecycle import DEAD, MONTHS, metrics

ROOT = Path(__file__).resolve().parent.parent

path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "lifecycle_s1.parquet"
df = pd.read_parquet(path)
m = Market()
top = df.head(20)
genes = [params(r) for _, r in top.iterrows()]

# stability across start periods: early starts (2017-2021) vs late (2022-2024)
_, eq_a, _ = m.lifecycles(genes, "2017-01", "2021-12", MONTHS)
_, eq_b, _ = m.lifecycles(genes, "2022-01", "2024-09", MONTHS)
ma, mb = metrics(eq_a), metrics(eq_b)
cols = ["dirs", "step", "step_atr", "step_grow", "mult", "max_orders", "tp", "tp_atr", "sl", "leverage",
        "risk", "ema", "er_win", "er_max", "sess", "vol_max", "T", "grow", "score", "s12", "life_med"]
show = top[cols].reset_index(drop=True).assign(
    score_17_21=ma.score, T_17_21=ma["T"], score_22_24=mb.score, T_22_24=mb["T"])
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("Топ-20 по оценке, все старты 2017-01..2024-09, и отдельно старты 2017-2021 / 2022-2024")
print(show.round(2).to_string())

best = genes[0]
print(f"\nЛучшая: {best}")
died, eq, starts = m.lifecycles([best], "2017-01", "2024-09", MONTHS)
died, eq = died[0], eq[0]
alive = eq >= DEAD
surv = alive.mean(axis=0)
print("Доля живых к концу месяца: " + " ".join(f"{k + 1}:{s:.0%}" for k, s in enumerate(surv)))
for k in (3, 6, 12, 24):
    e = eq[:, k - 1] / 1000
    print(f"  месяц {k:>2}: средний рост ×{e.mean():.2f}, медиана ×{np.median(e):.2f}, "
          f"25% ×{np.percentile(e, 25):.2f}, 75% ×{np.percentile(e, 75):.2f}")

death_t = pd.Series([m.index[d] for d in died if d >= 0])
print(f"\nСмертей: {len(death_t)} из {len(died)} стартов. Даты смертей (уникальные дни):")
print(death_t.dt.date.value_counts().sort_index().to_string())

# market state at death vs. typical
idx = died[died >= 0]
f = m.filt
day_move = np.abs(m.c - np.roll(m.c, 1440)) / m.atr  # 1-day move in ATR units
state = pd.DataFrame({"ER 1 день": f[4], "ATR / средний ATR": f[1], "движение за сутки, ATR": day_move})
print("\nСостояние рынка в момент смерти против обычного (медианы):")
print(pd.DataFrame({"в момент смерти": state.iloc[idx].median(), "обычно": state.iloc[1440:].median()})
      .round(2).to_string())

fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].step(range(1, MONTHS + 1), surv, where="post")
ax[0].axhline(0.9, color="gray", ls="--", lw=0.7)
ax[0].set_ylim(0, 1.02); ax[0].set_title("Доля живых запусков по месяцам от старта", loc="left", fontsize=9)
for s in range(len(starts)):
    ax[1].plot(range(1, MONTHS + 1), eq[s] / 1000, lw=0.6, alpha=0.5)
ax[1].set_yscale("symlog", linthresh=1); ax[1].set_title("Рост капитала каждого запуска, ×", loc="left", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "img" / "lifecycle_best.png", dpi=110)

"""Summarise evolve2 runs (trend-neutral fitness, regime genes)."""
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import Market, Params
from evolve import mirrored

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
GENES = list(Params.__dataclass_fields__)
INT_GENES = {"dirs", "step_atr", "max_orders", "tp_atr", "ema", "er_win", "sess"}

runs = sorted(Path(f).stem[len("evolve2_"):] for f in glob.glob(str(RES / "evolve2_*.csv"))
              if not f.endswith("_pop.csv"))
logs = {t: pd.read_csv(RES / f"evolve2_{t}.csv") for t in runs}

print("Капитал популяции по годам: реальная цена / зеркальная")
yr = {}
for t, g in logs.items():
    gy = g.assign(y=g.month.str[:4]).groupby("y")
    yr[t] = gy.apply(lambda d: f"{np.prod(1 + d.ret_real) - 1:+.0%} / {np.prod(1 + d.ret_mirror) - 1:+.0%}")
print(pd.DataFrame(yr).to_string())
for t, g in logs.items():
    print(f"{t}: месяцев с медианным фитнесом > 0: {(g.med_fit > 0).mean():.0%}, "
          f"доля особей с фитнесом > 0 (медиана по месяцам): {g.pos_fit.median():.0%}")

last = pd.DataFrame({t: g.iloc[-1] for t, g in logs.items()})
print("\nГены в конце:")
print(last.loc[["long", "short", "both", "trend", "er_on", "vol_on", "sess0", "sess1", "sess2", "sess3",
                "sess4", "mult1"]].astype(float).map(lambda v: f"{v:.0%}").to_string())

fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
for t, g in logs.items():
    x = pd.PeriodIndex(g.month, freq="M").to_timestamp()
    ax[0].plot(x, np.cumprod(1 + g.ret_real), label=f"{t} реальная")
    ax[0].plot(x, np.cumprod(1 + g.ret_mirror), "--", label=f"{t} зеркальная")
    ax[1].plot(x, g.trend, label=f"{t} по тренду"); ax[1].plot(x, g.long, ":", label=f"{t} покупки")
    ax[2].plot(x, g.er_on, label=f"{t} ER"); ax[2].plot(x, g.vol_on, ":", label=f"{t} волатильность")
    ax[3].plot(x, g.sess2, label=f"{t} Лондон"); ax[3].plot(x, g.sess0, ":", label=f"{t} любое время")
ax[0].set_yscale("log"); ax[0].set_title("Капитал популяции, ×", loc="left", fontsize=9)
ax[1].set_title("Направление", loc="left", fontsize=9)
ax[2].set_title("Доля с фильтрами режима", loc="left", fontsize=9)
ax[3].set_title("Сессия", loc="left", fontsize=9)
for a in ax:
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / "img" / "evolution_neutral.png", dpi=110)

markets = {"реальная": Market(), "зеркальная": Market(mirrored())}
print("\nКандидаты: 3 самых старых выживших каждого прогона, по годам с $1 000")
for t in runs:
    pop = pd.read_csv(RES / f"evolve2_{t}_pop.csv")
    for _, o in pop.sort_values(["born", "r"], ascending=[True, False]).head(3).iterrows():
        p = Params(**{k: int(o[k]) if k in INT_GENES else float(o[k]) for k in GENES})
        print(f"\n{t}: родился {o.born}\n  {p}")
        for name, m in markets.items():
            bal = [m.run(p, f"{y}-01-01", f"{y + 1}-01-01")["balance"] for y in range(2017, 2027)]
            print(f"  {name:>10}: " + " ".join(f"{b:,.0f}" for b in bal) + f" | сумма P&L {sum(bal) - 10000:+,.0f}")

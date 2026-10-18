"""Summarise evolve runs: real vs mirrored price, oldest survivor as candidate."""
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
INT_GENES = {"dirs", "step_atr", "max_orders", "tp_atr", "ema"}

runs = sorted(Path(f).stem[len("evolve_"):] for f in glob.glob(str(RES / "evolve_*.csv"))
              if not f.endswith("_pop.csv"))
logs = {t: pd.read_csv(RES / f"evolve_{t}.csv") for t in runs}

print("| Прогон | Капитал популяции, ×за период | Типичная особь, медиана мес. | Месяцев в плюсе (медиана) | Особей в конце | Доля «только покупки» в конце |")
print("|---|---|---|---|---|---|")
for t, g in logs.items():
    print(f"| {t} | ×{np.prod(1 + g.ret):.3g} | {g.med_ret.median():+.1%} | {(g.med_ret > 0).mean():.0%} "
          f"| {g['pop'].iloc[-1]} | {g.long.iloc[-1]:.0%} |")

# yearly capital-weighted return of the population
print("\nДоходность капитала популяции по годам")
yr = pd.DataFrame({t: g.assign(y=g.month.str[:4]).groupby("y").ret.apply(lambda r: np.prod(1 + r) - 1)
                   for t, g in logs.items()})
print(yr.map(lambda v: f"{v:+.0%}").to_string())

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for t, g in logs.items():
    x = pd.PeriodIndex(g.month, freq="M").to_timestamp()
    st = "-" if t.startswith("real") else "--"
    ax[0].plot(x, np.cumprod(1 + g.ret), st, label=t)
    ax[1].plot(x, g["pop"], st, label=t)
    ax[2].plot(x, g.long, st, label=t)
ax[0].set_yscale("log"); ax[0].set_title("Капитал популяции (без притока детей), ×", loc="left", fontsize=9)
ax[1].set_title("Размер популяции", loc="left", fontsize=9)
ax[2].set_title("Доля особей «только покупки»", loc="left", fontsize=9)
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "img" / "evolution.png", dpi=110)

# candidate: oldest member of the final real population
markets = {"real": Market(), "mirror": Market(mirrored())}
print("\nКандидаты — самые старые выжившие (реальная цена). Проверка по годам с $1 000 и на зеркальной цене:")
for t in [r for r in runs if r.startswith("real")]:
    pop = pd.read_csv(RES / f"evolve_{t}_pop.csv")
    old = pop.sort_values(["born", "bal"], ascending=[True, False]).iloc[0]
    p = Params(**{k: int(old[k]) if k in INT_GENES else float(old[k]) for k in GENES})
    print(f"\n{t}: родился {old.born}, баланс ${old.bal:,.0f}\n  гены {p}")
    for name, m in markets.items():
        bal = [m.run(p, f"{y}-01-01", f"{y + 1}-01-01")["balance"] for y in range(2017, 2027)]
        print(f"  {name:>6}: " + " ".join(f"{b:,.0f}" for b in bal) + f" | сумма P&L {sum(bal) - 10000:+,.0f}")

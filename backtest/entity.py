"""Deep dive into the chosen trading entity.

Starts every Monday 00:00 and every Wednesday 13:00 UTC from 2017 to 2024-09,
runs each for up to 24 months, and reports survival, growth, transfer across
start years and the pattern of its deaths. Also one continuous account from
2017-01 and the same starts on the mirrored price.

    python3 entity.py
"""
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import Market, Params
from evolve import mirrored

ROOT = Path(__file__).resolve().parent.parent

# candidate #1019 of select_entity.py (results/entity_candidates.csv)
ENTITY = Params(dirs=0, step=30.0, step_atr=0, step_grow=1.0, mult=1.6, max_orders=5, tp=3.0,
                tp_atr=1, sl=0.0, leverage=500.0, risk=0.0, ema=0, er_win=1, er_max=0.02,
                sess=1, vol_max=1.0)
MONTHS = 24
DAY = 1440


def after(t, months):
    return (t + pd.Timedelta(days=30.44 * months)).floor("min")


def runs(m, starts):
    rows = []
    for t in starts:
        end = after(t, MONTHS)
        r = m.run(ENTITY, t, end)
        eq = r["equity"]
        marks = {k: eq.asof(after(t, k)) / 1000 for k in (3, 6, 12, 24)}
        life = (r["died"] - t).days / 30.44 if r["died"] is not None else np.nan
        rows.append(dict(start=t, died=r["died"], life=life, maxdd=r["maxdd"], cycles=r["cycles"],
                         **{f"g{k}": v for k, v in marks.items()}))
    return pd.DataFrame(rows)


def summary(df, name):
    alive = lambda k: (df.died.isna() | (df.life > k)).mean()
    print(f"\n{name}: {len(df)} стартов")
    print("  живы: " + " ".join(f"{k} мес {alive(k):.0%}" for k in (3, 6, 12, 18, 24)))
    for k in (3, 6, 12, 24):
        g = df[f"g{k}"]
        print(f"  рост к {k:>2} мес: медиана ×{g.median():.2f}, 10% ×{g.quantile(.1):.2f}, "
              f"25% ×{g.quantile(.25):.2f}, 75% ×{g.quantile(.75):.2f}")
    surv = df[df.died.isna()]
    print(f"  макс. просадка у выживших 24 мес: медиана {surv.maxdd.median():.0%}, "
          f"90-й перцентиль {surv.maxdd.quantile(.9):.0%}, максимум {surv.maxdd.max():.0%}")


if __name__ == "__main__":
    m = Market()
    # the EMA gene is inert for a long-only entity
    ref = replace(ENTITY, ema=1440)
    assert m.run(ref, "2019-01-01", "2020-01-01")["balance"] == m.run(ENTITY, "2019-01-01", "2020-01-01")["balance"]

    mon = pd.date_range("2017-01-02", "2024-09-30", freq="W-MON", tz="UTC")
    wed = pd.date_range("2017-01-04 13:00", "2024-09-30", freq="W-WED", tz="UTC")
    a, b = runs(m, mon), runs(m, wed)
    summary(a, "Старт в понедельник 00:00")
    summary(b, "Старт в среду 13:00")
    df = pd.concat([a, b], ignore_index=True)

    print("\nПереносимость по году старта (оба набора стартов):")
    by = df.assign(y=df.start.dt.year).groupby("y").agg(
        стартов=("g12", "size"),
        живы_12=("life", lambda x: (x.isna() | (x > 12)).mean()),
        живы_24=("died", lambda x: x.isna().mean()),
        рост12_мед=("g12", "median"), рост12_25=("g12", lambda x: x.quantile(.25)),
        рост24_мед=("g24", "median"), просадка_мед=("maxdd", "median"))
    print(by.round(2).to_string())

    dead = df[df.died.notna()].sort_values("died")
    print(f"\nСмерти: {len(dead)} из {len(df)} стартов")
    if len(dead):
        ep = (dead.died.diff() > pd.Timedelta(days=5)).cumsum()
        c = m.bars.close
        for _, g in dead.groupby(ep):
            t = g.died.iloc[0]
            i = int(m.index.searchsorted(t))
            p_hi = c.iloc[max(i - 10 * DAY, 0):i].max()
            print(f"  {t:%Y-%m-%d}…{g.died.iloc[-1]:%m-%d}: погибло {len(g)}, возраст {g.life.min():.1f}–"
                  f"{g.life.max():.1f} мес, старты {g.start.min():%Y-%m-%d}…{g.start.max():%Y-%m-%d}; "
                  f"золото от максимума 10 дней {c.iloc[i] / p_hi - 1:+.1%} "
                  f"(${p_hi:,.0f} → ${c.iloc[i]:,.0f})")

    # continuous account
    r = m.run(ENTITY, "2017-01-01", "2026-10-01")
    eq = r["equity"]
    print(f"\nНепрерывный счёт с 2017-01: итог ${r['balance']:,.0f}, макс. просадка {r['maxdd']:.0%}, "
          f"смерть {r['died']}")
    print("  на конец года: " + ", ".join(f"{t.year}: ${v:,.0f}" for t, v in eq.resample("YE").last().items()))

    mm = Market(mirrored())
    summary(runs(mm, mon), "Зеркальная цена (тренд вниз), старт в понедельник")

    fig, ax = plt.subplots(3, 1, figsize=(10, 11))
    ks = np.arange(0, MONTHS + 0.01, 0.25)
    for d, lab in ((a, "понедельник 00:00"), (b, "среда 13:00")):
        ax[0].plot(ks, [(d.died.isna() | (d.life > k)).mean() for k in ks], label=lab)
    ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=8)
    ax[0].set_title("Доля живых запусков, месяцев от старта", loc="left", fontsize=9)
    for k in ("g3", "g6", "g12", "g24"):
        ax[1].scatter(df.start, df[k], s=4, label=f"к {k[1:]} мес")
    ax[1].set_yscale("symlog", linthresh=1); ax[1].legend(fontsize=8)
    ax[1].set_title("Рост капитала, × от даты старта", loc="left", fontsize=9)
    ax[2].plot(eq.index, eq.values, lw=0.8)
    ax[2].set_yscale("log")
    ax[2].set_title("Один непрерывный счёт с января 2017, $", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "img" / "entity.png", dpi=110)

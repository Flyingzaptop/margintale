"""Entity #1019 in money terms: $100 vs $1000, by year and month, fixed vs proportional lot.

Proportional lot: base lot = 0.01 per DOLLARS_PER_LOT of balance at basket open
(rounded down, at least 0.01); later orders scale with it (x1.6 each).
A full grid (5 orders) plus one more $30 step against costs about $1 110 per
0.01 of base lot, so DOLLARS_PER_LOT is also "balance per that worst case".

    python3 entity_money.py
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Market
from entity import ENTITY, after

ROOT = Path(__file__).resolve().parent.parent
FULL = ("2017-01-01", "2026-10-01")
LOSS_PER_LOT = 1 * 150 + 2 * 120 + 3 * 90 + 4 * 60 + 7 * 30  # $ at 5 orders + one step, per 0.01 base


def proportional(dollars_per_lot):
    return replace(ENTITY, risk=1000.0 / dollars_per_lot)


def yearly(eq, deposit):
    ye = eq.resample("YE").last()
    start = pd.concat([pd.Series([deposit], index=[eq.index[0]]), ye.iloc[:-1]]).to_numpy()
    return pd.DataFrame({"начало": start, "конец": ye.to_numpy(),
                         "%": (ye.to_numpy() / start - 1) * 100}, index=ye.index.year)


def monthly(eq):
    me = eq.resample("ME").last()
    r = me.pct_change()
    r.iloc[0] = me.iloc[0] / eq.iloc[0] - 1
    t = pd.DataFrame({"y": r.index.year, "m": r.index.month, "r": r.to_numpy() * 100})
    return t.pivot(index="y", columns="m", values="r")


def lifecycle_stats(m, p, deposit):
    starts = pd.date_range("2017-01-02", "2024-09-30", freq="W-MON", tz="UTC")
    rows = []
    for t in starts:
        r = m.run(p, t, after(t, 24), deposit=deposit)
        eq = r["equity"]
        rows.append(dict(died=r["died"], dd=r["maxdd"], **{f"g{k}": eq.asof(after(t, k)) / deposit
                                                           for k in (6, 12, 24)}))
    d = pd.DataFrame(rows)
    life = d.died.isna()
    return dict(живы_24=life.mean(), рост12_мед=d.g12.median(), рост12_10=d.g12.quantile(.1),
                рост24_мед=d.g24.median(), просадка_мед=d.dd[life].median() if life.any() else 1.0)


def show(title, df, fmt="{:,.0f}"):
    print(f"\n{title}")
    print(df.to_string(float_format=lambda v: fmt.format(v) if pd.notna(v) else ""))


if __name__ == "__main__":
    m = Market()
    pd.set_option("display.width", 250)

    print("Фиксированный лот 0.01, непрерывный счёт с 2017-01-01:")
    curves = {}
    for dep in (100, 1000):
        r = m.run(ENTITY, *FULL, deposit=dep)
        eq = curves[dep] = r["equity"]
        peak = eq.max()
        print(f"  депозит ${dep}: пик ${peak:,.0f} ({peak / dep - 1:+,.0%}) {eq.idxmax():%Y-%m-%d}, "
              f"смерть {r['died']:%Y-%m-%d} " if r["died"] is not None else
              f"  депозит ${dep}: итог ${r['balance']:,.0f}")
        show(f"По годам, депозит ${dep}", yearly(eq, dep).round(1))
    show("По месяцам, %, депозит $1000", monthly(curves[1000]).round(1), "{:+.1f}")

    print(f"\nХудший случай сетки: 5 ордеров + ещё шаг $30 = ${LOSS_PER_LOT:,} убытка на каждые 0.01 стартового лота")
    rows = []
    for dpl in (1000, 1500, 2000, 3000, 5000, 10000):
        p = proportional(dpl)
        for dep in (100, 1000, 10000):
            r = m.run(p, *FULL, deposit=dep)
            eq = r["equity"]
            yrs = ((r["died"] or eq.index[-1]) - eq.index[0]).days / 365.25
            cagr = (eq.max() / dep) ** (1 / max(yrs, 0.1)) - 1 if r["died"] else (r["balance"] / dep) ** (1 / yrs) - 1
            rows.append(dict(**{"$ на 0.01 лота": dpl, "депозит": dep,
                                "итог": r["balance"], "пик": eq.max(),
                                "смерть": f"{r['died']:%Y-%m-%d}" if r["died"] is not None else "—",
                                "рост в год до пика/конца, %": cagr * 100,
                                "макс. просадка, %": r["maxdd"] * 100}))
    show("Лот пропорционально балансу, непрерывный счёт с 2017-01", pd.DataFrame(rows).set_index(
        ["$ на 0.01 лота", "депозит"]))

    rows = []
    for dpl in (None, 1000, 2000, 3000, 5000):
        p = ENTITY if dpl is None else proportional(dpl)
        rows.append(dict(лот="фикс 0.01" if dpl is None else f"0.01 на ${dpl}",
                         **lifecycle_stats(m, p, 1000)))
    show("Старты каждый понедельник 2017-2024, депозит $1000, до 24 мес",
         pd.DataFrame(rows).set_index("лот"), "{:.2f}")

    best = proportional(3000)
    r = m.run(best, *FULL, deposit=1000)
    show("По годам, лот 0.01 на $3000, депозит $1000", yearly(r["equity"], 1000).round(1))
    show("По месяцам, %, лот 0.01 на $3000, депозит $1000", monthly(r["equity"]).round(1), "{:+.1f}")

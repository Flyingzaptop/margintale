"""How long an entity lives (in closed baskets and in days) vs how much it has made.

For one genome, started every Monday 2017-01..2024-09 with DEPOSIT, run to death or
the end of the data. For each n (closed baskets):
  S(n)      share of starts that reach n baskets alive (Kaplan-Meier, censored at data end),
  M(n)      median equity multiple after basket n among those alive,
  stop(n)   value of the rule "stop after n baskets": the multiple you hold if you got there,
            the dead remnant if not; median and mean over starts.
The same by calendar day. A fitted per-basket growth g gives M(n) ~ (1 + g)^n.

    python3 life_curve.py [aggro_final.csv row] [deposit]
"""
import sys
from dataclasses import astuple

import numpy as np
import pandas as pd

from engine import DEAD, Market, _run_core, params_from

NS = [1, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
DAYS = [1, 3, 7, 14, 30, 60, 90, 180, 270, 365]
EMPTY = (np.empty(0, np.int64), np.empty(0))


def runs(m, p, deposit, starts):
    out = []
    end = len(m.index) - 1
    P = np.array(astuple(p), dtype=np.float64)
    for t in starts:
        i0 = m.span(t, "2100-01-01")[0]
        r = _run_core(m.o, m.h, m.l, m.c, m.atr, m.ema(p.ema), m.filt, i0, end, deposit, P,
                      EMPTY[0], EMPTY[1], True)
        n = min(r[3], len(r[11]))
        out.append(dict(i0=i0, died=r[2], n=n, bal=r[11][:n] / deposit, bar=r[12][:n],
                        remnant=r[0] / deposit if r[2] >= 0 else None))
    return out


def km(life, dead, grid):
    s, out = 1.0, []
    order = np.argsort(life)
    life, dead = life[order], dead[order]
    for g in grid:
        pass
    S, t_prev = [], 0
    s = 1.0
    events = np.unique(life[dead])
    j = 0
    for g in grid:
        while j < len(events) and events[j] <= g:
            at = (life >= events[j]).sum()
            d = ((life == events[j]) & dead).sum()
            s *= 1 - d / at
            j += 1
        S.append(s)
    return np.array(S)


def by_trades(rs):
    life = np.array([r["n"] for r in rs], dtype=float)
    dead = np.array([r["died"] >= 0 for r in rs])
    S = km(life, dead, NS)
    rows = []
    for n, s in zip(NS, S):
        alive = [r["bal"][n - 1] for r in rs if r["n"] >= n]
        stop = [r["bal"][n - 1] if r["n"] >= n else (r["remnant"] if r["remnant"] is not None else np.nan)
                for r in rs]
        stop = np.array([v for v in stop if v == v])  # drop censored-before-n
        rows.append({"корзин": n, "доживут": s, "рост у живых, медиана": np.median(alive) if alive else np.nan,
                     "стоп после n: медиана": np.median(stop) if len(stop) else np.nan,
                     "стоп после n: среднее": stop.mean() if len(stop) else np.nan,
                     "под наблюдением": len(alive)})
    return pd.DataFrame(rows)


def by_days(m, rs):
    rows = []
    for d in DAYS:
        held, alive_n, known = [], 0, 0
        for r in rs:
            t_end = m.index[r["i0"]] + pd.Timedelta(days=d)
            if t_end > m.index[-1]:
                continue
            known += 1
            k = m.index.searchsorted(t_end)
            if r["died"] >= 0 and r["died"] <= k:
                held.append(r["remnant"])
                continue
            alive_n += 1
            j = np.searchsorted(r["bar"], k, side="right") - 1
            held.append(r["bal"][j] if j >= 0 else 1.0)
        held = np.array(held)
        rows.append({"дней": d, "живы": alive_n / max(known, 1), "капитал, медиана": np.median(held),
                     "среднее": held.mean(), "25%": np.percentile(held, 25), "75%": np.percentile(held, 75)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    row = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dep = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    src = sys.argv[3] if len(sys.argv) > 3 else f"../results/aggro_d{int(dep)}_s1_final.csv"
    p = params_from(pd.read_csv(src).iloc[row])
    m = Market()
    starts = pd.date_range("2017-01-02", "2024-09-30", freq="W-MON", tz="UTC")
    rs = runs(m, p, dep, starts)
    print(f"Особь: {p}\nдепозит ${dep:.0f}, стартов {len(rs)}")
    life = np.array([r["n"] for r in rs])
    dead = np.array([r["died"] >= 0 for r in rs])
    days_alive = np.array([((m.index[r["died"]] if r["died"] >= 0 else m.index[-1]) - m.index[r["i0"]]).days
                           for r in rs])
    tpd = np.median(life / np.maximum(days_alive, 1))
    print(f"умерли {dead.mean():.0%}; жизнь в корзинах: медиана {np.median(life[dead]) if dead.any() else np.nan:.0f} "
          f"(у умерших), в днях: медиана {np.median(days_alive[dead]) if dead.any() else np.nan:.0f}; "
          f"корзин в день: {tpd:.2f}")
    t = by_trades(rs)
    pd.set_option("display.width", 250)
    print("\nПо числу закрытых корзин:")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    ok = t.dropna(subset=["рост у живых, медиана"])
    ok = ok[(ok["рост у живых, медиана"] > 0) & (ok["под наблюдением"] >= 20)]
    g = np.exp(np.polyfit(ok["корзин"], np.log(ok["рост у живых, медиана"]), 1)[0]) - 1
    print(f"\nРост у живых ≈ (1 + {g:.3%})^n за корзину; при {tpd:.2f} корзин в день ≈ {(1 + g) ** tpd - 1:.2%} в день")
    print("\nПо календарным дням от запуска:")
    print(by_days(m, rs).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

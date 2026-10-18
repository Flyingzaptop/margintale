"""Survival analysis: if I start the algorithm on a random day, how likely is it
to be alive t days later?

Starts every day at 00:00 UTC from 2017-01-01 to 2026-08-31; every start runs
until death or the end of the data (then it is censored). Kaplan-Meier estimate
of S(t) with a Greenwood 95% interval, daily hazard by age, the equity multiple
of the living, and S(t) separately for start eras.

    python3 survival.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import Market
from engine2 import Genes, Market2
from entity import ENTITY

ROOT = Path(__file__).resolve().parent.parent
STARTS = pd.date_range("2017-01-01", "2026-08-31", freq="D", tz="UTC")
DAYS = [1, 2, 3, 7, 14, 30, 60, 90, 180, 270, 365, 540, 730, 1095]
ERAS = {"2017–2019": ("2017", "2019"), "2020–2022": ("2020", "2022"), "2023–2026": ("2023", "2026")}

# best genome of search2 (results/search2_s1_final.csv, row 0)
MARTIN = Genes(entry=0, dirs=0, look=1, ema=1, offset=0.0, sl=5.0, tp=12.0, trail_start=0.0, trail=1.0,
               be=0.0, max_hold=0, pend_life=60, sess=0, vol_max=0.0, er_win=0, er_max=0.0,
               risk=0.08, loss_mult=2.0, loss_cap=8.0, deposit=100.0, exit_sig=0)


def kaplan_meier(t, dead, grid):
    """t: time to death or censoring (days), dead: bool. Returns S, lo, hi on grid."""
    order = np.argsort(t)
    t, dead = t[order], dead[order]
    times = np.unique(t[dead])
    s, var, at = 1.0, 0.0, len(t)
    S, V, T = [1.0], [0.0], [0.0]
    for u in times:
        at_risk = (t >= u).sum()
        d = ((t == u) & dead).sum()
        s *= 1 - d / at_risk
        if at_risk > d:
            var += d / (at_risk * (at_risk - d))
        S.append(s), V.append(var), T.append(u)
    S, V, T = np.array(S), np.array(V), np.array(T)
    idx = np.searchsorted(T, grid, side="right") - 1
    s = S[idx]
    se = s * np.sqrt(V[idx])
    at_risk = np.array([(t >= g).sum() for g in grid])
    return s, np.clip(s - 1.96 * se, 0, 1), np.clip(s + 1.96 * se, 0, 1), at_risk


def table(name, t, dead, mult_at):
    s, lo, hi, n = kaplan_meier(t, dead, np.array(DAYS, dtype=float))
    rows = []
    for k, d in enumerate(DAYS):
        alive = t >= d
        mm = mult_at.get(d)
        rows.append({"день": d, "доживут": f"{s[k]:.1%}", "95% интервал": f"{lo[k]:.1%}–{hi[k]:.1%}",
                     "под наблюдением": int(n[k]),
                     "рост у живых, медиана": f"×{np.nanmedian(mm[alive]):.2f}" if mm is not None and alive.any() else "",
                     "25%–75%": f"×{np.nanpercentile(mm[alive], 25):.2f}–×{np.nanpercentile(mm[alive], 75):.2f}"
                     if mm is not None and alive.any() else ""})
    print(f"\n{name}: {len(t)} стартов, умерли {dead.sum()}, "
          f"медиана жизни {median_life(t, dead)}")
    print(pd.DataFrame(rows).to_string(index=False))
    return s


def median_life(t, dead):
    grid = np.arange(0, t.max() + 1)
    s, *_ = kaplan_meier(t, dead, grid.astype(float))
    below = np.nonzero(s <= 0.5)[0]
    return f"{grid[below[0]]:.0f} дней" if len(below) else f"> {t.max():.0f} дней (не достигнута)"


def hazard(t, dead, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        at = (t >= a).sum()
        d = ((t >= a) & (t < b) & dead).sum()
        exposure = np.clip(np.minimum(t, b) - a, 0, None)[t >= a].sum()
        out.append((a, b, at, d, d / exposure if exposure else np.nan))
    return out


def analyse(name, t, dead, mult_at, starts):
    table(name, t, dead, mult_at)
    print("  вероятность умереть в ближайшие сутки, если дожил до возраста:")
    for a, b, at, d, h in hazard(t, dead, [0, 7, 30, 90, 180, 365, 730, 1500]):
        print(f"    {a:>4}–{b:<4} дней: {h:.3%} в день  (смертей {d}, под наблюдением {at})")
    curves = {}
    for era, (a, b) in ERAS.items():
        sel = (starts.year >= int(a)) & (starts.year <= int(b))
        s, lo, hi, n = kaplan_meier(t[sel], dead[sel], np.array(DAYS, dtype=float))
        curves[era] = (t[sel], dead[sel])
        print(f"  старты {era} ({sel.sum()}): " + " ".join(
            f"{d}д {v:.0%}" for d, v, nn in zip(DAYS, s, n) if nn >= 20))
    return curves


def martin_runs(m):
    months = 117
    L = m.lifecycles([MARTIN], STARTS, months)
    end = m.index[-1]
    died = L["died"][0]
    t = np.array([((m.index[d] - s).total_seconds() if d >= 0 else (end - s).total_seconds()) / 86400
                  for d, s in zip(died, STARTS)])
    eq = L["eq"][0] / MARTIN.deposit
    mult = {d: eq[:, min(int(d / 30.44), months - 1)] if d >= 30 else np.full(len(t), np.nan) for d in DAYS}
    return t, died >= 0, mult


def grid_runs(m):
    months = 117
    died, eq, _ = m.lifecycles([ENTITY], months=months, start_times=STARTS)
    end = m.index[-1]
    died = died[0]
    t = np.array([((m.index[d] - s).total_seconds() if d >= 0 else (end - s).total_seconds()) / 86400
                  for d, s in zip(died, STARTS)])
    mult = {d: eq[0][:, min(int(d / 30.44), months - 1)] / 1000 if d >= 30 else np.full(len(t), np.nan)
            for d in DAYS}
    return t, died >= 0, mult


if __name__ == "__main__":
    res = {}
    t, dead, mult = martin_runs(Market2())
    res["мартингейл-пробой, $100, риск 8%, ×2 после убытка"] = analyse(
        "Мартингейл-пробой (search2 #0), депозит $100", t, dead, mult, STARTS)
    t, dead, mult = grid_runs(Market())
    res["сетка #1019, $1000, лот 0.01"] = analyse("Сетка #1019, депозит $1000", t, dead, mult, STARTS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    grid = np.arange(0, 1096, 1.0)
    for ax, (name, curves) in zip(axes, res.items()):
        for era, (tt, dd) in curves.items():
            s, lo, hi, _ = kaplan_meier(tt, dd, grid)
            ax.plot(grid, s, label=f"старты {era}")
            ax.fill_between(grid, lo, hi, alpha=0.15)
        ax.set_title(name, loc="left", fontsize=9)
        ax.set_xlabel("дней от запуска")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("вероятность быть живым")
    fig.tight_layout()
    fig.savefig(ROOT / "img" / "survival.png", dpi=110)

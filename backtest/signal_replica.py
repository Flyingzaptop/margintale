"""Replica of MQL5 signal 2304847 in the grid engine, fitted to its own trades.

The signal is a trend-following flat grid: enter with the 1-day trend, small TP,
add the same lot every few $ against, close the whole basket together, cut losers.
Its P&L is normalised to a fixed 0.01 lot ($ per oz) to compare with replicas
that trade 0.01 lot. Replicas are fitted on its period (2024-11-27..2026-09-03),
then the best ones are run on 2017-2024, which the signal never traded.

    python3 signal_replica.py
"""
from itertools import product

import numpy as np
import pandas as pd

from engine import Market, Params

FIT = ("2024-11-27", "2026-09-03")
DEP = 451.49


def signal_curve():
    t = pd.read_csv("../results/signal_trades.csv", parse_dates=["close_utc"])
    per_oz = t.profit / (t.vol * 100)  # $ per 1 oz = per 0.01 lot
    s = pd.Series(per_oz.to_numpy(), index=pd.DatetimeIndex(t.close_utc)).sort_index()
    return s.groupby(s.index.floor("D")).sum()


if __name__ == "__main__":
    m = Market()
    sig = signal_curve()
    days = pd.date_range(FIT[0], FIT[1], freq="D", tz="UTC")
    sig_cum = sig.reindex(days, fill_value=0.0).cumsum()
    print(f"Сигнал, нормировано на 0.01 лота: итог ${sig_cum.iloc[-1]:,.0f} за период, "
          f"сделок {len(pd.read_csv('../results/signal_trades.csv'))}")

    rows = []
    grid = product((1440, 7200), (1.5, 2.0, 3.0, 4.0), (5, 8, 12), (1.0, 1.5, 2.0, 3.0), (0.0, 0.03, 0.05, 0.1))
    for ema, step, mo, tp, sl in grid:
        p = Params(dirs=3, ema=ema, step=step, step_atr=0, mult=1.0, max_orders=mo, tp=tp, tp_atr=0, sl=sl,
                   leverage=100.0, risk=0.0, sess=0, vol_max=0.0, er_win=0)
        r = m.run(p, *FIT, deposit=DEP)
        eq = (r["equity"] - DEP).resample("D").last().reindex(days).ffill().fillna(0.0)
        dead = r["died"] is not None
        corr = np.corrcoef(eq.diff().fillna(0), sig_cum.diff().fillna(0))[0, 1]
        rows.append(dict(ema=ema, step=step, max_orders=mo, tp=tp, sl=sl, итог=eq.iloc[-1], смерть=dead,
                         корр_дней=corr, корр_кривой=np.corrcoef(eq, sig_cum)[0, 1], циклов=r["cycles"],
                         просадка=r["maxdd"]))
    df = pd.DataFrame(rows)
    df["сходство"] = df.корр_кривой - np.abs(np.log(np.maximum(df.итог, 1) / sig_cum.iloc[-1]))
    df = df.sort_values("сходство", ascending=False)
    df.to_csv("../results/signal_replica.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nЛучшие реплики на периоде сигнала (лот 0.01, депозит $451):")
    print(df.head(12).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nТе же реплики на годах, когда сигнала не было (старт $451, лот 0.01, каждый год с 1 января):")
    for _, r0 in df.head(5).iterrows():
        p = Params(dirs=3, ema=int(r0.ema), step=r0.step, step_atr=0, mult=1.0, max_orders=int(r0.max_orders),
                   tp=r0.tp, tp_atr=0, sl=r0.sl, leverage=100.0, risk=0.0, sess=0, vol_max=0.0, er_win=0)
        res = []
        for y in range(2017, 2025):
            rr = m.run(p, f"{y}-01-01", f"{y + 1}-01-01", deposit=DEP)
            res.append(f"{y}: {rr['balance'] / DEP:.2f}" + ("†" if rr["died"] is not None else ""))
        print(f"  ema {int(r0.ema)}, шаг ${r0.step}, до {int(r0.max_orders)}, TP ${r0.tp}, SL {r0.sl:.0%}: " + ", ".join(res))

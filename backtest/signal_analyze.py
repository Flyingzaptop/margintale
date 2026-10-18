"""Reverse-engineering MQL5 signal 2304847 (MSC SuperGold Pro): what its trades look like
and what the market looked like at its entries, against random moments.

Server time is Europe/Athens (UTC+2 / UTC+3 in summer): entry prices match our M1 bars
best with that offset.

    python3 signal_analyze.py
"""
import numpy as np
import pandas as pd

from engine import load_m1
from signal_load import load


def to_utc(t):
    return (pd.DatetimeIndex(t).tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="shift_forward")
            .tz_convert("UTC").floor("min").as_unit("ms"))


def features(m, idx, side):
    """Market state at bar positions idx (the bar of entry), sign by trade side (+1 buy, -1 sell)."""
    c, a = m.close.to_numpy(), m.atr.to_numpy()
    hh = {n: m.high.rolling(n).max().shift(1).to_numpy() for n in (15, 60, 240)}
    ll = {n: m.low.rolling(n).min().shift(1).to_numpy() for n in (15, 60, 240)}
    out = {}
    p = c[idx - 1]
    for n in (5, 15, 60, 240, 1440):
        out[f"ход_{n}м_ATR"] = side * (p - c[idx - 1 - n]) / a[idx - 1]  # >0: move in trade direction before entry
    for n in (15, 60, 240):
        rng = np.maximum(hh[n][idx] - ll[n][idx], 1e-9)
        pos = (p - ll[n][idx]) / rng
        out[f"место_в_диапазоне_{n}м"] = np.where(side > 0, pos, 1 - pos)  # 0: at the low for a buy
    out["ATR_$"] = a[idx - 1]
    out["час_UTC"] = m.index[idx].hour
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = load()
    x = d[d.symbol == "XAUUSD.f"].copy()
    x["side"] = np.where(x.type == "Buy", 1, -1)
    x["open_utc"] = to_utc(x.t_open)
    x["close_utc"] = to_utc(x.t_close)
    x["dist"] = x.side * (x.p_close - x.p_open)
    x["dur_min"] = (x.t_close - x.t_open).dt.total_seconds() / 60
    m = load_m1()
    x = x[(x.open_utc >= m.index[2000]) & (x.open_utc <= m.index[-2])].dropna(subset=["open_utc"])
    idx = m.index.get_indexer(x.open_utc, method="bfill")
    f = features(m, idx, x.side.to_numpy())
    f["dist_ATR"] = x.dist.to_numpy() / f["ATR_$"].to_numpy()
    f["side"] = x.side.to_numpy()

    rng = np.random.default_rng(1)
    ridx = rng.choice(np.arange(m.index.searchsorted(x.open_utc.min()), m.index.searchsorted(x.open_utc.max())), 20000)
    base = pd.concat([features(m, ridx, np.ones(len(ridx))), features(m, ridx, -np.ones(len(ridx)))])

    pd.set_option("display.width", 250)
    print(f"{len(x)} сделок XAUUSD с данными, {x.open_utc.min():%Y-%m-%d} … {x.open_utc.max():%Y-%m-%d}")
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    cols = [c for c in f.columns if c.startswith(("ход", "место"))]
    print("\nСостояние рынка перед входом (знак — по направлению сделки), его входы против случайных моментов:")
    print(pd.DataFrame({"его, медиана": f[cols].median(), "его, 25%": f[cols].quantile(.25),
                        "его, 75%": f[cols].quantile(.75), "случайно, медиана": base[cols].median()}).round(2))
    print("\nЧасы входа UTC (доля сделок):")
    print((f["час_UTC"].value_counts(normalize=True).sort_index() * 100).round(1).to_string())
    print("\nРезультат сделки в ATR (квантили):", f.dist_ATR.quantile([.01, .05, .1, .25, .5, .75, .9, .95, .99]).round(2).to_dict())
    print("Результат в $:", x.dist.quantile([.01, .05, .1, .25, .5, .75, .9, .95, .99]).round(2).to_dict())
    print("Длительность, мин:", x.dur_min.quantile(q).round(1).to_dict())
    wins, losses = x[x.dist > 0], x[x.dist <= 0]
    print(f"Прибыльные: {len(wins) / len(x):.1%}, медиана +${wins.dist.median():.2f} за {wins.dur_min.median():.0f} мин; "
          f"убыточные: медиана ${losses.dist.median():.2f} за {losses.dur_min.median():.0f} мин")
    print("Покупки / продажи по году:", x.groupby(x.open_utc.dt.year).side.apply(lambda s: f"{(s > 0).mean():.0%} buy").to_dict())

    # overlap: several positions at once, larger lots
    ev = pd.concat([pd.DataFrame({"t": x.t_open, "d": 1}), pd.DataFrame({"t": x.t_close, "d": -1})]).sort_values(["t", "d"])
    print("Открыто одновременно (по событиям):", ev.d.cumsum().value_counts().sort_index().to_dict())
    big = x[x.vol > 0.01]
    print(f"\nСделок с лотом > 0.01: {len(big)}; первые 20:")
    print(big[["t_open", "type", "vol", "p_open", "t_close", "p_close", "profit", "dur_min"]].head(20).to_string())
    x.assign(**{k: f[k].to_numpy() for k in f.columns}).to_csv("../results/signal_trades.csv", index=False)

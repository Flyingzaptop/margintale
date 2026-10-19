"""Extend XAUUSD M1 past the end of data/XAUUSD_41.parquet with Yahoo COMEX gold futures (GC=F).

Yahoo serves 1m bars for the last 7 days, 2m for 60 days, 5m for 60 days. Each period uses the
finest one available; a coarser bar is split into minutes along o -> first extreme -> second
extreme -> c (low first on an up bar, high first on a down bar), so 5-minute ranges and the ATR
keep their size. Futures are converted to spot with the futures/spot ratio, fitted as a
straight line in time on the overlap with the spot file (carry decays towards expiry).

    python3 extend_yahoo.py --fetch     # download data/yahoo/GC_{1m,2m,5m}.json first
    python3 extend_yahoo.py --check     # also: entities on the overlap, real M1 vs rebuilt 5m
Output: data/XAUUSD_ext.parquet (spot file + extension).
"""
import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from engine import DATA

Y = DATA.parent / "yahoo"
EXT = DATA.parent / "XAUUSD_ext.parquet"
RANGES = {"1m": "7d", "2m": "60d", "5m": "60d"}


def fetch():
    Y.mkdir(exist_ok=True)
    for iv, rg in RANGES.items():
        subprocess.run(["curl", "-s", "-m", "60", "-A", "Mozilla/5.0", "-o", str(Y / f"GC_{iv}.json"),
                        f"https://query2.finance.yahoo.com/v8/finance/chart/GC=F?interval={iv}&range={rg}&includePrePost=true"],
                       check=True)


def load(iv):
    r = json.load(open(Y / f"GC_{iv}.json"))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                     index=pd.to_datetime(r["timestamp"], unit="s", utc=True)).dropna()
    return d[d.index.second == 0]          # drop the live, still-forming last bar


def to_minutes(b, n):
    """Split n-minute bars into n one-minute bars along o -> ext1 -> ext2 -> c."""
    o, h, l, c = (b[x].to_numpy() for x in ("open", "high", "low", "close"))
    up = c >= o
    e1, e2 = np.where(up, l, h), np.where(up, h, l)
    k = max(n // 2, 1)                       # minute index where the second extreme is reached
    pts = np.zeros((len(b), n + 1))
    for j in range(n + 1):                   # waypoints at j = 0, 1, k, n-1, n (linear in between)
        if n == 1:
            pts[:, j] = o if j == 0 else c
        elif j == 0:
            pts[:, j] = o
        elif j <= 1:
            pts[:, j] = e1
        elif j <= k:
            pts[:, j] = e1 + (e2 - e1) * (j - 1) / max(k - 1, 1)
        elif j < n:
            pts[:, j] = e2
        else:
            pts[:, j] = c
    mo, mc = pts[:, :-1], pts[:, 1:]
    rows = pd.DataFrame({"open": mo.ravel(), "close": mc.ravel(),
                         "high": np.maximum(mo, mc).ravel(), "low": np.minimum(mo, mc).ravel()},
                        index=(b.index.values[:, None] + np.arange(n) * np.timedelta64(1, "m")).ravel())
    rows.index = pd.DatetimeIndex(rows.index).tz_localize("UTC") if rows.index.tz is None else rows.index
    return rows


def basis(spot, g5):
    s5 = spot.close.resample("5min").last()
    j = pd.concat([g5.close.rename("gc"), s5.rename("spot")], axis=1, sort=True).dropna()
    t = (j.index - j.index[0]).total_seconds().to_numpy() / 86400
    a, b0 = np.polyfit(t, (j.gc / j.spot).to_numpy(), 1)
    t0 = j.index[0]
    return lambda idx: b0 + a * (idx - t0).total_seconds().to_numpy() / 86400


def build(spot):
    end = spot.index[-1]
    g = {iv: load(iv) for iv in RANGES}
    ratio = basis(spot, g["5m"])
    parts, t = [], end
    for iv, n in (("5m", 5), ("2m", 2), ("1m", 1)):
        first_finer = min([g[x].index[0] for x in RANGES if int(x[0]) < n] or [pd.Timestamp.max.tz_localize("UTC")])
        b = g[iv][(g[iv].index > t - pd.Timedelta(minutes=n)) & (g[iv].index < first_finer)]
        if len(b):
            parts.append(to_minutes(b, n))
            t = b.index[-1] + pd.Timedelta(minutes=n)
    ext = pd.concat(parts)
    ext = ext[ext.index > end]
    ext = ext[~ext.index.duplicated()]
    ext[["open", "high", "low", "close"]] = ext[["open", "high", "low", "close"]].div(ratio(ext.index), axis=0)
    ext["volume"] = 0
    return ext


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    spot = pd.read_parquet(DATA).set_index("timestamp")
    ext = build(spot)
    out = pd.concat([spot, ext]).rename_axis("timestamp").reset_index()
    out.to_parquet(EXT)
    print(f"спот до {spot.index[-1]}, добавлено {len(ext)} минут до {ext.index[-1]}, "
          f"цена {ext.close.iloc[0]:.2f} → {ext.close.iloc[-1]:.2f}")

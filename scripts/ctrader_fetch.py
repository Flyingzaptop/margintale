"""Download XAUUSD M1 bars from cTrader Open API (JSON over WebSocket, port 5036) and append them
to the spot file: data/XAUUSD_ext.parquet = data/XAUUSD_41.parquet + cTrader bars after its end.

Needs (environment variables):
  CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET  - an application at https://openapi.ctrader.com/apps
  CTRADER_ACCESS_TOKEN                      - OAuth token of your cTrader ID (the app's Playground)
  CTRADER_ACCOUNT_ID (optional)             - ctidTraderAccountId; the first account is used if unset
Run on a machine that can reach demo.ctraderapi.com:5036 (or live.ctraderapi.com with --live):

    pip install websockets pandas pyarrow
    python3 scripts/ctrader_fetch.py [--symbol XAUUSD] [--from 2026-09-04] [--live] [--no-merge]

Bars are BID trendbars; prices come as integers in 1/100000.
"""
import argparse
import json
import os
import time
from itertools import count
from pathlib import Path

import pandas as pd
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
SPOT = ROOT / "data" / "XAUUSD_41.parquet"
OUT = ROOT / "data" / "XAUUSD_ctrader.parquet"
EXT = ROOT / "data" / "XAUUSD_ext.parquet"

APP_AUTH, ACC_AUTH, SYMBOLS, TRENDBARS, ACCOUNTS = 2100, 2102, 2114, 2137, 2149
ERROR, HEARTBEAT = 2142, 51
M1 = 1
CHUNK_MS = 3 * 86400 * 1000      # M1 requests are limited to a few days each
_ids = count(1)


class Api:
    def __init__(self, host):
        self.ws = connect(f"wss://{host}:5036", max_size=None, open_timeout=20)

    def call(self, ptype, payload):
        mid = str(next(_ids))
        self.ws.send(json.dumps({"clientMsgId": mid, "payloadType": ptype, "payload": payload}))
        while True:
            msg = json.loads(self.ws.recv(timeout=60))
            if msg.get("payloadType") == HEARTBEAT:
                self.ws.send(json.dumps({"payloadType": HEARTBEAT}))
                continue
            if msg.get("clientMsgId") != mid:
                continue
            if msg.get("payloadType") == ERROR or "errorCode" in msg.get("payload", {}):
                raise RuntimeError(f"cTrader error on {ptype}: {msg.get('payload')}")
            return msg["payload"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--from", dest="start", default=None, help="UTC date; default: end of the spot file")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    env = os.environ
    api = Api("live.ctraderapi.com" if a.live else "demo.ctraderapi.com")
    api.call(APP_AUTH, {"clientId": env["CTRADER_CLIENT_ID"], "clientSecret": env["CTRADER_CLIENT_SECRET"]})
    acc = env.get("CTRADER_ACCOUNT_ID")
    if not acc:
        accs = api.call(ACCOUNTS, {"accessToken": env["CTRADER_ACCESS_TOKEN"]})["ctidTraderAccount"]
        accs = [x for x in accs if bool(x.get("isLive")) == a.live] or accs
        acc = accs[0]["ctidTraderAccountId"]
        print("счёт", acc, "из", [x["ctidTraderAccountId"] for x in accs])
    acc = int(acc)
    api.call(ACC_AUTH, {"ctidTraderAccountId": acc, "accessToken": env["CTRADER_ACCESS_TOKEN"]})
    syms = api.call(SYMBOLS, {"ctidTraderAccountId": acc})["symbol"]
    sym = next(s for s in syms if s.get("symbolName", "").upper() == a.symbol.upper())
    print("символ", sym["symbolName"], "id", sym["symbolId"])

    spot = pd.read_parquet(SPOT)
    t0 = pd.Timestamp(a.start, tz="UTC") if a.start else spot.timestamp.iloc[-1] - pd.Timedelta(days=2)
    t, t_end = int(t0.timestamp() * 1000), int(time.time() * 1000)
    rows = []
    while t < t_end:
        res = api.call(TRENDBARS, {"ctidTraderAccountId": acc, "symbolId": sym["symbolId"], "period": M1,
                                   "fromTimestamp": t, "toTimestamp": min(t + CHUNK_MS, t_end)})
        for b in res.get("trendbar", []):
            low = b.get("low", 0)
            rows.append((b["utcTimestampInMinutes"], low + b.get("deltaOpen", 0), low + b.get("deltaHigh", 0),
                         low, low + b.get("deltaClose", 0), b.get("volume", 0)))
        print(pd.Timestamp(t, unit="ms", tz="UTC").date(), "баров", len(res.get("trendbar", [])))
        t += CHUNK_MS
        time.sleep(0.25)        # historical requests are rate limited
    df = pd.DataFrame(rows, columns=["m", "open", "high", "low", "close", "volume"]).drop_duplicates("m")
    df["timestamp"] = pd.to_datetime(df.m * 60, unit="s", utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] / 1e5
    df = df.sort_values("timestamp")[["timestamp", "open", "high", "low", "close", "volume"]]
    df.to_parquet(OUT)
    print(f"cTrader: {len(df)} минут, {df.timestamp.iloc[0]} … {df.timestamp.iloc[-1]} → {OUT}")
    if not a.no_merge:
        last = spot.timestamp.iloc[-1]
        both = df[df.timestamp > last]
        ov = spot.merge(df, on="timestamp", suffixes=("", "_c"))
        if len(ov):
            print(f"на пересечении {len(ov)} минут: средняя разница close cTrader − файл "
                  f"{(ov.close_c - ov.close).mean():+.2f}")
        ext = pd.concat([spot, both.astype(spot.dtypes.to_dict())], ignore_index=True)
        ext.to_parquet(EXT)
        print(f"склеено: {EXT} до {ext.timestamp.iloc[-1]}")


if __name__ == "__main__":
    main()

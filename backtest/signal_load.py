"""Load the MQL5 signal 2304847 trade history (data/signal/positions.csv)."""
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parent.parent / "data" / "signal" / "positions.csv"


def load(path=PATH):
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", header=0,
                      names=["t_open", "type", "vol", "symbol", "p_open", "vol2", "t_close", "p_close",
                             "commission", "swap", "profit"])
    raw["t_open"] = pd.to_datetime(raw.t_open, format="%Y.%m.%d %H:%M:%S")
    raw["t_close"] = pd.to_datetime(raw.t_close, format="%Y.%m.%d %H:%M:%S", errors="coerce")
    for c in ("vol", "p_open", "p_close", "commission", "swap", "profit"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0) if c in ("commission", "swap") \
            else pd.to_numeric(raw[c], errors="coerce")
    return raw.sort_values("t_open").reset_index(drop=True)

"""Grid + martingale on XAUUSD M5, as in README section 3.

Long-only basket: open 0.01 lot, add a lot x MULT (rounded to 0.01) every STEP dollars against,
close the whole basket when bid reaches average entry + TP. Each run starts on
Jan 1 with DEPOSIT and stops at the end of the year or when equity hits zero.

Intrabar order is pessimistic: the bar's low (adds, equity check) is processed
before its high (take profit).
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "XAUUSD_41.parquet"

STEP = 6.0       # $ between grid orders
MULT = 1.6       # lot multiplier per add
TP = 5.0         # $ above average entry to close basket
BASE_OZ = 1.0    # 0.01 lot = 1 oz
SPREAD = 0.30    # $ paid on every entry (bid data, buy at bid + spread)
DEPOSIT = 1000.0


@dataclass
class Result:
    year: int
    end_balance: float
    cycles: int
    max_orders: int
    died: pd.Timestamp | None
    equity: pd.Series = field(repr=False)


def load_m5(path=DATA):
    m1 = pd.read_parquet(path).set_index("timestamp")
    return m1.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()


def run_year(bars, year):
    b = bars.loc[str(year)]
    balance = DEPOSIT
    lots, prices = [], []  # open orders: oz and entry price
    cycles, max_orders, died = 0, 0, None
    eq_t, eq_v = [], []

    for t, o, h, l, c in zip(b.index, b.open, b.high, b.low, b.close):
        if not lots:
            lots, prices = [BASE_OZ], [o + SPREAD]
        # adds on the way down
        while l + SPREAD <= prices[-1] - STEP:
            lots.append(max(round(lots[-1] * MULT), 1.0))  # whole 0.01 lots
            prices.append(prices[-1] - STEP)
        max_orders = max(max_orders, len(lots))
        oz = sum(lots)
        avg = sum(q * p for q, p in zip(lots, prices)) / oz
        if balance + oz * (l - avg) <= 0:
            balance, died = 0.0, t
            eq_t.append(t), eq_v.append(0.0)
            break
        if h >= avg + TP:
            balance += oz * TP
            cycles += 1
            lots, prices = [], []
            eq_t.append(t), eq_v.append(balance)
        else:
            eq_t.append(t), eq_v.append(balance + oz * (c - avg))
    else:
        if lots:  # close open basket at the last close of the year
            oz = sum(lots)
            avg = sum(q * p for q, p in zip(lots, prices)) / oz
            balance += oz * (c - avg)
            eq_v[-1] = balance

    return Result(year, balance, cycles, max_orders, died, pd.Series(eq_v, eq_t))


def main(years):
    bars = load_m5()
    return [run_year(bars, y) for y in years]


if __name__ == "__main__":
    for r in main([int(a) for a in sys.argv[1:]] or [2024]):
        print(r)

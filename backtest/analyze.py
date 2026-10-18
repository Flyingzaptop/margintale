"""Summarise results/search.parquet: what survives, what earns, out of sample."""
import numpy as np
import pandas as pd

df = pd.read_parquet("results/search.parquet")
P = ["dirs", "step", "step_atr", "step_grow", "mult", "max_orders", "tp", "tp_atr",
     "sl", "leverage", "risk", "ema"]
Y = [f"y{y}" for y in range(2017, 2027)]
D = [f"d{y}" for y in range(2017, 2027)]
TRAIN, TEST = Y[:6], Y[6:]  # 2017-2022 / 2023-2026

df["deaths"] = df[D].sum(axis=1)
df["win"] = (df[Y] > 1000).sum(axis=1)
df["mean"] = df[Y].mean(axis=1)
df["med"] = df[Y].median(axis=1)
df["worst"] = df[Y].min(axis=1)
df["pnl"] = (df[Y] - 1000).sum(axis=1)  # cumulative, $1000 per year
df["train"] = (df[TRAIN] - 1000).sum(axis=1)
df["test"] = (df[TEST] - 1000).sum(axis=1)
df["train_win"] = (df[TRAIN] > 1000).sum(axis=1)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(f"{len(df)} configs; share with cumulative P&L > 0: {(df.pnl > 0).mean():.1%}; "
      f"no deaths in 10 years: {(df.deaths == 0).mean():.1%}; profitable all 10 years: {(df.win == 10).sum()}")

for col in ["dirs", "mult", "max_orders", "sl", "leverage", "step_atr", "tp_atr", "step_grow", "risk"]:
    g = df.groupby(col).agg(median_pnl=("pnl", "median"), share_pos=("pnl", lambda x: (x > 0).mean()),
                            deaths=("deaths", "mean"))
    print(f"\n{col}\n{g.round(2)}")

S = ["deaths", "win", "worst", "med", "pnl", "train", "test"]
print("\nTop 15 by cumulative P&L")
print(df.sort_values("pnl", ascending=False)[P + S].head(15).round(1).to_string())
print("\nTop 15 by worst year")
print(df.sort_values("worst", ascending=False)[P + S].head(15).round(1).to_string())

top = df[df.train_win == 6].sort_values("train", ascending=False).head(50)
print(f"\nOut of sample: {len(df[df.train_win == 6])} configs won every year 2017-2022; "
      f"top 50 of them by train P&L in 2023-2026: median test {top.test.median():.0f}, "
      f"positive {(top.test > 0).mean():.0%}")
print(top[P + S].head(15).round(1).to_string())

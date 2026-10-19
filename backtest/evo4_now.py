"""Where the evo4 winners stand now, on data/XAUUSD_ext.parquet (spot + Yahoo extension):
every daily start since 2026-06-01 run to the last bar (no target): equity now, best, whether
x2 was reached and when, dead or not; and the shadow rule for launching today
(the same entity started 60 days ago: is it in profit?).

    python3 extend_yahoo.py --fetch && python3 evo4_now.py
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from engine import load_m1
from engine3 import Genes, Market3
from extend_yahoo import EXT
from evolve import OUT

if __name__ == "__main__":
    m = Market3(load_m1(EXT))
    last = m.index[-1]
    starts = pd.date_range(pd.Timestamp("2026-06-01", tz="UTC"), last.floor("D") - pd.Timedelta(days=1), freq="D")
    fields = list(Genes.__dataclass_fields__)
    print(f"данные до {last}")
    for ent, cut in (("A", "202410"), ("B", "202310")):
        r = pd.read_csv(OUT / f"evo4_{cut}_s1.csv").sort_values(["t_p", "score"], ascending=False).iloc[0]
        g = Genes(**{k: (int(r[k]) if isinstance(getattr(Genes, k), int) else float(r[k])) for k in fields})
        days = (last - starts[0]).days
        eq, st, _ = m.batch([replace(g, target=1e9)], starts, days)
        x = eq[0] / g.deposit
        rows = []
        for i, t in enumerate(starts):
            n = (last - t).days            # days this start has lived
            path = x[i, :max(n, 1)]
            hit = np.where(path >= 1.998)[0]
            rows.append(dict(старт=t.date(), дней=n, сейчас=path[-1], максимум=path.max(), минимум=path.min(),
                             x2_на_дне=hit[0] + 1 if len(hit) else None, умерла=bool(st[0][i] >= 2)))
        d = pd.DataFrame(rows)
        d["месяц"] = pd.to_datetime(d.старт).dt.to_period("M")
        print(f"\n=== особь {ent} (риск {g.risk:.0%}, {'лонг' if g.dirs == 1 else 'лонг+шорт'})")
        print(d.groupby("месяц").agg(стартов=("старт", "size"), x2_уже=("x2_на_дне", lambda s: s.notna().mean()),
                                     умерли=("умерла", "mean"), счёт_сейчас_мед=("сейчас", "median"),
                                     дно_мед=("минимум", "median")).round(2).to_string())
        sh = d[d.старт == (last - pd.Timedelta(days=60)).date()]
        print("тень (запуск 60 дней назад): " + ("нет данных" if sh.empty else
              f"старт {sh.старт.iloc[0]}, счёт сейчас ×{sh.сейчас.iloc[0]:.2f} → "
              f"{'ЗАПУСКАТЬ' if sh.сейчас.iloc[0] >= 1 else 'НЕ запускать'}"))
        print("тень по дням за последние 3 недели (счёт запуска 60 дней назад на этот день):")
        sh_hist = []
        for k in range(21, -1, -1):
            day = last.floor("D") - pd.Timedelta(days=k)
            s0 = day - pd.Timedelta(days=60)
            i = np.where(starts == s0)[0]
            if len(i):
                sh_hist.append(f"{day:%m-%d}:×{x[i[0], 59]:.2f}")
        print("  " + " ".join(sh_hist))
        d.drop(columns="месяц").to_csv(OUT / f"evo4_now_{ent}.csv", index=False)

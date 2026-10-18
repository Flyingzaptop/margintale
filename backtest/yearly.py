"""Run the martingale from Jan 1 of every year; print a table, save a chart."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import martingale as m

YEARS = range(2017, 2027)  # 2016 starts in September, 2026 ends in September

bars = m.load_m5()
results = [m.run_year(bars, y) for y in YEARS]

total = 0.0
print("| Год | Итог из $1 000 | P&L | Кумулятивно | Пик средств | Циклов | Макс. ордеров | Слив |")
print("|---|---|---|---|---|---|---|---|")
for r in results:
    pnl = r.end_balance - m.DEPOSIT
    total += pnl
    died = r.died.strftime("%d.%m %H:%M") if r.died is not None else "—"
    print(f"| {r.year} | ${r.end_balance:,.0f} | {pnl:+,.0f} | {total:+,.0f} "
          f"| ${r.equity.max():,.0f} | {r.cycles} | {r.max_orders} | {died} |".replace(",", " "))

fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.2 * len(results)), sharey=False)
for ax, r in zip(axes, results):
    eq = r.equity.resample("1h").last().dropna()
    ax.plot(eq.index, eq.values, lw=1, color="tab:red" if r.died is not None else "tab:green")
    ax.axhline(m.DEPOSIT, color="gray", lw=0.6, ls="--")
    ax.set_title(f"{r.year}: ${r.end_balance:,.0f}"
                 + (f", слив {r.died:%d.%m}" if r.died is not None else ""), fontsize=9, loc="left")
    ax.set_xlim(eq.index[0], eq.index[0].replace(month=12, day=31))
fig.suptitle(f"XAUUSD M5, шаг ${m.STEP:g}, ×{m.MULT}, TP +${m.TP:g}, депозит ${m.DEPOSIT:,.0f}")
fig.tight_layout()
fig.savefig("img/martingale_yearly.png", dpi=110)

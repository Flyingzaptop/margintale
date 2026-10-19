"""Heat map image: P(reached xk BEFORE death) by trading day after launch, for the evo4 winners.
Rows x1.5..x20 (blue) and 'death' (red); cell = share of launches that by trading day N already
reached xk without dying first (only launches with N trading days in the data are counted).
Input results/evo4_launches.csv (evo4_heatmap.py), output docs/img/evo4_heat_<entity>.png.

    python3 evo4_heatpng.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from evolve import OUT

TD = [1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 250, 375, 500]
ROWS = [("x1_5", "×1.5"), ("x2", "×2"), ("x3", "×3"), ("x5", "×5"), ("x10", "×10"), ("x20", "×20"), ("death", "смерть")]
BLUE = LinearSegmentedColormap.from_list("b", ["#f4f7fb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
RED = LinearSegmentedColormap.from_list("r", ["#fbf4f3", "#f6cfc9", "#ec8f82", "#d6503f", "#a8281c"])
NAMES = {"A": "Особь A — Азия ORB, только лонг, риск 8%", "B": "Особь B — Азия ORB, лонг+шорт, риск 20%"}
GROUPS = [("с 2023 года", lambda d: d.year >= 2023), ("с теневым правилом (тень 60 дн в плюсе)", lambda d: d.shadow == 1),
          ("вся история 2017–2026", lambda d: d.year > 0)]


def to_td(start, cal):
    """calendar day count (1 = first day) -> trading days (Mon-Fri) elapsed; NaN stays NaN."""
    out = np.full(len(cal), np.nan)
    ok = ~np.isnan(cal)
    s = start.values.astype("datetime64[D]")
    out[ok] = np.busday_count(s[ok], s[ok] + cal[ok].astype(int))
    return out


def matrix(d):
    avail = to_td(d.start, d.avail.to_numpy(float))
    m = np.full((len(ROWS), len(TD)), np.nan)
    n = np.zeros(len(TD), int)
    for j, t in enumerate(TD):
        ok = avail >= t
        n[j] = ok.sum()
        if n[j] < 30:
            continue
        for i, (c, _) in enumerate(ROWS):
            v = to_td(d.start[ok], d[c][ok].to_numpy(float))
            m[i, j] = np.mean(np.nan_to_num(v, nan=1e9) <= t)
    return m, n


if __name__ == "__main__":
    df = pd.read_csv(OUT / "evo4_launches.csv")
    df["start"] = pd.to_datetime(df.start)
    df["year"] = df.start.dt.year
    for ent in ("A", "B"):
        fig, axes = plt.subplots(3, 1, figsize=(15, 13.5))
        fig.suptitle(f"{NAMES[ent]}\nВероятность дойти до ×k РАНЬШЕ смерти к N-му торговому дню после запуска",
                     fontsize=15, x=0.06, ha="left", y=0.985)
        for ax, (gname, f) in zip(axes, GROUPS):
            d = df[(df.entity == ent) & f(df)].reset_index(drop=True)
            m, n = matrix(d)
            for i in range(len(ROWS)):
                cmap = RED if ROWS[i][0] == "death" else BLUE
                row = np.ma.masked_invalid(m[i:i + 1])
                ax.imshow(row, cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(-0.5, len(TD) - 0.5, i + 0.5, i - 0.5))
                for j in range(len(TD)):
                    v = m[i, j]
                    if np.isnan(v):
                        ax.text(j, i, "нет\nданных", ha="center", va="center", fontsize=7, color="#9a9a9a")
                        continue
                    ax.text(j, i, f"{v:.0%}" if v >= 0.005 or v == 0 else "<1%", ha="center", va="center", fontsize=11,
                            color="white" if v > 0.55 else "#1d1d1d", fontweight="bold" if ROWS[i][0] == "x2" else None)
            ax.set_xlim(-0.5, len(TD) - 0.5)
            ax.set_ylim(len(ROWS) - 0.5, -0.5)
            ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS], fontsize=11)
            ax.set_xticks(range(len(TD)), [f"{t}" for t in TD], fontsize=10)
            ax.set_xlabel("торговых дней с запуска  (≈21 в месяце, ≈250 в году)", fontsize=10, color="#555")
            ax.set_title(f"Старты {gname}: {len(d):,} запусков".replace(",", " "), loc="left", fontsize=12)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks(np.arange(-0.5, len(TD)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(ROWS)), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.axhline(len(ROWS) - 1.5, color="#333", linewidth=1.2)
        fig.text(0.06, 0.008, "Синие строки: доля запусков, которые к этому дню уже достигли ×k и ни разу до этого не умерли "
                 "(смерть = −90% от пика). Красная строка: доля умерших к этому дню.\n"
                 "Бэктест XAUUSD M1, запуск каждый день в 00:00 UTC, без остановки на ×2. Считаются только запуски, "
                 "у которых в данных есть все N торговых дней (данные до 2026-09-04).", fontsize=9, color="#555")
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        out = OUT.parent / "docs" / "img" / f"evo4_heat_{ent}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        print(out)

import pandas as pd

DAYS = [-2, -1, 0]
LIMIT = 80 # hardcoded position limit per product

rows = []
grand = 0.0
for d in DAYS:
    df = pd.read_csv(f"prices_round_1_day_{d}.csv", sep=";")
    day_total = 0.0
    for prod, g in df.groupby("product"):
        g = g.sort_values("timestamp")
        mid = g["mid_price"].dropna().values
        if len(mid) < 2:
            continue
        diffs = pd.Series(mid).diff().dropna().abs().sum()
        pnl = LIMIT * diffs
        rows.append((d, prod, pnl))
        day_total += pnl
    print(f"day {d}: {day_total:.0f}")
    grand += day_total

print(f"grand total: {grand:.0f}")
print()
out = pd.DataFrame(rows, columns=["day", "product", "max_pnl"])
print(out.pivot(index="product", columns="day", values="max_pnl").fillna(0).round(0))

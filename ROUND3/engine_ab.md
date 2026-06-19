## Current strategy summary

**File:** `ROUND3/engine_ab.py`
**Products:** VELVETFRUIT_EXTRACT (underlying) + 10 VEV_K vouchers (4000–6500)

---

### Engine A — VELVETFRUIT_EXTRACT (underlying)

`VelvetTrader` (template_skew style):
- `fair_value = wall_mid` (current best bid / ask average)
- **Taker pass:** lift any ask ≤ FV-1, hit any bid ≥ FV+1
- **Maker pass:** quote ±6 around FV, skewed toward AR(2) MEAN (5250.95)
  - Skew formula: `(fair_value - MEAN) / VEV_STD × SKEW_MAX`, clipped ±6 ticks
  - When price > MEAN → quotes shift down → biased toward shorting
  - When price < MEAN → quotes shift up → biased toward longing
- Maker size scaled by σ-deviation from MEAN: bigger size on the favored side at extremes

Constants: `VEV_STD=15.63`, `SKEW_MAX=3`, `U_LIMIT=200`

---

### Engine B — voucher MM (per strike)

`VoucherTrader` for each of 10 strikes:

**1. Live smile fit** (per tick, in `Trader._update_smile`):
- For each strike, invert BS to extract IV from market mid: σ_K from `bs_call(spot, K, T, σ) = mid_K`
- Aggregate (log_mon, IV) into 8-float EWMA sums (γ=0.999, ~1000-tick memory)
- Solve 3×3 normal equations → polynomial smile coeffs `(a, b, c)` where `IV(x) = a·x² + b·x + c`, `x = ln(spot/K)`
- Warmup: 2000 obs (~200 ticks) before quoting
- State stored in `traderData["smile_sums"]` — only 8 floats persisted

**2. Per-voucher fair value:**
- `σ_K = a·log_mon_K² + b·log_mon_K + c`
- `FV_K = bs_call(spot, K, T_EXPIRY=5/365, σ_K)`

**3. Quote prices:**
- `inventory_lean = portfolio_delta / REGRET_THRESHOLD`
- `bid = FV - BASE_EDGE - lean·SPREAD`
- `ask = FV + BASE_EDGE - lean·SPREAD`
- (lean shifts both quotes against accumulated delta — biases toward flattening)

**4. Throttle:**
- Skip bid if portfolio_delta > +REGRET_THRESHOLD (too long, stop buying)
- Skip ask if portfolio_delta < -REGRET_THRESHOLD (too short, stop selling)

**5. Adaptive per-strike cap (just restored):**
- `strike_cap = max(50, 300·|delta_K|)`
- Skip bid if `position ≥ strike_cap` (don't deepen long past delta-scaled limit)
- Skip ask if `position ≤ -strike_cap` (don't deepen short past)
- Effect: deep ITM → full 300 cap (mean-reversion winners). Far OTM → 50 cap (avoid smile-overshoot pinning)

Constants: `BASE_EDGE=1`, `SPREAD=4`, `VOUCHER_QUOTE_SIZE=7`, `REGRET_THRESHOLD=1500`, `O_LIMIT=300`

---

### Engine C — cross-strike arb (just added)

`Trader._call_spread_arb`:
- Scan all 45 strike pairs each tick
- Check no-arb bound: `0 ≤ Call(K_low) - Call(K_high) ≤ K_high - K_low`
- Violation → fire both legs simultaneously:
  - **Direction A:** `bid_low - ask_high > strike_diff` → sell low at bid, buy high at ask
  - **Direction B:** `bid_high > ask_low` → buy low at ask, sell high at bid
- Locked profit at expiry. MTM volatile in interim.
- Updates per-strike `arb_used` budget so VoucherTrader's MM doesn't double-count capacity

---

### Tick flow (`Trader.run`)

```
1. spot = wall_mid(VELVETFRUIT_EXTRACT)
2. update smile EWMA sums, solve 3×3 → (a, b, c)
3. portfolio_delta = pos_underlying + Σ pos_K × BS_delta(K)
4. Engine A → quote underlying
5. Engine C → scan pairs, fire arbs (consumes per-strike capacity)
6. Engine B → for each voucher, post bid/ask using smile FV (capacity reduced by arb)
7. flush logger + persist smile_sums to traderData
```

---

### Persisted state (`traderData`)

- `smile_sums`: 8 floats (live smile EWMA)
- `prev_mid`, `prev_prev_mid`, `vev_wall_mid`: 3 floats (AR(2) state for future scalping; unused now)

Total state ~12 floats. Tiny.

---

### Tunables

- `VEV_STD`, `SKEW_MAX` (Engine A)
- `BASE_EDGE`, `SPREAD`, `VOUCHER_QUOTE_SIZE`, `REGRET_THRESHOLD`, `DELTA_CAP_MIN` (Engine B)
- `SMILE_GAMMA`, `SMILE_WARMUP_N` (smile fit speed/stability)
- Edge threshold in arb (currently `> 0`, may want `> 1` buffer)

---



# VEV Options Strategy Space

## Market Structure

- **Underlying:** VELVETFRUIT_EXTRACT (spot ~5250, pos limit 200), AR(2) mean-reverting around 5250.95
- **Vouchers:** VEV_K calls expiring at ts=8,000,000 (~5 days from round start), pos limit 100 each
- **Strike tiers:**
  - Deep ITM: 4000, 4500 — TV≈0, priced at intrinsic. Wide spread.
  - Active: 5000–5500 — meaningful IV, SSVI-tradeable (rho≈0, phi=1, only theta varies)
  - Deep OTM: 6000, 6500 — TV≈0, bid=0/ask=1. Worthless unless spot spikes.
- **Smile:** Flat in log-moneyness space (rho=0, phi=1). Only theta (ATM total variance) moves. Price deviations from smile are stationary and mean-reverting.

---

## Baseline (template.py)

Three-tier approach:
- Deep ITM: intrinsic ± 1 MM, DITM_TAKE_THRESH=2
- Active (excl. 5300): SSVI fair value, VEV_TAKE_THRESH=5, VEV_IV_LEAN=1.0, VEV_MAKER_SIZE=7, VEV_GAMMA_OPT=0.05
- Deep OTM: bid=0, ask=1
- **Result: 2,392** (82/1051/1259 per day)

---

## Strategy Catalog

### S1: v_take1 — Aggressive taker (take_thresh=1)
**Hypothesis:** The current take_thresh=5 is too conservative. Even a 1-tick misprice is exploitable. Earlier v1 used thresh=1.0 and earned 3,746. Lower threshold captures more fills.
**Changes:** `VEV_TAKE_THRESH = 1.0`, `VEV_IV_LEAN = 0.5` (v1 params)
**Risk:** More adverse selection on small misprices. May hurt if 1-tick fills are random noise.

---

### S2: v_no_take — Pure maker, no taking
**Hypothesis:** Taker leg adds adverse selection. The MM spread income is the real edge; crossing the spread to buy/sell mispriced options is noise-trading.
**Changes:** `VEV_TAKE_THRESH = 9999` (effectively disabled), `DITM_TAKE_THRESH = 9999`
**Risk:** Miss genuine arbitrages on deep ITM options.

---

### S3: v_include_5300 — Add VEV_5300 back to active strikes
**Hypothesis:** VEV_5300 was a bleeder under v1 params but the current SSVI+IV lean logic may handle it correctly since it's near-ATM with the best IV signal. It was excluded from template because it was a "consistent bleeder."
**Changes:** `ACTIVE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]` (add 5300 back)
**Risk:** ATM has highest gamma risk + adverse selection. May bleed again.

---

### S4: v_call_spread_arb — Cross-strike call spread no-arb
**Hypothesis:** Call spreads must satisfy `0 ≤ C(K_low) - C(K_high) ≤ K_high - K_low`. When violated, a locked profit exists. Engine C in engine_ab.md implements this. Cross-strike arb provides risk-free locked PnL at expiry.
**Changes:** Add `_call_spread_arb()` method: scan all 45 strike pairs each tick, fire when:
- bid(K_low) > ask(K_high) → buy high, sell low (C_low - C_high < 0, impossible since call price falls with K)
- bid(K_high) - ask(K_low) > K_high - K_low → sell low at bid, buy high at ask
**Risk:** MTM volatile between now and expiry. Backtester may not capture locked expiry profit correctly.

---

### S5: v_delta_hedge — Underlying delta hedge
**Hypothesis:** Net portfolio delta from option positions creates directional risk. Trading VEV underlying to flatten delta removes this risk and may improve Sharpe (reduce drawdown) even if mean PnL stays similar. AR(2) mean-reversion on underlying also provides spread income.
**Changes:** Add VelvetTrader: when `base_delta > DELTA_THRESH` (e.g., 20), sell underlying to flatten. When `base_delta < -DELTA_THRESH`, buy underlying. Use mean-reversion MM on underlying around AR(2) mean=5250.
**Risk:** Delta hedging costs spread. If option positions are short-lived, delta hedge creates P&L drag. Underlying position limit 200.

---

### S6: v_larger_size — Bigger maker quotes (size=15)
**Hypothesis:** Current maker size=7 is conservative. If the spread edge is real (and v1 showed it is), posting larger size earns proportionally more. The fill rate per order is independent of size.
**Changes:** `VEV_MAKER_SIZE = 15`
**Risk:** Larger quotes attract more adverse flow per fill. Inventory builds faster.

---

### S7: v_otm_ask2 — Deep OTM ask=2 instead of 1
**Hypothesis:** Deep OTM calls (6000, 6500) are priced at bid=0/ask=1. We currently post ask=1 to collect 1 tick. If we raise ask=2, we earn 2 ticks per sell. Market participants wanting to hedge upside tail risk may still accept ask=2.
**Changes:** `_trade_deep_otm`: post `ask=2` instead of `ask=1`
**Risk:** Fewer fills. If the ask=1 is already at the edge of market demand, raising to 2 may kill all volume.

---

## Combinations to consider post-S1–S7

- S1+S6: Aggressive taking + large size (high-volume version of v1)
- S4+S5: Call spread arb + delta hedge (full-stack risk-managed)
- S2+S6: Pure maker + large size (pure spread income, max volume)

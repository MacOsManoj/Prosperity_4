# Strategy Spec: Relative Value Z-Score Options Market Maker

**Version:** 1.0  
**Target file:** `rv_zscore.py` (copy from `engine_ab.py` as base)  
**Purpose:** Replace engine_ab's symmetric smile-centered quoting with inventory targeting
driven by per-strike relative-value z-scores. Remain delta-neutral via underlying hedge.

---

## 1. Motivation

From `trevor_sabr.ipynb` (Finding 5), each active strike has a **persistent deviation** from
the global smile fit. These biases are large and slow-reverting — they are not noise:

| Strike | mean_dev (ticks) | std_dev | AR(1) φ | half-life | mean_delta |
|--------|-----------------|---------|---------|-----------|------------|
| VEV_5000 | −0.06 | 0.56 | 0.22 | 0.5 ticks | 0.935 |
| VEV_5100 | −0.08 | 0.92 | 0.79 | 2.9 ticks | 0.821 |
| VEV_5200 | +0.72 | 0.93 | 0.84 | 4.0 ticks | 0.625 |
| VEV_5300 | +1.31 | 1.16 | 0.93 | 10.0 ticks | 0.390 |
| VEV_5400 | −2.19 | 0.78 | 0.94 | 11.0 ticks | 0.195 |
| VEV_5500 | +0.53 | 0.44 | 0.80 | 3.1 ticks | 0.079 |

`mean_dev` = time-average of `(market_mid − BS_FV_smile)` per strike.

**Engine_ab fails** on VEV_5200/5300/5400 because it treats the smile FV as ground truth.
But VEV_5200 consistently trades 0.72 ticks above the smile: selling it because it's
"expensive vs smile" means selling into its own persistent premium — pure adverse selection.
VEV_5400 consistently trades 2.19 ticks BELOW the smile: buying it because it's "cheap vs
smile" means buying into a persistent discount — also adverse selection.

**The fix**: each strike has its own "typical level" = `BS_FV + mean_dev`. We z-score
deviations around THAT level. Only when the current price departs from its own historical
pattern do we build a position. Delta neutrality across strikes prevents directional bias.

---

## 2. Signal: Per-Strike Relative Value Z-Score

### 2.1 Smile fair value

Use engine_ab's live EWMA quadratic smile fit (unchanged — 8-float `smile_sums` state).
For active strike K with smile coefficients `(a, b, c)`:
```
log_mon_K = log(spot / K)
sigma_K   = a * log_mon_K² + b * log_mon_K + c      (vol/√year in T_EXPIRY units)
BS_FV_K   = bs_call(spot, K, T_EXPIRY, sigma_K)
```

### 2.2 Per-strike deviation tracking (EWMA)

For each active strike K, maintain EWMA estimates of `mean_dev_K` and `var_dev_K` in
`traderData`, bootstrapped from historical priors on tick 0:

```
PRIOR_MEAN = {5000: -0.06, 5100: -0.08, 5200: 0.72, 5300: 1.31, 5400: -2.19, 5500: 0.53}
PRIOR_STD  = {5000:  0.56, 5100:  0.92, 5200: 0.93, 5300:  1.16, 5400:  0.78, 5500: 0.44}
```

**Update rule per tick** (only when market mid is observable for strike K):
```
mid_K      = (best_bid_K + best_ask_K) / 2.0
dev_K      = mid_K - BS_FV_K                       # raw deviation from smile
mean_dev_K = ALPHA * mean_dev_K + (1 - ALPHA) * dev_K
var_dev_K  = ALPHA * var_dev_K  + (1 - ALPHA) * (dev_K - mean_dev_K)**2
std_dev_K  = sqrt(max(var_dev_K, MIN_STD**2))
```

Use `ALPHA = 0.9990` (EWM gamma → memory ~1000 ticks ≈ 1 day). `MIN_STD = 0.30` ticks
to prevent division by zero when variance is low.

**Adjusted fair value** (the typical market price for this strike):
```
adj_FV_K = BS_FV_K + mean_dev_K
```

### 2.3 Z-score

```
relative_dev_K = mid_K - adj_FV_K           # deviation from THIS strike's typical level
z_K            = relative_dev_K / std_dev_K
```

- `z_K < 0`: option cheaper than usual → want to be LONG
- `z_K > 0`: option more expensive than usual → want to be SHORT

---

## 3. Position Targeting

### 3.1 Target inventory from z-score

The target fraction of max inventory follows the normal CDF sigmoid:
```
f(|z|) = 2 * Φ(|z|) - 1    where Φ is the standard normal CDF
```

This gives: `f(1) ≈ 0.68`, `f(2) ≈ 0.95`, `f(3) ≈ 1.0` — matches the user's intent
(64%, 90%, 100% at z = −1, −2, −3, with the approximation being round numbers).

```
target_pos_K = POS_LIMIT_K * (-sign(z_K)) * f(|z_K|)
```

Where `POS_LIMIT_K = 100` for all active strikes.

**Important:** This target is a desired long-run equilibrium, not a hard constraint. We
approach it via passive maker orders, not aggressively via taker orders.

### 3.2 Delta scaling of target

Before using `target_pos_K`, scale it so that the portfolio's NET TARGET DELTA is bounded.
This prevents the sum of all position targets from creating a large directional exposure:

```
raw_target_delta = Σ_K (target_pos_K × delta_K)
if |raw_target_delta| > DELTA_TARGET_CAP:
    scale = DELTA_TARGET_CAP / |raw_target_delta|
    for each K: target_pos_K *= scale
```

Use `DELTA_TARGET_CAP = 100` (equivalent to ~1 ATM option at full position).

This does NOT mean we are delta neutral before trading. It means the intended positions
don't require a delta hedge larger than 100 underlying units.

---

## 4. Quote Generation (Active Strikes: 5000–5500)

For each active strike K per tick:

### 4.1 Inventory lean

```
inventory_gap_K = target_pos_K - current_pos_K     # positive = want to buy
gap_fraction_K  = clip(inventory_gap_K / POS_LIMIT_K, -1, +1)
```

### 4.2 Asymmetric spread

```
BASE_HALF_SPREAD = 1     # minimum edge on each side (ticks)
MAX_LEAN         = 2     # maximum additional lean per direction (ticks)

lean_toward_target = gap_fraction_K * MAX_LEAN    # positive = buy lean

bid_edge = BASE_HALF_SPREAD - lean_toward_target    # tighter bid when buying (lean > 0)
ask_edge = BASE_HALF_SPREAD + lean_toward_target    # wider ask when buying

bid_edge = max(0, bid_edge)     # never negative (would create taker order)
ask_edge = max(1, ask_edge)     # always at least 1 tick above FV
```

**Example:** z = -2, target = +90, current = 0 → gap = +90 → gap_frac = 0.90:
- lean_toward_target = 0.90 * 2 = 1.80
- bid_edge = 1 - 1.80 = -0.80 → clamped to 0 → **bid at adj_FV** (crosses immediately!)
- ask_edge = 1 + 1.80 = 2.80 → **ask at adj_FV + 3** (wide, unlikely to fill)

This means: at extreme z, the bid is posted AT fair value (aggressive limit order),
and the ask is posted wide. The bid will get filled if anyone sells at or below fair.
This is the "aggressive maker" behavior: still technically a passive order, but very
competitive.

**Taker override:** If `bid_edge <= 0` (z extreme enough that bid ≥ adj_FV), AND the
market ask is at or below `adj_FV`, immediately cross the market ask with a taker order.
Similarly on the sell side. This is the only scenario where we take; normally we make.

```
# Taker logic
if mkt_ask <= adj_FV_K - 0 and gap_fraction_K > 0.5:
    take_qty = min(mkt_ask_size, POS_LIMIT_K - current_pos_K, QUOTE_SIZE)
    fire BUY taker at mkt_ask, qty=take_qty

if mkt_bid >= adj_FV_K + 0 and gap_fraction_K < -0.5:
    take_qty = min(mkt_bid_size, POS_LIMIT_K + current_pos_K, QUOTE_SIZE)
    fire SELL taker at mkt_bid, qty=take_qty
```

### 4.3 Quote prices

```
bid_price = round(adj_FV_K - bid_edge)
ask_price = round(adj_FV_K + ask_edge)

# Safety: never bid above best_bid (would be taker), never ask below best_ask
if mkt_bid and bid_price >= mkt_ask: bid_price = mkt_ask - 1
if mkt_ask and ask_price <= mkt_bid: ask_price = mkt_bid + 1

# Never bid above adj_FV, never ask below adj_FV + 1 (maintain edge vs our own reference)
bid_price = min(bid_price, int(adj_FV_K))
ask_price = max(ask_price, int(adj_FV_K) + 1)
```

### 4.4 Quote sizes

```
QUOTE_SIZE = 7    # per-side size per tick (same as engine_ab)
```

Use `QUOTE_SIZE` for both bid and ask. The asymmetric competitiveness (not size) drives
fill preference toward the target direction.

### 4.5 Throttle: position cap

Do not post a bid if `current_pos_K >= POS_LIMIT_K`.
Do not post an ask if `current_pos_K <= -POS_LIMIT_K`.
(Engine_ab used strike-cap scaled by delta; this is simplified to hard limit.)

---

## 5. Delta Neutrality (VELVETFRUIT_EXTRACT)

### 5.1 Portfolio delta computation

```
net_delta = pos_underlying + Σ_K (pos_K × delta_K)
```

Delta for each active strike comes from BS:
```
delta_K = bs_call_delta(spot, K, T_EXPIRY, sigma_K)
```

For deep ITM (4000/4500): delta_K ≈ 1.0 (effectively underlying-like).
For deep OTM (6000/6500): delta_K ≈ 0.0 (skip from delta sum).

### 5.2 Underlying hedge

If `|net_delta| > DELTA_HEDGE_THRESHOLD`:

```
DELTA_HEDGE_THRESHOLD = 20    # tolerate up to 20 delta units before hedging

hedge_units = round(-net_delta)    # number of underlying units to buy (negative = sell)
```

Execute hedge using TAKER orders (cross the spread) because delta accuracy is more
important than spread income on the underlying:
- If `hedge_units > 0`: buy `min(hedge_units, mkt_ask_qty, UND_LIMIT - pos_underlying)`
  units at `mkt_ask`.
- If `hedge_units < 0`: sell `min(-hedge_units, mkt_bid_qty, UND_LIMIT + pos_underlying)`
  units at `mkt_bid`.

**Also** post underlying MAKER orders using engine_ab's Engine A logic (mean-reversion MM
around AR(2) mean = 5250.95) to earn spread income on the underlying independent of the
options portfolio. These are additive to the taker hedge orders but capped by position
limits.

---

## 6. Deep ITM / OTM Handling (Unchanged from engine_ab)

**Deep ITM (VEV_4000, VEV_4500):** Use template-style intrinsic MM.
```
fair = max(spot - K, 0)
bid  = int(fair - DITM_GAMMA * pos_K) - 1
ask  = int(fair - DITM_GAMMA * pos_K) + 1
```
`DITM_GAMMA = 0.05`. Post QUOTE_SIZE on each side. Taker if ask_K < fair - 2 or bid_K > fair + 2.

**Deep OTM (VEV_6000, VEV_6500):** bid=0, ask=1. No z-score signal (TV < 1 tick).
```
post bid=0, qty=QUOTE_SIZE   (free lottery — pick up if anyone sells at 0)
post ask=1, qty=QUOTE_SIZE   (earn 1 tick if anyone buys at 1)
```

---

## 7. State in traderData

All EWMA state persists across ticks via JSON in `traderData`:

```python
{
    "smile_sums": [8 floats],         # engine_ab smile EWMA state
    "prev_mid": float,                # for AR(2) underlying
    "prev_prev_mid": float,
    "vev_wall_mid": float,
    "dev_mean": {                     # per-strike mean_dev EWMA (dict key = str(K))
        "5000": float, "5100": float, "5200": float,
        "5300": float, "5400": float, "5500": float
    },
    "dev_var": {                      # per-strike var_dev EWMA
        "5000": float, ...
    }
}
```

**Bootstrap** on tick 0 (when `traderData` is empty or `dev_mean` absent):
```
dev_mean = PRIOR_MEAN   # from historical analysis above
dev_var  = {K: PRIOR_STD[K]**2 for K in PRIOR_STD}
```

This means the algo starts with calibrated priors from the notebook and refines them live.

---

## 8. Per-Tick Execution Flow

```
def run(state):
    # 0. Load state
    trader_data = json.loads(state.traderData) or {}
    dev_mean = trader_data.get("dev_mean", PRIOR_MEAN.copy())
    dev_var  = trader_data.get("dev_var",  {K: PRIOR_STD[K]**2 for K in PRIOR_STD})

    # 1. Spot
    spot = wall_mid(VELVETFRUIT_EXTRACT order book)

    # 2. Smile fit (engine_ab _update_smile, unchanged)
    smile_sums, smile_coeffs = _update_smile(state, smile_sums, spot)

    # 3. For each active strike K in [5000, 5100, 5200, 5300, 5400, 5500]:
    for K in ACTIVE_STRIKES:
        if smile_coeffs is None: skip (pre-warmup)
        log_mon = log(spot / K)
        sigma   = a*log_mon² + b*log_mon + c
        BS_FV   = bs_call(spot, K, T_EXPIRY, sigma)
        delta_K = bs_call_delta(spot, K, T_EXPIRY, sigma)

        mid_K = (best_bid_K + best_ask_K) / 2  [skip if book empty]
        dev_K = mid_K - BS_FV

        # Update EWMA
        dev_mean[K] = ALPHA * dev_mean[K] + (1 - ALPHA) * dev_K
        dev_var[K]  = ALPHA * dev_var[K]  + (1 - ALPHA) * (dev_K - dev_mean[K])**2
        std_dev_K   = sqrt(max(dev_var[K], MIN_STD**2))

        adj_FV_K      = BS_FV + dev_mean[K]
        relative_dev  = mid_K - adj_FV_K
        z_K           = relative_dev / std_dev_K

        f_z       = 2 * norm_cdf(abs(z_K)) - 1
        target_K  = POS_LIMIT * (-sign(z_K)) * f_z

    # 4. Delta scale target positions
    raw_target_delta = Σ(target_K × delta_K)
    if |raw_target_delta| > DELTA_TARGET_CAP:
        scale = DELTA_TARGET_CAP / |raw_target_delta|
        target_K *= scale for all K

    # 5. Compute current net delta
    net_delta = pos_underlying + Σ(pos_K × delta_K)

    # 6. Deep ITM: intrinsic MM (unchanged from template)
    for K in [4000, 4500]: trade_deep_itm(K)

    # 7. Active strikes: z-score maker + optional taker
    for K in [5000, 5100, 5200, 5300, 5400, 5500]:
        gap_frac    = clip((target_K - pos_K) / POS_LIMIT, -1, +1)
        lean        = gap_frac * MAX_LEAN
        bid_edge    = max(0, BASE_HALF_SPREAD - lean)
        ask_edge    = max(1, BASE_HALF_SPREAD + lean)
        bid_price   = round(adj_FV - bid_edge)
        ask_price   = round(adj_FV + ask_edge)
        [apply safety clamps]
        post_maker_bid(bid_price, QUOTE_SIZE)
        post_maker_ask(ask_price, QUOTE_SIZE)

        # Taker override at extreme z
        if gap_frac > 0.5 and mkt_ask <= adj_FV:
            fire_taker_buy(mkt_ask, QUOTE_SIZE)
        if gap_frac < -0.5 and mkt_bid >= adj_FV:
            fire_taker_sell(mkt_bid, QUOTE_SIZE)

    # 8. Deep OTM: bid=0 / ask=1
    for K in [6000, 6500]: post_bid_0_ask_1(K)

    # 9. Delta hedge underlying
    if |net_delta| > DELTA_HEDGE_THRESHOLD:
        hedge underlying with taker order
    # + Engine A maker orders on underlying (mean-reversion MM, unchanged from engine_ab)

    # 10. Persist state
    trader_data["dev_mean"] = dev_mean
    trader_data["dev_var"]  = dev_var
    trader_data["smile_sums"] = smile_sums
    return result, 0, json.dumps(trader_data)
```

---

## 9. Parameters

```python
# Active strikes for z-score strategy
ACTIVE_STRIKES    = [5000, 5100, 5200, 5300, 5400, 5500]
DEEP_ITM_STRIKES  = [4000, 4500]
DEEP_OTM_STRIKES  = [6000, 6500]

# Position limits
POS_LIMIT         = 100     # per option strike
UND_LIMIT         = 200     # underlying

# Z-score quoting
BASE_HALF_SPREAD  = 1       # minimum ticks of edge on each side
MAX_LEAN          = 2       # max additional ticks shifted toward target
QUOTE_SIZE        = 7       # lots per quote (same as engine_ab)

# EWMA for per-strike deviation tracking
ALPHA             = 0.9990  # decay per tick (~1000-tick memory)
MIN_STD           = 0.30    # floor on std_dev_K to prevent z runaway

# Position targeting sigmoid
# target = POS_LIMIT * (-sign(z)) * (2*Φ(|z|) - 1)

# Delta management
DELTA_TARGET_CAP      = 100    # max net target delta across all strikes
DELTA_HEDGE_THRESHOLD = 20     # start hedging when |net_delta| exceeds this

# Deep ITM
DITM_GAMMA        = 0.05    # inventory skew per unit (ticks)
DITM_TAKE_THRESH  = 2.0

# Smile fit (engine_ab constants, unchanged)
T_EXPIRY          = 5.0 / 365.0
SMILE_GAMMA       = 0.999
SMILE_WARMUP_N    = 500     # start quoting sooner (was 2000; ea_v2 showed this helps)

# Historical priors (from trevor_sabr.ipynb analysis)
PRIOR_MEAN = {5000: -0.06, 5100: -0.08, 5200: 0.72, 5300: 1.31, 5400: -2.19, 5500: 0.53}
PRIOR_STD  = {5000:  0.56, 5100:  0.92, 5200: 0.93, 5300:  1.16, 5400:  0.78, 5500: 0.44}

# Engine A (underlying, unchanged from engine_ab)
AR2_ALPHA = 10.727097  AR2_BETA1 = 0.840362  AR2_BETA2 = 0.157596
AR2_MEAN  = 5250.95    VEV_STD   = 15.63     SKEW_MAX  = 3
```

---

## 10. Key Design Decisions and Rationale

### Why adjusted_FV not raw smile FV?

VEV_5400 has mean_dev = −2.19. The raw smile FV says "this option should trade at X." The
market consistently says "no, it trades at X − 2.19." If we quote at X ± 1, we'll have
a bid at X − 1 (above market's typical X − 2.19) and get filled by sellers every tick —
building an unwanted long that then reverts down. The adjusted FV X − 2.19 is the
market-consistent reference; quoting around it means we MM at the market's equilibrium.

### Why z-score the relative deviation, not the raw price?

VEV_5300 has mean_dev = +1.31, std = 1.16. At any moment, if VEV_5300 is trading 2.5
ticks above smile (raw price above model), the z-score relative to its own pattern is
(2.5 − 1.31) / 1.16 ≈ +1.0 → modestly expensive → we want to be 68% short.
Without the mean adjustment, we'd think it's very expensive (raw z ≈ 2.5/1.16 ≈ 2.2)
and try to be 95% short — overreacting to a structural premium that won't revert.

### Why target via makers, not aggressive takers?

- Maker orders earn the spread (edge) rather than paying it.
- Price deviations are stationary with half-lives of 3–11 ticks. We have time.
- Exception: at z ≈ ±2+ with taker condition (mkt_ask ≤ adj_FV), the option is so
  dislocated that crossing the spread has positive expected value even after paying spread.

### Why delta hedge via underlying rather than pairs trading?

Delta-neutral via underlying is simpler and more liquid than option pairs. The AR(2)
mean-reversion engine on the underlying also earns spread income while hedging. Two birds.

### Why SMILE_WARMUP_N = 500 not 2000?

The ea_v2 experiment showed the smile fits the market quickly (all 6 active strikes
report IV every tick → 6 obs/tick × 500 ticks = 3000 obs before warmup with GAMMA=0.999,
effective count saturates quickly). Starting earlier means capturing more early-day PnL.

---

## 11. Expected Behavior

**VEV_5300 example (typical: +1.31 above smile, std=1.16):**
- If market trades at +2.5 above smile: z = (+2.5 − 1.31)/1.16 = +1.03 → target = −68
  → we're short 68 units → our ask is at adj_FV + (1 − 0.68×2) = adj_FV − 0.36 → very competitive ask, wide bid
- If market trades at +0.5 above smile: z = (0.5 − 1.31)/1.16 = −0.70 → target = +51
  → we're long 51 units → our bid is at adj_FV − (1 − 0.70×2) = adj_FV + 0.40 → aggressive bid
  → buying 5300 when it's cheap relative to its OWN typical premium

**VEV_5400 example (typical: −2.19 below smile, std=0.78):**
- If market trades at −3.0 below smile: z = (−3.0 − (−2.19))/0.78 = −1.04 → target = +69 → build long
- If market trades at −1.5 below smile: z = (−1.5 − (−2.19))/0.78 = +0.88 → target = −59 → build short
  → selling 5400 when it's less cheap than usual (expensive in relative terms)

**Net effect on 5300+5400 pair (when 5300 expensive AND 5400 cheap):**
- Short 5300 (delta 0.39) + Long 5400 (delta 0.195) → net delta = −0.39×pos + 0.195×pos
- With pos=70: net delta from pair = −27.3 + 13.65 = −13.65
- Hedge: buy 14 units of underlying → delta neutral
- This is effectively a bear call spread position, earning when the 5300/5400 spread normalizes

---

## 12. Implementation Notes for AI Agent

1. **Base on `engine_ab.py`**: Keep the Logger, BS helpers, `solve_smile`, `ProductTrader`
   base class, `VelvetTrader` (Engine A), `_call_spread_arb`, `_update_smile`,
   `_portfolio_delta` functions unchanged. Replace `VoucherTrader.get_orders()`.

2. **Add `norm_cdf`** (already in engine_ab as `norm_cdf`). Use it for the sigmoid:
   ```python
   def position_sigmoid(z_abs: float) -> float:
       return 2.0 * norm_cdf(z_abs) - 1.0
   ```

3. **Add deep ITM handler**: Port `_trade_deep_itm` from `template.py` (the current
   template already has this; engine_ab does not handle 4000/4500 separately).

4. **State dict keys** must be strings (JSON serialization): use `str(K)` for strike keys.

5. **Handle pre-warmup**: During `smile_sums[0] < SMILE_WARMUP_N`, use only the PRIOR
   adjusted FV (skip z-score sizing entirely; post symmetric quotes at prior adj_FV ± 1).
   Don't sit idle during warmup.

6. **Order deduplication**: After taker orders fire, reduce available capacity for maker
   orders on the same side by the quantity taken (same pattern as engine_ab arb + MM).

7. **Call spread arb (Engine C)**: Keep unchanged. It rarely fires but is free option.

8. **Logging**: Log per-tick: `z_K`, `target_K`, `adj_FV_K`, `net_delta` for diagnostics.

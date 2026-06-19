"""
Round 5 — Cherry Picking Winners.

Three-layer stack:

  Layer 1 — Stationary spreads (the main earner).
    1a. SNACKPACK_CHOCOLATE + SNACKPACK_VANILLA: ρ(returns) = -0.96.
        The sum of mids is mean-reverting around ~20,000. Trade the
        z-score: short both when the sum is rich, long both when cheap.
    1b. PEBBLES_XL vs basket {XS, S, M, L}: ρ ≈ -0.5 of XL with each
        small pebble, small pebbles uncorrelated with each other.
        Trade the sum of all 5 mids the same way.

  Layer 2 — Per-group directional (placeholder).
    Short-horizon mean reversion on ret_100 for the four groups where
    section 8 of EDA showed any consistent signal: MICROCHIP, GALAXY,
    UV_VISOR, OXYGEN_SHAKE. Tiny size by design until per-group ridge
    weights are trained and slotted in.

  Layer 3 — Passive market making on the calmest fillers.
    Small inventory-skewed quotes on low-vol products that aren't
    being used by Layers 1 or 2.

Risk overlay:
  Hard cap = 10 (IMC limit). Soft cap = 7. No quote may push past 7
  on the building side. Conflicts between layers are resolved by the
  order they execute: spreads first, directional second, MM last.
  Each layer reads state.position fresh.

Why no NN/transformer/RL: signal magnitudes (ρ ≈ 0.05–0.22 on the
strongest features) and only 3 days of training data make model
capacity the wrong axis to scale. The two structural spreads above
are the only ρ-large signals in the data; everything else is noise
on top of those.
"""

import json
import math
from collections import deque
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [state.timestamp, trader_data, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations)]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, obs in observations.conversionObservations.items():
            conversion_observations[product] = [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate; lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ============================================================================
# Universal constants
# ============================================================================

POSITION_LIMIT = 10
SOFT_LIMIT     = 7    # don't add to a side past this; flatten allowed past it


# ============================================================================
# Layer 1a — SNACKPACK pair
# ============================================================================

SNACK_CFG = {
    "leg_a":         "SNACKPACK_CHOCOLATE",
    "leg_b":         "SNACKPACK_VANILLA",
    "window":        500,    # rolling stats window in ticks
    "warmup":        200,    # min observations before trading
    "z_entry":       1.5,
    "z_exit":        0.3,
    "z_max":         3.0,    # at this z, target full size
    "max_legs":      10,     # max position per leg (also IMC hard cap)
}


# ============================================================================
# Layer 1b — PEBBLES basket
# ============================================================================
# All 5 mids summed, traded as one direction. The 1:4 sizing (1 XL : 4
# basket-units) emerges naturally from equal sizing each leg.

PEB_CFG = {
    "leg_xl":        "PEBBLES_XL",
    "legs_basket":   ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"],
    "window":        500,
    "warmup":        200,
    "z_entry":       1.8,    # looser than SNACK because correlation is -0.5 not -0.96
    "z_exit":        0.3,
    "z_max":         3.0,
    "max_legs":      10,
}


# ============================================================================
# Layer 2 — directional (per-group ret_100 mean reversion)
# ============================================================================
# Placeholder: section 8 of EDA showed mild reversion at h=100 for these
# groups (ρ between -0.05 and -0.15). Replace with trained ridge weights
# once a per-group fit beats hit-rate 0.52 on day 4.

DIR_PRODUCTS = [
    # MICROCHIP — exclude SQUARE (the high-vol one is the trap)
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    # GALAXY — all 5 fit, BLACK_HOLES is the noisiest
    "GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES", "GALAXY_SOUNDS_SOLAR_WINDS",
    # UV_VISOR — best persistence group (only 3/10 sign flips)
    "UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE", "UV_VISOR_RED", "UV_VISOR_YELLOW",
    # OXYGEN_SHAKE — mid-vol, drifts both ways
    "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH",
]

DIR_LOOKBACK    = 100        # ret_100 horizon
DIR_RET_THRESH  = 0.0015     # min |log return| to act on (filters noise)
DIR_MAX_SIZE    = 3          # tiny while this layer is unvalidated


# ============================================================================
# Layer 3 — market making fillers
# ============================================================================
# Calmest members of groups where we are NOT trading anything else.
# Tight quote, small size, inventory skew.

MM_PRODUCTS = [
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
    "ROBOT_LAUNDRY",
    "SLEEP_POD_LAMB_WOOL",
    "PANEL_2X2",
]

MM_BASE_OFFSET   = 1
MM_SKEW_OFFSET   = 2
MM_SKEW_TRIGGER  = 3
MM_BASE_SIZE     = 2


# ============================================================================
# Helpers
# ============================================================================

def _mid(depth: OrderDepth):
    if not depth or not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders)
    ba = min(depth.sell_orders)
    if bb >= ba:
        return None
    return (bb + ba) / 2.0


def _best_bid_ask(depth: OrderDepth):
    if not depth or not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


def _l1_imbalance(depth: OrderDepth):
    if not depth or not depth.buy_orders or not depth.sell_orders:
        return 0.0
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv = abs(depth.buy_orders[bb])
    av = abs(depth.sell_orders[ba])
    tot = bv + av
    return 0.0 if tot <= 0 else (bv - av) / tot


def _scale_size(z_abs: float, z_entry: float, z_max: float, max_size: int) -> int:
    """Linear scale of position size between entry and max z-score."""
    if z_abs <= z_entry:
        return 0
    if z_abs >= z_max:
        return max_size
    frac = (z_abs - z_entry) / (z_max - z_entry)
    return max(1, int(round(frac * max_size)))


def _take_to_target(sym: str, depth: OrderDepth, current_pos: int, target_pos: int) -> list[Order]:
    """
    Issue aggressive orders to move position toward target.
    Crosses the spread by hitting best bid/ask. No multi-level walking
    (size per tick is tiny, top of book is enough).
    """
    delta = target_pos - current_pos
    if delta == 0:
        return []
    bb, ba = _best_bid_ask(depth)
    if bb is None:
        return []
    if delta > 0:
        # buy at best ask
        size = min(delta, abs(depth.sell_orders[ba]))
        return [Order(sym, ba, size)] if size > 0 else []
    else:
        # sell at best bid
        size = min(-delta, abs(depth.buy_orders[bb]))
        return [Order(sym, bb, -size)] if size > 0 else []


def _apply_position_limits(sym: str, current_pos: int, target_pos: int) -> int:
    """Clamp target to [-POSITION_LIMIT, +POSITION_LIMIT]. Soft limit
    only blocks NEW size on the building side; flattening always allowed."""
    target = max(-POSITION_LIMIT, min(POSITION_LIMIT, target_pos))
    # soft limit applies only when growing the position
    if target > 0 and target > current_pos and target > SOFT_LIMIT:
        target = max(current_pos, SOFT_LIMIT)
    if target < 0 and target < current_pos and target < -SOFT_LIMIT:
        target = min(current_pos, -SOFT_LIMIT)
    return target


# ============================================================================
# Spread engine — shared by SNACK pair and PEBBLES basket
# ============================================================================

def _spread_target_position(z: float, cfg: dict, current_leg_pos: int) -> int:
    """Map z-score to a target per-leg position."""
    if abs(z) < cfg["z_exit"]:
        return 0
    if abs(z) < cfg["z_entry"]:
        # holding zone: keep current direction if any, else flat
        return current_leg_pos
    sign = -1 if z > 0 else +1   # high z → short the spread → target < 0
    size = _scale_size(abs(z), cfg["z_entry"], cfg["z_max"], cfg["max_legs"])
    return sign * size


def _spread_orders(state: TradingState, history: deque, cfg: dict, legs: list) -> dict:
    """
    Generic z-score spread trader.
      - history: rolling deque of past sum-of-mids
      - legs: list of product symbols whose mids we sum
      - All legs traded in the SAME direction with equal size.
    """
    out = {}

    # require all legs visible
    mids = []
    for sym in legs:
        m = _mid(state.order_depths.get(sym))
        if m is None:
            return out
        mids.append(m)
    s = sum(mids)
    history.append(s)

    if len(history) < cfg["warmup"]:
        return out

    arr = list(history)
    mu = sum(arr) / len(arr)
    var = sum((x - mu) ** 2 for x in arr) / len(arr)
    sd = math.sqrt(var)
    if sd <= 1e-6:
        return out
    z = (s - mu) / sd

    # use leg_a / leg_xl position as the reference for "where we are in the trade"
    ref_pos = state.position.get(legs[0], 0)
    target = _spread_target_position(z, cfg, ref_pos)

    for sym in legs:
        depth = state.order_depths.get(sym)
        if not depth:
            continue
        pos = state.position.get(sym, 0)
        clamped = _apply_position_limits(sym, pos, target)
        ors = _take_to_target(sym, depth, pos, clamped)
        if ors:
            out[sym] = ors
    return out


# ============================================================================
# Layer 2 — directional ret_100 reversion
# ============================================================================

def _directional_orders(state: TradingState, mid_history: dict) -> dict:
    """
    For each product in DIR_PRODUCTS, target a small position opposite to
    the last DIR_LOOKBACK-tick log return. Tiny size, just enough to
    capture mild reversion shown in EDA.
    """
    out = {}
    for sym in DIR_PRODUCTS:
        depth = state.order_depths.get(sym)
        m = _mid(depth)
        if m is None:
            continue
        hist = mid_history.setdefault(sym, deque(maxlen=DIR_LOOKBACK + 1))
        hist.append(m)
        if len(hist) < DIR_LOOKBACK + 1:
            continue
        past = hist[0]
        if past <= 0:
            continue
        log_ret = math.log(m / past)
        if abs(log_ret) < DIR_RET_THRESH:
            target = 0
        else:
            sign = -1 if log_ret > 0 else +1   # mean revert
            mag = min(DIR_MAX_SIZE, max(1, int(abs(log_ret) / DIR_RET_THRESH)))
            target = sign * mag

        pos = state.position.get(sym, 0)
        clamped = _apply_position_limits(sym, pos, target)
        ors = _take_to_target(sym, depth, pos, clamped)
        if ors:
            out[sym] = ors
    return out


# ============================================================================
# Layer 3 — passive MM
# ============================================================================

def _mm_orders(state: TradingState) -> dict:
    out = {}
    for sym in MM_PRODUCTS:
        depth = state.order_depths.get(sym)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            continue
        bb, ba = _best_bid_ask(depth)
        if bb is None or bb >= ba:
            continue
        mid = (bb + ba) / 2.0
        pos = state.position.get(sym, 0)

        if pos > MM_SKEW_TRIGGER:
            bid_off, ask_off = MM_BASE_OFFSET + MM_SKEW_OFFSET, MM_BASE_OFFSET
        elif pos < -MM_SKEW_TRIGGER:
            bid_off, ask_off = MM_BASE_OFFSET, MM_BASE_OFFSET + MM_SKEW_OFFSET
        else:
            bid_off, ask_off = MM_BASE_OFFSET, MM_BASE_OFFSET

        bid_px = int(math.floor(mid)) - bid_off
        ask_px = int(math.ceil(mid))  + ask_off
        if bid_px >= ba: bid_px = ba - 1
        if ask_px <= bb: ask_px = bb + 1
        if bid_px >= ask_px:
            continue

        # respect soft limit on the building side
        bid_size = MM_BASE_SIZE if pos < SOFT_LIMIT else 0
        ask_size = MM_BASE_SIZE if pos > -SOFT_LIMIT else 0
        # never quote into hard limit
        bid_size = min(bid_size, POSITION_LIMIT - pos)
        ask_size = min(ask_size, POSITION_LIMIT + pos)

        # imbalance veto: don't post on the side the book is fading
        imb = _l1_imbalance(depth)
        if imb >  0.20: ask_size = 0
        if imb < -0.20: bid_size = 0

        ors = []
        if bid_size > 0:
            ors.append(Order(sym, bid_px, int(bid_size)))
        if ask_size > 0:
            ors.append(Order(sym, ask_px, -int(ask_size)))
        if ors:
            out[sym] = ors
    return out


# ============================================================================
# Trader
# ============================================================================

class Trader:
    def __init__(self):
        # rolling spread sums
        self.snack_hist = deque(maxlen=SNACK_CFG["window"])
        self.peb_hist   = deque(maxlen=PEB_CFG["window"])
        # per-product mid history for directional layer
        self.mid_hist: dict = {}

    def bid(self):
        return 0

    def run(self, state: TradingState):
        orders: dict[str, list[Order]] = {}

        # ---- Layer 1a: SNACKPACK pair ----
        snack_legs = [SNACK_CFG["leg_a"], SNACK_CFG["leg_b"]]
        for sym, ors in _spread_orders(state, self.snack_hist, SNACK_CFG, snack_legs).items():
            orders[sym] = ors

        # ---- Layer 1b: PEBBLES basket ----
        peb_legs = [PEB_CFG["leg_xl"]] + PEB_CFG["legs_basket"]
        for sym, ors in _spread_orders(state, self.peb_hist, PEB_CFG, peb_legs).items():
            orders[sym] = ors

        # ---- Layer 2: directional mean reversion ----
        # (skip products already touched by Layer 1)
        used = set(orders.keys())
        dir_out = _directional_orders(state, self.mid_hist)
        for sym, ors in dir_out.items():
            if sym in used:
                continue
            orders[sym] = ors

        # ---- Layer 3: market making fillers ----
        used = set(orders.keys())
        mm_out = _mm_orders(state)
        for sym, ors in mm_out.items():
            if sym in used:
                continue
            orders[sym] = ors

        return orders, 0, ""
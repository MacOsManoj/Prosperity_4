#10.6k
import json
import math
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[str, List[Order]],
              conversions: int, trader_data: str) -> None:
        base = [
            self.compress_state(state),
            self.compress_orders(orders),
            conversions, trader_data, self.logs,
        ]
        print(json.dumps(base, cls=ProsperityEncoder, separators=(",", ":")))
        self.logs = ""

    def compress_state(self, state: TradingState) -> list:
        return [
            state.timestamp, "",
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for tl in state.own_trades.values() for t in tl],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for tl in state.market_trades.values() for t in tl],
            state.position,
            [state.observations.plainValueObservations,
             {p: [c.bidPrice, c.askPrice, c.transportFees, c.exportTariff,
                  c.importTariff, c.sunlight, c.humidity]
              for p, c in state.observations.conversionObservations.items()}],
        ]

    def compress_orders(self, orders) -> dict:
        return {s: [[o.price, o.quantity] for o in ol] for s, ol in orders.items()}


logger = Logger()

# ════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════

POSITION_LIMITS = {
    "EMERALDS": 80,
    "ASH_COATED_OSMIUM": 80,
    "TOMATOES": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

# ── EMERALDS: stationary MM at 10000 ──
EMERALDS_CFG = {
    "fair_value": 10000, "take_width": 1, "clear_width": 0,
    "disregard_edge": 1, "join_edge": 2, "default_edge": 4,
    "soft_limit": 50, "base_size": 40, "skew_per": 10,
}

# ── ASH_COATED_OSMIUM: cycle-aware MM at 10000 ──
# FFT reveals consistent cycles at ~91s, ~333s, ~500s periods.
# We use multi-timescale EMAs to detect cycle phase and shift fair value.
# Strong mean reversion: lag-1 autocorr = -0.50, OU half-life ~3 ticks.
OSMIUM_CFG = {
    "fair_value": 10000, "take_width": 1, "clear_width": 0,
    "disregard_edge": 1, "join_edge": 2, "default_edge": 4,
    "soft_limit": 50, "base_size": 40, "skew_per": 10,
}
# EMA spans (in ticks, ~100ms each) for cycle detection
# ~91s cycle = 910 ticks, ~333s = 3330 ticks, ~500s = 5000 ticks
OSMIUM_EMA_SPANS = [10, 100, 500, 1000, 3000]
OSMIUM_CYCLE_WEIGHT = 0.5    # how much to shift fair based on cycle signal
OSMIUM_REVERSION_COEFF = 0.3  # mean reversion on last tick move

# ── TOMATOES: drifter with volume-filtered mid ──
TOMATOES_CFG = {
    "take_width": 1, "default_edge": 3, "soft_limit": 20,
    "hard_limit": 40, "base_size": 15, "skew_per": 5, "vol_filter": 15,
}

# ── INTARIAN_PEPPER_ROOT: deterministic upward drift ──
# Price drifts +0.001/tick (~1000/day). Strategy: accumulate max long.
PEPPER_SLOPE = 0.001
PEPPER_TAKE_PREMIUM = 8     # buy asks up to 8 above trend fair
PEPPER_SELL_FLOOR = 15       # only sell if bid is 15+ above fair
PEPPER_BID_EDGE = 2          # passive bid at fair - 2
PEPPER_ASK_EDGE = 20         # passive ask very wide (don't want to sell)
PEPPER_BASE_BID_SIZE = 30    # large bids to accumulate fast
PEPPER_BASE_ASK_SIZE = 5     # small asks


class Trader:

    def run(self, state: TradingState):
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                trader_data = {}

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            if product == "EMERALDS":
                result[product] = self.trade_stationary(product, state, EMERALDS_CFG)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_stationary(product, state, OSMIUM_CFG)
            elif product == "TOMATOES":
                orders, trader_data = self.trade_drifter(product, state, TOMATOES_CFG, trader_data)
                result[product] = orders
            elif product == "INTARIAN_PEPPER_ROOT":
                orders, trader_data = self.trade_pepper(state, trader_data)
                result[product] = orders

        conversions = 0
        traderData = json.dumps(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    # ════════════════════════════════════════════════════════
    # Stationary: EMERALDS
    # Three-phase: Take -> Clear -> Make around known fair value
    # ════════════════════════════════════════════════════════

    def trade_stationary(self, product: str, state: TradingState, cfg: dict) -> List[Order]:
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = POSITION_LIMITS[product]
        fair = cfg["fair_value"]
        orders: List[Order] = []
        bv = 0
        sv = 0

        # ── TAKE ──
        for price in sorted(od.sell_orders.keys()):
            if price > fair - cfg["take_width"]:
                break
            vol = -od.sell_orders[price]
            qty = min(vol, limit - (pos + bv))
            if qty > 0:
                orders.append(Order(product, price, qty))
                bv += qty

        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price < fair + cfg["take_width"]:
                break
            vol = od.buy_orders[price]
            qty = min(vol, limit + (pos - sv))
            if qty > 0:
                orders.append(Order(product, price, -qty))
                sv += qty

        # ── CLEAR ──
        cur = pos + bv - sv
        if cur > cfg["soft_limit"]:
            qty = min(cur - cfg["soft_limit"], limit + (pos - sv))
            if qty > 0:
                orders.append(Order(product, fair + cfg["clear_width"], -qty))
                sv += qty
        elif cur < -cfg["soft_limit"]:
            qty = min(-cur - cfg["soft_limit"], limit - (pos + bv))
            if qty > 0:
                orders.append(Order(product, fair - cfg["clear_width"], qty))
                bv += qty

        # ── MAKE ──
        cur = pos + bv - sv
        skew = math.floor(cur / cfg["skew_per"])

        bid_price = fair - cfg["default_edge"] - skew
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price >= fair - cfg["disregard_edge"]:
                continue
            if price >= fair - cfg["join_edge"]:
                bid_price = price
            else:
                bid_price = price + 1
            break

        ask_price = fair + cfg["default_edge"] - skew
        for price in sorted(od.sell_orders.keys()):
            if price <= fair + cfg["disregard_edge"]:
                continue
            if price <= fair + cfg["join_edge"]:
                ask_price = price
            else:
                ask_price = price - 1
            break

        buy_size = cfg["base_size"] - max(0, cur) // 5
        sell_size = cfg["base_size"] + min(0, cur) // 5
        buy_size = max(5, min(40, buy_size))
        sell_size = max(5, min(40, sell_size))

        if abs(cur) >= limit - 5:
            if cur > 0:
                buy_size = max(1, buy_size // 2)
                sell_size = min(50, sell_size * 2)
            else:
                sell_size = max(1, sell_size // 2)
                buy_size = min(50, buy_size * 2)

        qty = min(buy_size, limit - (pos + bv))
        if qty > 0:
            orders.append(Order(product, bid_price, qty))

        qty = min(sell_size, limit + (pos - sv))
        if qty > 0:
            orders.append(Order(product, ask_price, -qty))

        return orders

    # ════════════════════════════════════════════════════════
    # ASH_COATED_OSMIUM: cycle-aware market making
    # Multi-EMA cycle detection + mean reversion
    # ════════════════════════════════════════════════════════

    def trade_osmium(self, state: TradingState, trader_data: dict):
        product = "ASH_COATED_OSMIUM"
        cfg = OSMIUM_CFG
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = POSITION_LIMITS[product]
        base_fair = cfg["fair_value"]  # 10000
        orders: List[Order] = []

        if not od.buy_orders or not od.sell_orders:
            return orders, trader_data

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        # ── Retrieve persisted state ──
        aco_data = trader_data.get("ACO", {})
        emas = aco_data.get("emas", [mid] * len(OSMIUM_EMA_SPANS))
        prev_mid = aco_data.get("prev_mid", mid)

        # ── Update EMAs ──
        for i, span in enumerate(OSMIUM_EMA_SPANS):
            alpha = 2.0 / (span + 1)
            emas[i] = alpha * mid + (1 - alpha) * emas[i]

        # ── SIGNAL 1: Cycle detection via EMA crossovers ──
        # Fast EMA above slow EMA → upswing, below → downswing
        # Use multiple pairs for robustness
        # emas[0]=EMA10, emas[1]=EMA100, emas[2]=EMA500,
        # emas[3]=EMA1000, emas[4]=EMA3000
        cycle_signal = 0.0
        # Short cycle (~91s): EMA10 vs EMA500
        cycle_signal += (emas[0] - emas[2]) * 0.3
        # Medium cycle (~333s): EMA100 vs EMA1000
        cycle_signal += (emas[1] - emas[3]) * 0.4
        # Long cycle (~500s): EMA100 vs EMA3000
        cycle_signal += (emas[1] - emas[4]) * 0.3

        # ── SIGNAL 2: Mean reversion on last tick ──
        last_move = mid - prev_mid
        reversion_signal = -last_move * OSMIUM_REVERSION_COEFF

        # ── Adjusted fair value ──
        # Cycle signal predicts direction, reversion fades last move
        # Both shift fair value: if we expect price to rise, raise fair
        # (makes us buy more aggressively, sell less aggressively)
        fair_shift = cycle_signal * OSMIUM_CYCLE_WEIGHT + reversion_signal
        fair = round(base_fair + fair_shift)

        # Persist
        trader_data["ACO"] = {
            "emas": emas,
            "prev_mid": mid,
        }

        bv = 0
        sv = 0

        # ── TAKE ──
        for price in sorted(od.sell_orders.keys()):
            if price > fair - cfg["take_width"]:
                break
            vol = -od.sell_orders[price]
            qty = min(vol, limit - (pos + bv))
            if qty > 0:
                orders.append(Order(product, price, qty))
                bv += qty

        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price < fair + cfg["take_width"]:
                break
            vol = od.buy_orders[price]
            qty = min(vol, limit + (pos - sv))
            if qty > 0:
                orders.append(Order(product, price, -qty))
                sv += qty

        # ── CLEAR ──
        cur = pos + bv - sv
        if cur > cfg["soft_limit"]:
            qty = min(cur - cfg["soft_limit"], limit + (pos - sv))
            if qty > 0:
                orders.append(Order(product, fair + cfg["clear_width"], -qty))
                sv += qty
        elif cur < -cfg["soft_limit"]:
            qty = min(-cur - cfg["soft_limit"], limit - (pos + bv))
            if qty > 0:
                orders.append(Order(product, fair - cfg["clear_width"], qty))
                bv += qty

        # ── MAKE: join/improve with cycle-aware skew ──
        cur = pos + bv - sv
        skew = math.floor(cur / cfg["skew_per"])

        bid_price = fair - cfg["default_edge"] - skew
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price >= fair - cfg["disregard_edge"]:
                continue
            if price >= fair - cfg["join_edge"]:
                bid_price = price
            else:
                bid_price = price + 1
            break

        ask_price = fair + cfg["default_edge"] - skew
        for price in sorted(od.sell_orders.keys()):
            if price <= fair + cfg["disregard_edge"]:
                continue
            if price <= fair + cfg["join_edge"]:
                ask_price = price
            else:
                ask_price = price - 1
            break

        buy_size = cfg["base_size"] - max(0, cur) // 5
        sell_size = cfg["base_size"] + min(0, cur) // 5
        buy_size = max(5, min(40, buy_size))
        sell_size = max(5, min(40, sell_size))

        if abs(cur) >= limit - 5:
            if cur > 0:
                buy_size = max(1, buy_size // 2)
                sell_size = min(50, sell_size * 2)
            else:
                sell_size = max(1, sell_size // 2)
                buy_size = min(50, buy_size * 2)

        qty = min(buy_size, limit - (pos + bv))
        if qty > 0:
            orders.append(Order(product, bid_price, qty))

        qty = min(sell_size, limit + (pos - sv))
        if qty > 0:
            orders.append(Order(product, ask_price, -qty))

        return orders, trader_data

    # ════════════════════════════════════════════════════════
    # Drifter: TOMATOES — adaptive fair value from volume-filtered mid
    # ════════════════════════════════════════════════════════

    def trade_drifter(self, product: str, state: TradingState,
                      cfg: dict, trader_data: dict) -> tuple:
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        hard = cfg["hard_limit"]
        orders: List[Order] = []

        if not od.buy_orders or not od.sell_orders:
            return orders, trader_data

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        vf = cfg["vol_filter"]
        fb = {p: v for p, v in od.buy_orders.items() if v >= vf}
        fa = {p: v for p, v in od.sell_orders.items() if -v >= vf}

        if fb and fa:
            mid = (max(fb.keys()) + min(fa.keys())) / 2
        else:
            mid = (best_bid + best_ask) / 2

        fair = round(mid)

        bv = 0
        sv = 0

        # ── TAKE ──
        for price in sorted(od.sell_orders.keys()):
            if price > fair - cfg["take_width"]:
                break
            vol = -od.sell_orders[price]
            qty = min(vol, hard - (pos + bv))
            if qty > 0:
                orders.append(Order(product, price, qty))
                bv += qty

        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price < fair + cfg["take_width"]:
                break
            vol = od.buy_orders[price]
            qty = min(vol, hard + (pos - sv))
            if qty > 0:
                orders.append(Order(product, price, -qty))
                sv += qty

        # ── CLEAR ──
        cur = pos + bv - sv
        if cur > cfg["soft_limit"]:
            qty = min(cur - cfg["soft_limit"], hard + (pos - sv))
            if qty > 0:
                orders.append(Order(product, fair, -qty))
                sv += qty
        elif cur < -cfg["soft_limit"]:
            qty = min(-cur - cfg["soft_limit"], hard - (pos + bv))
            if qty > 0:
                orders.append(Order(product, fair, qty))
                bv += qty

        # ── MAKE ──
        cur = pos + bv - sv
        skew = math.floor(cur / cfg["skew_per"])

        bid_price = fair - cfg["default_edge"] - skew
        ask_price = fair + cfg["default_edge"] - skew

        buy_size = cfg["base_size"] - max(0, cur) // 3
        sell_size = cfg["base_size"] + min(0, cur) // 3
        buy_size = max(3, min(20, buy_size))
        sell_size = max(3, min(20, sell_size))

        qty = min(buy_size, hard - (pos + bv))
        if qty > 0:
            orders.append(Order(product, bid_price, qty))

        qty = min(sell_size, hard + (pos - sv))
        if qty > 0:
            orders.append(Order(product, ask_price, -qty))

        return orders, trader_data

    # ════════════════════════════════════════════════════════
    # INTARIAN_PEPPER_ROOT: trend-following buy & hold
    # Price drifts +0.001/tick deterministically (~1000/day)
    # Strategy: accumulate max long (80 units) ASAP and hold
    # ════════════════════════════════════════════════════════

    def trade_pepper(self, state: TradingState, trader_data: dict):
        product = "INTARIAN_PEPPER_ROOT"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = POSITION_LIMITS[product]
        orders: List[Order] = []
        ts = state.timestamp

        if not od.buy_orders or not od.sell_orders:
            return orders, trader_data

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        # Use max-volume prices for stable mid
        max_bid = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
        max_ask = min(od.sell_orders.keys(), key=lambda p: -od.sell_orders[p])
        raw_mid = (max_bid + max_ask) / 2

        # Detect new day (timestamp resets) and track day_start_price
        pepper_data = trader_data.get("INTARIAN_PEPPER_ROOT", {})
        last_ts = pepper_data.get("last_ts", ts)
        day_start_price = pepper_data.get("day_start_price", raw_mid)

        if ts < last_ts:
            day_start_price = raw_mid

        # Trend-adjusted fair value
        fair = round(day_start_price + ts * PEPPER_SLOPE)

        trader_data["INTARIAN_PEPPER_ROOT"] = {
            "last_ts": ts,
            "day_start_price": day_start_price,
        }

        bv = 0
        sv = 0

        # ── TAKE: buy aggressively — even above trend fair ──
        for price in sorted(od.sell_orders.keys()):
            if price > fair + PEPPER_TAKE_PREMIUM:
                break
            vol = -od.sell_orders[price]
            qty = min(vol, limit - (pos + bv))
            if qty > 0:
                orders.append(Order(product, price, qty))
                bv += qty

        # Only sell if bids are very far above fair
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price < fair + PEPPER_SELL_FLOOR:
                break
            vol = od.buy_orders[price]
            qty = min(vol, limit + (pos - sv))
            if qty > 0:
                orders.append(Order(product, price, -qty))
                sv += qty

        # ── CLEAR: aggressively unwind shorts (being short in uptrend is bad) ──
        cur = pos + bv - sv
        if cur < -5:
            clear_qty = min(-cur, 30)
            qty = min(clear_qty, limit - (pos + bv))
            if qty > 0:
                orders.append(Order(product, best_ask + 1, qty))
                bv += qty

        # ── MAKE: aggressive bid, very passive ask ──
        cur = pos + bv - sv
        long_excess = max(0, cur - 60)
        bid_edge = PEPPER_BID_EDGE + long_excess // 10

        bid_price = fair - bid_edge
        qty = min(PEPPER_BASE_BID_SIZE, limit - (pos + bv))
        if qty > 0:
            orders.append(Order(product, bid_price, qty))

        ask_price = fair + PEPPER_ASK_EDGE
        qty = min(PEPPER_BASE_ASK_SIZE, limit + (pos - sv))
        if qty > 0:
            orders.append(Order(product, ask_price, -qty))

        return orders, trader_data
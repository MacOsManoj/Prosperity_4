#10.2k
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    # Minimal local fallbacks so the file can be imported outside the simulator.
    @dataclass
    class Order:  # type: ignore
        symbol: str
        price: int
        quantity: int

    @dataclass
    class OrderDepth:  # type: ignore
        buy_orders: Dict[int, int]
        sell_orders: Dict[int, int]

    @dataclass
    class TradingState:  # type: ignore
        timestamp: int
        traderData: str
        order_depths: Dict[str, OrderDepth]
        position: Dict[str, int]


PEPPER = "INTARIAN_PEPPER_ROOT"
ASH = "ASH_COATED_OSMIUM"

POSITION_LIMITS: Dict[str, int] = {
    PEPPER: 80,
    ASH: 80,
}

DAY_END = 999_900
PEPPER_UNWIND_START = 900_000
PEPPER_FORCE_EXIT_START = 975_000

ASH_ANCHOR_FAIR = 10_000.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def floor_int(x: float) -> int:
    return math.floor(x)


def ceil_int(x: float) -> int:
    return math.ceil(x)


class Trader:
    """
    Two-product strategy:
    - INTARIAN_PEPPER_ROOT: deterministic upward-carry product.
      Main goal is to stay long early/mid session and unwind late.
    - ASH_COATED_OSMIUM: classic spread-capture market making around ~10000.

    Returns: (orders, conversions, traderData)
    """

    def run(self, state: TradingState):
        cache = self._load_cache(state.traderData)
        orders: Dict[str, List[Order]] = {}

        pepper_depth = state.order_depths.get(PEPPER)
        if pepper_depth is not None:
            pepper_orders = self._trade_pepper(
                timestamp=state.timestamp,
                order_depth=pepper_depth,
                position=state.position.get(PEPPER, 0),
                cache=cache,
            )
            if pepper_orders:
                orders[PEPPER] = pepper_orders

        ash_depth = state.order_depths.get(ASH)
        if ash_depth is not None:
            ash_orders = self._trade_ash(
                timestamp=state.timestamp,
                order_depth=ash_depth,
                position=state.position.get(ASH, 0),
                cache=cache,
            )
            if ash_orders:
                orders[ASH] = ash_orders

        trader_data = self._dump_cache(cache)
        return orders, 0, trader_data

    # ---------------------------------------------------------------------
    # PEPPER
    # ---------------------------------------------------------------------

    def _trade_pepper(
        self,
        timestamp: int,
        order_depth: OrderDepth,
        position: int,
        cache: Dict[str, float],
    ) -> List[Order]:
        limit = POSITION_LIMITS[PEPPER]
        bids, asks = self._extract_book(order_depth)
        best_bid, best_bid_qty = (bids[0] if bids else (None, 0))
        best_ask, best_ask_qty = (asks[0] if asks else (None, 0))

        wall_mid = self._wall_mid(best_bid, best_ask)
        if cache.get("pepper_base") is None:
            if wall_mid is not None:
                inferred = 1000 * round((wall_mid - timestamp / 1000.0) / 1000)
                cache["pepper_base"] = float(inferred)
            else:
                cache["pepper_base"] = 13_000.0

        pepper_base = cache.get("pepper_base", 13_000.0)
        fair = pepper_base + timestamp / 1000.0

        imbalance = self._imbalance(best_bid_qty, best_ask_qty)
        fair += 1.0 * imbalance

        target = self._pepper_target_position(timestamp, limit)
        remaining_drift = max(0.0, (DAY_END - timestamp) / 1000.0)

        buy_reservation = fair + min(8.0, 0.80 * remaining_drift)

        sell_discount = 0.20 * remaining_drift if timestamp < PEPPER_FORCE_EXIT_START else 0.0
        sell_reservation = fair - min(3.0, sell_discount)

        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        # 1) Aggressively take asks to build long position toward target.
        desired_buy = max(0, target - position)
        for ask_price, ask_qty in asks:
            if desired_buy <= 0:
                break
            if ask_price <= floor_int(buy_reservation):
                qty = min(ask_qty, desired_buy, limit - (position + buy_used))
                if qty > 0:
                    orders.append(Order(PEPPER, int(ask_price), int(qty)))
                    buy_used += qty
                    desired_buy -= qty

        # 2) Unwind when above target (late-day position reduction).
        desired_sell = max(0, (position + buy_used) - target)
        for bid_price, bid_qty in bids:
            if desired_sell <= 0:
                break
            must_unwind = timestamp >= PEPPER_FORCE_EXIT_START
            if must_unwind or bid_price >= ceil_int(sell_reservation):
                qty = min(bid_qty, desired_sell, limit + (position + buy_used - sell_used))
                if qty > 0:
                    orders.append(Order(PEPPER, int(bid_price), int(-qty)))
                    sell_used += qty
                    desired_sell -= qty

        position_after_takes = position + buy_used - sell_used
        remaining_buy_cap = limit - position_after_takes
        remaining_sell_cap = limit + position_after_takes

        # 3) Passive quoting with strong bid bias for carry capture.
        inventory_gap = target - position_after_takes
        pos_skew = 4.0 * inventory_gap / max(1, limit)
        quote_carry = min(8.0, 0.70 * remaining_drift)

        if remaining_buy_cap > 0 and inventory_gap > 0:
            raw_bid = floor_int(fair + quote_carry + pos_skew)
            if best_bid is not None:
                raw_bid = max(raw_bid, int(best_bid) + 1)
            if best_ask is not None:
                raw_bid = min(raw_bid, int(best_ask) - 1)

            if best_ask is None or raw_bid < best_ask:
                desired_qty = min(remaining_buy_cap, 25 + max(0, inventory_gap))
                if desired_qty > 0:
                    orders.append(Order(PEPPER, int(raw_bid), int(desired_qty)))

        should_post_ask = position_after_takes >= target or timestamp >= PEPPER_UNWIND_START
        if remaining_sell_cap > 0 and should_post_ask:
            late_factor = 1.0 if timestamp >= PEPPER_FORCE_EXIT_START else 0.0
            raw_ask = ceil_int(fair + (0.10 if late_factor else max(2.0, quote_carry + 1.0)) - 2.0 * pos_skew)
            if timestamp < PEPPER_UNWIND_START and position_after_takes <= target:
                raw_ask = max(raw_ask, ceil_int(fair + 9))
            if best_ask is not None:
                raw_ask = min(raw_ask, int(best_ask) - 1)
            if timestamp < PEPPER_UNWIND_START and position_after_takes <= target:
                raw_ask = max(raw_ask, ceil_int(fair + 9))
            if best_bid is not None:
                raw_ask = max(raw_ask, int(best_bid) + 1)

            if best_bid is None or raw_ask > best_bid:
                desired_qty = min(remaining_sell_cap, 15 + max(0, position_after_takes - target))
                if desired_qty > 0:
                    orders.append(Order(PEPPER, int(raw_ask), int(-desired_qty)))

        return self._deduplicate_and_clip(orders, limit)

    def _pepper_target_position(self, timestamp: int, limit: int) -> int:
        if timestamp <= PEPPER_UNWIND_START:
            return limit
        if timestamp >= DAY_END:
            return 0
        frac = (DAY_END - timestamp) / (DAY_END - PEPPER_UNWIND_START)
        return int(round(limit * clamp(frac, 0.0, 1.0)))

    # ---------------------------------------------------------------------
    # ASH
    # ---------------------------------------------------------------------

    def _trade_ash(
        self,
        timestamp: int,
        order_depth: OrderDepth,
        position: int,
        cache: Dict[str, float],
    ) -> List[Order]:
        del timestamp
        limit = POSITION_LIMITS[ASH]
        bids, asks = self._extract_book(order_depth)
        best_bid, best_bid_qty = (bids[0] if bids else (None, 0))
        best_ask, best_ask_qty = (asks[0] if asks else (None, 0))

        prev_ema = float(cache.get("ash_ema", ASH_ANCHOR_FAIR))

        bid_wall = bids[-1][0] if bids else None
        ask_wall = asks[-1][0] if asks else None
        if bid_wall is not None and ask_wall is not None:
            wall_mid = (bid_wall + ask_wall) / 2.0
        else:
            wall_mid = prev_ema

        ash_ema = 0.85 * prev_ema + 0.15 * wall_mid
        cache["ash_ema"] = ash_ema

        fair = wall_mid
        anchor = ASH_ANCHOR_FAIR
        deviation = fair - anchor

        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        # 1) Take: walk full book for any positive-edge fills.
        take_buy_threshold = floor_int(fair - 1.0)
        for ask_price, ask_qty in asks:
            if ask_price > take_buy_threshold:
                break
            qty = min(ask_qty, limit - (position + buy_used))
            if qty > 0:
                orders.append(Order(ASH, int(ask_price), int(qty)))
                buy_used += qty

        take_sell_threshold = ceil_int(fair + 1.0)
        for bid_price, bid_qty in bids:
            if bid_price < take_sell_threshold:
                break
            qty = min(bid_qty, limit + (position + buy_used - sell_used))
            if qty > 0:
                orders.append(Order(ASH, int(bid_price), int(-qty)))
                sell_used += qty

        # 2) Session-level mean reversion: aggressively take when price
        #    deviates from anchor AND the book offers an entry within
        #    anchor ± 5. At cap=5, profit/lot on exit >= 2.
        MR_THRESH = 4.0
        MR_MAX = 12
        MR_ENTRY_CAP = 4.0

        pos_after_std = position + buy_used - sell_used

        if deviation < -MR_THRESH and pos_after_std < MR_MAX and pos_after_std < 20:
            mr_budget = max(0, min(MR_MAX - max(0, pos_after_std), limit - pos_after_std))
            mr_price_ceil = floor_int(anchor + MR_ENTRY_CAP)
            for ask_price, ask_qty in asks:
                if mr_budget <= 0 or ask_price > mr_price_ceil:
                    break
                if ask_price <= take_buy_threshold:
                    continue
                qty = min(ask_qty, mr_budget, limit - (position + buy_used))
                if qty > 0:
                    orders.append(Order(ASH, int(ask_price), int(qty)))
                    buy_used += qty
                    mr_budget -= qty

        elif deviation > MR_THRESH and pos_after_std > -MR_MAX and pos_after_std > -20:
            mr_budget = max(0, min(MR_MAX + min(0, pos_after_std), limit + pos_after_std))
            mr_price_floor = ceil_int(anchor - MR_ENTRY_CAP)
            for bid_price, bid_qty in bids:
                if mr_budget <= 0 or bid_price < mr_price_floor:
                    break
                if bid_price >= take_sell_threshold:
                    continue
                qty = min(bid_qty, mr_budget, limit + (position + buy_used - sell_used))
                if qty > 0:
                    orders.append(Order(ASH, int(bid_price), int(-qty)))
                    sell_used += qty
                    mr_budget -= qty

        position_after_takes = position + buy_used - sell_used
        remaining_buy_cap = limit - position_after_takes
        remaining_sell_cap = limit + position_after_takes

        # 3) Passive quoting: penny-jump the BBO for queue priority.
        inv_shift = 8.0 * position_after_takes / max(1, limit)
        bid_edge = max(1.0, 3.0 + inv_shift)
        ask_edge = max(1.0, 3.0 - inv_shift)

        bid_cap = floor_int(fair - bid_edge)
        ask_floor = ceil_int(fair + ask_edge)

        primary_bid: Optional[int] = None
        if remaining_buy_cap > 0 and best_bid is not None:
            candidate = int(best_bid) + 1
            if best_ask is not None:
                candidate = min(candidate, int(best_ask) - 1)
            primary_bid = min(candidate, bid_cap)
            if best_ask is not None and primary_bid >= best_ask:
                primary_bid = None

        primary_ask: Optional[int] = None
        if remaining_sell_cap > 0 and best_ask is not None:
            candidate = int(best_ask) - 1
            if best_bid is not None:
                candidate = max(candidate, int(best_bid) + 1)
            primary_ask = max(candidate, ask_floor)
            if best_bid is not None and primary_ask <= best_bid:
                primary_ask = None

        pos_frac = position_after_takes / max(1, limit)
        buy_mult = max(0.15, min(1.0, 1.0 - max(0.0, pos_frac * 2.0 - 0.5)))
        sell_mult = max(0.15, min(1.0, 1.0 - max(0.0, -pos_frac * 2.0 - 0.5)))

        if primary_bid is not None and remaining_buy_cap > 0:
            qty = max(1, int(remaining_buy_cap * buy_mult))
            orders.append(Order(ASH, int(primary_bid), int(qty)))
            remaining_buy_cap -= qty

        if primary_ask is not None and remaining_sell_cap > 0:
            qty = max(1, int(remaining_sell_cap * sell_mult))
            orders.append(Order(ASH, int(primary_ask), int(-qty)))
            remaining_sell_cap -= qty

        # 4) Fallback quotes for one-sided books (~8% of timesteps).
        if primary_bid is None and remaining_buy_cap > 0:
            tight = floor_int(ash_ema - 7)
            wide = floor_int(ash_ema - 10)
            tight_qty = remaining_buy_cap * 2 // 3
            wide_qty = remaining_buy_cap - tight_qty
            if tight_qty > 0 and (best_ask is None or tight < best_ask):
                orders.append(Order(ASH, int(tight), int(tight_qty)))
            if wide_qty > 0 and (best_ask is None or wide < best_ask):
                orders.append(Order(ASH, int(wide), int(wide_qty)))

        if primary_ask is None and remaining_sell_cap > 0:
            tight = ceil_int(ash_ema + 7)
            wide = ceil_int(ash_ema + 10)
            tight_qty = remaining_sell_cap * 2 // 3
            wide_qty = remaining_sell_cap - tight_qty
            if tight_qty > 0 and (best_bid is None or tight > best_bid):
                orders.append(Order(ASH, int(tight), int(-tight_qty)))
            if wide_qty > 0 and (best_bid is None or wide > best_bid):
                orders.append(Order(ASH, int(wide), int(-wide_qty)))

        return self._deduplicate_and_clip(orders, limit)

    # ---------------------------------------------------------------------
    # Shared helpers
    # ---------------------------------------------------------------------

    def _extract_book(self, order_depth: OrderDepth) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = sorted(
            [(int(price), int(volume)) for price, volume in order_depth.buy_orders.items() if volume > 0],
            key=lambda x: x[0],
            reverse=True,
        )
        asks = sorted(
            [(int(price), int(abs(volume))) for price, volume in order_depth.sell_orders.items() if volume != 0],
            key=lambda x: x[0],
        )
        return bids, asks

    def _wall_mid(self, best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
        if best_bid is None and best_ask is None:
            return None
        if best_bid is None:
            return float(best_ask)
        if best_ask is None:
            return float(best_bid)
        return (best_bid + best_ask) / 2.0

    def _microprice(
        self,
        best_bid: Optional[int],
        bid_qty: int,
        best_ask: Optional[int],
        ask_qty: int,
    ) -> Optional[float]:
        if best_bid is None or best_ask is None:
            return None
        denom = bid_qty + ask_qty
        if denom <= 0:
            return (best_bid + best_ask) / 2.0
        return (best_ask * bid_qty + best_bid * ask_qty) / denom

    def _imbalance(self, bid_qty: int, ask_qty: int) -> float:
        denom = bid_qty + ask_qty
        if denom <= 0:
            return 0.0
        return (bid_qty - ask_qty) / denom

    def _flatten_at_fair(
        self,
        product: str,
        bids: List[Tuple[int, int]],
        asks: List[Tuple[int, int]],
        fair: float,
        position: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []
        if position > 0:
            for bid_price, bid_qty in bids:
                if bid_price >= ceil_int(fair - 1.0):
                    qty = min(position, bid_qty, limit + position)
                    if qty > 0:
                        orders.append(Order(product, int(bid_price), int(-qty)))
                        position -= qty
                if position <= 0:
                    break
        elif position < 0:
            need = -position
            for ask_price, ask_qty in asks:
                if ask_price <= floor_int(fair + 1.0):
                    qty = min(need, ask_qty, limit - position)
                    if qty > 0:
                        orders.append(Order(product, int(ask_price), int(qty)))
                        need -= qty
                if need <= 0:
                    break
        return orders

    def _deduplicate_and_clip(self, orders: List[Order], limit: int) -> List[Order]:
        merged: Dict[Tuple[str, int], int] = {}
        for order in orders:
            key = (order.symbol, int(order.price))
            merged[key] = merged.get(key, 0) + int(order.quantity)

        clean: List[Order] = []
        for (symbol, price), qty in merged.items():
            if qty == 0:
                continue
            qty = max(-limit, min(limit, qty))
            clean.append(Order(symbol, int(price), int(qty)))

        clean.sort(key=lambda o: (o.symbol, o.price, o.quantity))
        return clean

    def _load_cache(self, trader_data: Optional[str]) -> Dict[str, float]:
        if not trader_data:
            return {"pepper_base": None, "ash_ema": ASH_ANCHOR_FAIR}
        try:
            parsed = json.loads(trader_data)
            return {
                "pepper_base": parsed.get("pepper_base"),
                "ash_ema": float(parsed.get("ash_ema", ASH_ANCHOR_FAIR)),
            }
        except Exception:
            return {"pepper_base": None, "ash_ema": ASH_ANCHOR_FAIR}

    def _dump_cache(self, cache: Dict[str, float]) -> str:
        safe_cache = {
            "pepper_base": cache.get("pepper_base"),
            "ash_ema": float(cache.get("ash_ema", ASH_ANCHOR_FAIR)),
        }
        return json.dumps(safe_cache, separators=(",", ":"))
        
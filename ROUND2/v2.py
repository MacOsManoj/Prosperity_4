#9.4k
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datamodel import Order, OrderDepth, TradingState


class Product:
    ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
    INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"


PARAMS = {
    Product.ASH_COATED_OSMIUM: {
        "fair_value": 10000,
        "ewma_alpha": 0.12,
        "take_width": 0.5,
        "clear_width": 0.0,
        "disregard_edge": 0.5,
        "join_edge": 1.0,
        "default_edge": 95.5,
        "soft_position_limit": 16,
        "inventory_skew": 0.65,
    },
    Product.INTARIAN_PEPPER_ROOT: {
        "price_slope": 0.00100001,
        "base_update_weight": 0.2,
        "lookahead_timestamp": 8000,
        "max_timestamp": 1000000000,
        "take_width": 1.0,
        "take_alpha_scale": 0.5,
        "clear_width": 0.0,
        "disregard_edge": 1.0,
        "join_edge": 0.0,
        "default_edge": 1.5,
        "soft_position_limit": 70,
        "residual_weight": 0.8,
        "gap_weight": 0.0,
        "inventory_skew": 0.05,
        "alpha_clip": 8.0,
        "bid_skew_strength": 0.25,
        "ask_skew_strength": 0.1,
    },
}

LIMITS = {
    Product.ASH_COATED_OSMIUM: 80,
    Product.INTARIAN_PEPPER_ROOT: 80,
}


class SharedBookOps:
    def __init__(self, params: Dict[str, Dict[str, float]], limits: Dict[str, int]) -> None:
        self.params = params
        self.limits = limits

    @staticmethod
    def build_depth_from_books(buy_orders: dict[int, int], sell_orders: dict[int, int]) -> OrderDepth:
        depth = OrderDepth()
        depth.buy_orders = dict(buy_orders)
        depth.sell_orders = dict(sell_orders)
        return depth

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        buy_quantity = self.limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))

        sell_quantity = self.limits[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        reservation_price: float,
        width: int,
        orders: List[Order],
        sell_orders: Dict[int, int],
        buy_orders: Dict[int, int],
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(reservation_price - width)
        fair_for_ask = round(reservation_price + width)

        buy_quantity = self.limits[product] - (position + buy_order_volume)
        sell_quantity = self.limits[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -sent_quantity))
                sell_order_volume += sent_quantity

        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, sent_quantity))
                buy_order_volume += sent_quantity

        return buy_order_volume, sell_order_volume

    def compute_reservation_price(self, product: str, fair_value: float, position: int) -> float:
        params = self.params[product]
        soft_position_limit = max(1.0, float(params.get("soft_position_limit", 1)))
        inventory_skew = float(params.get("inventory_skew", 0.0))
        skew_ratio = max(-1.0, min(1.0, position / soft_position_limit))
        return fair_value - skew_ratio * inventory_skew


class OsmiumEngine(SharedBookOps):
    product = Product.ASH_COATED_OSMIUM
    namespace = "osmium"

    def compute_book_state(self, order_depth: OrderDepth) -> dict[str, float] | None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_volume = abs(order_depth.buy_orders[best_bid])
        ask_volume = abs(order_depth.sell_orders[best_ask])
        mid = (best_bid + best_ask) / 2
        denom = bid_volume + ask_volume
        microprice = (best_ask * bid_volume + best_bid * ask_volume) / denom if denom else mid
        return {
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "bid_volume": float(bid_volume),
            "ask_volume": float(ask_volume),
            "mid": float(mid),
            "microprice": float(microprice),
        }

    def estimate_fair_value(
        self,
        order_depth: OrderDepth,
        timestamp: int,
        trader_data: dict[str, Any],
    ) -> tuple[float, dict[str, float] | None]:
        params = self.params[self.product]
        default_fair_value = float(params["fair_value"])
        alpha = min(max(float(params.get("ewma_alpha", 0.1)), 0.0), 1.0)

        product_state = trader_data.setdefault(self.product, {})
        last_timestamp = product_state.get("last_timestamp")
        if isinstance(last_timestamp, int) and timestamp < last_timestamp:
            product_state.clear()

        book = self.compute_book_state(order_depth)
        if book is None:
            last_fair_value = product_state.get("last_fair_value")
            if isinstance(last_fair_value, (int, float)):
                return float(last_fair_value), None
            return default_fair_value, None

        prev_ewma = product_state.get("ewma_mid", default_fair_value)
        if not isinstance(prev_ewma, (int, float)):
            prev_ewma = default_fair_value

        fair_value = alpha * float(book["mid"]) + (1.0 - alpha) * float(prev_ewma)
        product_state["last_timestamp"] = int(timestamp)
        product_state["last_mid"] = float(book["mid"])
        product_state["last_microprice"] = float(book["microprice"])
        product_state["ewma_mid"] = fair_value
        product_state["last_fair_value"] = fair_value
        return fair_value, book

    def take_orders(
        self,
        reservation_price: float,
        take_width: float,
        orders: List[Order],
        sell_orders: Dict[int, int],
        buy_orders: Dict[int, int],
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        position_limit = self.limits[self.product]

        for ask_price in sorted(list(sell_orders.keys())):
            if ask_price > reservation_price - take_width:
                break
            ask_amount = abs(sell_orders[ask_price])
            quantity = min(ask_amount, position_limit - position - buy_order_volume)
            if quantity <= 0:
                break
            orders.append(Order(self.product, ask_price, quantity))
            buy_order_volume += quantity
            remaining = sell_orders[ask_price] + quantity
            if remaining == 0:
                del sell_orders[ask_price]
            else:
                sell_orders[ask_price] = remaining

        for bid_price in sorted(list(buy_orders.keys()), reverse=True):
            if bid_price < reservation_price + take_width:
                break
            bid_amount = buy_orders[bid_price]
            quantity = min(bid_amount, position_limit + position - sell_order_volume)
            if quantity <= 0:
                break
            orders.append(Order(self.product, bid_price, -quantity))
            sell_order_volume += quantity
            remaining = buy_orders[bid_price] - quantity
            if remaining == 0:
                del buy_orders[bid_price]
            else:
                buy_orders[bid_price] = remaining

        return buy_order_volume, sell_order_volume

    def make_orders(
        self,
        order_depth: OrderDepth,
        reservation_price: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        asks_above_res = [price for price in order_depth.sell_orders if price > reservation_price + disregard_edge]
        bids_below_res = [price for price in order_depth.buy_orders if price < reservation_price - disregard_edge]
        best_ask_above_res = min(asks_above_res) if asks_above_res else None
        best_bid_below_res = max(bids_below_res) if bids_below_res else None

        bid = reservation_price - default_edge
        ask = reservation_price + default_edge

        if best_ask_above_res is not None:
            ask = (
                best_ask_above_res
                if abs(best_ask_above_res - reservation_price) <= join_edge
                else best_ask_above_res - 1
            )
        if best_bid_below_res is not None:
            bid = (
                best_bid_below_res
                if abs(reservation_price - best_bid_below_res) <= join_edge
                else best_bid_below_res + 1
            )

        buy_order_volume, sell_order_volume = self.market_make(
            self.product,
            orders,
            round(bid),
            round(ask),
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def generate_orders(self, state: TradingState, trader_data: dict[str, Any]) -> List[Order]:
        if self.product not in state.order_depths:
            return []

        params = self.params[self.product]
        position = state.position.get(self.product, 0)
        fair_value, _ = self.estimate_fair_value(state.order_depths[self.product], state.timestamp, trader_data)
        reservation_price = self.compute_reservation_price(self.product, fair_value, position)
        product_state = trader_data.setdefault(self.product, {})
        product_state["last_reservation_price"] = reservation_price
        product_state["last_position"] = position

        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0
        sell_orders = dict(state.order_depths[self.product].sell_orders)
        buy_orders = dict(state.order_depths[self.product].buy_orders)

        buy_volume, sell_volume = self.take_orders(
            reservation_price,
            params["take_width"],
            orders,
            sell_orders,
            buy_orders,
            position,
            buy_volume,
            sell_volume,
        )

        clear_orders: List[Order] = []
        buy_volume, sell_volume = self.clear_position_order(
            self.product,
            reservation_price,
            params["clear_width"],
            clear_orders,
            sell_orders,
            buy_orders,
            position,
            buy_volume,
            sell_volume,
        )

        post_take_depth = self.build_depth_from_books(buy_orders, sell_orders)
        mm_orders, _, _ = self.make_orders(
            post_take_depth,
            reservation_price,
            position,
            buy_volume,
            sell_volume,
            params["disregard_edge"],
            params["join_edge"],
            params["default_edge"],
        )
        return orders + clear_orders + mm_orders


class PepperEngine(SharedBookOps):
    product = Product.INTARIAN_PEPPER_ROOT
    namespace = "pepper"

    def compute_gap_signal(self, order_depth: OrderDepth) -> float:
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        asks = sorted(order_depth.sell_orders.keys())
        if len(bids) < 2 or len(asks) < 2:
            return 0.0
        return float((bids[0] - bids[1]) - (asks[1] - asks[0]))

    def compute_book_state(self, order_depth: OrderDepth) -> dict[str, float] | None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_volume = abs(order_depth.buy_orders[best_bid])
        ask_volume = abs(order_depth.sell_orders[best_ask])
        mid = (best_bid + best_ask) / 2
        spread = max(1.0, best_ask - best_bid)
        denom = bid_volume + ask_volume
        microprice = (best_ask * bid_volume + best_bid * ask_volume) / denom if denom else mid
        imbalance = (bid_volume - ask_volume) / denom if denom else 0.0
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "mid": mid,
            "spread": spread,
            "microprice": microprice,
            "imbalance": imbalance,
            "gap_signal": self.compute_gap_signal(order_depth),
        }

    def estimate_fair_value(self, book: dict[str, float], timestamp: int, trader_data: Dict[str, Any]) -> float:
        params = self.params[self.product]
        last_timestamp = trader_data.get("last_timestamp")
        if last_timestamp is not None and timestamp < last_timestamp:
            trader_data.clear()

        observed_price = book["microprice"]
        base_observation = observed_price - params["price_slope"] * timestamp
        previous_base = trader_data.get("base_price")
        if previous_base is None:
            base_price = base_observation
        else:
            weight = params["base_update_weight"]
            base_price = (1 - weight) * previous_base + weight * base_observation

        trader_data["base_price"] = base_price
        trader_data["last_timestamp"] = timestamp
        fair_value = base_price + params["price_slope"] * timestamp
        trader_data["trend_fair"] = fair_value
        trader_data["trend_residual"] = observed_price - fair_value
        return fair_value

    def compute_alpha(
        self,
        book: dict[str, float],
        fair_value: float,
        timestamp: int,
        position: int,
        trader_data: Dict[str, Any],
    ) -> float:
        params = self.params[self.product]
        residual = book["microprice"] - fair_value
        remaining_time = max(0, params["max_timestamp"] - timestamp)
        forward_edge = params["price_slope"] * min(params["lookahead_timestamp"], remaining_time)
        alpha = (
            forward_edge
            - params["residual_weight"] * residual
            - params["gap_weight"] * book["gap_signal"]
            - params["inventory_skew"] * position
        )
        clipped_alpha = max(-params["alpha_clip"], min(params["alpha_clip"], alpha))
        trader_data["forward_edge"] = forward_edge
        trader_data["gap_signal"] = book["gap_signal"]
        trader_data["imbalance"] = book["imbalance"]
        trader_data["alpha"] = clipped_alpha
        return clipped_alpha

    def take_orders(
        self,
        reservation_price: float,
        take_width: float,
        orders: List[Order],
        sell_orders: Dict[int, int],
        buy_orders: Dict[int, int],
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        position_limit = self.limits[self.product]

        if sell_orders:
            best_ask = min(sell_orders)
            best_ask_amount = abs(sell_orders[best_ask])
            if best_ask <= reservation_price - take_width:
                quantity = min(best_ask_amount, position_limit - position - buy_order_volume)
                if quantity > 0:
                    orders.append(Order(self.product, best_ask, quantity))
                    buy_order_volume += quantity
                    remaining = sell_orders[best_ask] + quantity
                    if remaining == 0:
                        del sell_orders[best_ask]
                    else:
                        sell_orders[best_ask] = remaining

        if buy_orders:
            best_bid = max(buy_orders)
            best_bid_amount = buy_orders[best_bid]
            if best_bid >= reservation_price + take_width:
                quantity = min(best_bid_amount, position_limit + position - sell_order_volume)
                if quantity > 0:
                    orders.append(Order(self.product, best_bid, -quantity))
                    sell_order_volume += quantity
                    remaining = buy_orders[best_bid] - quantity
                    if remaining == 0:
                        del buy_orders[best_bid]
                    else:
                        buy_orders[best_bid] = remaining

        return buy_order_volume, sell_order_volume

    def make_orders(
        self,
        order_depth: OrderDepth,
        reservation_price: float,
        alpha: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        soft_position_limit: int,
    ) -> tuple[List[Order], int, int]:
        params = self.params[self.product]
        orders: List[Order] = []
        asks_above_res = [price for price in order_depth.sell_orders if price > reservation_price + disregard_edge]
        bids_below_res = [price for price in order_depth.buy_orders if price < reservation_price - disregard_edge]
        best_ask_above_res = min(asks_above_res) if asks_above_res else None
        best_bid_below_res = max(bids_below_res) if bids_below_res else None

        bid = reservation_price - default_edge
        ask = reservation_price + default_edge

        if best_ask_above_res is not None:
            ask = (
                best_ask_above_res
                if abs(best_ask_above_res - reservation_price) <= join_edge
                else best_ask_above_res - 1
            )
        if best_bid_below_res is not None:
            bid = (
                best_bid_below_res
                if abs(reservation_price - best_bid_below_res) <= join_edge
                else best_bid_below_res + 1
            )

        bid += max(alpha, 0.0) * params["bid_skew_strength"]
        ask += min(alpha, 0.0) * params["ask_skew_strength"]

        if position > soft_position_limit:
            ask -= 1
        elif position < -soft_position_limit:
            bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            self.product,
            orders,
            round(bid),
            round(ask),
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def generate_orders(self, state: TradingState, trader_data: dict[str, Any]) -> List[Order]:
        if self.product not in state.order_depths:
            return []

        params = self.params[self.product]
        position = state.position.get(self.product, 0)
        book = self.compute_book_state(state.order_depths[self.product])
        if book is None:
            return []

        fair_value = self.estimate_fair_value(book, state.timestamp, trader_data)
        alpha = self.compute_alpha(book, fair_value, state.timestamp, position, trader_data)
        reservation_price = fair_value + alpha

        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0
        sell_orders = dict(state.order_depths[self.product].sell_orders)
        buy_orders = dict(state.order_depths[self.product].buy_orders)

        take_width = max(0.0, params["take_width"] - params["take_alpha_scale"] * min(1.0, abs(alpha)))
        buy_volume, sell_volume = self.take_orders(
            reservation_price,
            take_width,
            orders,
            sell_orders,
            buy_orders,
            position,
            buy_volume,
            sell_volume,
        )

        clear_orders: List[Order] = []
        buy_volume, sell_volume = self.clear_position_order(
            self.product,
            reservation_price,
            params["clear_width"],
            clear_orders,
            sell_orders,
            buy_orders,
            position,
            buy_volume,
            sell_volume,
        )

        mm_orders, _, _ = self.make_orders(
            state.order_depths[self.product],
            reservation_price,
            alpha,
            position,
            buy_volume,
            sell_volume,
            params["disregard_edge"],
            params["join_edge"],
            params["default_edge"],
            params["soft_position_limit"],
        )
        return orders + clear_orders + mm_orders


class Trader:
    def __init__(self, params: Dict[str, Dict[str, float]] | None = None):
        self.params = params or PARAMS
        self.limits = dict(LIMITS)
        self.engines = {
            Product.ASH_COATED_OSMIUM: OsmiumEngine(self.params, self.limits),
            Product.INTARIAN_PEPPER_ROOT: PepperEngine(self.params, self.limits),
        }
        self.namespaces = {
            Product.ASH_COATED_OSMIUM: "osmium",
            Product.INTARIAN_PEPPER_ROOT: "pepper",
        }

    def load_trader_data(self, trader_data_raw: str) -> dict[str, dict[str, Any]]:
        empty = {"osmium": {}, "pepper": {}}
        if not trader_data_raw:
            return empty
        try:
            payload = json.loads(trader_data_raw)
        except json.JSONDecodeError:
            return empty
        if not isinstance(payload, dict):
            return empty

        namespaced = {
            "osmium": payload.get("osmium") if isinstance(payload.get("osmium"), dict) else None,
            "pepper": payload.get("pepper") if isinstance(payload.get("pepper"), dict) else None,
        }

        if namespaced["osmium"] is None:
            namespaced["osmium"] = payload if Product.ASH_COATED_OSMIUM in payload else {}
        if namespaced["pepper"] is None:
            namespaced["pepper"] = (
                payload
                if (
                    Product.INTARIAN_PEPPER_ROOT in payload
                    or "base_price" in payload
                    or "trend_fair" in payload
                    or "alpha" in payload
                )
                else {}
            )
        return namespaced

    def run(self, state: TradingState):
        trader_data = self.load_trader_data(state.traderData)
        result: Dict[str, List[Order]] = {product: [] for product in state.order_depths}

        for product, engine in self.engines.items():
            if product not in state.order_depths:
                continue
            namespace = self.namespaces[product]
            result[product] = engine.generate_orders(state, trader_data[namespace])

        for product in state.order_depths:
            result.setdefault(product, [])

        return result, 0, json.dumps(trader_data)
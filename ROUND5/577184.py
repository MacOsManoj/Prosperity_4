from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
import math
from typing import Any


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 2500

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = max(0, (self.max_log_length - base_length) // 3)
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Round 5 has 50 products, so the standard visualizer payload can exceed
        # the platform log parser's line limit. Keep stdout tiny and valid; the
        # actual orders are still returned by Trader.run().
        return [state.timestamp, trader_data, [], {}, [], [], {}, [{}, {}]]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {
            product: [
                obs.bidPrice, obs.askPrice, obs.transportFees, obs.exportTariff,
                obs.importTariff, obs.sugarPrice, obs.sunlightIndex,
            ]
            for product, obs in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        # Omit order echo from logs to prevent invalid truncated lambdaLog lines.
        return []

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                result = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return result


logger = Logger()


GROUPS = {
    "GALAXY": [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "SLEEP": [
        "SLEEP_POD_SUEDE",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_COTTON",
    ],
    "MICROCHIP": [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_TRIANGLE",
    ],
    "PEBBLES": ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
    "ROBOT": ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES", "ROBOT_LAUNDRY", "ROBOT_IRONING"],
    "UV_VISOR": ["UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE", "UV_VISOR_RED", "UV_VISOR_MAGENTA"],
    "TRANSLATOR": [
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ],
    "PANEL": ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"],
    "OXYGEN": [
        "OXYGEN_SHAKE_MORNING_BREATH",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_GARLIC",
    ],
    "SNACKPACK": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ],
}

PRODUCTS = [product for products in GROUPS.values() for product in products]
PRODUCT_TO_GROUP = {product: group for group, products in GROUPS.items() for product in products}
LIMIT = 10
LIMITS = {product: LIMIT for product in PRODUCTS}

DRIFT_BIAS = {
    "PANEL_2X4": 10.0,
    "MICROCHIP_OVAL": -10.0,
    "GALAXY_SOUNDS_BLACK_HOLES": 9.4,
    "OXYGEN_SHAKE_GARLIC": 9.0,
    "UV_VISOR_AMBER": -7.0,
    "SNACKPACK_CHOCOLATE": -5.3,
    "SNACKPACK_STRAWBERRY": 4.7,
    "UV_VISOR_RED": 4.7,
    "SNACKPACK_PISTACHIO": -4.4,
    "PEBBLES_S": -4.0,
    "MICROCHIP_SQUARE": 0.4,
    "PEBBLES_XL": 0.4,
    "PEBBLES_XS": -0.4,
}
DRIFT_PRODUCTS = set(DRIFT_BIAS)
DRIFT_SIGNAL_SCALE = {
    "MICROCHIP_SQUARE": 1.50,
    "UV_VISOR_RED": 1.50,
    "SNACKPACK_STRAWBERRY": 1.50,
    "PEBBLES_XL": 0.80,
    "PEBBLES_S": 0.80,
    "UV_VISOR_AMBER": 0.80,
    "SNACKPACK_PISTACHIO": 0.80,
    "SNACKPACK_CHOCOLATE": 0.80,
    "PEBBLES_XS": 0.80,
}

PRODUCT_SIGNAL_MULTIPLIER = {
    "PEBBLES_M": 0.35,
    "PEBBLES_S": 0.60,
    "PEBBLES_L": 0.60,
    "PANEL_1X2": 0.55,
    "GALAXY_SOUNDS_SOLAR_FLAMES": 0.55,
    "SLEEP_POD_LAMB_WOOL": 0.45,
    "UV_VISOR_MAGENTA": 0.60,
    "MICROCHIP_TRIANGLE": 0.70,
    "OXYGEN_SHAKE_MINT": 0.70,
    "UV_VISOR_RED": 0.75,
    "ROBOT_DISHES": 0.75,
}

CAUTIOUS_PRODUCTS = set(PRODUCT_SIGNAL_MULTIPLIER)
VERY_CAUTIOUS_PRODUCTS = {
    "PEBBLES_M",
    "PEBBLES_L",
    "PANEL_1X2",
    "SLEEP_POD_LAMB_WOOL",
    "UV_VISOR_MAGENTA",
    "MICROCHIP_TRIANGLE",
}
ULTRA_CAUTIOUS_PRODUCTS = {"PEBBLES_M"}
NO_TRADE_PRODUCTS = {
    "PANEL_1X2",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "OXYGEN_SHAKE_MINT",
    "PANEL_2X2",
    "SNACKPACK_VANILLA",
}
ROBOT_SHOCK_PRODUCTS = {"ROBOT_DISHES", "ROBOT_IRONING"}
ROBOT_SHOCK_THRESHOLD = 0.005
ROBOT_SHOCK_MAX_HOLD = 20
ROBOT_SHOCK_ENTRY_SIZE = 10

# Offline scan of day 2-4 showed product-level medium-horizon behavior that the
# pair/basket strategy was leaving on the table.  direction=+1 follows a move,
# direction=-1 fades it.  lookback controls the online EMA horizon, and strength
# is scaled from the out-of-sample t-stat bucket rather than hand-tuned per name.
INDIVIDUAL_ALPHA = {
    "MICROCHIP_TRIANGLE": (-1, 500, 3.2),
    "UV_VISOR_AMBER": (-1, 1000, 2.8),
    "GALAXY_SOUNDS_DARK_MATTER": (-1, 1000, 2.8),
    "GALAXY_SOUNDS_SOLAR_WINDS": (1, 1000, 2.4),
    "SNACKPACK_RASPBERRY": (-1, 1000, 2.4),
    "SLEEP_POD_NYLON": (-1, 1000, 2.4),
    "ROBOT_DISHES": (-1, 1000, 2.4),
    "MICROCHIP_RECTANGLE": (-1, 500, 2.3),
    "OXYGEN_SHAKE_MORNING_BREATH": (1, 500, 2.3),
    "SNACKPACK_PISTACHIO": (-1, 1000, 2.1),
    "TRANSLATOR_VOID_BLUE": (-1, 1000, 1.9),
    "SNACKPACK_STRAWBERRY": (-1, 1000, 1.9),
    "PEBBLES_S": (-1, 500, 1.9),
    "PANEL_1X4": (1, 200, 1.8),
    "ROBOT_LAUNDRY": (-1, 1000, 1.8),
    "PANEL_2X4": (-1, 1000, 1.8),
    "UV_VISOR_RED": (-1, 1000, 1.7),
    "PEBBLES_M": (-1, 500, 1.6),
    "TRANSLATOR_GRAPHITE_MIST": (-1, 1000, 1.6),
    "TRANSLATOR_ECLIPSE_CHARCOAL": (-1, 1000, 1.6),
    "TRANSLATOR_ASTRO_BLACK": (-1, 1000, 1.6),
    "ROBOT_VACUUMING": (-1, 1000, 1.6),
    "PEBBLES_XL": (-1, 100, 1.5),
    "ROBOT_MOPPING": (1, 200, 1.4),
    "OXYGEN_SHAKE_GARLIC": (-1, 500, 1.4),
    "MICROCHIP_OVAL": (-1, 500, 1.4),
    "PANEL_4X4": (1, 1000, 1.3),
    "PEBBLES_XS": (-1, 500, 1.3),
    "SNACKPACK_CHOCOLATE": (-1, 200, 1.3),
    "SLEEP_POD_POLYESTER": (-1, 200, 1.3),
    "SLEEP_POD_SUEDE": (1, 200, 1.2),
    "TRANSLATOR_SPACE_GRAY": (1, 200, 1.2),
    "UV_VISOR_YELLOW": (1, 200, 1.2),
    "GALAXY_SOUNDS_PLANETARY_RINGS": (1, 1000, 1.1),
    "GALAXY_SOUNDS_BLACK_HOLES": (1, 1000, 1.1),
    "MICROCHIP_CIRCLE": (1, 1000, 1.1),
    "OXYGEN_SHAKE_EVENING_BREATH": (-1, 200, 1.0),
    "SNACKPACK_VANILLA": (-1, 200, 1.0),
    "OXYGEN_SHAKE_CHOCOLATE": (-1, 50, 1.0),
    "UV_VISOR_MAGENTA": (-1, 1000, 1.0),
    "SLEEP_POD_COTTON": (1, 200, 0.9),
    "MICROCHIP_SQUARE": (-1, 500, 0.9),
    "PANEL_2X2": (-1, 200, 0.8),
    "SLEEP_POD_LAMB_WOOL": (-1, 500, 0.8),
    "PEBBLES_L": (-1, 500, 0.8),
    "UV_VISOR_ORANGE": (-1, 500, 0.6),
}

# asset_a = beta * asset_b + intercept + stationary-ish spread.
# If spread is high, sell asset_a and buy asset_b.  These came from the
# executed alpha notebooks; online EMA/std is used so they can adapt live.
SPREAD_CONFIGS = [
    ("PANEL_2X2", "PANEL_4X4", -0.807164, 17890.61179, 260.9, 0.70),
    ("UV_VISOR_AMBER", "UV_VISOR_MAGENTA", -1.240911, 21922.884435, 371.3, 0.55),
    ("OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_GARLIC", -0.406479, 13843.508074, 358.6, 0.50),
    ("UV_VISOR_YELLOW", "UV_VISOR_MAGENTA", 0.744220, 3006.040784, 406.2, 0.45),
    ("GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_SOLAR_FLAMES", 0.063507, 9559.949840, 326.5, 0.45),
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE", 0.292957, 6153.576042, 782.5, 0.45),
    ("ROBOT_LAUNDRY", "ROBOT_IRONING", 0.118598, 9134.334229, 273.8, 0.45),
    ("SLEEP_POD_NYLON", "SLEEP_POD_COTTON", -0.144960, 10970.990143, 319.4, 0.40),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", -0.869762, 18667.165060, 58.6, 0.50),
    ("SNACKPACK_VANILLA", "SNACKPACK_RASPBERRY", 0.012307, 9921.009379, 157.2, 0.35),
]

# leader return, after lag_steps rows of 100 timestamp units, predicts follower.
# direction=+1 follows leader; direction=-1 fades leader.
LEAD_LAG_CONFIGS = [
    ("SNACKPACK_VANILLA", "SNACKPACK_CHOCOLATE", 180, 1, 1.20),
    ("SNACKPACK_VANILLA", "SNACKPACK_RASPBERRY", 40, -1, 0.90),
    ("MICROCHIP_SQUARE", "MICROCHIP_OVAL", 100, 1, 0.90),
    ("MICROCHIP_OVAL", "MICROCHIP_SQUARE", 80, -1, 0.80),
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE", 140, 1, 0.75),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", 40, -1, 0.70),
    ("PEBBLES_S", "PEBBLES_M", 30, -1, 0.75),
    ("PANEL_2X4", "PANEL_1X4", 180, 1, 0.60),
    ("PANEL_2X4", "PANEL_4X4", 20, 1, 0.75),
    ("GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_BLACK_HOLES", 120, -1, 0.75),
    ("GALAXY_SOUNDS_SOLAR_WINDS", "GALAXY_SOUNDS_SOLAR_FLAMES", 70, 1, 0.55),
    ("SLEEP_POD_SUEDE", "SLEEP_POD_POLYESTER", 10, 1, 0.65),
    ("SLEEP_POD_SUEDE", "SLEEP_POD_COTTON", 30, 1, 0.55),
    ("ROBOT_MOPPING", "ROBOT_IRONING", 10, -1, 0.65),
    ("OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC", 10, 1, 0.60),
]

FLOW_CONFIGS = {
    ("PANEL", "BUY", 3): 1.40,
    ("SLEEP", "BUY", 2): 1.20,
    ("ROBOT", "SELL", 1): -1.10,
    ("ROBOT", "SELL", 3): -1.00,
    ("UV_VISOR", "BUY", 3): 1.00,
    ("TRANSLATOR", "SELL", 2): -1.00,
    ("MICROCHIP", "SELL", 2): -1.35,
    ("OXYGEN", "BUY", 1): 0.75,
    ("GALAXY", "BUY", 1): 0.65,
}

PRODUCT_FLOW_CONFIGS = {
    ("OXYGEN_SHAKE_GARLIC", "BUY", 1): 5.5,
    ("OXYGEN_SHAKE_GARLIC", "BUY", 2): 5.0,
    ("OXYGEN_SHAKE_GARLIC", "BUY", 3): 4.5,
    ("OXYGEN_SHAKE_GARLIC", "BUY", 4): 4.5,
    ("PEBBLES_XS", "SELL", 2): -5.0,
    ("PEBBLES_XS", "SELL", 3): -5.0,
    ("PEBBLES_XS", "SELL", 4): -6.0,
    ("PEBBLES_XS", "SELL", 5): -5.0,
    ("GALAXY_SOUNDS_BLACK_HOLES", "BUY", 1): 4.0,
    ("GALAXY_SOUNDS_BLACK_HOLES", "BUY", 2): 4.5,
    ("GALAXY_SOUNDS_BLACK_HOLES", "BUY", 4): 5.0,
    ("TRANSLATOR_SPACE_GRAY", "SELL", 1): -3.5,
    ("TRANSLATOR_SPACE_GRAY", "SELL", 3): -5.0,
    ("ROBOT_VACUUMING", "SELL", 2): -4.0,
    ("ROBOT_IRONING", "SELL", 2): -4.0,
    ("ROBOT_IRONING", "SELL", 3): -3.5,
    ("MICROCHIP_OVAL", "SELL", 1): -4.0,
    ("MICROCHIP_OVAL", "SELL", 2): -5.0,
    ("MICROCHIP_OVAL", "SELL", 3): -5.5,
    ("MICROCHIP_TRIANGLE", "SELL", 2): -5.0,
    ("SNACKPACK_STRAWBERRY", "BUY", 3): 3.5,
    ("PANEL_4X4", "SELL", 3): -4.5,
    ("PANEL_2X4", "BUY", 1): 3.5,
    ("PANEL_2X4", "BUY", 3): 4.5,
    ("PANEL_2X4", "BUY", 4): 4.0,
    ("SLEEP_POD_POLYESTER", "BUY", 2): 3.5,
    ("SLEEP_POD_POLYESTER", "BUY", 3): 4.5,
    ("SLEEP_POD_POLYESTER", "BUY", 4): 4.0,
    ("UV_VISOR_AMBER", "SELL", 3): -3.0,
    ("UV_VISOR_AMBER", "SELL", 4): -3.0,
    ("UV_VISOR_MAGENTA", "BUY", 1): 3.0,
    ("UV_VISOR_MAGENTA", "BUY", 4): 3.5,
    ("UV_VISOR_RED", "BUY", 3): 3.0,
    ("UV_VISOR_RED", "BUY", 4): 3.5,
    ("PANEL_1X2", "BUY", 1): 3.0,
    ("UV_VISOR_ORANGE", "SELL", 2): -3.0,
    ("SLEEP_POD_COTTON", "BUY", 3): 3.0,
    ("ROBOT_DISHES", "BUY", 2): 3.0,
}

MAX_RETURN_HISTORY = max(lag for _, _, lag, _, _ in LEAD_LAG_CONFIGS) + 5
RETURN_TRACK_PRODUCTS = sorted({
    "SLEEP_POD_LAMB_WOOL",
    *(leader for leader, _, _, _, _ in LEAD_LAG_CONFIGS),
})
RETURN_HISTORY_LIMIT = {
    symbol: max([lag for leader, _, lag, _, _ in LEAD_LAG_CONFIGS if leader == symbol] + [120 if symbol == "SLEEP_POD_LAMB_WOOL" else 0]) + 5
    for symbol in RETURN_TRACK_PRODUCTS
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def best_bid_ask(depth: OrderDepth) -> tuple[int | None, int | None]:
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    return best_bid, best_ask


def mid_price(depth: OrderDepth) -> float | None:
    best_bid, best_ask = best_bid_ask(depth)
    if best_bid is None or best_ask is None:
        return None
    return 0.5 * (best_bid + best_ask)


def top_obi(depth: OrderDepth) -> float:
    best_bid, best_ask = best_bid_ask(depth)
    if best_bid is None or best_ask is None:
        return 0.0
    bid_vol = abs(depth.buy_orders.get(best_bid, 0))
    ask_vol = abs(depth.sell_orders.get(best_ask, 0))
    denom = bid_vol + ask_vol
    return 0.0 if denom <= 0 else (bid_vol - ask_vol) / denom


class ProductBook:
    def __init__(self, symbol: str, state: TradingState, limit: int = LIMIT):
        self.symbol = symbol
        self.depth = state.order_depths.get(symbol, OrderDepth())
        self.limit = limit
        self.position = state.position.get(symbol, 0)
        self.pending = 0
        self.pending_buy = 0
        self.pending_sell = 0
        self.orders: list[Order] = []
        self.buy_orders = dict(sorted(self.depth.buy_orders.items(), reverse=True))
        self.sell_orders = dict(sorted(self.depth.sell_orders.items()))
        self.best_bid, self.best_ask = best_bid_ask(self.depth)
        self.mid = mid_price(self.depth)

    def projected_position(self) -> int:
        return self.position + self.pending

    def room_to_buy(self) -> int:
        return max(0, self.limit - self.position - self.pending_buy)

    def room_to_sell(self) -> int:
        return max(0, self.limit + self.position - self.pending_sell)

    def buy(self, price: int, quantity: int) -> int:
        quantity = min(int(quantity), self.room_to_buy())
        if quantity <= 0:
            return 0
        self.orders.append(Order(self.symbol, int(price), quantity))
        self.pending += quantity
        self.pending_buy += quantity
        return quantity

    def sell(self, price: int, quantity: int) -> int:
        quantity = min(int(quantity), self.room_to_sell())
        if quantity <= 0:
            return 0
        self.orders.append(Order(self.symbol, int(price), -quantity))
        self.pending -= quantity
        self.pending_sell += quantity
        return quantity


class Trader:
    def fresh_memory(self) -> dict[str, Any]:
        return {
            "last_mid": {},
            "ret_hist": {},
            "spread": {},
            "flow": {},
            "product_flow": {},
            "ema": {},
            "drift_guard": {},
            "robot_shock": {},
            "last_ts": None,
            "tick": 0,
        }

    def ensure_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        defaults = self.fresh_memory()
        for key, value in defaults.items():
            memory.setdefault(key, value)
        return memory

    def load_memory(self, trader_data: str) -> dict[str, Any]:
        if trader_data:
            try:
                decoded = json.loads(trader_data)
                if isinstance(decoded, dict):
                    return self.ensure_memory(decoded)
            except Exception:
                pass
        return self.fresh_memory()

    def dump_memory(self, memory: dict[str, Any]) -> str:
        # Keep only compact rolling state. This stays under the platform log limit.
        return json.dumps(memory, separators=(",", ":"))

    def update_mid_returns(self, mids: dict[str, float], memory: dict[str, Any]) -> None:
        last_mid = memory.setdefault("last_mid", {})
        ret_hist = memory.setdefault("ret_hist", {})
        for symbol, mid in mids.items():
            if symbol not in RETURN_TRACK_PRODUCTS:
                continue
            prev = last_mid.get(symbol)
            keep = RETURN_HISTORY_LIMIT.get(symbol, MAX_RETURN_HISTORY)
            ret = 0.0
            if isinstance(prev, (int, float)) and prev > 0 and mid > 0:
                ret = math.log(mid / float(prev))
            hist = [int(x) for x in ret_hist.get(symbol, [])[-keep + 1:]]
            hist.append(int(round(ret * 1_000_000)))
            ret_hist[symbol] = hist
            last_mid[symbol] = round(mid, 4)
        for symbol in list(last_mid.keys()):
            if symbol not in RETURN_TRACK_PRODUCTS:
                last_mid.pop(symbol, None)
        for symbol in list(ret_hist.keys()):
            if symbol not in RETURN_TRACK_PRODUCTS:
                ret_hist.pop(symbol, None)

    def decay_flow(self, memory: dict[str, Any]) -> None:
        flow = memory.setdefault("flow", {})
        product_flow = memory.setdefault("product_flow", {})
        for key in list(flow.keys()):
            value = 0.72 * float(flow.get(key, 0.0))
            if abs(value) < 0.03:
                flow.pop(key, None)
            else:
                flow[key] = round(value, 4)
        for symbol in list(product_flow.keys()):
            value = 0.9965 * float(product_flow.get(symbol, 0.0))
            if abs(value) < 0.05:
                product_flow.pop(symbol, None)
            else:
                product_flow[symbol] = round(value, 4)

    def infer_trade_side(self, trade: Trade, book: ProductBook) -> str | None:
        if book.best_ask is not None and trade.price >= book.best_ask:
            return "BUY"
        if book.best_bid is not None and trade.price <= book.best_bid:
            return "SELL"
        if book.mid is not None:
            if trade.price > book.mid:
                return "BUY"
            if trade.price < book.mid:
                return "SELL"
        return None

    def ingest_market_trades(self, state: TradingState, books: dict[str, ProductBook], memory: dict[str, Any]) -> None:
        self.decay_flow(memory)
        flow = memory.setdefault("flow", {})
        product_flow = memory.setdefault("product_flow", {})
        for symbol, trades in state.market_trades.items():
            if symbol not in PRODUCT_TO_GROUP or symbol not in books:
                continue
            group = PRODUCT_TO_GROUP[symbol]
            for trade in trades:
                side = self.infer_trade_side(trade, books[symbol])
                if side is None:
                    continue
                key = f"{group}:{side}:{int(trade.quantity)}"
                contrib = FLOW_CONFIGS.get((group, side, int(trade.quantity)))
                if contrib is None:
                    contrib = 0.12 if side == "BUY" else -0.12
                flow[key] = round(float(flow.get(key, 0.0)) + contrib * max(1, int(trade.quantity)), 4)
                product_contrib = PRODUCT_FLOW_CONFIGS.get((symbol, side, int(trade.quantity)))
                if product_contrib is not None:
                    product_flow[symbol] = round(float(product_flow.get(symbol, 0.0)) + product_contrib, 4)

    def add_pebbles_basket_signal(self, mids: dict[str, float], signals: dict[str, float]) -> None:
        products = GROUPS["PEBBLES"]
        if not all(p in mids for p in products):
            return
        basket = sum(mids[p] for p in products)
        z = (basket - 50000.0) / 2.8
        z = clamp(z, -5.0, 5.0)
        if abs(z) < 1.15:
            return
        # Sum too high: all components are collectively rich.
        skew = -1.15 * z
        for p in products:
            signals[p] += skew

    def add_spread_signals(self, mids: dict[str, float], signals: dict[str, float], memory: dict[str, Any]) -> None:
        spread_mem = memory.setdefault("spread", {})
        for a, b, beta, intercept, default_std, weight in SPREAD_CONFIGS:
            if a not in mids or b not in mids:
                continue
            spread = mids[a] - beta * mids[b] - intercept
            key = f"{a}|{b}"
            item = spread_mem.get(key, {"ema": spread, "var": default_std * default_std})
            ema = float(item.get("ema", spread))
            var = max(25.0, float(item.get("var", default_std * default_std)))
            alpha = 0.025
            resid = spread - ema
            ema = ema + alpha * resid
            var = (1.0 - alpha) * var + alpha * resid * resid
            spread_mem[key] = {"ema": round(ema, 4), "var": round(var, 4)}
            z = clamp((spread - ema) / max(default_std * 0.35, math.sqrt(var)), -4.0, 4.0)
            if abs(z) < 0.75:
                continue
            edge = weight * z
            signals[a] -= edge
            signals[b] += edge * clamp(abs(beta), 0.25, 1.25)

    def add_lead_lag_signals(self, signals: dict[str, float], memory: dict[str, Any]) -> None:
        ret_hist = memory.setdefault("ret_hist", {})
        for leader, follower, lag, direction, weight in LEAD_LAG_CONFIGS:
            hist = ret_hist.get(leader, [])
            if len(hist) <= lag:
                continue
            leader_ret = float(hist[-lag]) / 1_000_000.0
            # Convert log return into a tick-sized fair skew. Clamp keeps noisy
            # random walks from overwhelming the stronger reversion signals.
            skew = clamp(leader_ret * 10000.0, -2.0, 2.0) * direction * weight
            if abs(skew) >= 0.12:
                signals[follower] += skew

    def add_flow_signals(self, signals: dict[str, float], memory: dict[str, Any]) -> None:
        flow = memory.setdefault("flow", {})
        for key, raw_value in flow.items():
            try:
                group, side, qty_raw = key.split(":")
                qty = int(qty_raw)
            except Exception:
                continue
            value = float(raw_value)
            base = FLOW_CONFIGS.get((group, side, qty))
            if base is None:
                continue
            group_signal = clamp(value / max(1.0, abs(base) * 5.0), -2.5, 2.5)
            for product in GROUPS.get(group, []):
                signals[product] += group_signal
        product_flow = memory.setdefault("product_flow", {})
        for symbol, value in product_flow.items():
            if symbol in signals:
                signals[symbol] += clamp(float(value), -7.0, 7.0)

    def add_individual_alpha(self, mids: dict[str, float], signals: dict[str, float], memory: dict[str, Any]) -> None:
        ema_state = memory.setdefault("ema", {})
        tick = int(memory.get("tick", 0))
        for symbol, mid in mids.items():
            config = INDIVIDUAL_ALPHA.get(symbol)
            if config is None:
                continue
            direction, lookback, strength = config
            alpha = 2.0 / (lookback + 1.0)
            item = ema_state.get(symbol)
            if not isinstance(item, dict):
                item = {"ema": mid, "dev": 25.0, "n": 0}
            ema = float(item.get("ema", mid))
            dev = max(6.0, float(item.get("dev", 25.0)))
            n = int(item.get("n", 0)) + 1
            gap = mid - ema
            ema += alpha * gap
            dev = (1.0 - alpha) * dev + alpha * abs(gap)
            ema_state[symbol] = {"ema": round(ema, 3), "dev": round(dev, 3), "n": min(n, 5000)}
            warmup = max(30, min(250, lookback // 3))
            if tick < warmup:
                continue
            z = clamp(gap / max(dev, 4.0), -5.0, 5.0)
            if abs(z) < 0.45:
                continue
            signals[symbol] += direction * strength * z

    def add_lamb_wool_reversal(self, signals: dict[str, float], memory: dict[str, Any]) -> None:
        hist = memory.setdefault("ret_hist", {}).get("SLEEP_POD_LAMB_WOOL", [])
        if len(hist) < 120:
            return
        run = sum(float(x) for x in hist[-120:]) / 1_000_000.0
        # Notebook found strongest at 200 blocks, but live memory is constrained.
        # This shorter proxy fades sustained Lamb Wool runs.
        signals["SLEEP_POD_LAMB_WOOL"] += clamp(-run * 3500.0, -1.0, 1.0)

    def guarded_drift_bias(self, symbol: str, mid: float, memory: dict[str, Any]) -> float:
        bias = DRIFT_SIGNAL_SCALE.get(symbol, 1.0) * DRIFT_BIAS.get(symbol, 0.0)
        if bias == 0.0:
            return 0.0
        if symbol in {"PANEL_2X4", "GALAXY_SOUNDS_BLACK_HOLES"}:
            tick = int(memory.get("tick", 0))
            if tick < 80:
                bias *= tick / 80.0

        guard = memory.setdefault("drift_guard", {})
        item = guard.get(symbol)
        if not isinstance(item, dict):
            item = {"fast": mid, "slow": mid, "dev": 35.0, "bad": 0}

        fast = float(item.get("fast", mid))
        slow = float(item.get("slow", mid))
        dev = max(10.0, float(item.get("dev", 35.0)))
        bad = int(item.get("bad", 0))

        fast += 0.060 * (mid - fast)
        slow += 0.008 * (mid - slow)
        trend = fast - slow
        dev = 0.985 * dev + 0.015 * abs(mid - slow)

        signed_trend = trend if bias > 0 else -trend
        danger = max(45.0, 1.4 * dev)
        recovery = max(12.0, 0.4 * dev)
        if signed_trend < -danger:
            bad = min(80, bad + 3)
        elif signed_trend > recovery:
            bad = max(0, bad - 3)
        else:
            bad = max(0, bad - 1)

        item.update({"fast": round(fast, 3), "slow": round(slow, 3), "dev": round(dev, 3), "bad": bad})
        guard[symbol] = item

        if bad >= 35:
            return 0.20 * bias
        if bad >= 14:
            return 0.55 * bias
        return bias

    def add_generic_inventory_and_obi(self, books: dict[str, ProductBook], signals: dict[str, float], memory: dict[str, Any]) -> None:
        for symbol, book in books.items():
            if book.mid is None:
                continue
            signals[symbol] += self.guarded_drift_bias(symbol, book.mid, memory)

    def build_signals(self, books: dict[str, ProductBook], memory: dict[str, Any]) -> dict[str, float]:
        mids = {symbol: book.mid for symbol, book in books.items() if book.mid is not None}
        self.update_mid_returns(mids, memory)
        signals = {symbol: 0.0 for symbol in PRODUCTS}
        self.add_pebbles_basket_signal(mids, signals)
        self.add_spread_signals(mids, signals, memory)
        self.add_lead_lag_signals(signals, memory)
        self.add_flow_signals(signals, memory)
        self.add_individual_alpha(mids, signals, memory)
        self.add_lamb_wool_reversal(signals, memory)
        self.add_generic_inventory_and_obi(books, signals, memory)
        return {
            symbol: clamp(value * PRODUCT_SIGNAL_MULTIPLIER.get(symbol, 1.0), -28.0, 28.0)
            for symbol, value in signals.items()
        }

    def trade_product(self, book: ProductBook, signal: float) -> None:
        if book.mid is None or book.best_bid is None or book.best_ask is None:
            return
        spread = max(1, book.best_ask - book.best_bid)
        fair = book.mid + signal
        abs_signal = abs(signal)
        is_pebble = PRODUCT_TO_GROUP.get(book.symbol) == "PEBBLES"
        is_cautious = book.symbol in CAUTIOUS_PRODUCTS
        is_very_cautious = book.symbol in VERY_CAUTIOUS_PRODUCTS
        is_ultra_cautious = book.symbol in ULTRA_CAUTIOUS_PRODUCTS
        take_edge = 0.7 if abs_signal >= 2.2 else 1.2
        max_take = 3 if abs_signal < 3.0 else 5
        if abs_signal >= 6.0:
            take_edge = 0.35
            max_take = 10
        if is_pebble:
            take_edge = 1.6
            max_take = 2 if abs_signal >= 3.0 else 0
        if is_cautious:
            take_edge += 0.8
            max_take = min(max_take, 1 if abs_signal >= 3.5 else 0)
        if is_very_cautious:
            take_edge += 1.0
            max_take = min(max_take, 1 if abs_signal >= 5.0 else 0)
        if is_ultra_cautious:
            take_edge += 1.0
            max_take = 0 if abs_signal < 6.5 else max_take

        if max_take > 0:
            for ask_price, ask_volume in book.sell_orders.items():
                if ask_price > fair - take_edge:
                    break
                size = min(abs(ask_volume), max_take)
                book.buy(ask_price, size)

            for bid_price, bid_volume in book.buy_orders.items():
                if bid_price < fair + take_edge:
                    break
                size = min(abs(bid_volume), max_take)
                book.sell(bid_price, size)

        quote_edge = 1.0 if spread <= 4 else 1.6
        if abs_signal >= 3.0:
            quote_edge = 1.0
        if is_cautious:
            quote_edge += 0.7
        if is_very_cautious:
            quote_edge += 1.8
        if is_ultra_cautious:
            quote_edge += 2.0

        bid_quote = min(book.best_ask - 1, math.floor(fair - quote_edge))
        ask_quote = max(book.best_bid + 1, math.ceil(fair + quote_edge))
        base_size = 1
        if abs_signal >= 1.5:
            base_size = 2
        if abs_signal >= 3.0:
            base_size = 3
        if abs_signal >= 5.0:
            base_size = 5
        if abs_signal >= 8.0:
            base_size = 8
        if is_pebble:
            base_size = 1
            quote_edge += 0.8

        # Quote both sides for every product, but skew size toward the signal.
        if bid_quote < book.best_ask:
            buy_size = base_size + (1 if signal > 1.0 else 0)
            book.buy(int(bid_quote), buy_size)
        if ask_quote > book.best_bid:
            sell_size = base_size + (1 if signal < -1.0 else 0)
            book.sell(int(ask_quote), sell_size)

    def trade_robot_shock(self, book: ProductBook, memory: dict[str, Any]) -> None:
        if book.mid is None or book.best_bid is None or book.best_ask is None:
            return
        shock_mem = memory.setdefault("robot_shock", {})
        item = shock_mem.setdefault(book.symbol, {"last_mid": book.mid, "anchor": None, "hold": 0})
        last_mid = float(item.get("last_mid", book.mid))
        log_ret = math.log(book.mid / last_mid) if last_mid > 0 else 0.0
        shock = abs(log_ret) > ROBOT_SHOCK_THRESHOLD
        pos = book.position

        if pos != 0 and not shock:
            hold = int(item.get("hold", 0)) + 1
            item["hold"] = hold
            anchor = item.get("anchor")
            if not isinstance(anchor, (int, float)):
                if pos > 0:
                    book.sell(book.best_bid, abs(pos))
                else:
                    book.buy(book.best_ask, abs(pos))
            elif hold >= ROBOT_SHOCK_MAX_HOLD:
                if pos > 0:
                    book.sell(book.best_bid, abs(pos))
                else:
                    book.buy(book.best_ask, abs(pos))
            else:
                if pos > 0:
                    book.sell(max(int(round(anchor)), book.best_bid + 1), abs(pos))
                else:
                    book.buy(min(int(round(anchor)), book.best_ask - 1), abs(pos))

        if shock:
            item["anchor"] = last_mid
            item["hold"] = 0
            if log_ret > 0:
                target_pos = -ROBOT_SHOCK_ENTRY_SIZE
                delta = target_pos - book.position
                if delta < 0:
                    book.sell(book.best_bid, -delta)
            else:
                target_pos = ROBOT_SHOCK_ENTRY_SIZE
                delta = target_pos - book.position
                if delta > 0:
                    book.buy(book.best_ask, delta)

        if book.position == 0 and not shock:
            item["anchor"] = None
            item["hold"] = 0
        item["last_mid"] = round(book.mid, 4)

    def trim_orders(self, symbol: str, raw_orders: list[Order], position: int) -> list[Order]:
        trimmed: list[Order] = []
        buy_total = 0
        sell_total = 0
        for order in raw_orders:
            qty = int(order.quantity)
            if qty > 0:
                room = LIMITS[symbol] - position - buy_total
                allowed = min(qty, max(0, room))
                if allowed > 0:
                    trimmed.append(Order(symbol, int(order.price), allowed))
                    buy_total += allowed
            elif qty < 0:
                room = LIMITS[symbol] + position - sell_total
                allowed = min(-qty, max(0, room))
                if allowed > 0:
                    trimmed.append(Order(symbol, int(order.price), -allowed))
                    sell_total += allowed
        return trimmed

    def run(self, state: TradingState):
        memory = self.load_memory(state.traderData)
        memory["tick"] = int(memory.get("tick", 0)) + 1
        memory["last_ts"] = state.timestamp

        books = {symbol: ProductBook(symbol, state, LIMITS[symbol]) for symbol in PRODUCTS if symbol in state.order_depths}
        self.ingest_market_trades(state, books, memory)
        signals = self.build_signals(books, memory)

        orders: dict[Symbol, list[Order]] = {}
        active = []
        for symbol in PRODUCTS:
            book = books.get(symbol)
            if book is None:
                orders[symbol] = []
                continue
            if symbol in NO_TRADE_PRODUCTS:
                orders[symbol] = []
                continue
            if symbol in ROBOT_SHOCK_PRODUCTS:
                self.trade_robot_shock(book, memory)
                orders[symbol] = book.orders
                continue
            self.trade_product(book, signals.get(symbol, 0.0))
            orders[symbol] = book.orders
            if abs(signals.get(symbol, 0.0)) > 1.5:
                active.append(f"{symbol}:{signals[symbol]:.1f}")

        if active:
            logger.print("signals", ",".join(active[:12]))
        for symbol, symbol_orders in list(orders.items()):
            orders[symbol] = self.trim_orders(symbol, symbol_orders, state.position.get(symbol, 0))
        conversions = 0
        trader_data = self.dump_memory(memory)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
import math
from typing import Any


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
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                for trade_list in trades.values() for trade in trade_list]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {
            product: [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sugarPrice,
                obs.sunlightIndex,
            ]
            for product, obs in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[order.symbol, order.price, order.quantity] for order_list in orders.values() for order in order_list]

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

HYDRO = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
OPTION_BY_STRIKE = {strike: f"VEV_{strike}" for strike in STRIKES}

LIMIT = {HYDRO: 200, VELVET: 200}
for _strike in STRIKES:
    LIMIT[OPTION_BY_STRIKE[_strike]] = 300

ENABLE_OPTIONS = True
CP_DECAY = 0.90
ANCHOR = {HYDRO: 9990.95, VELVET: 5250.71}

LOSER_BUYERS = {
    HYDRO: {"Mark 38"},
    VELVET: {"Mark 55", "Mark 67"},
    "VEV_4000": {"Mark 38"},
    "VEV_5300": {"Mark 01"},
    "VEV_5400": {"Mark 01"},
    "VEV_5500": {"Mark 01"},
}
LOSER_SELLERS = {
    HYDRO: {"Mark 38"},
    VELVET: {"Mark 55"},
    "VEV_4000": {"Mark 38"},
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def best_bid_ask(depth: OrderDepth) -> tuple[int | None, int | None]:
    bid = max(depth.buy_orders) if depth.buy_orders else None
    ask = min(depth.sell_orders) if depth.sell_orders else None
    return bid, ask


def mid_price(depth: OrderDepth) -> float | None:
    bid, ask = best_bid_ask(depth)
    if bid is None or ask is None:
        return None
    return 0.5 * (bid + ask)


def append(values: list[Any], value: float, limit: int) -> list[float]:
    result = [float(v) for v in values[-limit + 1:]]
    result.append(round(float(value), 4))
    return result


def rolling_z(memory: dict[str, Any], bucket: str, key: str, value: float, length: int = 120, warmup: int = 30) -> tuple[float, float, float, int]:
    table = memory.setdefault(bucket, {})
    hist = [float(v) for v in table.get(key, [])]
    if len(hist) >= warmup:
        mean = sum(hist) / len(hist)
        var = sum((x - mean) ** 2 for x in hist) / max(1, len(hist) - 1)
        std = math.sqrt(max(var, 1e-6))
        z = (value - mean) / max(std, 0.08)
    else:
        mean, std, z = value, 1.0, 0.0
    table[key] = append(hist, value, length)
    return clamp(z, -4.0, 4.0), mean, std, len(hist)


def room_to_buy(state: TradingState, symbol: str, pending: int = 0) -> int:
    return LIMIT[symbol] - state.position.get(symbol, 0) - pending


def room_to_sell(state: TradingState, symbol: str, pending: int = 0) -> int:
    return LIMIT[symbol] + state.position.get(symbol, 0) + pending


class Trader:
    def fresh_memory(self) -> dict[str, Any]:
        return {
            "tick": 0,
            "ema": {},
            "last_mid": {},
            "hist": {},
            "cp": {},
        }

    def load_memory(self, data: str) -> dict[str, Any]:
        if data:
            try:
                loaded = json.loads(data)
                if isinstance(loaded, dict):
                    base = self.fresh_memory()
                    for key, value in base.items():
                        loaded.setdefault(key, value)
                    return loaded
            except Exception:
                pass
        return self.fresh_memory()

    def save_memory(self, memory: dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def update_counterparty_flow(self, state: TradingState, memory: dict[str, Any]) -> None:
        cp = memory.setdefault("cp", {})
        for symbol, flow in list(cp.items()):
            flow["buy"] = round(float(flow.get("buy", 0.0)) * CP_DECAY, 4)
            flow["sell"] = round(float(flow.get("sell", 0.0)) * CP_DECAY, 4)
            if flow["buy"] + flow["sell"] < 0.05:
                cp.pop(symbol, None)

        for trades in state.market_trades.values():
            for trade in trades:
                flow = cp.setdefault(trade.symbol, {"buy": 0.0, "sell": 0.0})
                if trade.seller in LOSER_SELLERS.get(trade.symbol, set()):
                    flow["buy"] = round(float(flow.get("buy", 0.0)) + trade.quantity, 4)
                if trade.buyer in LOSER_BUYERS.get(trade.symbol, set()):
                    flow["sell"] = round(float(flow.get("sell", 0.0)) + trade.quantity, 4)

    def flow(self, memory: dict[str, Any], symbol: str) -> tuple[float, float]:
        flow = memory.get("cp", {}).get(symbol, {})
        return float(flow.get("buy", 0.0)), float(flow.get("sell", 0.0))

    def trade_mean_reverter(self, state: TradingState, memory: dict[str, Any], symbol: str, base_edge: float) -> list[Order]:
        depth = state.order_depths.get(symbol)
        if depth is None:
            return []
        bid, ask = best_bid_ask(depth)
        mid = mid_price(depth)
        if bid is None or ask is None or mid is None:
            return []

        prev_mid = float(memory["last_mid"].get(symbol, mid))
        memory["last_mid"][symbol] = mid
        ema = float(memory["ema"].get(symbol, ANCHOR[symbol]))
        ema = 0.985 * ema + 0.015 * mid
        memory["ema"][symbol] = ema
        momentum = mid - prev_mid
        anchor = ANCHOR[symbol]
        z, _, _, hist_len = rolling_z(memory, "hist", symbol, mid - anchor, 200, 60)
        buy_flow, sell_flow = self.flow(memory, symbol)

        if symbol == HYDRO:
            fair = mid + 0.42 * (anchor - ema) - 0.10 * momentum + clamp((buy_flow - sell_flow) / 12.0, -2.0, 2.0)
            quote_size = 16
        else:
            fair = mid + 0.16 * (anchor - mid) - 0.12 * momentum + clamp((buy_flow - sell_flow) / 12.0, -2.0, 2.0)
            memory["velvet_edge"] = round(fair - mid, 4)
            memory["velvet_z"] = round(float(z), 4)
            quote_size = 10
        pos = state.position.get(symbol, 0)
        orders: list[Order] = []
        pending = 0

        if ask <= fair - base_edge and room_to_buy(state, symbol, pending) > 0:
            qty = min(abs(depth.sell_orders[ask]), room_to_buy(state, symbol, pending), 26)
            orders.append(Order(symbol, ask, qty))
            pending += qty
        if bid >= fair + base_edge and room_to_sell(state, symbol, pending) > 0:
            qty = min(abs(depth.buy_orders[bid]), room_to_sell(state, symbol, pending), 26)
            orders.append(Order(symbol, bid, -qty))
            pending -= qty

        if hist_len >= 20 and abs(pos + pending) < 175:
            if mid < fair - 1.0 and room_to_buy(state, symbol, pending) > 0:
                price = min(bid + 1, ask - 1)
                if price < ask:
                    qty = min(room_to_buy(state, symbol, pending), max(3, quote_size - max(0, pos) // 15))
                    orders.append(Order(symbol, price, qty))
                    pending += qty
            elif mid > fair + 1.0 and room_to_sell(state, symbol, pending) > 0:
                price = max(ask - 1, bid + 1)
                if price > bid:
                    qty = min(room_to_sell(state, symbol, pending), max(3, quote_size + min(0, pos) // 15))
                    orders.append(Order(symbol, price, -qty))
                    pending -= qty

        return orders

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, spot: float, strike: int, t: float, sigma: float) -> float:
        if t <= 0 or sigma <= 0:
            return max(0.0, spot - strike)
        vol = sigma * math.sqrt(t)
        d1 = (math.log(max(spot, 1e-9) / strike) + 0.5 * sigma * sigma * t) / vol
        d2 = d1 - vol
        return spot * self.norm_cdf(d1) - strike * self.norm_cdf(d2)

    def bs_delta(self, spot: float, strike: int, t: float, sigma: float = 0.22) -> float:
        if t <= 0 or sigma <= 0:
            return 1.0 if spot >= strike else 0.0
        vol = sigma * math.sqrt(t)
        d1 = (math.log(max(spot, 1e-9) / strike) + 0.5 * sigma * sigma * t) / vol
        return self.norm_cdf(d1)

    def avellaneda_stoikov(self, mid: float, pos: int, sigma: float, gamma: float = 0.002, kappa: float = 1.5) -> tuple[float, float]:
        sigma = max(sigma, 0.5)
        reservation = mid - pos * gamma * sigma * sigma
        half_spread = (gamma * sigma * sigma + (2.0 / gamma) * math.log(1.0 + gamma / kappa)) / 2.0
        half_spread = max(half_spread, 1.0)
        return reservation - half_spread, reservation + half_spread

    def trade_options(self, state: TradingState, memory: dict[str, Any], spot: float | None) -> dict[str, list[Order]]:
        result: dict[str, list[Order]] = {}
        if spot is None or not ENABLE_OPTIONS:
            return result

        velvet_edge = float(memory.get("velvet_edge", 0.0))
        velvet_z = float(memory.get("velvet_z", 0.0))
        t = max(0.2, 4.0 - state.timestamp / 1_000_000.0) / 365.0

        SIGNAL_Z = 1.1   # velvet z-score threshold for directional mode
        AS_GAMMA = 0.002
        AS_KAPPA = 1.5
        DIR_CAP = 300
        MM_CAP = 80

        for strike in STRIKES:
            symbol = OPTION_BY_STRIKE[strike]
            if strike in {6000, 6500}:
                result[symbol] = []
                continue
            depth = state.order_depths.get(symbol)
            if depth is None:
                result[symbol] = []
                continue
            bid, ask = best_bid_ask(depth)
            mid = mid_price(depth)
            if bid is None or ask is None or mid is None:
                result[symbol] = []
                continue

            orders: list[Order] = []
            pending = 0
            pos = state.position.get(symbol, 0)
            spread = ask - bid
            buy_flow, sell_flow = self.flow(memory, symbol)

            # Option fair value = mid + delta * velvet_edge
            # Same mean-reversion signal as VELVET, scaled by delta.
            # delta = self.bs_delta(spot, strike, t)
            delta = 1.4 if strike in {5000, 5100} else 1.5
            fair = mid + delta * velvet_edge
            base_edge = max(delta * 1.4, 1.0)
            quote_size = max(2, int(delta * 10))

            if abs(velvet_z) >= SIGNAL_Z:
                # Directional: mirrors trade_mean_reverter, scaled by delta
                max_qty = 40
                if ask <= fair - base_edge and room_to_buy(state, symbol, pending) > 0 and pos + pending < DIR_CAP:
                    qty = min(abs(depth.sell_orders[ask]), room_to_buy(state, symbol, pending), max_qty, DIR_CAP - pos - pending)
                    if qty > 0:
                        orders.append(Order(symbol, ask, qty))
                        pending += qty
                if bid >= fair + base_edge and room_to_sell(state, symbol, pending) > 0 and pos + pending > -DIR_CAP:
                    qty = min(abs(depth.buy_orders[bid]), room_to_sell(state, symbol, pending), max_qty, DIR_CAP + pos + pending)
                    if qty > 0:
                        orders.append(Order(symbol, bid, -qty))
                        pending -= qty

                # Passive follow-up inside spread, inventory-adjusted
                if spread >= 2 and abs(pos + pending) < DIR_CAP * 0.9:
                    if mid < fair - 1.0 and room_to_buy(state, symbol, pending) > 0:
                        price = min(bid + 1, ask - 1)
                        if price < ask:
                            qty = min(room_to_buy(state, symbol, pending), max(2, quote_size - max(0, pos) // 15))
                            if qty > 0:
                                orders.append(Order(symbol, price, qty))
                                pending += qty
                    elif mid > fair + 1.0 and room_to_sell(state, symbol, pending) > 0:
                        price = max(ask - 1, bid + 1)
                        if price > bid:
                            qty = min(room_to_sell(state, symbol, pending), max(2, quote_size + min(0, pos) // 15))
                            if qty > 0:
                                orders.append(Order(symbol, price, -qty))
                                pending -= qty
            else:
                # Avellaneda-Stoikov market making when no spot signal.
                _, _, sigma_hist, _ = rolling_z(memory, "hist", f"{symbol}:mm", mid, 200, 40)
                as_bid_f, as_ask_f = self.avellaneda_stoikov(mid, pos + pending, sigma_hist, AS_GAMMA, AS_KAPPA)
                bid_price = min(int(math.floor(as_bid_f)), ask - 1)
                ask_price = max(int(math.ceil(as_ask_f)), bid + 1)
                if bid_price < ask_price:
                    mm_qty = 3
                    if room_to_buy(state, symbol, pending) > 0 and pos + pending < MM_CAP:
                        qty = min(mm_qty, room_to_buy(state, symbol, pending), MM_CAP - pos - pending)
                        if qty > 0:
                            orders.append(Order(symbol, bid_price, qty))
                            pending += qty
                    if room_to_sell(state, symbol, pending) > 0 and pos + pending > -MM_CAP:
                        qty = min(mm_qty, room_to_sell(state, symbol, pending), MM_CAP + pos + pending)
                        if qty > 0:
                            orders.append(Order(symbol, ask_price, -qty))
                            pending -= qty

            # Trader-id layer: exploit known losing counterparties.
            if symbol in LOSER_BUYERS and room_to_sell(state, symbol, pending) > 0:
                qty = 2 if symbol in {"VEV_5300", "VEV_5400", "VEV_5500"} else 1
                cap = 45 if symbol in {"VEV_5300", "VEV_5400", "VEV_5500"} else 25
                if pos + pending > -cap:
                    price = max(ask - 1, bid + 1) if spread >= 2 else ask
                    orders.append(Order(symbol, price, -min(qty, room_to_sell(state, symbol, pending))))
                    pending -= min(qty, room_to_sell(state, symbol, pending))
            if symbol in LOSER_SELLERS and room_to_buy(state, symbol, pending) > 0:
                qty = 1
                if pos + pending < 25:
                    price = min(bid + 1, ask - 1) if spread >= 2 else bid
                    orders.append(Order(symbol, price, min(qty, room_to_buy(state, symbol, pending))))

            # Same-tick tape imitation. The public tape repeatedly shows a profitable
            # participant improving against a predictable counterparty. In raw-csv
            # replay, matching at the visible price leaves us behind the visible queue,
            # so we only try this when we can improve the touch by one tick.
            if spread >= 3 and symbol == "VEV_4000":
                for trade in state.market_trades.get(symbol, []):
                    if trade.seller == "Mark 38" and pos + pending < 85 and room_to_buy(state, symbol, pending) > 0:
                        price = min(int(trade.price) + 1, ask - 1)
                        if price < ask:
                            qty = min(int(trade.quantity) + 1, 4, room_to_buy(state, symbol, pending), 85 - (pos + pending))
                            if qty > 0:
                                orders.append(Order(symbol, price, qty))
                                pending += qty
                    elif trade.buyer == "Mark 38" and pos + pending > -85 and room_to_sell(state, symbol, pending) > 0:
                        price = max(int(trade.price) - 1, bid + 1)
                        if price > bid:
                            qty = min(int(trade.quantity) + 1, 4, room_to_sell(state, symbol, pending), 85 + (pos + pending))
                            if qty > 0:
                                orders.append(Order(symbol, price, -qty))
                                pending -= qty

            if strike in {5400, 5500}:
                for trade in state.market_trades.get(symbol, []):
                    if trade.seller == "Mark 22" and trade.buyer in {"Mark 01", "Mark 14"}:
                        cap = 300
                        if pos + pending > -cap and room_to_sell(state, symbol, pending) > 0:
                            qty = min(int(trade.quantity), 5, room_to_sell(state, symbol, pending), cap + (pos + pending))
                            if qty > 0:
                                orders.append(Order(symbol, bid, -qty))
                                pending -= qty

            result[symbol] = orders
        return result

    def run(self, state: TradingState):
        memory = self.load_memory(state.traderData)
        memory["tick"] = int(memory.get("tick", 0)) + 1
        self.update_counterparty_flow(state, memory)

        orders: dict[Symbol, list[Order]] = {}
        orders[HYDRO] = self.trade_mean_reverter(state, memory, HYDRO, 2.0)
        orders[VELVET] = self.trade_mean_reverter(state, memory, VELVET, 1.2)

        spot = mid_price(state.order_depths[VELVET]) if VELVET in state.order_depths else None
        orders.update(self.trade_options(state, memory, spot))

        trader_data = self.save_memory(memory)
        conversions = 0
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
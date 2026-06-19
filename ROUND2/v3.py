#8.7k
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Trade
from typing import List, Dict, Any, Tuple, Optional
import jsonpickle
import math

### --- ARTIFACTS ---

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750
 
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end
 
    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""
 
    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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
 
    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed
 
    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed
 
    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed
 
    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]
 
    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed
 
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
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out

logger = Logger()

STATIC_SYMBOL = "EMERALDS"

class GeneralTrader:
    def decode_trader_data(self, state: TradingState) -> Dict[str, Any]:
        if state.traderData:
            return jsonpickle.decode(state.traderData)
        return {"iteration": 0, "price_history": {}}

    def encode_trader_data(self, memory: Dict[str, Any]) -> str:
        return jsonpickle.encode(memory)

    def get_order_depth(self, state: TradingState, product: str) -> OrderDepth:
        return state.order_depths.get(product, OrderDepth())

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[int, int]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        return best_bid, best_ask

    def get_walls(self, orders: Dict[int, int], volume_threshold: int) -> List[int]:
        return sorted([p for p, v in orders.items() if abs(v) >= volume_threshold])

    def get_current_position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def calculate_available_limits(self, current_position: int, position_limit: int) -> Tuple[int, int]:
        buy_limit = position_limit - current_position
        sell_limit = -position_limit - current_position
        return buy_limit, sell_limit

    def create_order(self, product: str, price: int, quantity: int) -> Order:
        return Order(product, price, quantity)


class StaticPriceTrader(GeneralTrader):
    def compute_wall_mid(self, order_depth: OrderDepth, volume_threshold: int):
        bid_walls = self.get_walls(order_depth.buy_orders, volume_threshold)
        ask_walls = self.get_walls(order_depth.sell_orders, volume_threshold)
        if not bid_walls or not ask_walls:
            return None
        return (max(bid_walls) + min(ask_walls)) / 2

    def size_fraction(self, deviation: int) -> float:
        if deviation >= 10:
            return 1.0
        elif deviation >= 7:
            return 0.6
        elif deviation >= 4:
            return 0.3
        return 0.0

    def place_band_trades(self, product: str, order_depth: OrderDepth,
                          buy_limit: int, sell_limit: int,
                          mid: int = 10000) -> List[Order]:
        orders: List[Order] = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

        deviation_sell = best_bid - mid
        deviation_buy  = mid - best_ask

        logger.print(f"BAND_CHECK|{product}|best_bid={best_bid}|best_ask={best_ask}|dev_sell={deviation_sell}|dev_buy={deviation_buy}|sell_frac={self.size_fraction(deviation_sell)}|buy_frac={self.size_fraction(deviation_buy)}")

        # Sell side — price spiked up
        sell_fraction = self.size_fraction(deviation_sell)
        if sell_fraction > 0 and sell_limit < 0:
            available_sell = round(abs(sell_limit) * sell_fraction)
            remaining = available_sell
            for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                if price < mid or remaining <= 0:
                    break
                sell_qty = max(-remaining, -quantity)
                orders.append(Order(product, price, sell_qty))
                remaining -= abs(sell_qty)

        # Buy side — price dumped down
        buy_fraction = self.size_fraction(deviation_buy)
        if buy_fraction > 0 and buy_limit > 0:
            available_buy = round(buy_limit * buy_fraction)
            remaining = available_buy
            for price, quantity in sorted(order_depth.sell_orders.items()):
                if price > mid or remaining <= 0:
                    break
                buy_qty = min(remaining, abs(quantity))
                orders.append(Order(product, price, buy_qty))
                remaining -= buy_qty

        return orders

    def place_aggressive_trades(self, product: str, order_depth: OrderDepth, true_price: int,
                                buy_limit: int, sell_limit: int,
                                mid: int = 10000) -> List[Order]:
        orders: List[Order] = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

        deviation_sell = best_bid - mid
        deviation_buy  = mid - best_ask

        # If band is active, aggressive layer stands down entirely
        if self.size_fraction(deviation_sell) > 0 or self.size_fraction(deviation_buy) > 0:
            return []

        for price, quantity in sorted(order_depth.sell_orders.items()):
            if price <= true_price and buy_limit > 0:
                buy_quantity = min(buy_limit, abs(quantity))
                orders.append(Order(product, price, buy_quantity))
                buy_limit -= buy_quantity
        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
            if price >= true_price and sell_limit < 0:
                sell_quantity = max(sell_limit, -quantity)
                orders.append(Order(product, price, sell_quantity))
                sell_limit -= sell_quantity
        return orders

    def place_passive_orders(self, product: str, order_depth: OrderDepth, true_price: int,
                             buy_limit: int, sell_limit: int, config: dict,
                             state: TradingState, memory: dict) -> List[Order]:
        orders: List[Order] = []
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        position_limit = config.get("position_limit", 80)
        current_position = self.get_current_position(state, product)
        inv_ratio = current_position / position_limit if position_limit > 0 else 0
        risk_aversion = config.get("risk_aversion", 2)
        skew = round(inv_ratio * risk_aversion)

        # Volatility-adjusted spread
        vol_key = f"mid_history_{product}"
        mid_history = memory.get(vol_key, [])
        if best_bid and best_ask:
            current_mid = (best_bid + best_ask) / 2
            mid_history.append(current_mid)
            if len(mid_history) > 20:
                mid_history = mid_history[-20:]
            memory[vol_key] = mid_history

        vol_spread = 0
        if len(mid_history) >= 5:
            diffs = [abs(mid_history[i] - mid_history[i-1]) for i in range(1, len(mid_history))]
            avg_move = sum(diffs) / len(diffs)
            vol_spread = round(avg_move)

        vol_spread = max(1, min(vol_spread, config.get("max_vol_spread", 5)))

        logger.print(f"VOL_SPREAD|{product}|vol_spread={vol_spread}|skew={skew}")

        bid_price = min(best_bid + 1, true_price - 1 - skew - vol_spread) if best_bid is not None else None
        ask_price = max(best_ask - 1, true_price + 1 - skew + vol_spread) if best_ask is not None else None

        if buy_limit > 0 and bid_price is not None:
            if ask_price is None or bid_price < ask_price:
                orders.append(Order(product, bid_price, buy_limit))

        if sell_limit < 0 and ask_price is not None:
            if bid_price is None or ask_price > bid_price:
                orders.append(Order(product, ask_price, sell_limit))

        return orders

    def get_product_orders(self, state: TradingState, product: str,
                           config: dict, memory: dict) -> List[Order]:
        order_depth = self.get_order_depth(state, product)
        # Calculate standard mid_price directly instead of wall_mid
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
            true_price = int(mid_price + 0.5)
        else:
            true_price = config.get("true_price", 10000)
        current_position = self.get_current_position(state, product)
        buy_limit, sell_limit = self.calculate_available_limits(current_position, config["position_limit"])
        mid = config.get("mid", 10000)

        orders: List[Order] = []

        # --- Band strategy with tranching (highest priority) ---
        band_orders = self.place_band_trades(
            product, order_depth, buy_limit, sell_limit, mid=mid,
        )
        orders.extend(band_orders)
        for o in band_orders:
            side = "BUY" if o.quantity > 0 else "SELL"
            logger.print(f"TRADE_LOG|{state.timestamp}|{product}|BAND|{side}|{o.price}|{abs(o.quantity)}")
            if o.quantity > 0:
                buy_limit -= o.quantity
            else:
                sell_limit -= o.quantity

        # --- Aggressive layer (stands down when band is active) ---
        agg_orders = self.place_aggressive_trades(
            product, order_depth, true_price, buy_limit, sell_limit, mid=mid,
        )
        orders.extend(agg_orders)
        for o in agg_orders:
            side = "BUY" if o.quantity > 0 else "SELL"
            logger.print(f"TRADE_LOG|{state.timestamp}|{product}|STATIC_AGG|{side}|{o.price}|{abs(o.quantity)}")
            if o.quantity > 0:
                buy_limit -= o.quantity
            else:
                sell_limit -= o.quantity

        # --- Passive layer (volatility-adjusted spread) ---
        if buy_limit > 0 or sell_limit < 0:
            pass_orders = self.place_passive_orders(
                product, order_depth, true_price, buy_limit, sell_limit, config, state, memory,
            )
            orders.extend(pass_orders)
            for o in pass_orders:
                side = "BUY" if o.quantity > 0 else "SELL"
                logger.print(f"TRADE_LOG|{state.timestamp}|{product}|STATIC_PASS|{side}|{o.price}|{abs(o.quantity)}")

        return orders

    def send_orders(self, state: TradingState, product_configurations: Dict,
                    memory: dict) -> Dict[str, List[Order]]:
        result = {}
        for product, config in product_configurations.items():
            if product not in state.order_depths:
                continue
            result[product] = self.get_product_orders(state, product, config, memory)
        return result

class LinearTrader(GeneralTrader):
    def buy_everything(self, product: str, order_depth: OrderDepth,
                       buy_limit: int) -> List[Order]:
        orders: List[Order] = []
        for price, quantity in sorted(order_depth.sell_orders.items()):
            if buy_limit > 0:
                buy_quantity = min(buy_limit, abs(quantity))
                orders.append(Order(product, price, buy_quantity))
                buy_limit -= buy_quantity
        return orders

    def get_product_orders(self, state: TradingState, product: str, config: dict, memory: dict) -> List[Order]:
        order_depth = self.get_order_depth(state, product)

        current_position = self.get_current_position(state, product)
        buy_limit, _ = self.calculate_available_limits(current_position, config["position_limit"])

        # Already at max position — hold, do nothing
        if buy_limit <= 0:
            return []

        # Not yet at max — sweep all asks immediately
        return self.buy_everything(product, order_depth, buy_limit)

    def send_orders(self, state: TradingState, product_configurations: Dict, memory: dict) -> Dict[str, List[Order]]:
        result = {}
        for product, config in product_configurations.items():
            if product not in state.order_depths:
                continue
            result[product] = self.get_product_orders(state, product, config, memory)
        return result

class Trader:
    def bid(self):
        return 1002
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        static_product_configurations = {
            "ASH_COATED_OSMIUM": {
                "true_price": 10000,
                "position_limit": 80,
                "wall_volume_threshold": 20,
                "risk_aversion": 2,
                "mid": 10000,
                "max_vol_spread": 5,
            },
        }

        linear_product_configurations = {
            "INTARIAN_PEPPER_ROOT": {
                "true_price": 12000,          # update each session
                "position_limit": 80,
                "wall_volume_threshold": 5,  # was 30 — too high, wall_mid was never found
                "risk_aversion": 1,          # no skew — price is trending, don't fight inventory
            }
        }

        dynamic_product_configurations = {
            "INTARIAN_PEPPER_ROOT": {
                "x0": 20000, # Fallback initial mid-price for the asset.
                "sigma0": 4.686e-4, # Baseline initial volatility for price movements
                "ewma_lambda": 0.93, # high lambda = slow reactions + smooth to price shocks. low lambda = faster and aggressive
                "position_limit": 80,
                "wall_volume_threshold": 5, # Minimum resting volume to establish walls
                # Avellaneda-Stoikov parameters
                "as_gamma": 3.0, # risk aversion: higher = wider spread + aggressively shifts quotes to dump inventory
                "as_kappa": 1.0, # order arrival intensity: higher = tighter quotes to capture frequent trades
                "session_length": 200000,
                # Inventory gate
                "inv_gate": 0.95, # Inventory utilization limit for aggressive orders
                "momentum_weight": 0.5, # Impact of trend drift on pricing
                "mr_threshold": 10, # Price deviation required for mean reversion
            }
        }

        if state.traderData:
            memory = jsonpickle.decode(state.traderData)
        else:
            memory = {"iteration": 0, "price_history": {}}

        result.update(StaticPriceTrader().send_orders(state, static_product_configurations, memory))
        result.update(LinearTrader().send_orders(state, linear_product_configurations, memory))
        memory["iteration"] += 1
        traderData = jsonpickle.encode(memory)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
class RandomWalkTrader(GeneralTrader):
    """
    Avellaneda-Stoikov market maker.

    Fair value = AS reservation price:
        r = mid - q * gamma * sigma^2 * T
    where q = inventory ratio, T = fraction of session remaining.

    Optimal spread:
        spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)

    As T shrinks toward session end, both the inventory penalty and the
    spread shrink — the model naturally tightens quotes and flattens
    inventory near close without any extra logic.

    Parameters:
        as_gamma        risk aversion (higher = wider spread, stronger inv penalty)
        as_kappa        order arrival intensity (higher = tighter spread)
        session_length  total timestamps in a session (for T computation)
        inv_gate        hard stop on opening side beyond this inv_ratio
    """

    def update_ewma_var(self, prev_ewma_var: float, log_return: float, lam: float) -> float:
        return lam * prev_ewma_var + (1.0 - lam) * log_return ** 2

    def compute_vwap_mid(self, order_depth: OrderDepth, levels: int = 3) -> Optional[float]:
        bid_items = sorted(order_depth.buy_orders.items(), reverse=True)[:levels]
        ask_items = sorted(order_depth.sell_orders.items())[:levels]
        if not bid_items or not ask_items:
            return None
        bid_vwap = sum(p * v for p, v in bid_items) / (sum(v for _, v in bid_items) or 1)
        ask_vwap = sum(p * abs(v) for p, v in ask_items) / (sum(abs(v) for _, v in ask_items) or 1)
        return (bid_vwap + ask_vwap) / 2.0

    def compute_wall_mid(self, order_depth: OrderDepth, volume_threshold: int):
        bid_walls = [p for p, v in order_depth.buy_orders.items() if v >= volume_threshold]
        ask_walls = [p for p, v in order_depth.sell_orders.items() if abs(v) >= volume_threshold]
        if not bid_walls or not ask_walls:
            return None
        return (max(bid_walls) + min(ask_walls)) / 2

    def place_orders(self, product: str, order_depth: OrderDepth, fair_value: float,
                     buy_threshold: float, sell_threshold: float,
                     buy_limit: int, sell_limit: int) -> List[Order]:
        orders: List[Order] = []
        lower_band = fair_value - buy_threshold
        upper_band = fair_value + sell_threshold

        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            if ask_price >= lower_band or buy_limit <= 0:
                break
            fill = min(buy_limit, abs(ask_qty))
            orders.append(Order(product, ask_price, fill))
            buy_limit -= fill

        for bid_price, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price <= upper_band or sell_limit >= 0:
                break
            fill = max(sell_limit, -bid_qty)
            orders.append(Order(product, bid_price, fill))
            sell_limit -= fill

        return orders

    def get_product_orders(self, state: TradingState, product: str, config: dict,
                           memory: Dict[str, Any]) -> Tuple[List[Order], Dict[str, Any]]:

        # --- Parameters ---
        sigma0 = config["sigma0"]
        x0_key = f"prev_mid_{product}"
        lam = config.get("ewma_lambda", 0.97)
        position_limit = config["position_limit"]
        wall_threshold = config.get("wall_volume_threshold", 8)
        inv_gate = config.get("inv_gate", 0.4)

        # Avellaneda-Stoikov parameters
        gamma = config.get("as_gamma", 0.1)
        kappa = config.get("as_kappa", 1.5)
        session_length = config.get("session_length", 200000)

        # --- Market data ---
        order_depth = self.get_order_depth(state, product)
        best_bid, best_ask = self.get_best_bid_ask(order_depth)

        vwap_mid = self.compute_vwap_mid(order_depth, levels=config.get("vwap_levels", 3))
        current_mid = vwap_mid if vwap_mid is not None else (
            (best_bid + best_ask) / 2.0 if best_bid and best_ask else config.get("x0", 5000)
        )

        # --- EWMA volatility ---
        ewma_key = f"ewma_var_{product}"
        ewma_var = memory.get(ewma_key, sigma0 ** 2)
        prev_mid = memory.get(x0_key, current_mid)

        if prev_mid > 0:
            log_return = math.log(current_mid / prev_mid)
            ewma_var = self.update_ewma_var(ewma_var, log_return, lam)
        sigma_t = math.sqrt(ewma_var)

        # --- EWMA momentum ---
        ewma_ret_key = f"ewma_ret_{product}"
        ewma_ret = memory.get(ewma_ret_key, 0.0)
        if prev_mid > 0:
            ewma_ret = lam * ewma_ret + (1.0 - lam) * log_return

        # --- Inventory ---
        current_position = self.get_current_position(state, product)
        inv_ratio = current_position / position_limit if position_limit > 0 else 0

        # --- Avellaneda-Stoikov fair value and spread ---
        # T = fraction of session remaining, floored at 0.01 so spread never
        # collapses to zero. As T approaches 0 (end of session), the inventory
        # penalty shrinks and the model flattens positions naturally.
        T = max(1.0 - state.timestamp / session_length, 0.01)

        abs_sigma = sigma_t * current_mid
        abs_variance = abs_sigma ** 2
        reservation_price = current_mid - inv_ratio * gamma * abs_variance * T
        
        # Optimal half-spread: widens with vol, risk aversion, and time remaining.
        # The log term ensures spread stays positive even at low kappa.
        half_spread = (gamma * abs_variance * T / 2.0) + (1.0 / gamma) * math.log(1.0 + gamma / kappa)

        fair_value = reservation_price
        buy_threshold  = max(half_spread, 0)
        sell_threshold = max(half_spread, 0)

        # --- Buy/sell limits ---
        buy_limit, sell_limit = self.calculate_available_limits(current_position, position_limit)

        # --- Inventory gate ---
        agg_buy_limit  = 0 if inv_ratio > inv_gate else buy_limit
        agg_sell_limit = 0 if inv_ratio < -inv_gate else sell_limit

        # --- Aggressive orders ---
        orders = self.place_orders(product, order_depth, fair_value,
                                   buy_threshold, sell_threshold,
                                   agg_buy_limit, agg_sell_limit)

        for o in orders:
            if o.quantity > 0:
                buy_limit -= o.quantity
            else:
                sell_limit -= o.quantity

        # --- Mean reversion trigger ---
        rolling_mean_key = f"rolling_mean_{product}"
        rolling_mean = memory.get(rolling_mean_key, current_mid)
        rolling_mean = 0.99 * rolling_mean + 0.01 * current_mid  # slow EWMA of price

        deviation = current_mid - rolling_mean
        mr_threshold = config.get("mr_threshold", 8.0)  # ticks

        # If price is well below mean → aggressively buy the dip
        if deviation < -mr_threshold and buy_limit > 0:
            for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
                if buy_limit <= 0:
                    break
                fill = min(buy_limit, abs(ask_qty))
                orders.append(Order(product, ask_price, fill))
                buy_limit -= fill

        # If price is well above mean → aggressively sell the peak
        elif deviation > mr_threshold and sell_limit < 0:
            for bid_price, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
                if sell_limit >= 0:
                    break
                fill = max(sell_limit, -bid_qty)
                orders.append(Order(product, bid_price, fill))
                sell_limit -= fill

        memory[rolling_mean_key] = rolling_mean

        # --- Passive layer ---
        # Anchor to wall_mid when large resting orders define a range,
        # otherwise fall back to reservation price as the anchor.
        wall_mid = self.compute_wall_mid(order_depth, wall_threshold)
        anchor = wall_mid if wall_mid is not None else fair_value
        momentum_weight = config.get("momentum_weight", 0.0)
        momentum_adjustment = ewma_ret * momentum_weight * current_mid
        anchor_rounded = round(anchor + momentum_adjustment)

        if buy_limit > 0 and best_bid != 0:
            # bid_price = min(best_bid + 1, anchor_rounded - 1) # conservative trading
            bid_price = best_bid + 1 # aggressive trading
            if bid_price > 0:
                orders.append(Order(product, bid_price, buy_limit))

        if sell_limit < 0 and best_ask != 0:
            # ask_price = max(best_ask - 1, anchor_rounded + 1) # conservative trading
            ask_price = best_ask - 1 # aggressive trading
            orders.append(Order(product, ask_price, sell_limit))

        # --- Update memory ---
        memory[ewma_key]     = ewma_var
        memory[ewma_ret_key] = ewma_ret
        memory[x0_key]       = current_mid

        return orders, memory

    def send_orders(self, state: TradingState, product_configurations: Dict,
                    memory: Dict[str, Any]) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        result = {}
        for product, config in product_configurations.items():
            if product not in state.order_depths:
                continue
            orders, memory = self.get_product_orders(state, product, config, memory)
            result[product] = orders
        return result, memory
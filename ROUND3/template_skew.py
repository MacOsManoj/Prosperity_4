from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
from typing import Any

####### LOGGER #######

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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [order_depth.buy_orders, order_depth.sell_orders] for symbol, order_depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                for trade_list in trades.values() for trade in trade_list]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {product: [obs.bidPrice, obs.askPrice, obs.transportFees, obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex]
                          for product, obs in observations.conversionObservations.items()}
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[order.symbol, order.price, order.quantity] for order_list in orders.values() for order in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        low, high = 0, min(len(value), max_length)
        result = ""
        while low <= high:
            midpoint = (low + high) // 2
            candidate = value[:midpoint]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                result = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        return result

logger = Logger()

####### CONFIG #######

POS_LIMIT  = 200

PACK_SYMBOL  = "HYDROGEL_PACK"
PACK_MEAN    = 9990
PACK_STD     = 32   # tune from historical data
SKEW_MAX     = 3    # max ticks of skew at 1 std dev away from mean

####### BASE TRADER #######

class ProductTrader:

    def __init__(self, symbol, state):
        self.name  = symbol
        self.state : TradingState = state
        order_depth = state.order_depths.get(symbol, OrderDepth())
        position    = state.position.get(symbol, 0)

        self.initial_position = position
        self.position         = position
        self.orders           = []
        self.buy_volume       = 0
        self.sell_volume      = 0

        self.mkt_buy_orders  = {price: abs(volume) for price, volume in sorted(order_depth.buy_orders.items(),  reverse=True)} if order_depth.buy_orders  else {}
        self.mkt_sell_orders = {price: abs(volume) for price, volume in sorted(order_depth.sell_orders.items())} if order_depth.sell_orders else {}
        self.bid_wall        = max(self.mkt_buy_orders)  if self.mkt_buy_orders  else None
        self.ask_wall        = min(self.mkt_sell_orders) if self.mkt_sell_orders else None
        self.wall_mid        = (self.bid_wall + self.ask_wall) / 2.0 if self.bid_wall is not None and self.ask_wall is not None else None

    @property
    def max_allowed_buy_volume(self):
        return POS_LIMIT - self.initial_position - self.buy_volume

    @property
    def max_allowed_sell_volume(self):
        return POS_LIMIT + self.initial_position - self.sell_volume

    def bid(self, price, quantity, logging=False):
        remaining_capacity = POS_LIMIT - self.initial_position - self.buy_volume
        fill_volume = min(quantity, remaining_capacity)
        if fill_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), fill_volume))
        self.buy_volume += fill_volume
        self.position += fill_volume
        if logging:
            logger.print(f"BID {self.name} {int(price)} x{fill_volume}")

    def ask(self, price, quantity, logging=False):
        remaining_capacity = POS_LIMIT + self.initial_position - self.sell_volume
        fill_volume = min(quantity, remaining_capacity)
        if fill_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), -fill_volume))
        self.sell_volume += fill_volume
        self.position -= fill_volume
        if logging:
            logger.print(f"ASK {self.name} {int(price)} x{fill_volume}")

    def get_orders(self):
        return {self.name: self.orders}


# ASH_COATED_OSMIUM
class PACKTrader(ProductTrader):
    def __init__(self, state: TradingState, last_wall_mid=None):
        super().__init__(PACK_SYMBOL, state)
        if self.wall_mid is None:
            self.wall_mid = last_wall_mid

    def get_orders(self):
        if self.wall_mid is not None:
            fair_value = self.wall_mid
        
        # edge toward fair
        # fair_value = 0.9 * fair_value + 0.1 * PACK_MEAN
        
        ##########################################################
        ####### 1. TAKING
        ##########################################################
        for sell_price, sell_volume in self.mkt_sell_orders.items():
            if sell_price <= fair_value - 1:
                self.bid(sell_price, sell_volume, logging=False)
            elif sell_price <= fair_value and self.initial_position < 0:
                defensive_volume = min(sell_volume, abs(self.initial_position))
                self.bid(sell_price, defensive_volume, logging=False)

        for buy_price, buy_volume in self.mkt_buy_orders.items():
            if buy_price >= fair_value + 1:
                self.ask(buy_price, buy_volume, logging=False)
            elif buy_price >= fair_value and self.initial_position > 0:
                defensive_volume = min(buy_volume, self.initial_position)
                self.ask(buy_price, defensive_volume, logging=False)

        ###########################################################
        ####### 2. MAKING
        ###########################################################

        # position already reflects taker fills from step 1
        bid_ceiling = int(fair_value) if self.position < 0 else int(fair_value - 1)
        ask_floor   = int(fair_value) if self.position > 0 else int(fair_value + 1)

        thick_bid = next((p for p in self.mkt_buy_orders if p <= fair_value and self.mkt_buy_orders[p] > 1), None)
        thick_ask = next((p for p in self.mkt_sell_orders if p >= fair_value and self.mkt_sell_orders[p] > 1), None)

        # skew shifts both quotes toward the direction we want to be filled.
        # positive skew (price > mean): push quotes down → more aggressive ask, less aggressive bid
        # negative skew (price < mean): push quotes up   → more aggressive bid, less aggressive ask
        raw_skew = (fair_value - PACK_MEAN) / PACK_STD * SKEW_MAX
        skew = int(round(max(-2 * SKEW_MAX, min(2 * SKEW_MAX, raw_skew))))

        base_bid = (thick_bid + 1) if thick_bid is not None else int(fair_value - 6)
        base_ask = (thick_ask - 1) if thick_ask is not None else int(fair_value + 6)

        bid_price = min(base_bid - skew, bid_ceiling)
        ask_price = max(base_ask - skew, ask_floor)

        # scale the aggressive side's size from 6 → max_allowed linearly over 0–3 std devs
        std_devs = (fair_value - PACK_MEAN) / PACK_STD
        normalized = min(abs(std_devs) / 3, 1.0)
        if std_devs > 0:  # price above mean: inflate bid size to attract sellers
            maker_buy_volume  = min(int(6 + (self.max_allowed_buy_volume - 6) * normalized), self.max_allowed_buy_volume)
            maker_sell_volume = min(6, self.max_allowed_sell_volume)
        else:              # price below mean: inflate ask size to attract buyers
            maker_buy_volume  = min(6, self.max_allowed_buy_volume)
            maker_sell_volume = min(int(6 + (self.max_allowed_sell_volume - 6) * normalized), self.max_allowed_sell_volume)

        self.bid(bid_price, maker_buy_volume)
        self.ask(ask_price, maker_sell_volume)
        
        return {self.name: self.orders}

####### MAIN #######

class Trader:

    def bid(self): 
        return 1

    def __init__(self):
        self._ipr_fv_start = None

    def run(self, state: TradingState):
        result = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}

        for symbol in state.order_depths:
            order_depth = state.order_depths[symbol]

            try:
                if symbol == PACK_SYMBOL:
                    pack_trader = PACKTrader(state, last_wall_mid=trader_data.get("pack_wall_mid"))
                    result.update(pack_trader.get_orders())
                    if pack_trader.wall_mid is not None:
                        trader_data["pack_wall_mid"] = pack_trader.wall_mid

            except Exception as e:
                logger.print(f"ERROR {symbol}: {e}")

        out_trader_data = json.dumps(trader_data)
        logger.flush(state, result, 0, out_trader_data)
        return result, 0, out_trader_data

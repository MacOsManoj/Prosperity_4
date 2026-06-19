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

POS_LIMIT  = 80
IPR_SLOPE  = 0.001
IPR_MAX_TS = 999900

ASH_SYMBOL  = "ASH_COATED_OSMIUM"
ROOT_SYMBOL = "INTARIAN_PEPPER_ROOT"

# ASH_COATED_OSMIUM — AR(1) mean-reversion model
# X_{t+1} = X_t + k*(mu - X_t) + eps,  eps ~ N(0, sigma^2)
ACO_MU           = 10000.0
ACO_K            = 0.242392   # mean-reversion speed, calibrated from data
ACO_RESIDUAL_STD = 3.493654   # residual std (R^2 = 0.12)


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
class AshTrader(ProductTrader):
    def __init__(self, state: TradingState):
        super().__init__(ASH_SYMBOL, state)
        self.last_fair = ACO_MU

    def get_orders(self):

        if self.wall_mid is not None:
            # Calculate the effective K for 10 timesteps into the future
            # You could also compute this once in __init__ to save processing time
            LOOKAHEAD_STEPS = 10
            effective_k = 1 - (1 - ACO_K)**LOOKAHEAD_STEPS

            # AR(1) predicted fair value 10 timesteps out
            fair_value = self.wall_mid + effective_k * (ACO_MU - self.wall_mid)
            self.last_fair = fair_value
        else:
            fair_value = self.last_fair
        
        logger.print(f"ACO_FV_10_STEP:{fair_value:.4f}")


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

        bid_price = min((thick_bid + 1) if thick_bid is not None else int(fair_value - 7), bid_ceiling)
        ask_price = max((thick_ask - 1) if thick_ask is not None else int(fair_value + 7), ask_floor)

        maker_buy_volume  = self.max_allowed_buy_volume
        maker_sell_volume = self.max_allowed_sell_volume
        self.bid(bid_price, maker_buy_volume)
        self.ask(ask_price, maker_sell_volume)
        
        return {self.name: self.orders}


IPR_MAX_SPREAD = 7

# INTARIAN_PEPPER_ROOT
class RootTrader(ProductTrader):
    def __init__(self, state, end_of_day_fair_value, current_fair_value):
        super().__init__(ROOT_SYMBOL, state)
        self.end_of_day_fair_value = end_of_day_fair_value
        self.current_fair_value = current_fair_value

    def get_orders(self):
        for ask_price, ask_volume in self.mkt_sell_orders.items():
            if ask_price < self.end_of_day_fair_value and ask_price <= self.current_fair_value + IPR_MAX_SPREAD:
                self.bid(ask_price, ask_volume)
            else:
                break

        for bid_price, bid_volume in self.mkt_buy_orders.items():
            if bid_price > self.end_of_day_fair_value and bid_price >= self.current_fair_value - IPR_MAX_SPREAD:
                self.ask(bid_price, bid_volume)
            else:
                break

        return {self.name: self.orders}

####### MAIN #######

class Trader:

    def bid(self): 
        return 1

    def __init__(self):
        self._ipr_fv_start = None

    def _ipr_fv_eod(self, order_depth, timestamp):
        if self._ipr_fv_start is None:
            if order_depth.buy_orders and order_depth.sell_orders:
                mid_price = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
                raw_start = mid_price - timestamp * IPR_SLOPE
                self._ipr_fv_start = float(round(raw_start / 1000) * 1000)
            else:
                self._ipr_fv_start = 10000.0
        return self._ipr_fv_start + IPR_MAX_TS * IPR_SLOPE

    def _ipr_fv_now(self, timestamp):
        return self._ipr_fv_start + timestamp * IPR_SLOPE

    def run(self, state: TradingState):
        result = {}

        for symbol in state.order_depths:
            order_depth = state.order_depths[symbol]

            try:
                if symbol == ASH_SYMBOL:
                    result.update(AshTrader(state).get_orders())

                elif symbol == ROOT_SYMBOL:
                    end_of_day_fair_value = self._ipr_fv_eod(order_depth, state.timestamp)
                    current_fair_value = self._ipr_fv_now(state.timestamp)
                    result.update(RootTrader(state, end_of_day_fair_value, current_fair_value).get_orders())

            except Exception as e:
                logger.print(f"ERROR {symbol}: {e}")

        logger.flush(state, result, 0, "")
        return result, 0, ""
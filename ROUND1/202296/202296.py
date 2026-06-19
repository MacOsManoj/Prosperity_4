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
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                for p, o in observations.conversionObservations.items()}
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
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
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out

logger = Logger()

####### CONFIG #######

POS_LIMIT  = 80
IPR_SLOPE  = 0.001
IPR_MAX_TS = 999900

STATIC_SYMBOL  = "ASH_COATED_OSMIUM"
DYNAMIC_SYMBOL = "INTARIAN_PEPPER_ROOT"

LONG  = 1
SHORT = -1
INFORMED_TRADER = ""  # TODO: identify from competition logs

# ASH_COATED_OSMIUM — AR(1) mean-reversion model
# X_{t+1} = X_t + k*(mu - X_t) + eps,  eps ~ N(0, sigma^2)
ACO_MU           = 10000.0
ACO_K            = 0.242392   # mean-reversion speed, calibrated from data
ACO_RESIDUAL_STD = 3.493654   # residual std (R^2 = 0.12)


####### BASE TRADER #######

class ProductTrader:

    def __init__(self, symbol, state):
        self.name  = symbol
        self.state = state
        od         = state.order_depths.get(symbol, OrderDepth())
        position   = state.position.get(symbol, 0)

        self.initial_position = position
        self.position         = position
        self.orders           = []

        self.mkt_buy_orders  = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  reverse=True)} if od.buy_orders  else {}
        self.mkt_sell_orders = {p: abs(v) for p, v in sorted(od.sell_orders.items())} if od.sell_orders else {}
        self.bid_wall        = max(self.mkt_buy_orders)  if self.mkt_buy_orders  else None
        self.ask_wall        = min(self.mkt_sell_orders) if self.mkt_sell_orders else None
        self.wall_mid        = (self.bid_wall + self.ask_wall) / 2.0 if self.bid_wall is not None and self.ask_wall is not None else None

    @property
    def max_allowed_buy_volume(self):
        return POS_LIMIT - self.position

    @property
    def max_allowed_sell_volume(self):
        return POS_LIMIT + self.position

    def bid(self, price, qty, logging=False):
        room = POS_LIMIT - self.position
        vol  = min(qty, room)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, int(price), vol))
        self.position += vol
        if logging:
            logger.print(f"BID {self.name} {int(price)} x{vol}")

    def ask(self, price, qty, logging=False):
        room = POS_LIMIT + self.position
        vol  = min(qty, room)
        if vol <= 0:
            return
        self.orders.append(Order(self.name, int(price), -vol))
        self.position -= vol
        if logging:
            logger.print(f"ASK {self.name} {int(price)} x{vol}")

    def check_for_informed(self):
        direction = None
        bought_ts = None
        sold_ts   = None
        if not INFORMED_TRADER:
            return direction, bought_ts, sold_ts
        for trade in self.state.market_trades.get(self.name, []):
            if trade.buyer == INFORMED_TRADER:
                direction = LONG
                bought_ts = trade.timestamp
            elif trade.seller == INFORMED_TRADER:
                direction = SHORT
                sold_ts   = trade.timestamp
        return direction, bought_ts, sold_ts

    def get_orders(self):
        return {self.name: self.orders}


# ASH_COATED_OSMIUM
class StaticTrader(ProductTrader):
    def __init__(self, state):
        super().__init__(STATIC_SYMBOL, state)

    def get_orders(self):

        if self.wall_mid is not None:

            # AR(1) predicted fair value: shifts the quoting center toward mu
            # when mid > mu we lean to sell; when mid < mu we lean to buy
            fv = self.wall_mid + ACO_K * (ACO_MU - self.wall_mid)

            ##########################################################
            ####### 1. TAKING
            ##########################################################
            for sp, sv in self.mkt_sell_orders.items():
                if sp <= fv - 1:
                    self.bid(sp, sv, logging=False)
                elif sp <= fv and self.initial_position < 0:
                    volume = min(sv, abs(self.initial_position))
                    self.bid(sp, volume, logging=False)

            for bp, bv in self.mkt_buy_orders.items():
                if bp >= fv + 1:
                    self.ask(bp, bv, logging=False)
                elif bp >= fv and self.initial_position > 0:
                    volume = min(bv, self.initial_position)
                    self.ask(bp, volume, logging=False)

            ###########################################################
            ####### 2. MAKING
            ###########################################################
            bid_price = int(self.bid_wall + 1)
            ask_price = int(self.ask_wall - 1)

            # OVERBIDDING: overbid best bid that is still under fv
            for bp, bv in self.mkt_buy_orders.items():
                overbidding_price = bp + 1
                if bv > 1 and overbidding_price < fv:
                    bid_price = max(bid_price, overbidding_price)
                    break
                elif bp < fv:
                    bid_price = max(bid_price, bp)
                    break

            # UNDERBIDDING: underbid best ask that is still over fv
            for sp, sv in self.mkt_sell_orders.items():
                underbidding_price = sp - 1
                if sv > 1 and underbidding_price > fv:
                    ask_price = min(ask_price, underbidding_price)
                    break
                elif sp > fv:
                    ask_price = min(ask_price, sp)
                    break

            # POST ORDERS — capture volumes before bid() mutates self.position
            buy_vol  = self.max_allowed_buy_volume
            sell_vol = self.max_allowed_sell_volume
            self.bid(bid_price, buy_vol)
            self.ask(ask_price, sell_vol)

        return {self.name: self.orders}


# INTARIAN_PEPPER_ROOT
class DynamicTrader(ProductTrader):
    def __init__(self, state):
        super().__init__(DYNAMIC_SYMBOL, state)
        self.informed_direction, self.informed_bought_ts, self.informed_sold_ts = self.check_for_informed()

    def get_orders(self):

        if self.wall_mid is not None:

            bid_price  = self.bid_wall + 1
            bid_volume = self.max_allowed_buy_volume

            if self.informed_bought_ts is not None and self.informed_bought_ts + 5_00 >= self.state.timestamp:
                if self.initial_position < 40:
                    bid_price  = self.ask_wall
                    bid_volume = 40 - self.initial_position

            else:
                if self.wall_mid - bid_price < 1 and (self.informed_direction == SHORT and self.initial_position > -40):
                    bid_price = self.bid_wall

            self.bid(bid_price, bid_volume)

            ask_price  = self.ask_wall - 1
            ask_volume = self.max_allowed_sell_volume

            if self.informed_sold_ts is not None and self.informed_sold_ts + 5_00 >= self.state.timestamp:
                if self.initial_position > -40:
                    ask_price  = self.bid_wall
                    ask_volume = 40 + self.initial_position

            if ask_price - self.wall_mid < 1 and (self.informed_direction == LONG and self.initial_position < 40):
                ask_price = self.ask_wall

            self.ask(ask_price, ask_volume)

        return {self.name: self.orders}


####### MAIN #######

class Trader:

    def __init__(self):
        self._ipr_fv_start = None

    def _ipr_fv_eod(self, od, timestamp):
        if self._ipr_fv_start is None:
            if od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                raw = mid - timestamp * IPR_SLOPE
                self._ipr_fv_start = float(round(raw / 1000) * 1000)
            else:
                self._ipr_fv_start = 10000.0
        return self._ipr_fv_start + IPR_MAX_TS * IPR_SLOPE

    def run(self, state: TradingState):
        result = {}

        for symbol in state.order_depths:
            od       = state.order_depths[symbol]
            position = state.position.get(symbol, 0)

            try:
                if symbol == STATIC_SYMBOL:
                    trader = StaticTrader(state)
                    result.update(trader.get_orders())

                elif symbol == DYNAMIC_SYMBOL:
                    # Buy and hold: accumulate long at any ask below end-of-day FV
                    fv_eod = self._ipr_fv_eod(od, state.timestamp)
                    orders = []
                    for ask in sorted(od.sell_orders.keys()):
                        if ask >= fv_eod:
                            break
                        can_buy = POS_LIMIT - position
                        if can_buy <= 0:
                            break
                        qty = min(-od.sell_orders[ask], can_buy)
                        orders.append(Order(symbol, ask, qty))
                        position += qty
                    result[symbol] = orders

            except Exception as e:
                logger.print(f"ERROR {symbol}: {e}")

        logger.flush(state, result, 0, "")
        return result, 0, ""
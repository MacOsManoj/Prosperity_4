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

####### HARDENING CONFIG #######

# Intercept detection: snap detected values to nearest grid multiple
ACO_MU_GRID    = 1000   # ACO_MU is most likely 10000; snap to nearest 1000
IPR_START_GRID = 1000   # IPR intercept snapped to nearest 1000
MID_BUF_SIZE   = 20     # rolling buffer length for layer-2/3 detection fallbacks

# Regime-change detector thresholds
# ACO: residual std ~3.5, so GAP >> 3*std. CONFIRM consecutive ticks to fire.
ACO_REGIME_GAP     = 25
ACO_REGIME_CONFIRM = 5
ACO_REGIME_WARMUP  = 10   # ticks before arming (let filter settle)

# IPR: normal mid fluctuates within a few ticks of the trend
IPR_REGIME_GAP     = 100
IPR_REGIME_CONFIRM = 5
IPR_REGIME_WARMUP  = 10


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
    def __init__(self, state: TradingState, aco_mu: float = ACO_MU):
        super().__init__(ASH_SYMBOL, state)
        self.aco_mu    = aco_mu
        self.last_fair = aco_mu

    def get_orders(self):

        if self.wall_mid is not None:
            # Calculate the effective K for 10 timesteps into the future
            # You could also compute this once in __init__ to save processing time
            LOOKAHEAD_STEPS = 10
            effective_k = 1 - (1 - ACO_K)**LOOKAHEAD_STEPS

            # AR(1) predicted fair value 10 timesteps out
            fair_value = self.wall_mid + effective_k * (self.aco_mu - self.wall_mid)
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

    def __init__(self):
        # IPR state
        self._ipr_fv_start   = None
        self._ipr_est_buf    = []   # rolling buffer of (mid - t*slope) estimates
        self._ipr_consec_dev = 0    # consecutive deviation counter
        self._ipr_tick       = 0

        # ACO state
        self._aco_mu         = None  # detected; None until first valid tick
        self._aco_mid_buf    = []    # rolling buffer of observed mids
        self._aco_consec_dev = 0
        self._aco_tick       = 0

        # Safety wrapper trim counter (> 0 means upstream logic has a bug)
        self._trim_count = 0

    # ---- ACO intercept detection (Section 2) ----

    def _detect_aco_mu(self, order_depth):
        """4-layer ACO_MU detection. Always returns a float."""
        # Layer 1: snap current mid to nearest ACO_MU_GRID
        if order_depth.buy_orders and order_depth.sell_orders:
            mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
            return float(round(mid / ACO_MU_GRID) * ACO_MU_GRID)
        # Layer 2: median of buffered mids
        if len(self._aco_mid_buf) >= 3:
            med = sorted(self._aco_mid_buf)[len(self._aco_mid_buf) // 2]
            return float(round(med / ACO_MU_GRID) * ACO_MU_GRID)
        # Layer 3: mean of buffered mids
        if self._aco_mid_buf:
            mean = sum(self._aco_mid_buf) / len(self._aco_mid_buf)
            return float(round(mean / ACO_MU_GRID) * ACO_MU_GRID)
        # Layer 4: hardcoded fallback
        return ACO_MU

    # ---- ACO regime-change detector (Section 3) ----

    def _check_aco_regime(self, order_depth):
        """Update mid buffer; re-detect mu if consecutive deviation threshold fires."""
        if not (order_depth.buy_orders and order_depth.sell_orders):
            return
        mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        self._aco_mid_buf.append(mid)
        if len(self._aco_mid_buf) > MID_BUF_SIZE:
            self._aco_mid_buf.pop(0)
        self._aco_tick += 1
        if self._aco_tick < ACO_REGIME_WARMUP or self._aco_mu is None:
            return
        if abs(mid - self._aco_mu) >= ACO_REGIME_GAP:
            self._aco_consec_dev += 1
        else:
            self._aco_consec_dev = 0
        if self._aco_consec_dev >= ACO_REGIME_CONFIRM:
            new_mu = self._detect_aco_mu(order_depth)
            logger.print(f"ACO_REGIME: mu {self._aco_mu:.0f} -> {new_mu:.0f}")
            self._aco_mu         = new_mu
            self._aco_consec_dev = 0
            self._aco_mid_buf.clear()

    # ---- IPR intercept detection (Section 2) ----

    def _detect_ipr_start(self, order_depth, timestamp):
        """4-layer IPR intercept detection. Always returns a float."""
        # Layer 1: derive from current mid
        if order_depth.buy_orders and order_depth.sell_orders:
            mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
            raw = mid - timestamp * IPR_SLOPE
            return float(round(raw / IPR_START_GRID) * IPR_START_GRID)
        # Layer 2: median of buffered estimates
        if len(self._ipr_est_buf) >= 3:
            med = sorted(self._ipr_est_buf)[len(self._ipr_est_buf) // 2]
            return float(round(med / IPR_START_GRID) * IPR_START_GRID)
        # Layer 3: mean of buffered estimates
        if self._ipr_est_buf:
            mean = sum(self._ipr_est_buf) / len(self._ipr_est_buf)
            return float(round(mean / IPR_START_GRID) * IPR_START_GRID)
        # Layer 4: hardcoded fallback
        return 12000.0

    # ---- IPR regime-change detector (Section 3) ----

    def _check_ipr_regime(self, order_depth, timestamp):
        """Update estimate buffer; re-detect intercept if consecutive deviation threshold fires."""
        if not (order_depth.buy_orders and order_depth.sell_orders):
            return
        mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        est = mid - timestamp * IPR_SLOPE
        self._ipr_est_buf.append(est)
        if len(self._ipr_est_buf) > MID_BUF_SIZE:
            self._ipr_est_buf.pop(0)
        self._ipr_tick += 1
        if self._ipr_tick < IPR_REGIME_WARMUP or self._ipr_fv_start is None:
            return
        expected_mid = self._ipr_fv_start + timestamp * IPR_SLOPE
        if abs(mid - expected_mid) >= IPR_REGIME_GAP:
            self._ipr_consec_dev += 1
        else:
            self._ipr_consec_dev = 0
        if self._ipr_consec_dev >= IPR_REGIME_CONFIRM:
            new_start = self._detect_ipr_start(order_depth, timestamp)
            logger.print(f"IPR_REGIME: start {self._ipr_fv_start:.0f} -> {new_start:.0f}")
            self._ipr_fv_start   = new_start
            self._ipr_consec_dev = 0
            self._ipr_est_buf.clear()

    def _ipr_fv_eod(self, order_depth, timestamp):
        if self._ipr_fv_start is None:
            self._ipr_fv_start = self._detect_ipr_start(order_depth, timestamp)
        return self._ipr_fv_start + IPR_MAX_TS * IPR_SLOPE

    def _ipr_fv_now(self, timestamp):
        if self._ipr_fv_start is None:
            return 12000.0 + timestamp * IPR_SLOPE  # safe fallback before first detection
        return self._ipr_fv_start + timestamp * IPR_SLOPE

    # ---- Safety wrapper (Section 1) ----

    def _trim_orders(self, symbol, orders, position):
        """
        Trim order list so engine position constraints are satisfied.
        Should always be a no-op if upstream logic is correct.
        Logs TRIM_WARNING and increments self._trim_count if it ever fires.
        """
        buy_total  = 0
        sell_total = 0
        trimmed    = []
        n_trimmed  = 0
        for order in orders:
            if order.quantity > 0:
                cap     = POS_LIMIT - position - buy_total
                allowed = min(order.quantity, cap)
                if allowed < order.quantity:
                    n_trimmed += 1
                if allowed > 0:
                    trimmed.append(Order(symbol, order.price, allowed))
                    buy_total += allowed
            else:
                qty     = abs(order.quantity)
                cap     = POS_LIMIT + position - sell_total
                allowed = min(qty, cap)
                if allowed < qty:
                    n_trimmed += 1
                if allowed > 0:
                    trimmed.append(Order(symbol, order.price, -allowed))
                    sell_total += allowed
        if n_trimmed > 0:
            self._trim_count += n_trimmed
            # logger.print(f"TRIM_WARNING {symbol}: {n_trimmed} orders trimmed — upstream logic bug")
        return trimmed

    def run(self, state: TradingState):
        result = {}

        for symbol in state.order_depths:
            order_depth = state.order_depths[symbol]

            try:
                if symbol == ASH_SYMBOL:
                    # Intercept detection on first valid tick (Section 2)
                    if self._aco_mu is None:
                        self._aco_mu = self._detect_aco_mu(order_depth)
                    # Regime check — updates _aco_mu if regime changes (Section 3)
                    self._check_aco_regime(order_depth)
                    result.update(AshTrader(state, self._aco_mu).get_orders())

                elif symbol == ROOT_SYMBOL:
                    # Regime check — updates _ipr_fv_start if regime changes (Section 3)
                    self._check_ipr_regime(order_depth, state.timestamp)
                    end_of_day_fair_value = self._ipr_fv_eod(order_depth, state.timestamp)
                    current_fair_value    = self._ipr_fv_now(state.timestamp)
                    result.update(RootTrader(state, end_of_day_fair_value, current_fair_value).get_orders())

            except Exception as e:
                logger.print(f"ERROR {symbol}: {e}")

        # Safety wrapper: trim any orders violating engine position limits (Section 1)
        for symbol, orders in list(result.items()):
            position = state.position.get(symbol, 0)
            result[symbol] = self._trim_orders(symbol, orders, position)

        logger.flush(state, result, 0, "")
        return result, 0, ""

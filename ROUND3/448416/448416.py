from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
import math
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

# AR(2) on VELVETFRUIT_EXTRACT mid, fitted in round3_timeseries.ipynb
ALPHA = 10.727097081004104
BETA1 = 0.840361578360772
BETA2 = 0.15759553490214856
MEAN  = ALPHA / (1.0 - BETA1 - BETA2)   # ≈ 5250.95

# Symbols
UNDERLYING     = "VELVETFRUIT_EXTRACT"
VOUCHER_PREFIX = "VEV_"

def voucher_symbol(strike: int) -> str:
    return f"{VOUCHER_PREFIX}{strike}"

# Position limits
U_LIMIT = 200    # VELVETFRUIT_EXTRACT
O_LIMIT = 300    # per voucher

VEV_STD       = 15.630
SKEW_MAX      = 3

####### VOUCHER STRATEGY (per discord: trevor + pirey) #######
# Three groups, each with its own quote logic:
#   SMILE_STRIKES  — fit a hardcoded polynomial smile from training (pirey).
#                    Quote around BS fair value. Direction-gate by z-score residual.
#   DELTA1_STRIKES — pure intrinsic, no time value. MM tight around wall_mid.
#   WIDE_STRIKES   — bid=0, ask=1 in market. MM the 1-tick spread directly.

SMILE_STRIKES  = [5000, 5100, 5200, 5300, 5400, 5500]
DELTA1_STRIKES = [4000, 4500]
WIDE_STRIKES   = [6000, 6500]
VOUCHER_STRIKES = SMILE_STRIKES + DELTA1_STRIKES + WIDE_STRIKES

VOUCHER_QUOTE_SIZE = 7    # per rayray: 7 is the right limit

# Hardcoded polynomial smile from pirey's training fit (K=5000–5500, daily-unit IV).
# σ_daily(x) = SMILE_A · x² + SMILE_B · x + SMILE_C   where x = ln(spot/K)
SMILE_A       = 0.086020
SMILE_B       = -0.000806
SMILE_C       = 0.012536
T_EXPIRY_DAYS = 5.0   # 5 days; uses daily-unit σ from poly above

# Per-strike chronic mispricing stats from training (market_mid - poly_FV).
# (mean, std) — used to z-score the live deviation each tick.
STRIKE_DEV_STATS = {
    5000: (-0.05, 0.57),
    5100: (-0.07, 0.92),
    5200: (+0.72, 0.93),
    5300: (+1.32, 1.16),
    5400: (-2.19, 0.78),
    5500: (+0.53, 0.44),
}
Z_GATE          = 1.0    # |z| threshold for one-sided quoting
SMILE_BASE_EDGE = 1      # half-spread around poly FV
DELTA1_EDGE     = 1      # half-spread around wall_mid for K=4000/4500

####### BLACK-SCHOLES HELPERS #######

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return F * norm_cdf(d1) - K * norm_cdf(d2)

def bs_call_delta(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if F > K else 0.0
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def implied_vol(market_price: float, F: float, K: float, T: float,
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-5, max_iter: int = 50):
    """Bisection IV solver. Returns None if no valid bracket."""
    if T <= 0:
        return None
    intrinsic = max(F - K, 0.0)
    if market_price <= intrinsic + 1e-6:
        return None
    f_lo = bs_call(F, K, T, lo) - market_price
    f_hi = bs_call(F, K, T, hi) - market_price
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call(F, K, T, mid) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)

def solve_smile(s):
    """Cramer's rule on 3x3 normal equations. s = [Σ1, Σx, Σx², Σx³, Σx⁴, Σy, Σxy, Σx²y].
    Returns (a, b, c) for IV ≈ a·x² + b·x + c, or None if singular."""
    s0, s1, s2, s3, s4, y0, y1, y2 = s
    # M = [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]],  rhs = [y0, y1, y2],  unknowns [c, b, a]
    det = (s0*(s2*s4 - s3*s3) - s1*(s1*s4 - s3*s2) + s2*(s1*s3 - s2*s2))
    if abs(det) < 1e-18:
        return None
    det_c = (y0*(s2*s4 - s3*s3) - s1*(y1*s4 - s3*y2) + s2*(y1*s3 - s2*y2))
    det_b = (s0*(y1*s4 - s3*y2) - y0*(s1*s4 - s3*s2) + s2*(s1*y2 - y1*s2))
    det_a = (s0*(s2*y2 - y1*s3) - s1*(s1*y2 - y1*s2) + y0*(s1*s3 - s2*s2))
    return det_a/det, det_b/det, det_c/det

def smile_iv(coeffs, log_mon: float) -> float:
    a, b, c = coeffs
    return a * log_mon * log_mon + b * log_mon + c

def poly_iv(log_mon: float) -> float:
    """Hardcoded smile from training. Returns σ in daily units (x = ln(spot/K))."""
    return SMILE_A * log_mon * log_mon + SMILE_B * log_mon + SMILE_C

####### BASE TRADER #######

class ProductTrader:

    def __init__(self, symbol: str, state: TradingState, pos_limit: int):
        self.name      = symbol
        self.pos_limit = pos_limit
        self.state     = state
        order_depth    = state.order_depths.get(symbol, OrderDepth())
        position       = state.position.get(symbol, 0)

        self.initial_position = position
        self.position    = position
        self.orders      = []
        self.buy_volume  = 0
        self.sell_volume = 0

        self.mkt_buy_orders  = {p: abs(v) for p, v in sorted(order_depth.buy_orders.items(),  reverse=True)} if order_depth.buy_orders  else {}
        self.mkt_sell_orders = {p: abs(v) for p, v in sorted(order_depth.sell_orders.items())}              if order_depth.sell_orders else {}
        self.bid_wall = max(self.mkt_buy_orders)  if self.mkt_buy_orders  else None
        self.ask_wall = min(self.mkt_sell_orders) if self.mkt_sell_orders else None
        self.wall_mid = (self.bid_wall + self.ask_wall) / 2.0 if self.bid_wall is not None and self.ask_wall is not None else None

    @property
    def max_allowed_buy_volume(self):
        return self.pos_limit - self.initial_position - self.buy_volume

    @property
    def max_allowed_sell_volume(self):
        return self.pos_limit + self.initial_position - self.sell_volume

    def bid(self, price, quantity, logging=False):
        fill_volume = min(quantity, self.max_allowed_buy_volume)
        if fill_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), fill_volume))
        self.buy_volume += fill_volume
        self.position   += fill_volume
        if logging:
            logger.print(f"BID {self.name} {int(price)} x{fill_volume}")

    def ask(self, price, quantity, logging=False):
        fill_volume = min(quantity, self.max_allowed_sell_volume)
        if fill_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), -fill_volume))
        self.sell_volume += fill_volume
        self.position    -= fill_volume
        if logging:
            logger.print(f"ASK {self.name} {int(price)} x{fill_volume}")

    def get_orders(self):
        return {self.name: self.orders}


####### ENGINE A — VELVETFRUIT_EXTRACT MEAN REVERSION #######
# Anchored to wall_mid for fair value, quotes skewed toward AR(2) MEAN (5250.95).
# Below MEAN -> aggressive bid + inflated bid size. Above MEAN -> mirrored ask side.
class VelvetTrader(ProductTrader):

    def __init__(self, state: TradingState, last_wall_mid=None):
        super().__init__(UNDERLYING, state, U_LIMIT)
        if self.wall_mid is None:
            self.wall_mid = last_wall_mid

    def get_orders(self):
        if self.wall_mid is None:
            return {self.name: self.orders}
        fair_value = self.wall_mid

        # 1. TAKING
        for sell_price, sell_volume in self.mkt_sell_orders.items():
            if sell_price <= fair_value - 1:
                self.bid(sell_price, sell_volume)
            elif sell_price <= fair_value and self.initial_position < 0:
                self.bid(sell_price, min(sell_volume, abs(self.initial_position)))
        for buy_price, buy_volume in self.mkt_buy_orders.items():
            if buy_price >= fair_value + 1:
                self.ask(buy_price, buy_volume)
            elif buy_price >= fair_value and self.initial_position > 0:
                self.ask(buy_price, min(buy_volume, self.initial_position))

        # 2. MAKING — skew toward MEAN
        bid_ceiling = int(fair_value) if self.position < 0 else int(fair_value - 1)
        ask_floor   = int(fair_value) if self.position > 0 else int(fair_value + 1)

        thick_bid = next((p for p in self.mkt_buy_orders  if p <= fair_value and self.mkt_buy_orders[p]  > 1), None)
        thick_ask = next((p for p in self.mkt_sell_orders if p >= fair_value and self.mkt_sell_orders[p] > 1), None)

        raw_skew = (fair_value - MEAN) / VEV_STD * SKEW_MAX
        skew = int(round(max(-2 * SKEW_MAX, min(2 * SKEW_MAX, raw_skew))))

        base_bid = (thick_bid + 1) if thick_bid is not None else int(fair_value - 6)
        base_ask = (thick_ask - 1) if thick_ask is not None else int(fair_value + 6)

        bid_price = min(base_bid - skew, bid_ceiling)
        ask_price = max(base_ask - skew, ask_floor)

        std_devs = (fair_value - MEAN) / VEV_STD
        normalized = min(abs(std_devs) / 3, 1.0)
        if std_devs > 0:
            maker_buy_volume  = min(int(6 + (self.max_allowed_buy_volume  - 6) * normalized), self.max_allowed_buy_volume)
            maker_sell_volume = min(6, self.max_allowed_sell_volume)
        else:
            maker_buy_volume  = min(6, self.max_allowed_buy_volume)
            maker_sell_volume = min(int(6 + (self.max_allowed_sell_volume - 6) * normalized), self.max_allowed_sell_volume)

        self.bid(bid_price, maker_buy_volume)
        self.ask(ask_price, maker_sell_volume)
        return {self.name: self.orders}


####### ENGINE B — VOUCHER MARKET MAKING #######
# Three modes per strike group:
#   SMILE_STRIKES  → poly FV + z-residual directional gate (per pirey/trevor)
#   DELTA1_STRIKES → MM tight around wall_mid (no FV from poly — intrinsic only)
#   WIDE_STRIKES   → bid 0 / ask 1 (capture the chronic 1-tick spread)
class VoucherTrader(ProductTrader):

    def __init__(self, state: TradingState, strike: int, spot: float):
        super().__init__(voucher_symbol(strike), state, O_LIMIT)
        self.strike = strike
        self.spot   = spot

    def get_orders(self):
        if self.spot is None or self.spot <= 0 or self.wall_mid is None:
            return {self.name: self.orders}
        K = self.strike
        if K in WIDE_STRIKES:
            self._mm_wide()
        elif K in DELTA1_STRIKES:
            self._mm_delta1()
        elif K in SMILE_STRIKES:
            self._mm_smile()
        return {self.name: self.orders}

    def _mm_wide(self):
        # Market is bid 0 / ask 1. Sit on both sides, collect 1 tick when crossed.
        self.bid(0, VOUCHER_QUOTE_SIZE)
        self.ask(1, VOUCHER_QUOTE_SIZE)

    def _mm_delta1(self):
        # Pure intrinsic, behaves like underlying. MM tight around wall_mid.
        wall_mid = self.wall_mid
        bid_p = int(round(wall_mid - DELTA1_EDGE))
        ask_p = int(round(wall_mid + DELTA1_EDGE))
        if bid_p > 0:
            self.bid(bid_p, VOUCHER_QUOTE_SIZE)
        self.ask(ask_p, VOUCHER_QUOTE_SIZE)

    def _mm_smile(self):
        K = self.strike
        log_mon = math.log(self.spot / K)
        sigma   = poly_iv(log_mon)
        if sigma <= 0:
            return
        fv = bs_call(self.spot, K, T_EXPIRY_DAYS, sigma)

        mean_dev, std_dev = STRIKE_DEV_STATS.get(K, (0.0, 1.0))
        dev = self.wall_mid - fv
        z   = (dev - mean_dev) / std_dev if std_dev > 0 else 0.0

        my_bid = int(round(fv - SMILE_BASE_EDGE))
        my_ask = int(round(fv + SMILE_BASE_EDGE))

        # Directional gate: market chronically rich → only sell. Cheap → only buy.
        post_bid = True
        post_ask = True
        if   z >  Z_GATE: post_bid = False
        elif z < -Z_GATE: post_ask = False

        if post_bid and my_bid > 0:
            self.bid(my_bid, VOUCHER_QUOTE_SIZE)
        if post_ask:
            self.ask(my_ask, VOUCHER_QUOTE_SIZE)


####### MAIN #######

class Trader:

    def __init__(self):
        pass

    def _voucher_mid(self, state: TradingState, strike: int):
        od = state.order_depths.get(voucher_symbol(strike))
        if od is None or not od.buy_orders or not od.sell_orders:
            return None
        return 0.5 * (max(od.buy_orders) + min(od.sell_orders))

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}

        # Spot = underlying wall_mid
        u_od = state.order_depths.get(UNDERLYING)
        spot = None
        if u_od is not None and u_od.buy_orders and u_od.sell_orders:
            spot = 0.5 * (max(u_od.buy_orders) + min(u_od.sell_orders))
        if spot is None:
            spot = trader_data.get("vev_wall_mid")

        logger.print(f"spot={spot}")

        # Engine A
        try:
            vt = VelvetTrader(state, last_wall_mid=trader_data.get("vev_wall_mid"))
            result.update(vt.get_orders())
            if vt.wall_mid is not None:
                trader_data["prev_prev_mid"] = trader_data.get("prev_mid")
                trader_data["prev_mid"]      = vt.wall_mid
                trader_data["vev_wall_mid"]  = vt.wall_mid
        except Exception as e:
            logger.print(f"ERROR {UNDERLYING}: {e}")

        # Engine B — one VoucherTrader per strike (mode dispatched inside the class)
        for k in VOUCHER_STRIKES:
            try:
                vrt = VoucherTrader(state, k, spot)
                result.update(vrt.get_orders())
            except Exception as e:
                logger.print(f"ERROR {voucher_symbol(k)}: {e}")

        out_trader_data = json.dumps(trader_data)
        logger.flush(state, result, 0, out_trader_data)
        return result, 0, out_trader_data
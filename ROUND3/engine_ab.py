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
UNDERLYING      = "VELVETFRUIT_EXTRACT"
VOUCHER_PREFIX  = "VEV_"
ALL_STRIKES     = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
# Far-OTM "wide" strikes: book sits at bid=0/ask=1. Spread infinite (tick > price).
# MM the 0/1 spread directly — never take, only post passive. Strike-cap (50) limits damage.
WIDE_STRIKES    = [6000, 6500]

def voucher_symbol(strike: int) -> str:
    return f"{VOUCHER_PREFIX}{strike}"

# Position limits
U_LIMIT = 200    # VELVETFRUIT_EXTRACT
O_LIMIT = 300    # per voucher

VEV_STD       = 15.630
SKEW_MAX      = 3       # TODO: tune — max ticks of price skew at 1σ from MEAN
SPREAD_FRACTION = 0.6   # our quoted spread as a fraction of the market spread (50→30, 5→3)
EMA_ALPHA       = 0.02  # ~50-tick EMA for spot z-score
Z_TAKE_BUY      = 0.5   # only take buys if z-score < threshold (price not elevated vs history)
Z_TAKE_SELL     = -0.5  # only take sells if z-score > threshold

PACK_SYMBOL   = "HYDROGEL_PACK"
PACK_LIMIT    = 200
PACK_MEAN     = 9990
PACK_STD      = 32      # tune from historical data
PACK_SKEW_MAX = 3       # max ticks of skew at 1σ away from PACK_MEAN

# Engine B — voucher market making
T_EXPIRY  = 5.0 / 365.0   # TODO: held constant per user. 5 days till expiry. σ must match unit.
BASE_EDGE = 1             # TODO: tune — half-spread around BS fair value (FV safety rail)
SPREAD    = 4             # TODO: tune — magnitude of inventory-lean shift
VOUCHER_QUOTE_SIZE = 7    # per-side size on each voucher quote — small or won't get filled

REGRET_THRESHOLD = 1500   # TODO: tune — net portfolio delta threshold for throttle / lean denominator

# Live smile fit (8-float EWMA state in traderData) — see options.ipynb live-engine test.
# γ=0.999 → effective memory ~1000 ticks.
SMILE_GAMMA      = 0.999
# Seed sums from training (rounds 0-2 of round 3). Replays all 30k ticks with γ=0.999.
# These sums produce coeffs a≈9.62, b≈-0.065, c≈0.24 — end-of-training smile.
# Engine starts hot — no warmup window. Old data still decays because EWMA keeps applying γ each tick.
INITIAL_SMILE_SUMS = [
    8.336818e+03,
    -2.479861e+02,
    8.282092e+01,
    -7.868455e+00,
    3.140089e+00,
    2.807962e+03,
    -1.404019e+02,
    5.052927e+01,
]

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

####### BASE TRADER #######

class ProductTrader:

    def __init__(self, symbol: str, state: TradingState, pos_limit: int,
                 extra_buy_used: int = 0, extra_sell_used: int = 0):
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
        self.extra_buy_used  = extra_buy_used    # qty already promised by an upstream engine (e.g. arb)
        self.extra_sell_used = extra_sell_used

        self.mkt_buy_orders  = {p: abs(v) for p, v in sorted(order_depth.buy_orders.items(),  reverse=True)} if order_depth.buy_orders  else {}
        self.mkt_sell_orders = {p: abs(v) for p, v in sorted(order_depth.sell_orders.items())}              if order_depth.sell_orders else {}
        self.bid_wall = max(self.mkt_buy_orders)  if self.mkt_buy_orders  else None
        self.ask_wall = min(self.mkt_sell_orders) if self.mkt_sell_orders else None
        self.wall_mid = (self.bid_wall + self.ask_wall) / 2.0 if self.bid_wall is not None and self.ask_wall is not None else None

    @property
    def max_allowed_buy_volume(self):
        return self.pos_limit - self.initial_position - self.buy_volume - self.extra_buy_used

    @property
    def max_allowed_sell_volume(self):
        return self.pos_limit + self.initial_position - self.sell_volume - self.extra_sell_used

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

    def __init__(self, state: TradingState, last_wall_mid=None, vev_zscore: float = 0.0):
        super().__init__(UNDERLYING, state, U_LIMIT)
        if self.wall_mid is None:
            self.wall_mid = last_wall_mid
        self.vev_zscore = vev_zscore

    def get_orders(self):
        if self.wall_mid is None:
            return {self.name: self.orders}
        fair_value = self.wall_mid

        # 1. TAKING — only when z-score confirms price is favorable vs recent history
        if self.vev_zscore < Z_TAKE_BUY:
            for sell_price, sell_volume in self.mkt_sell_orders.items():
                if sell_price <= fair_value - 1:
                    self.bid(sell_price, sell_volume)
                elif sell_price <= fair_value and self.initial_position < 0:
                    self.bid(sell_price, min(sell_volume, abs(self.initial_position)))
        if self.vev_zscore > Z_TAKE_SELL:
            for buy_price, buy_volume in self.mkt_buy_orders.items():
                if buy_price >= fair_value + 1:
                    self.ask(buy_price, buy_volume)
                elif buy_price >= fair_value and self.initial_position > 0:
                    self.ask(buy_price, min(buy_volume, self.initial_position))

        # 2. MAKING — 5c inside thick wall, skew toward MEAN
        bid_ceiling = int(fair_value) if self.position < 0 else int(fair_value - 1)
        ask_floor   = int(fair_value) if self.position > 0 else int(fair_value + 1)

        thick_bid = next((p for p in self.mkt_buy_orders  if p <= fair_value and self.mkt_buy_orders[p]  > 1), None)
        thick_ask = next((p for p in self.mkt_sell_orders if p >= fair_value and self.mkt_sell_orders[p] > 1), None)

        raw_skew = (fair_value - MEAN) / VEV_STD * SKEW_MAX
        skew = int(round(max(-2 * SKEW_MAX, min(2 * SKEW_MAX, raw_skew))))

        if thick_bid is not None and thick_ask is not None:
            improve  = int((1 - SPREAD_FRACTION) / 2 * (thick_ask - thick_bid))
            base_bid = thick_bid + improve
            base_ask = thick_ask - improve
        elif thick_bid is not None:
            base_bid = thick_bid + 1
            base_ask = int(fair_value + 2)
        elif thick_ask is not None:
            base_bid = int(fair_value - 2)
            base_ask = thick_ask - 1
        else:
            base_bid = int(fair_value - 2)
            base_ask = int(fair_value + 2)

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



class PACKTrader(ProductTrader):

    def __init__(self, state: TradingState, last_wall_mid=None):
        super().__init__(PACK_SYMBOL, state, PACK_LIMIT)
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

        # 2. MAKING — skew toward PACK_MEAN
        bid_ceiling = int(fair_value) if self.position < 0 else int(fair_value - 1)
        ask_floor   = int(fair_value) if self.position > 0 else int(fair_value + 1)

        thick_bid = next((p for p in self.mkt_buy_orders  if p <= fair_value and self.mkt_buy_orders[p]  > 1), None)
        thick_ask = next((p for p in self.mkt_sell_orders if p >= fair_value and self.mkt_sell_orders[p] > 1), None)

        raw_skew = (fair_value - PACK_MEAN) / PACK_STD * PACK_SKEW_MAX
        skew = int(round(max(-2 * PACK_SKEW_MAX, min(2 * PACK_SKEW_MAX, raw_skew))))

        base_bid = (thick_bid + 1) if thick_bid is not None else int(fair_value - 6)
        base_ask = (thick_ask - 1) if thick_ask is not None else int(fair_value + 6)

        bid_price = min(base_bid - skew, bid_ceiling)
        ask_price = max(base_ask - skew, ask_floor)

        std_devs = (fair_value - PACK_MEAN) / PACK_STD
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
# FV = BS_call(F=spot, K=strike, T=T_EXPIRY, σ=smile_iv(K, spot))
# Smile σ comes from the live EWMA parabola fit maintained in Trader.run.
# Lean quotes against portfolio delta. Throttle = kill bid/ask side past threshold.
class VoucherTrader(ProductTrader):

    def __init__(self, state: TradingState, strike: int, portfolio_delta: float,
                 smile_coeffs, spot: float):
        super().__init__(voucher_symbol(strike), state, O_LIMIT)
        self.strike          = strike
        self.portfolio_delta = portfolio_delta
        self.smile_coeffs    = smile_coeffs
        self.spot            = spot

    def get_orders(self):
        # Wide-strike override: K=6000/6500 sit at bid=0/ask=1 in the book. Tick size = full spread.
        # Just post at 0 and 1 — pure passive MM, no take. Sanity-check the book first;
        # if the microstructure shifts (e.g. mid moves above 0.5), bail out.
        if self.strike in WIDE_STRIKES:
            if self.bid_wall == 0 and self.ask_wall == 1:
                self.bid(0, VOUCHER_QUOTE_SIZE)
                self.ask(1, VOUCHER_QUOTE_SIZE)
            return {self.name: self.orders}

        if self.smile_coeffs is None or self.spot is None or self.spot <= 0:
            return {self.name: self.orders}

        log_mon = math.log(self.spot / self.strike)
        sigma   = smile_iv(self.smile_coeffs, log_mon)
        if sigma <= 0:
            return {self.name: self.orders}

        fv = bs_call(self.spot, self.strike, T_EXPIRY, sigma)

        inventory_lean = self.portfolio_delta / REGRET_THRESHOLD

        # Quote at SPREAD_FRACTION of market spread; FV ± BASE_EDGE is the safety rail
        if self.bid_wall is not None and self.ask_wall is not None:
            improve = (1 - SPREAD_FRACTION) / 2 * (self.ask_wall - self.bid_wall)
            my_bid  = min(self.bid_wall + improve, fv - BASE_EDGE)
            my_ask  = max(self.ask_wall - improve, fv + BASE_EDGE)
        else:
            my_bid = fv - BASE_EDGE
            my_ask = fv + BASE_EDGE
        my_bid -= inventory_lean * SPREAD
        my_ask -= inventory_lean * SPREAD

        # 4. THROTTLING — skip the offending side rather than posting useless 0 / 999999 orders
        post_bid = self.portfolio_delta <=  REGRET_THRESHOLD
        post_ask = self.portfolio_delta >= -REGRET_THRESHOLD

        if post_bid and my_bid > 0:
            self.bid(int(round(my_bid)), VOUCHER_QUOTE_SIZE)
        if post_ask:
            self.ask(int(round(my_ask)), VOUCHER_QUOTE_SIZE)

        return {self.name: self.orders}


####### MAIN #######

class Trader:

    def __init__(self):
        pass

    def _voucher_mid(self, state: TradingState, strike: int):
        od = state.order_depths.get(voucher_symbol(strike))
        if od is None or not od.buy_orders or not od.sell_orders:
            return None
        return 0.5 * (max(od.buy_orders) + min(od.sell_orders))

    def _call_spread_arb(self, state: TradingState):
        """Scan all strike pairs for call-spread bound violations:
            0 ≤ Call(K_low) - Call(K_high) ≤ K_high - K_low
        Violations = locked-profit arb. Returns (orders_dict, arb_used dict, n_arbs).
        arb_used[k] = (buy_qty_used, sell_qty_used) consumed by arb on this tick."""
        arb_orders: dict[Symbol, list[Order]] = {}
        arb_used: dict[int, list[int]] = {k: [0, 0] for k in VOUCHER_STRIKES}   # [buy, sell]
        n_arbs = 0
        strikes = sorted(VOUCHER_STRIKES)

        def remaining_buy(k):
            pos = state.position.get(voucher_symbol(k), 0)
            return O_LIMIT - pos - arb_used[k][0]

        def remaining_sell(k):
            pos = state.position.get(voucher_symbol(k), 0)
            return O_LIMIT + pos - arb_used[k][1]

        for i, ki in enumerate(strikes):
            for kj in strikes[i + 1:]:
                sym_lo = voucher_symbol(ki)
                sym_hi = voucher_symbol(kj)
                od_lo = state.order_depths.get(sym_lo)
                od_hi = state.order_depths.get(sym_hi)
                if od_lo is None or od_hi is None:
                    continue
                strike_diff = kj - ki

                # Direction A: Call(K_low) too expensive vs Call(K_high)
                # Bound: bid_lo - ask_hi ≤ strike_diff. Violation → sell low, buy high.
                if od_lo.buy_orders and od_hi.sell_orders:
                    bid_lo  = max(od_lo.buy_orders)
                    ask_hi  = min(od_hi.sell_orders)
                    edge    = bid_lo - ask_hi - strike_diff
                    if edge > 0:
                        bidq_lo = od_lo.buy_orders[bid_lo]
                        askq_hi = -od_hi.sell_orders[ask_hi]
                        qty = min(bidq_lo, askq_hi, remaining_sell(ki), remaining_buy(kj))
                        if qty > 0:
                            arb_orders.setdefault(sym_lo, []).append(Order(sym_lo, bid_lo, -qty))
                            arb_orders.setdefault(sym_hi, []).append(Order(sym_hi, ask_hi,  qty))
                            arb_used[ki][1] += qty
                            arb_used[kj][0] += qty
                            n_arbs += 1
                            logger.print(f"ARB+ sell {sym_lo}@{bid_lo} buy {sym_hi}@{ask_hi} qty={qty} edge={edge:.1f}")

                # Direction B: Call(K_high) priced above Call(K_low) — also a violation.
                # Bound: bid_hi - ask_lo ≤ 0. Violation → buy low, sell high.
                if od_lo.sell_orders and od_hi.buy_orders:
                    ask_lo  = min(od_lo.sell_orders)
                    bid_hi  = max(od_hi.buy_orders)
                    edge    = bid_hi - ask_lo
                    if edge > 0:
                        askq_lo = -od_lo.sell_orders[ask_lo]
                        bidq_hi = od_hi.buy_orders[bid_hi]
                        qty = min(askq_lo, bidq_hi, remaining_buy(ki), remaining_sell(kj))
                        if qty > 0:
                            arb_orders.setdefault(sym_lo, []).append(Order(sym_lo, ask_lo,  qty))
                            arb_orders.setdefault(sym_hi, []).append(Order(sym_hi, bid_hi, -qty))
                            arb_used[ki][0] += qty
                            arb_used[kj][1] += qty
                            n_arbs += 1
                            logger.print(f"ARB- buy {sym_lo}@{ask_lo} sell {sym_hi}@{bid_hi} qty={qty} edge={edge:.1f}")

        return arb_orders, arb_used, n_arbs

    def _update_smile(self, state: TradingState, smile_sums, spot: float):
        """Decay state by γ, then add this tick's (log_mon, iv) contributions across all 10 strikes.
        Returns updated sums + fitted coeffs (or None if pre-warmup / singular)."""
        new_sums = [SMILE_GAMMA * s for s in smile_sums]
        for k in ALL_STRIKES:
            mid = self._voucher_mid(state, k)
            if mid is None:
                continue
            iv = implied_vol(mid, spot, k, T_EXPIRY)
            if iv is None or iv <= 0 or iv > 5:
                continue
            x = math.log(spot / k)
            new_sums[0] += 1.0
            new_sums[1] += x
            new_sums[2] += x * x
            new_sums[3] += x * x * x
            new_sums[4] += x * x * x * x
            new_sums[5] += iv
            new_sums[6] += x * iv
            new_sums[7] += x * x * iv
        coeffs = solve_smile(new_sums)
        return new_sums, coeffs

    def _portfolio_delta(self, state: TradingState, smile_coeffs, spot) -> float:
        # Underlying contributes delta = position. Each voucher contributes pos × BS delta.
        delta = float(state.position.get(UNDERLYING, 0))
        if smile_coeffs is None or spot is None or spot <= 0:
            return delta
        for k in VOUCHER_STRIKES:
            pos = state.position.get(voucher_symbol(k), 0)
            if pos == 0:
                continue
            log_mon = math.log(spot / k)
            sigma = smile_iv(smile_coeffs, log_mon)
            if sigma <= 0:
                continue
            delta += pos * bs_call_delta(spot, k, T_EXPIRY, sigma)
        return delta

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}

        # AR(2) state for future scalping (S_next currently unused by either engine)
        prev_mid      = trader_data.get("prev_mid")
        prev_prev_mid = trader_data.get("prev_prev_mid")
        s_next = None
        if prev_mid is not None and prev_prev_mid is not None:
            s_next = BETA1 * prev_mid + BETA2 * prev_prev_mid + ALPHA

        # Determine spot from underlying order book (used for smile fit + Engine B FV)
        u_od = state.order_depths.get(UNDERLYING)
        spot = None
        if u_od is not None and u_od.buy_orders and u_od.sell_orders:
            spot = 0.5 * (max(u_od.buy_orders) + min(u_od.sell_orders))
        if spot is None:
            spot = trader_data.get("vev_wall_mid")

        # Spot EMA z-score — gates VelvetTrader taking to price-history-favorable moments
        vev_ema     = trader_data.get("vev_ema")
        vev_ema_var = trader_data.get("vev_ema_var", VEV_STD * VEV_STD)
        if spot is not None:
            if vev_ema is None:
                vev_ema = spot
            diff        = spot - vev_ema
            vev_ema     = vev_ema + EMA_ALPHA * diff
            vev_ema_var = (1 - EMA_ALPHA) * (vev_ema_var + EMA_ALPHA * diff * diff)
            trader_data["vev_ema"]     = vev_ema
            trader_data["vev_ema_var"] = vev_ema_var
        ema_std    = math.sqrt(max(vev_ema_var, 1.0)) if vev_ema_var is not None else VEV_STD
        vev_zscore = (spot - vev_ema) / ema_std if (spot is not None and vev_ema is not None) else 0.0

        # Live smile fit — 8-float EWMA state. Cold-start from training-fitted sums.
        smile_sums = trader_data.get("smile_sums", list(INITIAL_SMILE_SUMS))
        smile_coeffs = None
        if spot is not None and spot > 0:
            smile_sums, smile_coeffs = self._update_smile(state, smile_sums, spot)
        trader_data["smile_sums"] = smile_sums

        portfolio_delta = self._portfolio_delta(state, smile_coeffs, spot)
        logger.print(f"delta={portfolio_delta:.1f} spot={spot} smile={smile_coeffs} n={smile_sums[0]:.1f}")

        # Engine A — VELVETFRUIT_EXTRACT
        try:
            vt = VelvetTrader(state, last_wall_mid=trader_data.get("vev_wall_mid"), vev_zscore=vev_zscore)
            result.update(vt.get_orders())
            if vt.wall_mid is not None:
                trader_data["prev_prev_mid"] = trader_data.get("prev_mid")
                trader_data["prev_mid"]      = vt.wall_mid
                trader_data["vev_wall_mid"]  = vt.wall_mid
        except Exception as e:
            logger.print(f"ERROR {UNDERLYING}: {e}")

        try:
            pt = PACKTrader(state, last_wall_mid=trader_data.get("pack_wall_mid"))
            result.update(pt.get_orders())
            if pt.wall_mid is not None:
                trader_data["pack_wall_mid"] = pt.wall_mid
        except Exception as e:
            logger.print(f"ERROR {PACK_SYMBOL}: {e}")

        # Engine C — cross-strike call-spread arbitrage (locked profit when book violates bounds).
        try:
            arb_orders, arb_used, n_arbs = self._call_spread_arb(state)
            for sym, orders in arb_orders.items():
                result.setdefault(sym, []).extend(orders)
            if n_arbs:
                logger.print(f"arbs_fired={n_arbs}")
        except Exception as e:
            logger.print(f"ERROR arb: {e}")
            arb_used = {k: [0, 0] for k in VOUCHER_STRIKES}

        # Engine B — one VoucherTrader per strike (capacity reduced by arb's per-strike usage)
        for k in VOUCHER_STRIKES:
            try:
                buy_used, sell_used = arb_used.get(k, (0, 0))
                vrt = VoucherTrader(state, k, portfolio_delta, smile_coeffs, spot)
                vrt.extra_buy_used  = buy_used
                vrt.extra_sell_used = sell_used
                voucher_orders = vrt.get_orders()
                for sym, orders in voucher_orders.items():
                    result.setdefault(sym, []).extend(orders)
            except Exception as e:
                logger.print(f"ERROR {voucher_symbol(k)}: {e}")

        out_trader_data = json.dumps(trader_data)
        logger.flush(state, result, 0, out_trader_data)
        return result, 0, out_trader_data

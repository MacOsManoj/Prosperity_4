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

# AR(2) on VELVETFRUIT_EXTRACT — from round3_timeseries.ipynb
AR2_C    = 10.727097081004104
AR2_B1   = 0.840361578360772
AR2_B2   = 0.15759553490214856
AR2_MEAN = AR2_C / (1.0 - AR2_B1 - AR2_B2)   # ≈ 5250.95

# Symbols
UNDERLYING       = "VELVETFRUIT_EXTRACT"
ALL_STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
ACTIVE_STRIKES   = [5000, 5100, 5200, 5300, 5400, 5500]   # z-score MM tier
DEEP_ITM_STRIKES = [4000, 4500]                             # intrinsic MM tier
DEEP_OTM_STRIKES = [6000, 6500]                             # bid=0 / ask=1 tier
VOUCHER_STRIKES  = ALL_STRIKES                              # for arb scanner

def voucher_symbol(k: int) -> str:
    return f"VEV_{k}"

# Position limits
U_LIMIT = 200    # VELVETFRUIT_EXTRACT
O_LIMIT = 300

# Engine A — underlying mean reversion
VEV_STD  = 15.630
SKEW_MAX = 3

# Smile fit (live EWMA quadratic parabola, 8-float state)
T_EXPIRY       = 5.0 / 365.0
SMILE_GAMMA    = 0.999
SMILE_WARMUP_N = 500    # obs count before trusting fit (~50 ticks with 6 active strikes/tick)

# Z-score MM — quoting
BASE_HALF_SPREAD = 1     # minimum edge per side (ticks)
MAX_LEAN         = 2     # max additional edge shift toward inventory target (ticks)
QUOTE_SIZE       = 7     # lots per quote

# Z-score MM — EWMA for per-strike deviation tracking
RV_ALPHA = 0.9990   # decay per tick (~1000-tick memory ≈ 1 day)
MIN_STD  = 0.30     # floor on std_dev to prevent z-score blowup

# Active strike inventory targeting (separate from O_LIMIT to avoid giant z-score positions)
ACTIVE_POS_LIMIT = 100   # max inventory targeted via z-score per active strike

# Delta scaling: keeps net target delta bounded so z-score bets don't all pile the same way
DELTA_TARGET_CAP = 100

# Deep ITM params
DITM_GAMMA       = 0.05    # reservation price skew per inventory unit (ticks)
DITM_TAKE_THRESH = 2.0     # taker threshold (ticks from intrinsic fair)

# Historical priors — from trevor_sabr.ipynb Finding 5 (time-average of market_mid - BS_FV_smile)
# These bootstrap the EWMA so the algo starts calibrated rather than cold
PRIOR_MEAN: dict[str, float] = {
    "5000": -0.06, "5100": -0.08, "5200":  0.72,
    "5300":  1.31, "5400": -2.19, "5500":  0.53,
}
PRIOR_STD: dict[str, float] = {
    "5000":  0.56, "5100":  0.92, "5200":  0.93,
    "5300":  1.16, "5400":  0.78, "5500":  0.44,
}
PRIOR_VAR: dict[str, float] = {k: PRIOR_STD[k] ** 2 for k in PRIOR_STD}

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
    s0, s1, s2, s3, s4, y0, y1, y2 = s
    det = (s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2))
    if abs(det) < 1e-18:
        return None
    det_c = (y0 * (s2 * s4 - s3 * s3) - s1 * (y1 * s4 - s3 * y2) + s2 * (y1 * s3 - s2 * y2))
    det_b = (s0 * (y1 * s4 - s3 * y2) - y0 * (s1 * s4 - s3 * s2) + s2 * (s1 * y2 - y1 * s2))
    det_a = (s0 * (s2 * y2 - y1 * s3) - s1 * (s1 * y2 - y1 * s2) + y0 * (s1 * s3 - s2 * s2))
    return det_a / det, det_b / det, det_c / det

def smile_iv(coeffs, log_mon: float) -> float:
    a, b, c = coeffs
    return a * log_mon * log_mon + b * log_mon + c

def position_sigmoid(z_abs: float) -> float:
    """Target fraction of max inventory. f(1)≈0.68, f(2)≈0.95, f(3)≈1.0."""
    return 2.0 * norm_cdf(z_abs) - 1.0

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
        self.extra_buy_used  = extra_buy_used
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

    def bid(self, price, quantity):
        fill = min(quantity, self.max_allowed_buy_volume)
        if fill <= 0:
            return
        self.orders.append(Order(self.name, int(price), fill))
        self.buy_volume += fill
        self.position   += fill

    def ask(self, price, quantity):
        fill = min(quantity, self.max_allowed_sell_volume)
        if fill <= 0:
            return
        self.orders.append(Order(self.name, int(price), -fill))
        self.sell_volume += fill
        self.position    -= fill

    def get_orders(self):
        return {self.name: self.orders}



class VelvetTrader(ProductTrader):

    def __init__(self, state: TradingState, last_wall_mid=None):
        super().__init__(UNDERLYING, state, U_LIMIT)
        if self.wall_mid is None:
            self.wall_mid = last_wall_mid

    def get_orders(self):
        if self.wall_mid is None:
            return {self.name: self.orders}
        fair_value = self.wall_mid

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

        bid_ceiling = int(fair_value) if self.position < 0 else int(fair_value - 1)
        ask_floor   = int(fair_value) if self.position > 0 else int(fair_value + 1)

        thick_bid = next((p for p in self.mkt_buy_orders  if p <= fair_value and self.mkt_buy_orders[p]  > 1), None)
        thick_ask = next((p for p in self.mkt_sell_orders if p >= fair_value and self.mkt_sell_orders[p] > 1), None)

        raw_skew = (fair_value - AR2_MEAN) / VEV_STD * SKEW_MAX
        skew = int(round(max(-2 * SKEW_MAX, min(2 * SKEW_MAX, raw_skew))))

        base_bid = (thick_bid + 1) if thick_bid is not None else int(fair_value - 6)
        base_ask = (thick_ask - 1) if thick_ask is not None else int(fair_value + 6)

        bid_price = min(base_bid - skew, bid_ceiling)
        ask_price = max(base_ask - skew, ask_floor)

        std_devs   = (fair_value - AR2_MEAN) / VEV_STD
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


####### MAIN #######

class Trader:

    def __init__(self):
        pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _voucher_mid(self, state: TradingState, strike: int):
        od = state.order_depths.get(voucher_symbol(strike))
        if od is None or not od.buy_orders or not od.sell_orders:
            return None
        return 0.5 * (max(od.buy_orders) + min(od.sell_orders))

    def _update_smile(self, state: TradingState, smile_sums, spot: float):
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
        coeffs = solve_smile(new_sums) if new_sums[0] >= SMILE_WARMUP_N else None
        return new_sums, coeffs

    def _portfolio_delta(self, state: TradingState, smile_coeffs, spot) -> float:
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

    def _call_spread_arb(self, state: TradingState):
        arb_orders: dict[Symbol, list[Order]] = {}
        arb_used: dict[int, list[int]] = {k: [0, 0] for k in VOUCHER_STRIKES}
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

                if od_lo.buy_orders and od_hi.sell_orders:
                    bid_lo = max(od_lo.buy_orders)
                    ask_hi = min(od_hi.sell_orders)
                    edge   = bid_lo - ask_hi - strike_diff
                    if edge > 0:
                        qty = min(od_lo.buy_orders[bid_lo], -od_hi.sell_orders[ask_hi],
                                  remaining_sell(ki), remaining_buy(kj))
                        if qty > 0:
                            arb_orders.setdefault(sym_lo, []).append(Order(sym_lo, bid_lo, -qty))
                            arb_orders.setdefault(sym_hi, []).append(Order(sym_hi, ask_hi,  qty))
                            arb_used[ki][1] += qty
                            arb_used[kj][0] += qty
                            n_arbs += 1

                if od_lo.sell_orders and od_hi.buy_orders:
                    ask_lo = min(od_lo.sell_orders)
                    bid_hi = max(od_hi.buy_orders)
                    edge   = bid_hi - ask_lo
                    if edge > 0:
                        qty = min(-od_lo.sell_orders[ask_lo], od_hi.buy_orders[bid_hi],
                                  remaining_buy(ki), remaining_sell(kj))
                        if qty > 0:
                            arb_orders.setdefault(sym_lo, []).append(Order(sym_lo, ask_lo,  qty))
                            arb_orders.setdefault(sym_hi, []).append(Order(sym_hi, bid_hi, -qty))
                            arb_used[ki][0] += qty
                            arb_used[kj][1] += qty
                            n_arbs += 1

        return arb_orders, arb_used, n_arbs

    # ── Strike tier handlers ───────────────────────────────────────────────────

    def _trade_deep_itm(self, K: int, state: TradingState, result: dict, spot: float,
                         arb_used: dict):
        """Pure maker: intrinsic ± 1 with inventory skew. No taking."""
        sym = voucher_symbol(K)
        od  = state.order_depths.get(sym, OrderDepth())
        pos = state.position.get(sym, 0)

        mkt_bid = max(od.buy_orders)  if od.buy_orders  else None
        mkt_ask = min(od.sell_orders) if od.sell_orders else None

        fair        = max(spot - K, 0.0)
        reservation = fair - pos * DITM_GAMMA

        arb_b, arb_s = arb_used.get(K, [0, 0])
        buy_cap  = O_LIMIT - pos - arb_b
        sell_cap = O_LIMIT + pos - arb_s

        bid_price = int(reservation) - 1
        ask_price = int(reservation) + 1
        if ask_price <= bid_price:
            ask_price = bid_price + 1
        if mkt_ask is not None:
            bid_price = min(bid_price, mkt_ask - 1)
        if mkt_bid is not None:
            ask_price = max(ask_price, mkt_bid + 1)

        orders = []
        if buy_cap > 0 and bid_price > 0:
            orders.append(Order(sym, bid_price,  min(QUOTE_SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(sym, ask_price, -min(QUOTE_SIZE, sell_cap)))
        if orders:
            result.setdefault(sym, []).extend(orders)

    def _trade_deep_otm(self, K: int, state: TradingState, result: dict, arb_used: dict):
        sym = voucher_symbol(K)
        od  = state.order_depths.get(sym, OrderDepth())
        pos = state.position.get(sym, 0)

        if od.buy_orders and max(od.buy_orders) != 0:
            return
        if od.sell_orders and min(od.sell_orders) != 1:
            return

        arb_b, arb_s = arb_used.get(K, [0, 0])
        buy_cap  = O_LIMIT - pos - arb_b
        sell_cap = O_LIMIT + pos - arb_s

        orders = []
        if buy_cap > 0:
            orders.append(Order(sym, 0,  min(QUOTE_SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(sym, 1, -min(QUOTE_SIZE, sell_cap)))
        if orders:
            result.setdefault(sym, []).extend(orders)

    def _trade_active_strike(self, K: int, state: TradingState, result: dict,
                              adj_fv: float, target_pos: float, arb_used: dict):
        """Pure maker: asymmetric spread leaned toward target. No taking."""
        sym = voucher_symbol(K)
        od  = state.order_depths.get(sym, OrderDepth())
        pos = state.position.get(sym, 0)

        mkt_bid = max(od.buy_orders)  if od.buy_orders  else None
        mkt_ask = min(od.sell_orders) if od.sell_orders else None

        arb_b, arb_s = arb_used.get(K, [0, 0])
        buy_cap  = O_LIMIT - pos - arb_b
        sell_cap = O_LIMIT + pos - arb_s

        # gap_frac normalised against ACTIVE_POS_LIMIT (not O_LIMIT) so lean is meaningful
        gap_frac = max(-1.0, min(1.0, (target_pos - pos) / ACTIVE_POS_LIMIT))
        lean     = gap_frac * MAX_LEAN   # positive = want to buy → tighter bid, wider ask

        bid_edge = max(0.0, BASE_HALF_SPREAD - lean)
        ask_edge = max(1.0, BASE_HALF_SPREAD + lean)

        bid_price = int(round(adj_fv - bid_edge))
        ask_price = int(round(adj_fv + ask_edge))

        # Hard limits and anti-cross clamps
        bid_price = min(bid_price, int(adj_fv))
        ask_price = max(ask_price, int(adj_fv) + 1)
        if mkt_ask is not None and bid_price >= mkt_ask:
            bid_price = mkt_ask - 1
        if mkt_bid is not None and ask_price <= mkt_bid:
            ask_price = mkt_bid + 1
        if bid_price >= ask_price:
            ask_price = bid_price + 1

        orders = []
        if buy_cap > 0 and bid_price > 0:
            orders.append(Order(sym, bid_price,  min(QUOTE_SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(sym, ask_price, -min(QUOTE_SIZE, sell_cap)))
        if orders:
            result.setdefault(sym, []).extend(orders)

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}

        # Load per-strike deviation EWMA (bootstrap with historical priors on cold start)
        dev_mean: dict[str, float] = trader_data.get("dev_mean", {k: PRIOR_MEAN[k] for k in PRIOR_MEAN})
        dev_var:  dict[str, float] = trader_data.get("dev_var",  {k: PRIOR_VAR[k]  for k in PRIOR_VAR})

        # Spot price from underlying book
        u_od = state.order_depths.get(UNDERLYING)
        spot: float | None = None
        if u_od is not None and u_od.buy_orders and u_od.sell_orders:
            spot = 0.5 * (max(u_od.buy_orders) + min(u_od.sell_orders))
        if spot is None:
            spot = trader_data.get("vev_wall_mid")

        # Live smile fit
        smile_sums = trader_data.get("smile_sums", [0.0] * 8)
        smile_coeffs = None
        if spot is not None and spot > 0:
            smile_sums, smile_coeffs = self._update_smile(state, smile_sums, spot)
        trader_data["smile_sums"] = smile_sums

        # ── First pass: compute z-scores, update EWMA, build targets ──────────
        targets:  dict[int, float] = {}
        deltas:   dict[int, float] = {}
        adj_fvs:  dict[int, float] = {}

        for K in ACTIVE_STRIKES:
            sk = str(K)
            od = state.order_depths.get(voucher_symbol(K), OrderDepth())
            if not od.buy_orders or not od.sell_orders:
                targets[K]  = 0.0
                adj_fvs[K]  = 0.0
                continue

            mkt_bid_K = max(od.buy_orders)
            mkt_ask_K = min(od.sell_orders)
            mid_K     = 0.5 * (mkt_bid_K + mkt_ask_K)

            if smile_coeffs is not None and spot is not None and spot > 0:
                log_mon = math.log(spot / K)
                sigma   = smile_iv(smile_coeffs, log_mon)
                if sigma <= 0:
                    targets[K] = 0.0
                    adj_fvs[K] = mid_K
                    continue

                bs_fv   = bs_call(spot, K, T_EXPIRY, sigma)
                delta_K = bs_call_delta(spot, K, T_EXPIRY, sigma)
                dev_K   = mid_K - bs_fv

                # EWMA update for mean and variance of this strike's deviation
                dm_prev = dev_mean.get(sk, PRIOR_MEAN.get(sk, 0.0))
                dv_prev = dev_var.get(sk,  PRIOR_VAR.get(sk,  0.09))
                dm_new  = RV_ALPHA * dm_prev + (1.0 - RV_ALPHA) * dev_K
                dv_new  = RV_ALPHA * dv_prev + (1.0 - RV_ALPHA) * (dev_K - dm_new) ** 2
                dev_mean[sk] = dm_new
                dev_var[sk]  = dv_new

                std_K   = math.sqrt(max(dv_new, MIN_STD ** 2))
                adj_fv  = bs_fv + dm_new
                rel_dev = mid_K - adj_fv
                z_K     = rel_dev / std_K

                f_z    = position_sigmoid(abs(z_K))
                sign_z = 1.0 if z_K > 0.0 else (-1.0 if z_K < 0.0 else 0.0)
                target  = ACTIVE_POS_LIMIT * (-sign_z) * f_z

                targets[K] = target
                deltas[K]  = delta_K
                adj_fvs[K] = adj_fv

                logger.print(f"K={K} z={z_K:.2f} tgt={target:.0f} adj={adj_fv:.2f} dm={dm_new:.3f}")

            else:
                # Pre-warmup: symmetric MM around market mid, no z-score lean
                targets[K] = 0.0
                adj_fvs[K] = mid_K

        # ── Delta scale: bound net target delta ───────────────────────────────
        raw_tgt_delta = sum(targets.get(K, 0.0) * deltas.get(K, 0.0) for K in ACTIVE_STRIKES if K in deltas)
        if abs(raw_tgt_delta) > DELTA_TARGET_CAP:
            scale = DELTA_TARGET_CAP / abs(raw_tgt_delta)
            targets = {K: v * scale for K, v in targets.items()}

        logger.print(f"raw_tgt_delta={raw_tgt_delta:.1f} smile_n={smile_sums[0]:.1f} spot={spot}")

        # ── Engine C: call spread arb ─────────────────────────────────────────
        try:
            arb_orders, arb_used, n_arbs = self._call_spread_arb(state)
            for sym, orders in arb_orders.items():
                result.setdefault(sym, []).extend(orders)
            if n_arbs:
                logger.print(f"arbs={n_arbs}")
        except Exception as e:
            logger.print(f"ERROR arb: {e}")
            arb_used = {k: [0, 0] for k in VOUCHER_STRIKES}

        # ── Engine A: underlying mean-reversion maker ─────────────────────────
        vt = None
        try:
            vt = VelvetTrader(state, last_wall_mid=trader_data.get("vev_wall_mid"))
            ea_orders = vt.get_orders().get(UNDERLYING, [])
            result.setdefault(UNDERLYING, []).extend(ea_orders)
            if vt.wall_mid is not None:
                trader_data["prev_prev_mid"] = trader_data.get("prev_mid")
                trader_data["prev_mid"]      = vt.wall_mid
                trader_data["vev_wall_mid"]  = vt.wall_mid
        except Exception as e:
            logger.print(f"ERROR {UNDERLYING}: {e}")

        # ── Deep ITM: intrinsic ± 1 MM ────────────────────────────────────────
        if spot is not None:
            for K in DEEP_ITM_STRIKES:
                try:
                    self._trade_deep_itm(K, state, result, spot, arb_used)
                except Exception as e:
                    logger.print(f"ERROR DITM_{K}: {e}")

        # ── Active strikes: z-score maker + conditional taker ─────────────────
        for K in ACTIVE_STRIKES:
            adj_fv = adj_fvs.get(K)
            if adj_fv is None or adj_fv <= 0:
                continue
            try:
                self._trade_active_strike(K, state, result, adj_fv, targets.get(K, 0.0), arb_used)
            except Exception as e:
                logger.print(f"ERROR ACTIVE_{K}: {e}")

        # ── Deep OTM: bid=0 / ask=1 ───────────────────────────────────────────
        for K in DEEP_OTM_STRIKES:
            try:
                self._trade_deep_otm(K, state, result, arb_used)
            except Exception as e:
                logger.print(f"ERROR DOTM_{K}: {e}")

        # ── Persist state ─────────────────────────────────────────────────────
        trader_data["dev_mean"] = dev_mean
        trader_data["dev_var"]  = dev_var

        out = json.dumps(trader_data)
        logger.flush(state, result, 0, out)
        return result, 0, out

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

####### MATH — Black-Scholes + SSVI (no scipy) #######

def _norm_cdf(x: float) -> float:
    if x < -6.0: return 0.0
    if x >  6.0: return 1.0
    neg = x < 0
    ax  = abs(x)
    t   = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p   = 1.0 - 0.3989422804014327 * math.exp(-0.5 * ax * ax) * poly
    return 1.0 - p if neg else p

def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 1e-8:
        return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    d2  = d1 - sigma * sqT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def _bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 1e-8:
        return 1.0 if S > K else 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    return _norm_cdf(d1)

def _implied_vol_nr(price: float, S: float, K: float, T: float, tol: float = 1e-5):
    if T <= 1e-8 or price - max(S - K, 0.0) < 1.0:
        return None
    sigma = math.sqrt(2 * math.pi / T) * price / S
    sigma = max(1e-7, min(sigma, 0.5))
    for _ in range(20):
        sqT  = math.sqrt(T)
        d1   = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
        vega = S * sqT * 0.3989422804014327 * math.exp(-0.5 * d1 * d1)
        if vega < 1e-10:
            break
        diff  = _bs_call(S, K, T, sigma) - price
        sigma = max(1e-8, sigma - diff / vega)
        if abs(diff) < tol:
            break
    return sigma if 1e-8 < sigma < 2.0 else None

def _ssvi_vol_tick(k: float, theta: float, T_ticks: float) -> float:
    if T_ticks <= 0:
        return 1e-8
    T_days = T_ticks / 10_000
    w_day  = (theta / 2.0) * (1.0 + math.sqrt(k * k + 1.0))
    return math.sqrt(max(w_day / T_days, 1e-12)) / math.sqrt(10_000)

####### CONFIG #######

PACK_SYMBOL    = "HYDROGEL_PACK"
PACK_POS_LIMIT = 200
PACK_MEAN      = 9990
PACK_STD       = 32
SKEW_MAX       = 3

UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
VEV_POS_LIMIT     = 100
VEV_EXPIRY_TS     = 8_000_000
VEV_TS_PER_TICK   = 100

# Strike groups
DEEP_ITM_STRIKES  = [4000, 4500]
ACTIVE_STRIKES    = [5000, 5100, 5200, 5400, 5500]  # 5300 excluded (consistent bleeder)
DEEP_OTM_STRIKES  = [6000, 6500]

# SSVI
VEV_THETA_DEFAULT = 0.001027
VEV_THETA_EWM     = 0.90

# Active trading params
# IV deviations from SSVI are slow-moving/persistent — no need to rush with takers.
# Pure patient market making: earn the spread by quoting both sides competitively,
# let the market come to us, use IV lean to accumulate naturally in the right direction.
VEV_TAKE_THRESH   = 5.0   # only cross for extreme mispricing (5+ ticks); normally 0 taking
VEV_IV_LEAN       = 1.0   # shift both quotes by full price deviation toward our edge
VEV_GAMMA_OPT     = 0.05  # per-option inventory skew (ticks/unit)
VEV_GAMMA_DELTA   = 0.01  # base-delta skew
VEV_MAKER_SIZE    = 7

# Deep ITM params
DITM_TAKE_THRESH  = 2.0   # only take clear arb on deep ITM (2+ ticks)
DITM_GAMMA        = 0.05  # inventory skew for deep ITM

####### VEV OPTIONS TRADER #######

class VEVTrader:
    """
    Three-tier strategy:
      Deep ITM  [4000,4500] — simple MM at intrinsic ± 1 (TV=0), inventory-skewed reservation
      Active    [5000–5500] — SSVI stat arb: IV-lean maker + aggressive taker (full capacity)
      Deep OTM  [6000,6500] — bid=0 ask=1 (mirror market: free lottery bids, collect 1-tick asks)
    """

    def __init__(self, state: TradingState, theta: float):
        self.state = state
        self.T     = max((VEV_EXPIRY_TS - state.timestamp) / VEV_TS_PER_TICK, 1.0)

        und = state.order_depths.get(UNDERLYING_SYMBOL, OrderDepth())
        if und.buy_orders and und.sell_orders:
            self.S = (max(und.buy_orders) + min(und.sell_orders)) / 2.0
        elif und.buy_orders:
            self.S = float(max(und.buy_orders))
        elif und.sell_orders:
            self.S = float(min(und.sell_orders))
        else:
            self.S = 5250.0

        self.theta      = self._update_theta(theta)
        self.base_delta = self._compute_base_delta()
        self._orders: dict = {}

    def _update_theta(self, prev: float) -> float:
        best_K = min(ACTIVE_STRIKES, key=lambda K: abs(math.log(self.S / K)))
        depth  = self.state.order_depths.get(f"VEV_{best_K}", OrderDepth())
        if not depth.buy_orders or not depth.sell_orders:
            return prev
        mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
        iv  = _implied_vol_nr(mid, self.S, best_K, self.T)
        if iv is None:
            return prev
        iv_day     = iv * math.sqrt(10_000)
        theta_live = iv_day * iv_day * (self.T / 10_000)
        return VEV_THETA_EWM * prev + (1.0 - VEV_THETA_EWM) * theta_live

    def _compute_base_delta(self) -> float:
        total = 0.0
        for K in ACTIVE_STRIKES:
            pos = self.state.position.get(f"VEV_{K}", 0)
            if pos == 0:
                continue
            k     = math.log(self.S / K)
            sigma = _ssvi_vol_tick(k, self.theta, self.T)
            total += pos * _bs_delta(self.S, K, self.T, sigma)
        return total

    # ── Deep ITM: MM at intrinsic ± 1 with inventory skew ───────────────────

    def _trade_deep_itm(self, K: int):
        sym   = f"VEV_{K}"
        depth = self.state.order_depths.get(sym, OrderDepth())
        pos   = self.state.position.get(sym, 0)

        bids = {p: abs(v) for p, v in sorted(depth.buy_orders.items(),  reverse=True)} if depth.buy_orders  else {}
        asks = {p: abs(v) for p, v in sorted(depth.sell_orders.items())} if depth.sell_orders else {}
        mkt_bid = max(bids) if bids else None
        mkt_ask = min(asks) if asks else None

        fair        = max(self.S - K, 0.0)
        reservation = fair - pos * DITM_GAMMA

        orders   = []
        buy_vol  = sell_vol = 0

        # Taker: only take clear arb (ask < fair - 1 or bid > fair + 1)
        for p, v in asks.items():
            if p < fair - DITM_TAKE_THRESH:
                fill = min(v, VEV_POS_LIMIT - pos - buy_vol)
                if fill > 0:
                    orders.append(Order(sym, p, fill))
                    buy_vol += fill
            else:
                break
        for p, v in bids.items():
            if p > fair + DITM_TAKE_THRESH:
                fill = min(v, VEV_POS_LIMIT + pos - sell_vol)
                if fill > 0:
                    orders.append(Order(sym, p, -fill))
                    sell_vol += fill
            else:
                break

        bid_price = int(reservation) - 1
        ask_price = int(reservation) + 1
        if ask_price <= bid_price: ask_price = bid_price + 1
        if mkt_ask is not None: bid_price = min(bid_price, mkt_ask - 1)
        if mkt_bid is not None: ask_price = max(ask_price, mkt_bid + 1)

        buy_cap  = VEV_POS_LIMIT - pos - buy_vol
        sell_cap = VEV_POS_LIMIT + pos - sell_vol
        if buy_cap > 0 and bid_price > 0:
            orders.append(Order(sym, bid_price, min(VEV_MAKER_SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(sym, ask_price, -min(VEV_MAKER_SIZE, sell_cap)))
        if orders:
            self._orders[sym] = orders

    # ── Active strikes: SSVI stat arb ────────────────────────────────────────

    def _trade_active(self, K: int):
        sym   = f"VEV_{K}"
        depth = self.state.order_depths.get(sym, OrderDepth())
        if not depth.buy_orders and not depth.sell_orders:
            return

        pos  = self.state.position.get(sym, 0)
        bids = {p: abs(v) for p, v in sorted(depth.buy_orders.items(),  reverse=True)} if depth.buy_orders  else {}
        asks = {p: abs(v) for p, v in sorted(depth.sell_orders.items())} if depth.sell_orders else {}
        mkt_bid = max(bids) if bids else None
        mkt_ask = min(asks) if asks else None

        market_mid = ((mkt_bid + mkt_ask) / 2.0 if mkt_bid and mkt_ask
                      else float(mkt_bid or mkt_ask))

        k     = math.log(self.S / K)
        sigma = _ssvi_vol_tick(k, self.theta, self.T)
        fair  = max(_bs_call(self.S, K, self.T, sigma), self.S - K)

        delta       = _bs_delta(self.S, K, self.T, sigma)
        price_dev   = market_mid - fair
        iv_adj      = price_dev * VEV_IV_LEAN

        reservation = (fair
                       - pos * VEV_GAMMA_OPT
                       - self.base_delta * delta * VEV_GAMMA_DELTA)

        # Competitive maker: post just inside the current best bid/ask.
        # res_adj encodes inventory + delta skew as an integer quote shift.
        # IV lean additionally shifts both quotes toward our edge:
        #   option rich  (iv_adj>0) → both shift down → ask inside mkt, bid below → sell more
        #   option cheap (iv_adj<0) → both shift up   → bid inside mkt, ask above → buy more
        res_adj    = int(round(reservation - fair))
        bid_target = (mkt_bid + 1 if mkt_bid is not None else int(fair) - 2) + res_adj
        ask_target = (mkt_ask - 1 if mkt_ask is not None else int(fair) + 2) + res_adj

        bid_price = int(bid_target - iv_adj)
        ask_price = int(ask_target - iv_adj)

        # Hard fair-value limits: never trade at an outright loss vs model
        bid_price = min(bid_price, int(fair))      # never bid above floor(fair)
        ask_price = max(ask_price, int(fair) + 1)  # never ask below floor(fair)+1

        if ask_price <= bid_price: ask_price = bid_price + 1
        if mkt_ask is not None and bid_price >= mkt_ask: bid_price = mkt_ask - 1
        if mkt_bid is not None and ask_price <= mkt_bid: ask_price = mkt_bid + 1

        orders   = []
        buy_vol  = sell_vol = 0

        # Patient taker: only cross for extreme mispricing (5+ ticks)
        for p, v in asks.items():
            if p <= fair - VEV_TAKE_THRESH:
                fill = min(v, VEV_POS_LIMIT - pos - buy_vol)
                if fill > 0:
                    orders.append(Order(sym, p, fill))
                    buy_vol += fill
            else:
                break

        for p, v in bids.items():
            if p >= fair + VEV_TAKE_THRESH:
                fill = min(v, VEV_POS_LIMIT + pos - sell_vol)
                if fill > 0:
                    orders.append(Order(sym, p, -fill))
                    sell_vol += fill
            else:
                break

        buy_cap  = VEV_POS_LIMIT - pos - buy_vol
        sell_cap = VEV_POS_LIMIT + pos - sell_vol
        if buy_cap > 0 and bid_price > 0:
            orders.append(Order(sym, bid_price, min(VEV_MAKER_SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(sym, ask_price, -min(VEV_MAKER_SIZE, sell_cap)))
        if orders:
            self._orders[sym] = orders

    # ── Deep OTM: bid=0 ask=1 ───────────────────────────────────────────────

    def _trade_deep_otm(self, K: int):
        sym  = f"VEV_{K}"
        pos  = self.state.position.get(sym, 0)
        orders = []
        if VEV_POS_LIMIT - pos > 0:
            orders.append(Order(sym, 0, min(VEV_MAKER_SIZE, VEV_POS_LIMIT - pos)))
        if VEV_POS_LIMIT + pos > 0:
            orders.append(Order(sym, 1, -min(VEV_MAKER_SIZE, VEV_POS_LIMIT + pos)))
        if orders:
            self._orders[sym] = orders

    # ── S4: Cross-strike call spread no-arb ─────────────────────────────────

    def _call_spread_arb(self):
        ALL_STRIKES = DEEP_ITM_STRIKES + ACTIVE_STRIKES + DEEP_OTM_STRIKES
        arb_used = {}  # track extra capacity used per strike

        for i, K_lo in enumerate(ALL_STRIKES):
            for K_hi in ALL_STRIKES[i+1:]:
                sym_lo = f"VEV_{K_lo}"
                sym_hi = f"VEV_{K_hi}"
                d_lo = self.state.order_depths.get(sym_lo, OrderDepth())
                d_hi = self.state.order_depths.get(sym_hi, OrderDepth())
                if not d_lo.buy_orders or not d_lo.sell_orders:
                    continue
                if not d_hi.buy_orders or not d_hi.sell_orders:
                    continue

                bid_lo = max(d_lo.buy_orders)
                ask_lo = min(d_lo.sell_orders)
                bid_hi = max(d_hi.buy_orders)
                ask_hi = min(d_hi.sell_orders)
                strike_diff = K_hi - K_lo

                # Direction A: C(K_lo) - C(K_hi) > strike_diff → impossible by no-arb
                # sell K_lo at bid, buy K_hi at ask; profit = bid_lo - ask_hi - strike_diff
                if bid_lo - ask_hi > strike_diff:
                    pos_lo = self.state.position.get(sym_lo, 0)
                    pos_hi = self.state.position.get(sym_hi, 0)
                    used_sell_lo = arb_used.get(sym_lo + "_sell", 0)
                    used_buy_hi  = arb_used.get(sym_hi + "_buy",  0)
                    sell_cap = VEV_POS_LIMIT + pos_lo - used_sell_lo
                    buy_cap  = VEV_POS_LIMIT - pos_hi - used_buy_hi
                    qty = min(sell_cap, buy_cap, abs(d_lo.buy_orders[bid_lo]), abs(d_hi.sell_orders[ask_hi]))
                    if qty > 0:
                        self._orders.setdefault(sym_lo, []).append(Order(sym_lo, bid_lo, -qty))
                        self._orders.setdefault(sym_hi, []).append(Order(sym_hi, ask_hi,  qty))
                        arb_used[sym_lo + "_sell"] = used_sell_lo + qty
                        arb_used[sym_hi + "_buy"]  = used_buy_hi  + qty

                # Direction B: C(K_hi) - C(K_lo) < 0 → bid_hi > ask_lo
                # buy K_lo at ask, sell K_hi at bid; profit = bid_hi - ask_lo > 0
                elif bid_hi > ask_lo:
                    pos_lo = self.state.position.get(sym_lo, 0)
                    pos_hi = self.state.position.get(sym_hi, 0)
                    used_buy_lo   = arb_used.get(sym_lo + "_buy",  0)
                    used_sell_hi  = arb_used.get(sym_hi + "_sell", 0)
                    buy_cap  = VEV_POS_LIMIT - pos_lo - used_buy_lo
                    sell_cap = VEV_POS_LIMIT + pos_hi - used_sell_hi
                    qty = min(buy_cap, sell_cap, abs(d_lo.sell_orders[ask_lo]), abs(d_hi.buy_orders[bid_hi]))
                    if qty > 0:
                        self._orders.setdefault(sym_lo, []).append(Order(sym_lo, ask_lo,  qty))
                        self._orders.setdefault(sym_hi, []).append(Order(sym_hi, bid_hi, -qty))
                        arb_used[sym_lo + "_buy"]  = used_buy_lo  + qty
                        arb_used[sym_hi + "_sell"] = used_sell_hi + qty

    def get_orders(self):
        try:
            self._call_spread_arb()
        except Exception as e:
            logger.print(f"arb err: {e}")
        for K in DEEP_ITM_STRIKES:
            try:
                self._trade_deep_itm(K)
            except Exception as e:
                logger.print(f"VEV_{K} err: {e}")
        for K in ACTIVE_STRIKES:
            try:
                self._trade_active(K)
            except Exception as e:
                logger.print(f"VEV_{K} err: {e}")
        for K in DEEP_OTM_STRIKES:
            try:
                self._trade_deep_otm(K)
            except Exception as e:
                logger.print(f"VEV_{K} err: {e}")
        return self._orders, self.theta


####### MAIN #######

class Trader:

    def run(self, state: TradingState):
        result      = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}

        try:
            vev_theta  = trader_data.get("vev_theta", VEV_THETA_DEFAULT)
            vev_trader = VEVTrader(state, vev_theta)
            vev_orders, new_theta = vev_trader.get_orders()
            result.update(vev_orders)
            trader_data["vev_theta"] = new_theta
            logger.print(f"VEV Δ:{vev_trader.base_delta:.1f} θ:{new_theta:.6f}")
        except Exception as e:
            logger.print(f"ERROR VEV: {e}")

        # PACK: add back after VEV is tuned
        # try:
        #     pack_trader = PACKTrader(state, ...)
        #     result.update(pack_trader.get_orders())
        # except Exception as e:
        #     logger.print(f"ERROR PACK: {e}")

        out = json.dumps(trader_data)
        logger.flush(state, result, 0, out)
        return result, 0, out

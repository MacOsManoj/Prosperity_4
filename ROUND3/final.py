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
        return [state.timestamp, trader_data, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations)]

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
OPTION_SYMBOLS = [f"VEV_{strike}" for strike in STRIKES]
OPTION_BY_STRIKE = {strike: f"VEV_{strike}" for strike in STRIKES}

LIMITS = {
    HYDRO: 200,
    VELVET: 200,
    **{symbol: 300 for symbol in OPTION_SYMBOLS},
}

HG_MU = 9990.95
HG_STD = 31.9
VF_MU = 5250.71
VF_STD = 15.63

VF_AR2_ALPHA = 10.727097081004104
VF_AR2_BETA1 = 0.840361578360772
VF_AR2_BETA2 = 0.15759553490214856

SMILE_GAMMA = 0.999
INITIAL_SMILE_SUMS = [
    8.336818e03,
    -2.479861e02,
    8.282092e01,
    -7.868455e00,
    3.140089e00,
    2.807962e03,
    -1.404019e02,
    5.052927e01,
]
OPTION_LIVE_START_DAYS = 5.0
OPTION_TTE_MIN_DAYS = 3.5
OPTION_TTE_MAX_DAYS = 8.5
OPTION_TTE_EMA_ALPHA = 0.08
STATIC_DAILY_SMILE_A = 0.086020
STATIC_DAILY_SMILE_B = -0.000806
STATIC_DAILY_SMILE_C = 0.012536
SQRT_365 = math.sqrt(365.0)

STATIC_PRICE_BIAS = {
    4000: 0.012,
    4500: 0.010,
    5000: -0.048,
    5100: -0.073,
    5200: 0.722,
    5300: 1.326,
    5400: -2.193,
    5500: 0.532,
    6000: 0.492,
    6500: 0.500,
}

RESID_STD = {
    4000: 0.85,
    4500: 0.80,
    5000: 0.565,
    5100: 0.924,
    5200: 0.926,
    5300: 1.157,
    5400: 0.778,
    5500: 0.441,
    6000: 0.20,
    6500: 0.20,
}

LINEAR_FAIR = {
    4000: (-3998.265, 0.9997),
    4500: (-4496.997, 0.9994),
    5000: (-4550.495, 0.9153),
    5100: (-3950.958, 0.7843),
    5200: (-2871.359, 0.5651),
    5300: (-1704.686, 0.3336),
    5400: (-644.239, 0.1257),
    5500: (-281.457, 0.0549),
}

LINEAR_BLEND = {
    4000: 0.95,
    4500: 0.90,
    5000: 0.72,
    5100: 0.58,
    5200: 0.34,
    5300: 0.24,
    5400: 0.12,
    5500: 0.10,
    6000: 0.00,
    6500: 0.00,
}

OPTION_POS_LIMIT = {
    4000: 140,
    4500: 120,
    5000: 140,
    5100: 140,
    5200: 160,
    5300: 160,
    5400: 180,
    5500: 150,
    6000: 80,
    6500: 80,
}

OPTION_QUOTE_SIZE = {
    4000: 6,
    4500: 6,
    5000: 10,
    5100: 12,
    5200: 13,
    5300: 13,
    5400: 12,
    5500: 10,
    6000: 20,
    6500: 20,
}

OPTION_BASE_EDGE = {
    4000: 5.0,
    4500: 2.8,
    5000: 1.0,
    5100: 0.85,
    5200: 0.8,
    5300: 0.7,
    5400: 0.6,
    5500: 0.5,
    6000: 0.5,
    6500: 0.5,
}

WIDE_WINGS = {6000, 6500}
OPTION_REVERSAL_COOLDOWN = 25
OPTION_OPEN_TAKER_MULT = {
    4000: 1.00,
    4500: 1.05,
    5000: 1.10,
    5100: 1.15,
    5200: 1.45,
    5300: 1.50,
    5400: 1.70,
    5500: 1.60,
    6000: 1.00,
    6500: 1.00,
}
OPTION_OPEN_TAKER_CAP = {
    4000: 18,
    4500: 14,
    5000: 12,
    5100: 12,
    5200: 10,
    5300: 9,
    5400: 8,
    5500: 7,
    6000: 8,
    6500: 8,
}
OPTION_TAKER_SIDE_CAP = {
    4000: 110,
    4500: 90,
    5000: 95,
    5100: 90,
    5200: 70,
    5300: 65,
    5400: 55,
    5500: 45,
    6000: 50,
    6500: 50,
}

USE_DAY3_SEED = True
DAY3_SEED_MEMORY = {
    "ema_mid": {HYDRO: 10015.193078484906},
    "last_mid": {HYDRO: 10010.0, VELVET: 5295.5},
    "vf_prev_mid": 5295.5,
    "vf_prev_prev": 5296.5,
    "smile_sums": [
        9635.96768,
        41.879512,
        152.049064,
        9.521171,
        7.662491,
        3128.838824,
        -63.966634,
        69.694207,
    ],
    "option_resid_ema": {
        "VEV_4000": -3.601968,
        "VEV_4500": 0.015798,
        "VEV_5000": -0.40751,
        "VEV_5100": -2.431348,
        "VEV_5200": -5.162883,
        "VEV_5300": -6.229948,
        "VEV_5400": -3.812512,
        "VEV_5500": -2.329329,
        "VEV_6000": 0.45337,
        "VEV_6500": 0.454545,
    },
    "last_fill_ts": {},
    "last_fill_price": {},
    "last_fill_side": {},
    "opt_tte_days": 5.0,
    "tick": 30000,
}


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_call(spot: float, strike: float, t_years: float, sigma: float) -> float:
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 1e-9:
        return max(spot - strike, 0.0)
    sqrt_t = math.sqrt(t_years)
    scaled = sigma * sqrt_t
    if scaled <= 1e-9:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / scaled
    d2 = d1 - scaled
    return spot * norm_cdf(d1) - strike * norm_cdf(d2)


def bs_delta(spot: float, strike: float, t_years: float, sigma: float) -> float:
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 1e-9:
        return 1.0 if spot > strike else 0.0
    sqrt_t = math.sqrt(t_years)
    scaled = sigma * sqrt_t
    if scaled <= 1e-9:
        return 1.0 if spot > strike else 0.0
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / scaled
    return norm_cdf(d1)


def implied_vol(market_price: float, spot: float, strike: float, t_years: float) -> float | None:
    if t_years <= 0 or spot <= 0 or strike <= 0:
        return None
    intrinsic = max(spot - strike, 0.0)
    if market_price <= intrinsic + 1e-6:
        return None
    lo, hi = 1e-4, 5.0
    f_lo = bs_call(spot, strike, t_years, lo) - market_price
    f_hi = bs_call(spot, strike, t_years, hi) - market_price
    if f_lo * f_hi > 0:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call(spot, strike, t_years, mid) - market_price
        if abs(f_mid) < 1e-5:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def solve_smile(sums: list[float]) -> tuple[float, float, float] | None:
    s0, s1, s2, s3, s4, y0, y1, y2 = sums
    det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2)
    if abs(det) < 1e-18:
        return None
    det_c = y0 * (s2 * s4 - s3 * s3) - s1 * (y1 * s4 - s3 * y2) + s2 * (y1 * s3 - s2 * y2)
    det_b = s0 * (y1 * s4 - s3 * y2) - y0 * (s1 * s4 - s3 * s2) + s2 * (s1 * y2 - y1 * s2)
    det_a = s0 * (s2 * y2 - y1 * s3) - s1 * (s1 * y2 - y1 * s2) + y0 * (s1 * s3 - s2 * s2)
    return det_a / det, det_b / det, det_c / det


def best_bid_ask(depth: OrderDepth) -> tuple[int | None, int | None]:
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    return best_bid, best_ask


def top_of_book_mid(depth: OrderDepth) -> float | None:
    best_bid, best_ask = best_bid_ask(depth)
    if best_bid is None or best_ask is None:
        return None
    return 0.5 * (best_bid + best_ask)


def top_obi(depth: OrderDepth) -> float:
    best_bid, best_ask = best_bid_ask(depth)
    if best_bid is None or best_ask is None:
        return 0.0
    bid_vol = abs(depth.buy_orders.get(best_bid, 0))
    ask_vol = abs(depth.sell_orders.get(best_ask, 0))
    denom = bid_vol + ask_vol
    if denom <= 0:
        return 0.0
    return (bid_vol - ask_vol) / denom


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ProductBook:
    def __init__(self, symbol: str, state: TradingState, limit: int):
        self.symbol = symbol
        self.depth = state.order_depths.get(symbol, OrderDepth())
        self.limit = limit
        self.position = state.position.get(symbol, 0)
        self.orders: list[Order] = []
        self.pending = 0
        self.buy_orders = dict(sorted(self.depth.buy_orders.items(), reverse=True)) if self.depth.buy_orders else {}
        self.sell_orders = dict(sorted(self.depth.sell_orders.items())) if self.depth.sell_orders else {}
        self.best_bid, self.best_ask = best_bid_ask(self.depth)
        self.mid = top_of_book_mid(self.depth)

    def projected_position(self) -> int:
        return self.position + self.pending

    def room_to_buy(self) -> int:
        return self.limit - self.projected_position()

    def room_to_sell(self) -> int:
        return self.limit + self.projected_position()

    def buy(self, price: int, quantity: int) -> int:
        quantity = min(quantity, self.room_to_buy())
        if quantity <= 0:
            return 0
        self.orders.append(Order(self.symbol, int(price), int(quantity)))
        self.pending += quantity
        return quantity

    def sell(self, price: int, quantity: int) -> int:
        quantity = min(quantity, self.room_to_sell())
        if quantity <= 0:
            return 0
        self.orders.append(Order(self.symbol, int(price), -int(quantity)))
        self.pending -= quantity
        return quantity


class HydrogelStrategy:
    def trade(self, state: TradingState, memory: dict[str, Any]) -> ProductBook:
        book = ProductBook(HYDRO, state, LIMITS[HYDRO])
        if book.mid is None or book.best_bid is None or book.best_ask is None:
            return book

        prev_ema = float(memory["ema_mid"].get(HYDRO, book.mid))
        ema_mid = prev_ema + 0.03 * (book.mid - prev_ema)
        prev_mid = float(memory["last_mid"].get(HYDRO, book.mid))
        momentum = book.mid - prev_mid
        obi = top_obi(book.depth)
        memory["ema_mid"][HYDRO] = ema_mid
        memory["last_mid"][HYDRO] = book.mid

        fair = book.mid + 0.48 * (HG_MU - ema_mid) - 0.18 * momentum + 2.2 * obi
        deviation = book.mid - HG_MU

        for ask_price, ask_volume in book.sell_orders.items():
            if ask_price > fair - 3.0:
                break
            edge = fair - ask_price
            current_long = max(0, book.projected_position())
            buy_band = min(book.limit, 95 + max(0, int(6 * (edge - 3.0))))
            if momentum < -2.0 and book.mid < prev_mid:
                buy_band -= 55
            if deviation < -35:
                buy_band -= 20
            buy_band = max(28, buy_band)
            room_in_band = max(0, buy_band - current_long)
            if room_in_band <= 0:
                break
            size = min(abs(ask_volume), 80, int(18 + 3 * edge), room_in_band)
            book.buy(ask_price, size)

        for bid_price, bid_volume in book.buy_orders.items():
            if bid_price < fair + 3.0:
                break
            edge = bid_price - fair
            current_short = max(0, -book.projected_position())
            sell_band = min(book.limit, 95 + max(0, int(6 * (edge - 3.0))))
            if momentum > 2.0 and book.mid > prev_mid:
                sell_band -= 55
            if deviation > 35:
                sell_band -= 20
            sell_band = max(28, sell_band)
            room_in_band = max(0, sell_band - current_short)
            if room_in_band <= 0:
                break
            size = min(abs(bid_volume), 80, int(18 + 3 * edge), room_in_band)
            book.sell(bid_price, size)

        projected = book.projected_position()
        fair -= 11.0 * (projected / book.limit)
        spread = book.best_ask - book.best_bid
        last_fill_ts = int(memory.get("last_fill_ts", {}).get(HYDRO, 0))
        quiet_for = state.timestamp - last_fill_ts

        # Small conditional MM layer. Tiny when near fair, one-sided when far from fair.
        # This keeps participation alive without the large PnL hit of an always-on joiner.
        if spread >= 15 and abs(projected) <= 120:
            if abs(deviation) <= 25:
                join_bid = min(book.best_bid + 1, book.best_ask - 1)
                join_ask = max(book.best_ask - 1, book.best_bid + 1)
                if join_bid < book.best_ask:
                    book.buy(int(join_bid), 3)
                if join_ask > book.best_bid:
                    book.sell(int(join_ask), 3)
            elif deviation <= -25 and momentum > 1.0:
                join_bid = min(book.best_bid + 1, book.best_ask - 1)
                join_ask = max(book.best_ask - 2, book.best_bid + 1)
                if join_bid < book.best_ask:
                    book.buy(int(join_bid), 4)
                if join_ask > book.best_bid:
                    book.sell(int(join_ask), 4)
            elif deviation >= 25 and momentum < -1.0:
                join_bid = min(book.best_bid + 2, book.best_ask - 1)
                join_ask = max(book.best_ask - 1, book.best_bid + 1)
                if join_bid < book.best_ask:
                    book.buy(int(join_bid), 4)
                if join_ask > book.best_bid:
                    book.sell(int(join_ask), 4)
            elif deviation <= -35:
                join_bid = min(book.best_bid + 1, book.best_ask - 1)
                if join_bid < book.best_ask:
                    book.buy(int(join_bid), 4)
            elif deviation >= 35:
                join_ask = max(book.best_ask - 1, book.best_bid + 1)
                if join_ask > book.best_bid:
                    book.sell(int(join_ask), 4)

        # If we have gone a long time without a fill while price is still far from the
        # long-run mean, tighten one-sided to re-engage on the mean-reversion side.
        if quiet_for >= 15000 and abs(projected) <= 80:
            if deviation <= -30 and spread >= 15:
                tight_bid = min(book.best_bid + 2, book.best_ask - 1)
                if tight_bid < book.best_ask:
                    book.buy(int(tight_bid), 4)
            elif deviation >= 30 and spread >= 15:
                tight_ask = max(book.best_ask - 2, book.best_bid + 1)
                if tight_ask > book.best_bid:
                    book.sell(int(tight_ask), 4)

        # If we already carry a large position and the move is still going against us,
        # bias toward peeling inventory on small bounces instead of averaging down hard.
        if projected >= 90 and momentum < -2.0 and spread >= 15:
            defensive_ask = max(book.best_bid + 1, min(book.best_ask - 1, int(book.mid + 2)))
            if defensive_ask > book.best_bid:
                book.sell(defensive_ask, 8)
        elif projected <= -90 and momentum > 2.0 and spread >= 15:
            defensive_bid = min(book.best_ask - 1, max(book.best_bid + 1, int(book.mid - 2)))
            if defensive_bid < book.best_ask:
                book.buy(defensive_bid, 8)

        quote_edge = 2.0 if abs(deviation) <= 35 and spread >= 15 else 3.0
        bid_quote = min(book.best_bid + 1, math.floor(fair - quote_edge))
        ask_quote = max(book.best_ask - 1, math.ceil(fair + quote_edge))

        if bid_quote < book.best_ask:
            size = max(5, int(24 * max(0.25, 1.0 - max(0, projected) / book.limit)))
            book.buy(int(bid_quote), size)
        if ask_quote > book.best_bid:
            size = max(5, int(24 * max(0.25, 1.0 + min(0, projected) / book.limit)))
            book.sell(int(ask_quote), size)

        logger.print(f"HG mid={book.mid:.1f} fair={fair:.1f} obi={obi:.2f} pos={book.projected_position()}")
        return book


class VelvetStrategy:
    def trade(self, state: TradingState, memory: dict[str, Any]) -> ProductBook:
        book = ProductBook(VELVET, state, LIMITS[VELVET])
        if book.mid is None or book.best_bid is None or book.best_ask is None:
            return book

        prev_mid = float(memory["last_mid"].get(VELVET, book.mid))
        prev_prev = memory["vf_prev_prev"]
        last_prev = memory["vf_prev_mid"]
        memory["vf_prev_prev"] = last_prev
        memory["vf_prev_mid"] = book.mid
        memory["last_mid"][VELVET] = book.mid

        ar2_fair = None
        if isinstance(last_prev, (int, float)) and isinstance(prev_prev, (int, float)):
            ar2_fair = VF_AR2_ALPHA + VF_AR2_BETA1 * float(last_prev) + VF_AR2_BETA2 * float(prev_prev)

        base_fair = book.mid if ar2_fair is None else 0.65 * book.mid + 0.35 * ar2_fair
        momentum = book.mid - prev_mid
        obi = top_obi(book.depth)
        fair = base_fair + 0.10 * (VF_MU - base_fair) - 0.12 * momentum + 0.75 * obi

        for ask_price, ask_volume in book.sell_orders.items():
            if ask_price > fair - 1.4:
                break
            size = min(abs(ask_volume), 32, int(10 + 5 * (fair - ask_price)))
            book.buy(ask_price, size)

        for bid_price, bid_volume in book.buy_orders.items():
            if bid_price < fair + 1.4:
                break
            size = min(abs(bid_volume), 32, int(10 + 5 * (bid_price - fair)))
            book.sell(bid_price, size)

        projected = book.projected_position()
        mean_dev = (book.mid - VF_MU) / VF_STD
        skew_ticks = int(round(clamp(mean_dev * 3.0 + 0.9 * obi, -5.0, 5.0)))
        inv_lean = 6.0 * (projected / book.limit)

        thick_bid = next((price for price, vol in book.buy_orders.items() if price <= fair and abs(vol) > 1), None)
        thick_ask = next((price for price, vol in book.sell_orders.items() if price >= fair and abs(vol) > 1), None)
        base_bid = (thick_bid + 1) if thick_bid is not None else int(fair - 3)
        base_ask = (thick_ask - 1) if thick_ask is not None else int(fair + 3)
        bid_quote = min(base_bid - skew_ticks - int(round(inv_lean)), math.floor(fair - 1))
        ask_quote = max(base_ask - skew_ticks - int(round(inv_lean)), math.ceil(fair + 1))

        normalized = min(abs(mean_dev) / 3.0, 1.0)
        if mean_dev > 0:
            bid_size = min(22, max(4, int(6 + 18 * normalized)))
            ask_size = 6
        else:
            bid_size = 6
            ask_size = min(22, max(4, int(6 + 18 * normalized)))

        if bid_quote < book.best_ask:
            book.buy(int(bid_quote), bid_size)
        if ask_quote > book.best_bid:
            book.sell(int(ask_quote), ask_size)

        logger.print(f"VF mid={book.mid:.1f} fair={fair:.1f} mean_dev={mean_dev:.2f} obi={obi:.2f} pos={book.projected_position()}")
        return book


class OptionComplexStrategy:
    def _mid(self, state: TradingState, strike: int) -> float | None:
        depth = state.order_depths.get(OPTION_BY_STRIKE[strike])
        if depth is None:
            return None
        return top_of_book_mid(depth)

    def update_smile(self, state: TradingState, memory: dict[str, Any], spot: float) -> tuple[list[float], tuple[float, float, float] | None]:
        sums = list(memory.get("smile_sums", INITIAL_SMILE_SUMS))
        sums = [SMILE_GAMMA * value for value in sums]
        t_years = float(memory.get("opt_tte_days", OPTION_LIVE_START_DAYS)) / 365.0
        for strike in STRIKES:
            mid = self._mid(state, strike)
            if mid is None:
                continue
            iv = implied_vol(mid, spot, strike, t_years)
            if iv is None or iv <= 0 or iv > 5:
                continue
            x = math.log(spot / strike)
            sums[0] += 1.0
            sums[1] += x
            sums[2] += x * x
            sums[3] += x * x * x
            sums[4] += x * x * x * x
            sums[5] += iv
            sums[6] += x * iv
            sums[7] += x * x * iv
        coeffs = solve_smile(sums)
        memory["smile_sums"] = sums
        return sums, coeffs

    def update_tte_days(self, state: TradingState, memory: dict[str, Any], spot: float) -> float:
        candidates: list[float] = []
        for strike in (5200, 5300):
            depth = state.order_depths.get(OPTION_BY_STRIKE[strike])
            if depth is None:
                continue
            best_bid, best_ask = best_bid_ask(depth)
            if best_bid is None or best_ask is None:
                continue
            mid = 0.5 * (best_bid + best_ask)
            intrinsic = max(spot - strike, 0.0)
            # Skip if almost purely intrinsic; TTE estimate becomes noisy.
            if mid <= intrinsic + 0.5:
                continue

            sigma = self.static_sigma(spot, strike)
            lo = OPTION_TTE_MIN_DAYS
            hi = OPTION_TTE_MAX_DAYS
            for _ in range(32):
                t_days = 0.5 * (lo + hi)
                fair = bs_call(spot, strike, t_days / 365.0, sigma) + STATIC_PRICE_BIAS[strike]
                if fair < mid:
                    lo = t_days
                else:
                    hi = t_days
            candidates.append(0.5 * (lo + hi))

        est = memory.get("opt_tte_days")
        if candidates:
            new_est = sum(candidates) / len(candidates)
            if isinstance(est, (int, float)):
                est = (1.0 - OPTION_TTE_EMA_ALPHA) * float(est) + OPTION_TTE_EMA_ALPHA * new_est
            else:
                est = new_est
        elif not isinstance(est, (int, float)):
            est = OPTION_LIVE_START_DAYS

        est = clamp(float(est), OPTION_TTE_MIN_DAYS, OPTION_TTE_MAX_DAYS)
        memory["opt_tte_days"] = est
        return est

    def static_sigma(self, spot: float, strike: int) -> float:
        x = math.log(spot / strike)
        daily_sigma = STATIC_DAILY_SMILE_A * x * x + STATIC_DAILY_SMILE_B * x + STATIC_DAILY_SMILE_C
        return max(0.05, daily_sigma * SQRT_365)

    def smile_sigma(self, live_coeffs: tuple[float, float, float] | None, spot: float, strike: int) -> float:
        static = self.static_sigma(spot, strike)
        if live_coeffs is None:
            return static
        a, b, c = live_coeffs
        x = math.log(spot / strike)
        live = a * x * x + b * x + c
        if live <= 0 or live > 5.0:
            return static
        return 0.7 * live + 0.3 * static

    def linear_fair(self, spot: float, strike: int) -> float | None:
        params = LINEAR_FAIR.get(strike)
        if params is None:
            return None
        intercept, slope = params
        return intercept + slope * spot

    def net_delta(self, state: TradingState, spot: float, t_years: float) -> float:
        delta = float(state.position.get(VELVET, 0))
        for strike in STRIKES:
            pos = state.position.get(OPTION_BY_STRIKE[strike], 0)
            if pos == 0:
                continue
            sigma = self.static_sigma(spot, strike)
            delta += pos * bs_delta(spot, strike, t_years, sigma)
        return delta

    def call_spread_arb(self, books: dict[int, ProductBook]) -> None:
        for idx, low_strike in enumerate(STRIKES):
            low_book = books[low_strike]
            for high_strike in STRIKES[idx + 1:]:
                high_book = books[high_strike]
                width = high_strike - low_strike

                if low_book.best_bid is not None and high_book.best_ask is not None:
                    edge = low_book.best_bid - high_book.best_ask - width
                    if edge > 0:
                        qty = min(
                            abs(low_book.buy_orders.get(low_book.best_bid, 0)),
                            abs(high_book.sell_orders.get(high_book.best_ask, 0)),
                            low_book.room_to_sell(),
                            high_book.room_to_buy(),
                            18,
                        )
                        if qty > 0:
                            low_book.sell(low_book.best_bid, qty)
                            high_book.buy(high_book.best_ask, qty)
                            logger.print(f"ARB+ {low_strike}/{high_strike} qty={qty} edge={edge:.1f}")

                if low_book.best_ask is not None and high_book.best_bid is not None:
                    edge = high_book.best_bid - low_book.best_ask
                    if edge > 0:
                        qty = min(
                            abs(low_book.sell_orders.get(low_book.best_ask, 0)),
                            abs(high_book.buy_orders.get(high_book.best_bid, 0)),
                            low_book.room_to_buy(),
                            high_book.room_to_sell(),
                            18,
                        )
                        if qty > 0:
                            low_book.buy(low_book.best_ask, qty)
                            high_book.sell(high_book.best_bid, qty)
                            logger.print(f"ARB- {low_strike}/{high_strike} qty={qty} edge={edge:.1f}")

    def soft_spread_trades(self, books: dict[int, ProductBook], fair_map: dict[int, float]) -> None:
        pairs = [
            (5300, 5400, 1.2, 0.8, 12),
            (5200, 5400, 1.3, 0.9, 10),
            (5000, 5100, 0.8, 0.8, 8),
        ]
        for rich_strike, cheap_strike, rich_thr, cheap_thr, max_qty in pairs:
            rich_book = books[rich_strike]
            cheap_book = books[cheap_strike]
            if rich_book.best_bid is None or cheap_book.best_ask is None:
                continue
            rich_edge = rich_book.best_bid - fair_map[rich_strike]
            cheap_edge = fair_map[cheap_strike] - cheap_book.best_ask
            if rich_edge < rich_thr or cheap_edge < cheap_thr:
                continue
            qty = min(
                max_qty,
                abs(rich_book.buy_orders.get(rich_book.best_bid, 0)),
                abs(cheap_book.sell_orders.get(cheap_book.best_ask, 0)),
                rich_book.room_to_sell(),
                cheap_book.room_to_buy(),
                self.side_room(rich_book, rich_strike, "SELL"),
                self.side_room(cheap_book, cheap_strike, "BUY"),
            )
            if qty > 0:
                rich_book.sell(rich_book.best_bid, qty)
                cheap_book.buy(cheap_book.best_ask, qty)
                logger.print(f"PAIR {rich_strike}>{cheap_strike} qty={qty} rich={rich_edge:.2f} cheap={cheap_edge:.2f}")

    def recent_reversal_block(self, memory: dict[str, Any], symbol: str, side: str, price: int, tick: int, threshold: float) -> bool:
        last_side = memory.get("last_taker_side", {}).get(symbol)
        last_price = memory.get("last_taker_price", {}).get(symbol)
        last_tick = memory.get("last_taker_tick", {}).get(symbol)
        if last_side is None or last_price is None or last_tick is None:
            return False
        if tick - int(last_tick) > OPTION_REVERSAL_COOLDOWN:
            return False
        if last_side == "SELL" and side == "BUY" and price > float(last_price) + max(1.0, 0.35 * threshold):
            return True
        if last_side == "BUY" and side == "SELL" and price < float(last_price) - max(1.0, 0.35 * threshold):
            return True
        return False

    def record_taker(self, memory: dict[str, Any], symbol: str, side: str, price: int, tick: int) -> None:
        memory.setdefault("last_taker_side", {})[symbol] = side
        memory.setdefault("last_taker_price", {})[symbol] = price
        memory.setdefault("last_taker_tick", {})[symbol] = tick

    def side_room(self, book: ProductBook, strike: int, side: str) -> int:
        projected = book.projected_position()
        same_side_cap = min(book.limit, OPTION_TAKER_SIDE_CAP[strike])
        if side == "BUY":
            return max(0, same_side_cap - max(0, projected))
        return max(0, same_side_cap - max(0, -projected))

    def trade(self, state: TradingState, memory: dict[str, Any], spot: float | None) -> dict[str, list[Order]]:
        orders: dict[str, list[Order]] = {}
        if spot is None or spot <= 0:
            return orders

        tick = int(memory.get("tick", 0))
        tte_days = self.update_tte_days(state, memory, spot)
        t_years = tte_days / 365.0
        residual_ema = memory.setdefault("option_resid_ema", {})
        books = {strike: ProductBook(OPTION_BY_STRIKE[strike], state, OPTION_POS_LIMIT[strike]) for strike in STRIKES}
        net_delta = self.net_delta(state, spot, t_years)
        delta_lean = clamp(net_delta / 180.0, -2.5, 2.5)

        fair_map: dict[int, float] = {}
        for strike, book in books.items():
            if book.mid is None or book.best_bid is None or book.best_ask is None:
                continue
            sigma = self.static_sigma(spot, strike)
            base_fair = bs_call(spot, strike, t_years, sigma)
            resid_key = OPTION_BY_STRIKE[strike]
            adaptive = clamp(0.05 * float(residual_ema.get(resid_key, 0.0)), -1.0, 1.0)
            fair = base_fair + STATIC_PRICE_BIAS[strike] + adaptive
            resid = book.mid - fair
            residual_ema[resid_key] = 0.94 * float(residual_ema.get(resid_key, 0.0)) + 0.06 * resid
            fair_map[strike] = fair

        self.call_spread_arb(books)
        self.soft_spread_trades(books, fair_map)

        for strike, book in books.items():
            if book.mid is None or book.best_bid is None or book.best_ask is None or strike not in fair_map:
                orders[book.symbol] = book.orders
                continue

            fair = fair_map[strike]
            projected = book.projected_position()
            if strike in WIDE_WINGS:
                if book.best_bid == 0 and book.best_ask == 1:
                    book.buy(0, OPTION_QUOTE_SIZE[strike])
                    book.sell(1, OPTION_QUOTE_SIZE[strike])
                orders[book.symbol] = book.orders
                continue

            spread = book.best_ask - book.best_bid
            edge = OPTION_BASE_EDGE[strike]
            threshold = max(0.55 * RESID_STD[strike], 0.35 * spread, edge)

            rich_edge = book.best_bid - fair
            cheap_edge = fair - book.best_ask

            open_threshold = threshold * OPTION_OPEN_TAKER_MULT[strike]
            last_fill_price = memory.get("last_fill_price", {}).get(book.symbol)
            last_fill_side = memory.get("last_fill_side", {}).get(book.symbol)

            if rich_edge > open_threshold:
                if not self.recent_reversal_block(memory, book.symbol, "SELL", book.best_bid, tick, threshold):
                    if not (projected > 0 and last_fill_side == "BUY" and last_fill_price is not None and book.best_bid < float(last_fill_price) - 0.30 * threshold and rich_edge < 1.8 * open_threshold):
                        size = min(
                            OPTION_QUOTE_SIZE[strike] + int(3 * rich_edge / max(RESID_STD[strike], 0.3)),
                            abs(book.buy_orders.get(book.best_bid, 0)),
                            book.room_to_sell(),
                            self.side_room(book, strike, "SELL"),
                            OPTION_OPEN_TAKER_CAP[strike],
                        )
                        filled = book.sell(book.best_bid, size)
                        if filled > 0:
                            self.record_taker(memory, book.symbol, "SELL", book.best_bid, tick)
            elif projected > 0 and rich_edge > -0.4 * threshold:
                unwind_floor = float(memory.get("last_taker_price", {}).get(book.symbol, last_fill_price if last_fill_price is not None else fair))
                if book.best_bid >= unwind_floor - max(1.0, 0.2 * spread):
                    size = min(abs(projected), abs(book.buy_orders.get(book.best_bid, 0)), 14)
                    filled = book.sell(book.best_bid, size)
                    if filled > 0:
                        self.record_taker(memory, book.symbol, "SELL", book.best_bid, tick)

            if cheap_edge > open_threshold:
                if not self.recent_reversal_block(memory, book.symbol, "BUY", book.best_ask, tick, threshold):
                    if not (projected < 0 and last_fill_side == "SELL" and last_fill_price is not None and book.best_ask > float(last_fill_price) + 0.30 * threshold and cheap_edge < 1.8 * open_threshold):
                        size = min(
                            OPTION_QUOTE_SIZE[strike] + int(3 * cheap_edge / max(RESID_STD[strike], 0.3)),
                            abs(book.sell_orders.get(book.best_ask, 0)),
                            book.room_to_buy(),
                            self.side_room(book, strike, "BUY"),
                            OPTION_OPEN_TAKER_CAP[strike],
                        )
                        filled = book.buy(book.best_ask, size)
                        if filled > 0:
                            self.record_taker(memory, book.symbol, "BUY", book.best_ask, tick)
            elif projected < 0 and cheap_edge > -0.4 * threshold:
                unwind_ceiling = float(memory.get("last_taker_price", {}).get(book.symbol, last_fill_price if last_fill_price is not None else fair))
                if book.best_ask <= unwind_ceiling + max(1.0, 0.2 * spread):
                    size = min(abs(projected), abs(book.sell_orders.get(book.best_ask, 0)), 14)
                    filled = book.buy(book.best_ask, size)
                    if filled > 0:
                        self.record_taker(memory, book.symbol, "BUY", book.best_ask, tick)

            # Quote continuously like a real MM. Fair value sets the lean, but the quote
            # anchor stays close to the touch so narrower strikes trade more often.
            signal = clamp((fair - book.mid) / max(RESID_STD[strike], 0.35), -2.0, 2.0)
            inv_lean = 0.6 * (book.projected_position() / max(OPTION_POS_LIMIT[strike], 1))
            tilt = signal - 0.5 * delta_lean - inv_lean
            if spread >= 3:
                bid_quote = book.best_bid + 1
                ask_quote = book.best_ask - 1
            elif spread == 2:
                bid_quote = book.best_bid + (1 if tilt >= 0 else 0)
                ask_quote = book.best_ask - (1 if tilt <= 0 else 0)
            else:
                bid_quote = book.best_bid
                ask_quote = book.best_ask

            if tilt > 0.75 and bid_quote < book.best_ask:
                bid_quote = min(bid_quote + 1, book.best_ask - 1)
            if tilt < -0.75 and ask_quote > book.best_bid:
                ask_quote = max(ask_quote - 1, book.best_bid + 1)

            model_bid = math.floor(fair - edge)
            model_ask = math.ceil(fair + edge)
            bid_quote = max(min(bid_quote, model_bid + 1), book.best_bid)
            ask_quote = min(max(ask_quote, model_ask - 1), book.best_ask)
            quote_size = OPTION_QUOTE_SIZE[strike]
            projected_after_takers = book.projected_position()
            if projected_after_takers > 0:
                quote_bid_size = quote_size
                quote_ask_size = max(2, int(round(quote_size * 1.35)))
                if last_fill_side == "BUY" and last_fill_price is not None:
                    ask_quote = max(ask_quote, int(math.ceil(float(last_fill_price))))
            elif projected_after_takers < 0:
                quote_bid_size = max(2, int(round(quote_size * 1.35)))
                quote_ask_size = quote_size
                if last_fill_side == "SELL" and last_fill_price is not None:
                    bid_quote = min(bid_quote, int(math.floor(float(last_fill_price))))
            else:
                quote_bid_size = quote_size
                quote_ask_size = quote_size

            if bid_quote < book.best_ask:
                book.buy(int(bid_quote), quote_bid_size)
            if ask_quote > book.best_bid:
                book.sell(int(ask_quote), quote_ask_size)

            logger.print(
                f"{book.symbol} fair={fair:.2f} mid={book.mid:.2f} edge=({cheap_edge:.2f},{rich_edge:.2f}) pos={book.projected_position()}"
            )
            orders[book.symbol] = book.orders

        logger.print(f"OPT delta={net_delta:.1f} lean={delta_lean:.2f} tte_days={tte_days:.3f}")
        return orders


class Trader:
    def __init__(self) -> None:
        self.hydro = HydrogelStrategy()
        self.velvet = VelvetStrategy()
        self.options = OptionComplexStrategy()

    def load_memory(self, trader_data: str) -> dict[str, Any]:
        if trader_data:
            try:
                memory = json.loads(trader_data)
                if isinstance(memory, dict):
                    return memory
            except Exception:
                pass
        if USE_DAY3_SEED:
            return json.loads(json.dumps(DAY3_SEED_MEMORY))
        return {
            "ema_mid": {},
            "last_mid": {},
            "vf_prev_mid": None,
            "vf_prev_prev": None,
            "smile_sums": list(INITIAL_SMILE_SUMS),
            "option_resid_ema": {},
            "last_fill_ts": {},
            "last_fill_price": {},
            "last_fill_side": {},
            "opt_tte_days": OPTION_LIVE_START_DAYS,
            "tick": 0,
        }

    def dump_memory(self, memory: dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def ingest_own_trades(self, state: TradingState, memory: dict[str, Any]) -> None:
        fill_ts = memory.setdefault("last_fill_ts", {})
        fill_price = memory.setdefault("last_fill_price", {})
        fill_side = memory.setdefault("last_fill_side", {})
        for trade_list in state.own_trades.values():
            for trade in trade_list:
                if trade.buyer == "SUBMISSION":
                    side = "BUY"
                elif trade.seller == "SUBMISSION":
                    side = "SELL"
                else:
                    continue
                prev_ts = int(fill_ts.get(trade.symbol, -1))
                if trade.timestamp >= prev_ts:
                    fill_ts[trade.symbol] = trade.timestamp
                    fill_price[trade.symbol] = trade.price
                    fill_side[trade.symbol] = side

    def run(self, state: TradingState):
        memory = self.load_memory(state.traderData)
        self.ingest_own_trades(state, memory)
        memory["tick"] = int(memory.get("tick", 0)) + 1

        orders: dict[Symbol, list[Order]] = {}

        hydro_book = self.hydro.trade(state, memory)
        velvet_book = self.velvet.trade(state, memory)
        orders[HYDRO] = hydro_book.orders
        orders[VELVET] = velvet_book.orders

        option_orders = self.options.trade(state, memory, velvet_book.mid)
        for symbol, product_orders in option_orders.items():
            orders[symbol] = product_orders

        trader_data = self.dump_memory(memory)
        conversions = 0
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
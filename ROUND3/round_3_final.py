#21.6k
import json
import math
from statistics import NormalDist
from datamodel import Order, OrderDepth, TradingState

STANDARD_NORMAL = NormalDist()

def black_scholes_call(spot, strike, time_to_expiry, volatility):
    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0: return max(spot - strike, 0.0)
    try:
        d1 = (math.log(spot / strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        return spot * STANDARD_NORMAL.cdf(d1) - strike * STANDARD_NORMAL.cdf(d2)
    except: return max(spot - strike, 0.0)

def implied_vol(call_price, spot, strike, time_to_expiry):
    intrinsic = max(spot - strike, 0)
    if call_price <= intrinsic + 1e-4 or call_price >= spot - 1e-4 or time_to_expiry <= 0: return None
    lo, hi = 1e-4, 3.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if black_scholes_call(spot, strike, time_to_expiry, mid) - call_price > 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

INNER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300]
OUTER_STRIKES = [5400, 5500]
FIT_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
POSITION_LIMITS = {'HYDROGEL_PACK': 200, 'VELVETFRUIT_EXTRACT': 200}
for strike in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
    POSITION_LIMITS[f'VEV_{strike}'] = 300

# Hydrogel market-making parameters
HYDROGEL_FAIR = 10000
HYDROGEL_HALF_SPREAD = 4
HYDROGEL_ORDER_FRACTION = 0.4


class Trader:
    def __init__(self):
        self.stored_data = {}

    def get_best_prices(self, depth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    def get_wall_midpoint(self, depth):
        bids = sorted(depth.buy_orders.keys()) if depth.buy_orders else []
        asks = sorted(depth.sell_orders.keys()) if depth.sell_orders else []
        if not bids or not asks: return None
        return (bids[0] + asks[-1]) / 2

    def get_volume_weighted_mid(self, depth):
        best_bid, best_ask = self.get_best_prices(depth)
        if best_bid is None or best_ask is None: return None
        bid_volume = depth.buy_orders[best_bid]
        ask_volume = -depth.sell_orders[best_ask]
        if bid_volume + ask_volume == 0: return (best_bid + best_ask) / 2
        return (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)

    # Hydrogel fixed fair-value market making
    def trade_hydrogel(self, depth, position):
        symbol = 'HYDROGEL_PACK'
        if not depth.buy_orders or not depth.sell_orders:
            return []
        limit = POSITION_LIMITS[symbol]
        orders = []
        best_bid, best_ask = self.get_best_prices(depth)
        fair_value = HYDROGEL_FAIR
        half_spread = HYDROGEL_HALF_SPREAD

        # Take: buy if best ask below fair - 1
        if best_ask is not None and best_ask < fair_value - 1:
            available = abs(sum(vol for price, vol in depth.sell_orders.items() if price <= fair_value - 1))
            buy_qty = min(limit - position, available)
            if buy_qty > 0:
                orders.append(Order(symbol, best_ask, buy_qty))
                position += buy_qty

        # Take: sell if best bid above fair + 1
        if best_bid is not None and best_bid > fair_value + 1:
            available = sum(vol for price, vol in depth.buy_orders.items() if price >= fair_value + 1)
            sell_qty = min(position + limit, available)
            if sell_qty > 0:
                orders.append(Order(symbol, best_bid, -sell_qty))
                position -= sell_qty

        # Make with inventory skew
        skew = int(position / limit * half_spread)
        our_bid = int(fair_value) - half_spread - skew
        our_ask = int(fair_value) + half_spread - skew

        if best_bid is not None: our_bid = min(our_bid, best_bid)
        if best_ask is not None: our_ask = max(our_ask, best_ask)

        buy_capacity = limit - position
        sell_capacity = limit + position
        bid_size = max(1, int(buy_capacity * HYDROGEL_ORDER_FRACTION))
        ask_size = max(1, int(sell_capacity * HYDROGEL_ORDER_FRACTION))

        if buy_capacity > 0 and our_bid > 0:
            orders.append(Order(symbol, our_bid, bid_size))
        if sell_capacity > 0:
            orders.append(Order(symbol, our_ask, -ask_size))

        return orders

    # Random-walk market making for VEV underlying
    def market_make_random_walk(self, symbol, depth, position, ema_key, alpha=0.2, take_width=1, clear_width=2, max_size=50, use_wall=False):
        limit = POSITION_LIMITS[symbol]
        orders = []
        raw_mid = self.get_wall_midpoint(depth) if use_wall else self.get_volume_weighted_mid(depth)
        if raw_mid is None: return orders

        prev_ema = self.stored_data.get(ema_key)
        fair_value = raw_mid if prev_ema is None else prev_ema + alpha * (raw_mid - prev_ema)
        self.stored_data[ema_key] = fair_value

        best_bid, best_ask = self.get_best_prices(depth)
        buy_capacity = limit - position
        sell_capacity = limit + position

        # Take underpriced asks
        for price in sorted(depth.sell_orders):
            if price <= fair_value - take_width and buy_capacity > 0:
                volume = min(-depth.sell_orders[price], buy_capacity)
                if volume > 0: orders.append(Order(symbol, price, volume)); buy_capacity -= volume

        # Take overpriced bids
        for price in sorted(depth.buy_orders, reverse=True):
            if price >= fair_value + take_width and sell_capacity > 0:
                volume = min(depth.buy_orders[price], sell_capacity)
                if volume > 0: orders.append(Order(symbol, price, -volume)); sell_capacity -= volume

        # Clear inventory toward fair value
        projected_position = position + sum(order.quantity for order in orders)
        if projected_position > 0:
            clear_price = int(round(fair_value + clear_width))
            if clear_price in depth.buy_orders and sell_capacity > 0:
                clear_vol = min(depth.buy_orders[clear_price], projected_position, sell_capacity)
                if clear_vol > 0: orders.append(Order(symbol, clear_price, -clear_vol)); sell_capacity -= clear_vol
        elif projected_position < 0:
            clear_price = int(round(fair_value - clear_width))
            if clear_price in depth.sell_orders and buy_capacity > 0:
                clear_vol = min(-depth.sell_orders[clear_price], -projected_position, buy_capacity)
                if clear_vol > 0: orders.append(Order(symbol, clear_price, clear_vol)); buy_capacity -= clear_vol

        # Passive quote inside the spread
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            edge = max(1, spread // 2)
            quote_bid = int(round(fair_value - edge))
            quote_ask = int(round(fair_value + edge))
            if quote_bid <= best_bid: quote_bid = best_bid + 1
            if quote_ask >= best_ask: quote_ask = best_ask - 1
            if quote_bid < quote_ask:
                if buy_capacity > 0: orders.append(Order(symbol, quote_bid, min(max_size, buy_capacity)))
                if sell_capacity > 0: orders.append(Order(symbol, quote_ask, -min(max_size, sell_capacity)))
        return orders

    # Inner-strike option trading with EMA fair value
    def trade_inner_option(self, symbol, depth, position, result):
        best_bid, best_ask = self.get_best_prices(depth)
        if best_bid is None or best_ask is None: return
        bid_volume = depth.buy_orders[best_bid]
        ask_volume = -depth.sell_orders[best_ask]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        ema_key = f'oema_{symbol}'
        prev_ema = self.stored_data.get(ema_key)
        fair_raw = mid_price if prev_ema is None else prev_ema
        fair_value = fair_raw - 0.005 * position

        dynamic_edge = max(14.0, spread * 1.5)
        limit = POSITION_LIMITS[symbol]
        buy_capacity = limit - position
        sell_capacity = limit + position

        if best_ask <= fair_value - dynamic_edge:
            qty = min(30, ask_volume, buy_capacity)
            if qty > 0: result.setdefault(symbol, []).append(Order(symbol, best_ask, qty))
        elif best_bid >= fair_value + dynamic_edge:
            qty = min(30, bid_volume, sell_capacity)
            if qty > 0: result.setdefault(symbol, []).append(Order(symbol, best_bid, -qty))

        updated_ema = fair_raw + 0.0003 * (mid_price - fair_raw)
        self.stored_data[ema_key] = updated_ema

    def solve_3x3_linear(self, coefficients, constants):
        augmented = [row[:] + [constants[i]] for i, row in enumerate(coefficients)]
        for i in range(3):
            pivot_row = max(range(i, 3), key=lambda r: abs(augmented[r][i]))
            augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]
            pivot = augmented[i][i]
            if abs(pivot) < 1e-12: raise ValueError
            for row in range(i + 1, 3):
                factor = augmented[row][i] / pivot
                for col in range(i, 4): augmented[row][col] -= factor * augmented[i][col]
        solution = [0] * 3
        for i in range(2, -1, -1):
            solution[i] = (augmented[i][3] - sum(augmented[i][col] * solution[col] for col in range(i + 1, 3))) / augmented[i][i]
        return solution

    # Volatility smile fitting and outer-strike trading
    def trade_smile_options(self, state, result):
        time_to_expiry = max(0.01, 5 - state.timestamp / 1_000_000)
        if 'VELVETFRUIT_EXTRACT' not in state.order_depths: return
        underlying_depth = state.order_depths['VELVETFRUIT_EXTRACT']
        underlying_bid, underlying_ask = self.get_best_prices(underlying_depth)
        if underlying_bid is None or underlying_ask is None: return
        spot = (underlying_bid + underlying_ask) / 2

        # Collect implied vols for fitting
        implied_vols = {}
        for strike in FIT_STRIKES:
            symbol = f'VEV_{strike}'
            if symbol not in state.order_depths: continue
            depth = state.order_depths[symbol]
            best_bid, best_ask = self.get_best_prices(depth)
            if best_bid is None or best_ask is None: continue
            call_mid = (best_bid + best_ask) / 2
            vol = implied_vol(call_mid, spot, strike, time_to_expiry)
            if vol is None: continue
            moneyness = math.log(strike / spot) / math.sqrt(time_to_expiry)
            implied_vols[strike] = (moneyness, vol, call_mid)
        if len(implied_vols) < 4: return

        # Fit quadratic smile: iv = a*m^2 + b*m + c
        moneyness_vals = [entry[0] for entry in implied_vols.values()]
        vol_vals = [entry[1] for entry in implied_vols.values()]
        n = len(moneyness_vals)
        sum_m = sum(moneyness_vals)
        sum_m2 = sum(m * m for m in moneyness_vals)
        sum_m3 = sum(m ** 3 for m in moneyness_vals)
        sum_m4 = sum(m ** 4 for m in moneyness_vals)
        sum_v = sum(vol_vals)
        sum_mv = sum(m * v for m, v in zip(moneyness_vals, vol_vals))
        sum_m2v = sum(m * m * v for m, v in zip(moneyness_vals, vol_vals))
        try:
            quad_coeff, lin_coeff, const_coeff = self.solve_3x3_linear(
                [[sum_m4, sum_m3, sum_m2], [sum_m3, sum_m2, sum_m], [sum_m2, sum_m, n]],
                [sum_m2v, sum_mv, sum_v]
            )
        except: return

        WARMUP_TICKS = 200
        EMA_SPAN = 300
        ema_alpha = 2.0 / (EMA_SPAN + 1)

        for strike in OUTER_STRIKES:
            if strike not in implied_vols: continue
            symbol = f'VEV_{strike}'
            moneyness, vol, call_mid = implied_vols[strike]

            # Theoretical IV from smile fit
            fitted_iv = quad_coeff * moneyness * moneyness + lin_coeff * moneyness + const_coeff
            theoretical_price = black_scholes_call(spot, strike, time_to_expiry, fitted_iv)

            # Track bias and variance via EMA
            raw_residual = call_mid - theoretical_price
            bias_key = f'b_{strike}'
            var_key = f'v_{strike}'
            count_key = f'c_{strike}'

            prev_bias = self.stored_data.get(bias_key, 0.0)
            bias = prev_bias + ema_alpha * (raw_residual - prev_bias)
            self.stored_data[bias_key] = bias

            deviation = raw_residual - bias
            prev_variance = self.stored_data.get(var_key, 1.0)
            variance = prev_variance + ema_alpha * (deviation * deviation - prev_variance)
            self.stored_data[var_key] = variance

            tick_count = self.stored_data.get(count_key, 0) + 1
            self.stored_data[count_key] = tick_count
            if tick_count < WARMUP_TICKS: continue

            fair_value = theoretical_price + bias
            std_dev = math.sqrt(max(variance, 1e-6))

            depth = state.order_depths[symbol]
            best_bid, best_ask = self.get_best_prices(depth)
            if best_bid is None or best_ask is None: continue
            position = state.position.get(symbol, 0)
            limit = POSITION_LIMITS[symbol]

            edge = max(1, int(round(std_dev * 1.5)))
            quote_bid = int(round(fair_value - edge))
            quote_ask = int(round(fair_value + edge))

            if quote_bid > best_bid: quote_bid = best_bid + 1 if best_bid + 1 < quote_ask else best_bid
            if quote_ask < best_ask: quote_ask = best_ask - 1 if best_ask - 1 > quote_bid else best_ask
            if quote_bid >= fair_value: quote_bid = int(math.floor(fair_value - 1))
            if quote_ask <= fair_value: quote_ask = int(math.ceil(fair_value + 1))

            buy_capacity = limit - position
            sell_capacity = limit + position
            order_size = 10

            if quote_bid < quote_ask and quote_bid > 0:
                if buy_capacity > 0: result.setdefault(symbol, []).append(Order(symbol, quote_bid, min(order_size, buy_capacity)))
                if sell_capacity > 0: result.setdefault(symbol, []).append(Order(symbol, quote_ask, -min(order_size, sell_capacity)))

            # Aggressive take on extreme z-scores
            z_score = (call_mid - fair_value) / std_dev
            if abs(z_score) > 3.5:
                if z_score > 0 and sell_capacity > 0:
                    qty = min(5, sell_capacity, depth.buy_orders.get(best_bid, 0))
                    if qty > 0: result.setdefault(symbol, []).append(Order(symbol, best_bid, -qty))
                elif z_score < 0 and buy_capacity > 0:
                    qty = min(5, buy_capacity, -depth.sell_orders.get(best_ask, 0))
                    if qty > 0: result.setdefault(symbol, []).append(Order(symbol, best_ask, qty))

    def run(self, state):
        if state.traderData:
            try: self.stored_data = json.loads(state.traderData)
            except: self.stored_data = {}
        else: self.stored_data = {}
        result = {}

        # Hydrogel: fixed fair-value market making
        if 'HYDROGEL_PACK' in state.order_depths:
            position = state.position.get('HYDROGEL_PACK', 0)
            result['HYDROGEL_PACK'] = self.trade_hydrogel(state.order_depths['HYDROGEL_PACK'], position)

        # Velvet Fruit Extract: random-walk market making
        if 'VELVETFRUIT_EXTRACT' in state.order_depths:
            position = state.position.get('VELVETFRUIT_EXTRACT', 0)
            result['VELVETFRUIT_EXTRACT'] = self.market_make_random_walk(
                'VELVETFRUIT_EXTRACT', state.order_depths['VELVETFRUIT_EXTRACT'], position, 'vfv', max_size=50
            )

        # Options: inner-strike takes + smile fit on outer strikes
        for strike in INNER_STRIKES:
            symbol = f'VEV_{strike}'
            if symbol in state.order_depths:
                position = state.position.get(symbol, 0)
                self.trade_inner_option(symbol, state.order_depths[symbol], position, result)
        self.trade_smile_options(state, result)

        return result, 0, json.dumps(self.stored_data)

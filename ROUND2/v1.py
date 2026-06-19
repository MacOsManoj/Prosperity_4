#9.05k
"""

REGRESSION FIT: fut5 = +3.22 × OBI1 − 0.56 × dev_20   (R² = 0.45 on 30K ticks)

Translate to quote skew. skew>0 pushes quotes DOWN (sell more).
We want skew>0 when fut5<0 (price expected to fall).
So skew = K × (−3.22 × obi + 0.56 × dev).

Quote shift magnitudes need integer rounding — tune K to match local data sweep.
"""
import json
from typing import List, Dict
from datamodel import OrderDepth, TradingState, Order


def vwap_mid(od):
    bb = sum(p*q for p,q in od.buy_orders.items()) / sum(od.buy_orders.values()) if od.buy_orders else None
    aa = sum(p*(-q) for p,q in od.sell_orders.items()) / sum(-q for q in od.sell_orders.values()) if od.sell_orders else None
    if bb is not None and aa is not None: return (bb+aa)/2.0
    return bb if bb is not None else aa


class Trader:
    POSITION_LIMIT = 80
    P_TAKE_BUY_WIDTH = 1.0
    P_DRIFT = 0.001
    P_INTERCEPT_ALPHA = 2.0 / 101.0
    P_WARMUP_END = 1000
    P_AGGRESSIVE_WIDTH = -10
    P_MAKE_BID_OFFSET = 2

    O_ROLL_WINDOW = 50
    O_JUMP_CLIP = 5.5
    O_TAKE_BUFFER = 0
    O_SOFT_LIMIT = 80
    O_SIZE_MAKE = 15
    O_SIZE_MAKE_JUMP = 20
    O_SKEW_K_POS = 7
    # Combined signal coefficients (from OLS regression)
    O_SKEW_GAIN = 1.0       # overall scaling — tune
    O_COEF_OBI = -3.22      # sign: skew>0 when price falls, OBI>0 predicts rise, so negate
    O_COEF_DEV = 0.56       # dev>0 predicts fall → skew positive

    def bid(self):
        return 150

    def run(self, state: TradingState):
        td = {}
        if state.traderData:
            try:
                p = json.loads(state.traderData)
                if isinstance(p, dict): td = p
            except Exception: pass
        if not isinstance(td.get("ip"), (int, float)): td.pop("ip", None)
        result: Dict[str, List[Order]] = {}

        # PEPPER V19
        product = "INTARIAN_PEPPER_ROOT"
        if product in state.order_depths:
            od = state.order_depths[product]
            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders); best_ask = min(od.sell_orders)
                position = state.position.get(product, 0)
                orders: List[Order] = []
                buy_alloc = sell_alloc = 0
                mid = (best_bid + best_ask) / 2.0
                det = mid - self.P_DRIFT * state.timestamp
                intercept = (self.P_INTERCEPT_ALPHA * det + (1 - self.P_INTERCEPT_ALPHA) * td["ip"]) if "ip" in td else det
                td["ip"] = intercept
                fair = intercept + self.P_DRIFT * state.timestamp
                max_buy = max(self.POSITION_LIMIT - position, 0)
                max_sell = max(self.POSITION_LIMIT + position, 0)
                tw = self.P_AGGRESSIVE_WIDTH if state.timestamp <= self.P_WARMUP_END else self.P_TAKE_BUY_WIDTH
                for ask in sorted(od.sell_orders):
                    if ask <= fair - tw:
                        avol = -od.sell_orders[ask]; can = max_buy - buy_alloc
                        if can <= 0: break
                        q = min(avol, can)
                        if q > 0: orders.append(Order(product, ask, q)); buy_alloc += q
                    else: break
                make_bid = best_bid + self.P_MAKE_BID_OFFSET
                make_ask = best_ask - 1
                if make_bid < make_ask:
                    can = max_buy - buy_alloc
                    if can > 0: orders.append(Order(product, make_bid, can)); buy_alloc += can
                    can = max_sell - sell_alloc
                    if can > 0: orders.append(Order(product, make_ask, -can)); sell_alloc += can
                result[product] = orders

        # OSMIUM — combined signal skew
        product = "ASH_COATED_OSMIUM"
        if product in state.order_depths:
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            orders: List[Order] = []
            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None
            if best_bid is None and best_ask is None:
                result[product] = orders
            else:
                vwap = vwap_mid(od)
                prev = td.get("o_prev_vwap")
                is_jump = (prev is not None and vwap is not None and abs(vwap - prev) > self.O_JUMP_CLIP)
                td["o_prev_vwap"] = vwap

                hist = td.get("o_hist", [])
                if vwap is not None and not is_jump:
                    hist.append(float(vwap))
                    if len(hist) > self.O_ROLL_WINDOW:
                        hist = hist[-self.O_ROLL_WINDOW:]
                td["o_hist"] = hist
                if len(hist) >= 3:
                    rmean = sum(hist) / len(hist)
                else:
                    rmean = vwap if vwap is not None else 10000.0
                fair = round(rmean)
                mid_now = vwap if vwap is not None else rmean
                dev = mid_now - rmean
                bv = od.buy_orders.get(best_bid, 0)
                av = -od.sell_orders.get(best_ask, 0)
                obi = (bv - av) / (bv + av + 1e-9)

                # Combined signal (negative of predicted fut5)
                signal = self.O_COEF_OBI * obi + self.O_COEF_DEV * dev  # negates fut5 prediction
                dev_skew = round(self.O_SKEW_GAIN * signal)

                buy_h = self.POSITION_LIMIT - position
                sell_h = self.POSITION_LIMIT + position

                # Standard takes
                for ask_price in sorted(od.sell_orders):
                    if ask_price >= fair - self.O_TAKE_BUFFER or buy_h <= 0: break
                    qty = min(-od.sell_orders[ask_price], buy_h)
                    if qty > 0: orders.append(Order(product, ask_price, qty)); buy_h -= qty
                for bid_price in sorted(od.buy_orders, reverse=True):
                    if bid_price <= fair + self.O_TAKE_BUFFER or sell_h <= 0: break
                    qty = min(od.buy_orders[bid_price], sell_h)
                    if qty > 0: orders.append(Order(product, bid_price, -qty)); sell_h -= qty

                # MM with combined skew
                pos_skew = round(self.O_SKEW_K_POS * position / self.POSITION_LIMIT)
                skew = pos_skew + dev_skew
                make_sz = self.O_SIZE_MAKE_JUMP if is_jump else self.O_SIZE_MAKE
                fb = 1 if is_jump else 6
                bid_q = min(best_bid + 1, fair - 1 - skew) if best_bid is not None else fair - fb - skew
                ask_q = max(best_ask - 1, fair + 1 - skew) if best_ask is not None else fair + fb - skew
                if buy_h > 0 and position < 80:
                    orders.append(Order(product, bid_q, min(make_sz, buy_h)))
                if sell_h > 0 and position > -80:
                    orders.append(Order(product, ask_q, -min(make_sz, sell_h)))
                result[product] = orders

        return result, 0, json.dumps(td)
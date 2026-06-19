# 6k
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json


class Trader:

    # ── Osmium (unchanged from strat5) ──────────────────────
    OSM_FV         = 10000
    OSM_LIMIT      = 80
    TAKE_MICRO     = 3
    TAKE_SMALL     = 5
    TAKE_MED       = 9
    TAKE_LARGE     = 15
    MIN_MAKER_EDGE = 2

    # ── Pepper Root ──────────────────────────────────────────
    PEPPER_LIMIT   = 80

    # Trend layer: shift all quotes UP by this many ticks
    # so the market drifts us long over time (+1000/day trend)
    TREND_SHIFT    = 1      # bid+1, ask+1 vs neutral

    # Zigzag layer: additional price offset per regime
    # (bid_extra, ask_extra) — positive = more aggressive on that side
    ZIGZAG = {
        'up':   (0,  1),   # after UP:   ask tighter, bid looser
        'down': (1,  0),   # after DOWN: bid tighter, ask looser
        'flat': (1,  1),   # flat:       tight both (standard pennying)
    }

    PEPPER_KEY = "pepper_last_mid"

    # ────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result = {}

        # decode last pepper mid
        pepper_last_mid = 0.0
        try:
            td = json.loads(state.traderData) if state.traderData else {}
            pepper_last_mid = float(td.get(self.PEPPER_KEY, 0))
        except Exception:
            pepper_last_mid = 0.0

        new_pepper_mid = pepper_last_mid

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            has_bids = len(order_depth.buy_orders) > 0
            has_asks = len(order_depth.sell_orders) > 0
            best_bid = max(order_depth.buy_orders.keys())  if has_bids else 0
            best_ask = min(order_depth.sell_orders.keys()) if has_asks else 0

            # ═══════════════════════════════════════════════
            # ASH_COATED_OSMIUM — unchanged from strat5
            # ═══════════════════════════════════════════════
            if product == "ASH_COATED_OSMIUM":
                fv  = self.OSM_FV
                lim = self.OSM_LIMIT
                pos = state.position.get(product, 0)
                buy_cap  =  lim - pos
                sell_cap = -lim - pos

                if has_bids and has_asks:
                    for ask_price, ask_vol_neg in sorted(order_depth.sell_orders.items()):
                        if buy_cap <= 0: break
                        diff = fv - ask_price
                        if diff <= 0: continue
                        avail = -ask_vol_neg
                        if   diff >= self.TAKE_LARGE: take = min(buy_cap, avail)
                        elif diff >= self.TAKE_MED:   take = min(buy_cap, avail * 2 // 3)
                        elif diff >= self.TAKE_SMALL: take = min(buy_cap, avail // 2)
                        elif diff >= self.TAKE_MICRO: take = min(buy_cap, min(avail, 5))
                        else: continue
                        if take > 0:
                            orders.append(Order(product, ask_price, take))
                            buy_cap -= take

                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if sell_cap >= 0: break
                        diff = bid_price - fv
                        if diff <= 0: continue
                        avail = bid_vol
                        if   diff >= self.TAKE_LARGE: take = min(-sell_cap, avail)
                        elif diff >= self.TAKE_MED:   take = min(-sell_cap, avail * 2 // 3)
                        elif diff >= self.TAKE_SMALL: take = min(-sell_cap, avail // 2)
                        elif diff >= self.TAKE_MICRO: take = min(-sell_cap, min(avail, 5))
                        else: continue
                        if take > 0:
                            orders.append(Order(product, bid_price, -take))
                            sell_cap += take

                if has_bids and has_asks and buy_cap > 0:
                    our_bid = min(fv - self.MIN_MAKER_EDGE, best_bid + 1)
                    orders.append(Order(product, our_bid, buy_cap))
                if has_bids and has_asks and sell_cap < 0:
                    our_ask = max(fv + self.MIN_MAKER_EDGE, best_ask - 1)
                    orders.append(Order(product, our_ask, sell_cap))

            # ═══════════════════════════════════════════════
            # INTARIAN_PEPPER_ROOT — price-skewing maker
            # ═══════════════════════════════════════════════
            elif product == "INTARIAN_PEPPER_ROOT":
                pos = state.position.get(product, 0)
                lim = self.PEPPER_LIMIT

                # one-sided guard
                if not (has_bids and has_asks):
                    result[product] = orders
                    continue

                cur_mid = (best_bid + best_ask) / 2.0
                new_pepper_mid = cur_mid

                # ── zigzag regime ────────────────────────
                if pepper_last_mid > 0:
                    move = cur_mid - pepper_last_mid
                    regime = 'up' if move > 0.5 else ('down' if move < -0.5 else 'flat')
                else:
                    regime = 'flat'

                bid_extra, ask_extra = self.ZIGZAG[regime]

                # ── base quotes: penny the book ──────────
                base_bid = best_bid + 1
                base_ask = best_ask - 1
                if base_bid >= base_ask:          # collapsed spread guard
                    base_bid = best_bid
                    base_ask = best_ask

                # ── apply trend shift + zigzag ────────────
                # Trend: shift both quotes UP by TREND_SHIFT
                #   → net effect: we sell cheaper to buyers
                #     (ask goes down → fills faster) and buy
                #     cheaper from sellers (bid stays or up)
                #     giving us a long inventory bias.
                our_bid = base_bid + self.TREND_SHIFT + (bid_extra - 1)
                our_ask = base_ask + self.TREND_SHIFT - (ask_extra - 1)

                # flat : our_bid = base_bid + TREND_SHIFT (bid+1+1=bid+2)
                #        our_ask = base_ask + TREND_SHIFT (ask-1+1=ask)
                # up   : our_bid = base_bid + TREND_SHIFT - 1 (= bid+1)
                #        our_ask = base_ask + TREND_SHIFT     (= ask)  tighter sell
                # down : our_bid = base_bid + TREND_SHIFT + 1 (= bid+3) tighter buy
                #        our_ask = base_ask + TREND_SHIFT + 1 (= ask+1) relaxed sell

                # collapsed spread guard after adjustments
                if our_bid >= our_ask:
                    our_bid = best_bid + self.TREND_SHIFT
                    our_ask = best_ask + self.TREND_SHIFT

                # ── full-capacity quoting ─────────────────
                # KEY FIX: always quote full remaining capacity,
                # no LOT size cap — matches strat4/5 behavior.
                buy_cap  =  lim - pos
                sell_cap = -lim - pos   # negative

                if buy_cap > 0:
                    orders.append(Order(product, int(our_bid), buy_cap))
                if sell_cap < 0:
                    orders.append(Order(product, int(our_ask), sell_cap))

            result[product] = orders

        traderData = json.dumps({self.PEPPER_KEY: new_pepper_mid})
        return result, 0, traderData
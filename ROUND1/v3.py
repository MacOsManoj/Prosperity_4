#8.9k
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle

LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
}

POSITION_BUFFER = 2  # never exceed limit - buffer


class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        data = {}
        if state.traderData and state.traderData != "":
            try:
                data = jsonpickle.decode(state.traderData)
            except Exception:
                data = {}

        if "osmium_mids" not in data:
            data["osmium_mids"] = []

        for product in state.order_depths:
            orders: List[Order] = []
            depth: OrderDepth = state.order_depths[product]
            pos = state.position.get(product, 0)
            limit = LIMITS.get(product, 80)

            best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
            best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

            if best_bid is None and best_ask is None:
                result[product] = orders
                continue

            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid = best_bid
            else:
                mid = best_ask

            # ==========================================================
            # INTARIAN_PEPPER_ROOT
            #
            # Price drifts +1 per 10 ticks = +1000 per full day.
            # Buy to 80 immediately. Never sell. Hold for drift.
            # 80 units * 1000 drift = 80,000 profit per day.
            # ==========================================================
            if product == "INTARIAN_PEPPER_ROOT":

                # Sweep every ask on the book to fill to 80
                if depth.sell_orders:
                    for ask_price in sorted(depth.sell_orders.keys()):
                        if pos >= limit:
                            break
                        ask_vol = -depth.sell_orders[ask_price]
                        can_buy = min(ask_vol, limit - pos)
                        if can_buy > 0:
                            orders.append(Order(product, ask_price, can_buy))
                            pos += can_buy

                # Passive bid to fill any remaining capacity
                if pos < limit and best_bid is not None:
                    bid_price = min(best_bid + 1, int(mid))
                    bid_qty = limit - pos
                    if bid_qty > 0:
                        orders.append(Order(product, bid_price, bid_qty))

                # NEVER SELL

            # ==========================================================
            # ASH_COATED_OSMIUM
            #
            # Mean-reverts ~10000. Typical book: 9993 / 10009 (spread 16).
            # Strategy:
            #   1. Take any ask < fair, any bid > fair (aggressive)
            #   2. Penny-jump: bid at best_bid+1, ask at best_ask-1
            #   3. Skew quotes hard to flatten inventory
            # ==========================================================
            elif product == "ASH_COATED_OSMIUM":
                data["osmium_mids"].append(mid)

                # Fair value: rolling mean of last 30 mids
                recent = data["osmium_mids"][-30:]
                fair = sum(recent) / len(recent)

                safe_limit = limit - POSITION_BUFFER  # 78

                # --- STEP 1: Aggressive taking ---
                # Buy anything offered at or below fair value
                if depth.sell_orders:
                    for ask_price in sorted(depth.sell_orders.keys()):
                        if ask_price > fair:
                            break
                        if pos >= safe_limit:
                            break
                        ask_vol = -depth.sell_orders[ask_price]
                        can_buy = min(ask_vol, safe_limit - pos)
                        if can_buy > 0:
                            orders.append(Order(product, ask_price, can_buy))
                            pos += can_buy

                # Sell to anything bidding at or above fair value
                if depth.buy_orders:
                    for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                        if bid_price < fair:
                            break
                        if pos <= -safe_limit:
                            break
                        bid_vol = depth.buy_orders[bid_price]
                        can_sell = min(bid_vol, safe_limit + pos)
                        if can_sell > 0:
                            orders.append(Order(product, bid_price, -can_sell))
                            pos -= can_sell

                # --- STEP 2: Penny-jump quoting ---
                # Place bid at best_bid+1, ask at best_ask-1
                # This puts us at the front of the book
                # But clamp to not cross fair value too aggressively

                # Inventory skew: stronger than v1
                skew = pos * 0.12  # at pos=50, skew=6

                if best_bid is not None and best_ask is not None:
                    # Primary quotes: penny-jump
                    my_bid = best_bid + 1
                    my_ask = best_ask - 1

                    # Apply skew
                    my_bid = int(round(my_bid - skew))
                    my_ask = int(round(my_ask - skew))

                    # Safety: don't let bid >= ask
                    if my_bid >= my_ask:
                        my_bid = int(fair - 1 - skew)
                        my_ask = int(fair + 1 - skew)
                        if my_bid >= my_ask:
                            my_ask = my_bid + 1

                    max_buy = max(0, safe_limit - pos)
                    max_sell = max(0, safe_limit + pos)

                    buy_qty = min(20, max_buy)
                    sell_qty = min(20, max_sell)

                    if buy_qty > 0:
                        orders.append(Order(product, my_bid, buy_qty))
                    if sell_qty > 0:
                        orders.append(Order(product, my_ask, -sell_qty))

                    # Secondary layer: deeper quotes for catching bigger moves
                    bid2 = int(round(fair - 6 - skew))
                    ask2 = int(round(fair + 6 - skew))

                    buy_qty2 = min(15, max(0, max_buy - buy_qty))
                    sell_qty2 = min(15, max(0, max_sell - sell_qty))

                    if buy_qty2 > 0:
                        orders.append(Order(product, bid2, buy_qty2))
                    if sell_qty2 > 0:
                        orders.append(Order(product, ask2, -sell_qty2))

            result[product] = orders

        # Trim data
        data["osmium_mids"] = data["osmium_mids"][-50:]

        traderData = jsonpickle.encode(data)
        return result, conversions, traderData
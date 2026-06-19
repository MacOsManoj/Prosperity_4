from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json


POS_CAP = 10

QUOTING_PARAMS = {
    "SNACKPACK_CHOCOLATE":  {"ag": 2, "ps": 2, "vol": 3, "tilt": 0.20},
    "SNACKPACK_VANILLA":    {"ag": 2, "ps": 2, "vol": 3, "tilt": 0.20},
    "SNACKPACK_PISTACHIO":  {"ag": 2, "ps": 2, "vol": 3, "tilt": 0.20},
    "SNACKPACK_STRAWBERRY": {"ag": 2, "ps": 2, "vol": 3, "tilt": 0.20},
    "SNACKPACK_RASPBERRY":  {"ag": 2, "ps": 2, "vol": 3, "tilt": 0.20},
    "PANEL_1X2": {"ag": 2, "ps": 2, "vol": 3, "tilt": 0.25},
    "PANEL_2X2": {"ag": 2, "ps": 2, "vol": 3, "tilt": 0.25},
    "PANEL_1X4": {"ag": 2, "ps": 2, "vol": 3, "tilt": 0.25},
    "PANEL_2X4": {"ag": 3, "ps": 3, "vol": 2, "tilt": 0.30},
    "PANEL_4X4": {"ag": 3, "ps": 3, "vol": 2, "tilt": 0.30},
    "PEBBLES_XS": {"ag": 5, "ps": 5, "vol": 2, "tilt": 0.50},
    "PEBBLES_S":  {"ag": 5, "ps": 5, "vol": 2, "tilt": 0.50},
    "PEBBLES_M":  {"ag": 5, "ps": 5, "vol": 2, "tilt": 0.50},
    "PEBBLES_L":  {"ag": 5, "ps": 5, "vol": 2, "tilt": 0.50},
    "PEBBLES_XL": {"ag": 6, "ps": 6, "vol": 2, "tilt": 0.60},
    "MICROCHIP_SQUARE": {"ag": 5, "ps": 5, "vol": 1, "tilt": 0.50},
    "ROBOT_DISHES":     {"ag": 4, "ps": 4, "vol": 1, "tilt": 0.50},
}

FALLBACK_PARAMS = {"ag": 3, "ps": 3, "vol": 2, "tilt": 0.30}

TUNED_OVERRIDES = {
    "UV_VISOR_ORANGE":     {"ag": 2, "ps": 1, "vol": 2, "tilt": 0.55,
                            "gap_floor": 4, "inv_ceil": 10},
    "SLEEP_POD_POLYESTER": {"ag": 3, "ps": 2, "vol": 1, "tilt": 0.60,
                            "gap_floor": 4, "inv_ceil": 8},
    "SNACKPACK_VANILLA":   {"ag": 2, "ps": 1, "vol": 3, "tilt": 0.35,
                            "gap_floor": 6, "inv_ceil": 10},
}

BLACKLIST = {
    "GALAXY_SOUNDS_PLANETARY_RINGS", "OXYGEN_SHAKE_MINT",
    "SLEEP_POD_LAMB_WOOL", "PEBBLES_XS", "PEBBLES_S",
    "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    "OXYGEN_SHAKE_EVENING_BREATH", "SLEEP_POD_POLYESTER",
    "UV_VISOR_MAGENTA", "UV_VISOR_YELLOW", "UV_VISOR_AMBER",
    "ROBOT_VACUUMING", "PANEL_2X2",
    "GALAXY_SOUNDS_BLACK_HOLES", "MICROCHIP_TRIANGLE",
    "OXYGEN_SHAKE_GARLIC", "PANEL_4X4", "ROBOT_LAUNDRY",
    "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_SPACE_GRAY", "UV_VISOR_ORANGE", "UV_VISOR_RED",
}

ROCK_GROUP = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
ROCK_ANCHOR = 50_000.0
ROCK_INV_MAX = 5

FOOD_GROUP = ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"]
FOOD_ANCHOR = 19_880.0

HL_FAST = 20
HL_SLOW = 100

SIGNAL_TABLE = {
    "GALAXY_SOUNDS_DARK_MATTER":
        {"hf": 50, "hs": 800, "enter": 21, "leave": 10.5, "bound": 10, "sign": -1},
    "GALAXY_SOUNDS_SOLAR_FLAMES":
        {"hf": 50, "hs": 800, "enter": 180, "leave": 90, "bound": 10, "sign": 1},
    "GALAXY_SOUNDS_SOLAR_WINDS":
        {"hf": 120, "hs": 500, "enter": 233, "leave": 116.5, "bound": 10, "sign": -1},
    "SLEEP_POD_COTTON":
        {"hf": 20, "hs": 200, "enter": 120, "leave": 60, "bound": 10, "sign": 1},
    "SLEEP_POD_SUEDE":
        {"hf": 10, "hs": 100, "enter": 120, "leave": 60, "bound": 10, "sign": -1},
    "MICROCHIP_RECTANGLE":
        {"hf": 120, "hs": 200, "enter": 3, "leave": 1.5, "bound": 10, "sign": -1},
    "MICROCHIP_CIRCLE":
        {"hf": 120, "hs": 500, "enter": 8, "leave": 4, "bound": 10, "sign": 1},
    "MICROCHIP_OVAL":
        {"hf": 5, "hs": 800, "enter": 144, "leave": 72, "bound": 10, "sign": 1},
    "MICROCHIP_SQUARE":
        {"hf": 20, "hs": 100, "enter": 144, "leave": 72, "bound": 10, "sign": 1},
    "ROBOT_MOPPING":
        {"hf": 20, "hs": 200, "enter": 34, "leave": 17, "bound": 10, "sign": 1},
    "ROBOT_IRONING":
        {"hf": 5, "hs": 800, "enter": 180, "leave": 90, "bound": 10, "sign": 1},
    "TRANSLATOR_ECLIPSE_CHARCOAL":
        {"hf": 50, "hs": 800, "enter": 377, "leave": 188.5, "bound": 10, "sign": -1},
    "TRANSLATOR_VOID_BLUE":
        {"hf": 5, "hs": 800, "enter": 233, "leave": 116.5, "bound": 10, "sign": -1},
    "PANEL_1X4":
        {"hf": 10, "hs": 100, "enter": 34, "leave": 17, "bound": 10, "sign": 1},
    "PANEL_2X4":
        {"hf": 120, "hs": 800, "enter": 144, "leave": 72, "bound": 10, "sign": -1},
    "OXYGEN_SHAKE_MORNING_BREATH":
        {"hf": 80, "hs": 300, "enter": 1, "leave": 0.5, "bound": 10, "sign": 1},
    "PANEL_1X2":
        {"hf": 120, "hs": 500, "enter": 144, "leave": 72, "bound": 10, "sign": -1},
    "ROBOT_DISHES":
        {"hf": 120, "hs": 500, "enter": 89, "leave": 44.5, "bound": 10, "sign": -1},
    "SLEEP_POD_NYLON":
        {"hf": 120, "hs": 800, "enter": 2, "leave": 1, "bound": 10, "sign": -1},
    "SNACKPACK_RASPBERRY":
        {"hf": 5, "hs": 800, "enter": 180, "leave": 90, "bound": 10, "sign": -1},
    "SNACKPACK_CHOCOLATE":
        {"hf": 120, "hs": 800, "enter": 1, "leave": 0.5, "bound": 10, "sign": -1},
    "SNACKPACK_VANILLA":
        {"hf": 120, "hs": 800, "enter": 5, "leave": 2.5, "bound": 10, "sign": -1},
    "SNACKPACK_PISTACHIO":
        {"hf": 120, "hs": 800, "enter": 8, "leave": 4, "bound": 10, "sign": -1},
    "SNACKPACK_STRAWBERRY":
        {"hf": 120, "hs": 200, "enter": 13, "leave": 6.5, "bound": 10, "sign": -1},
    "TRANSLATOR_SPACE_GRAY":
        {"hf": 20, "hs": 100, "enter": 34, "leave": 17, "bound": 10, "sign": 1},
}


def _alpha(hl):
    return 1.0 - math.exp(-math.log(2) / hl)


def _center(ob):
    if not ob.buy_orders or not ob.sell_orders:
        return None
    return (max(ob.buy_orders) + min(ob.sell_orders)) / 2


def _rock_gap(depths):
    s = 0.0
    for name in ROCK_GROUP:
        c = _center(depths.get(name, None)) if name in depths else None
        if c is None:
            return None
        s += c
    return s - ROCK_ANCHOR


def _food_gap(depths):
    s = 0.0
    for name in FOOD_GROUP:
        if name not in depths:
            return None
        c = _center(depths[name])
        if c is None:
            return None
        s += c
    return s - FOOD_ANCHOR


def _build_quotes(sym, ob, inv, cfg, mem, ref_px=None, extra_bias=0.0, hard_cap=None):
    if not ob.buy_orders or not ob.sell_orders:
        return []

    lim = POS_CAP
    eff_cap = lim if hard_cap is None else min(lim, int(hard_cap))
    cross_w = cfg["ag"]
    post_w = cfg["ps"]
    clip = int(cfg["vol"])
    lean = cfg["tilt"]

    hi_bid = max(ob.buy_orders)
    lo_ask = min(ob.sell_orders)
    mp = (hi_bid + lo_ask) / 2

    a_f = _alpha(HL_FAST)
    a_s = _alpha(HL_SLOW)
    ef = mem.get(f"{sym}_ema_f", mp)
    es = mem.get(f"{sym}_ema_s", mp)
    ef += a_f * (mp - ef)
    es += a_s * (mp - es)
    mem[f"{sym}_ema_f"] = ef
    mem[f"{sym}_ema_s"] = es

    drift = ef - es
    drift_adj = max(-3.0, min(3.0, 0.30 * drift))

    anchor = ref_px if ref_px is not None else mp
    theo = anchor - lean * inv + drift_adj + extra_bias

    room_up = max(0, eff_cap - inv)
    room_dn = max(0, eff_cap + inv)
    out = []

    for px in sorted(ob.sell_orders):
        if room_up <= 0:
            break
        sz = -ob.sell_orders[px]
        if theo - px >= cross_w:
            n = min(sz, room_up)
            if n > 0:
                out.append(Order(sym, px, n))
                room_up -= n
        else:
            break

    for px in sorted(ob.buy_orders, reverse=True):
        if room_dn <= 0:
            break
        sz = ob.buy_orders[px]
        if px - theo >= cross_w:
            n = min(sz, room_dn)
            if n > 0:
                out.append(Order(sym, px, -n))
                room_dn -= n
        else:
            break

    want_bid = math.floor(theo - post_w)
    my_bid = min(hi_bid + 1, want_bid, lo_ask - 1)
    if room_up > 0 and my_bid > hi_bid:
        out.append(Order(sym, my_bid, min(clip, room_up)))

    want_ask = math.ceil(theo + post_w)
    my_ask = max(lo_ask - 1, want_ask, hi_bid + 1)
    if room_dn > 0 and my_ask < lo_ask:
        out.append(Order(sym, my_ask, -min(clip, room_dn)))

    return out


def _chase_trend(sym, ob, inv, rule, mem):
    if not ob.buy_orders or not ob.sell_orders:
        return []

    hi_bid = max(ob.buy_orders)
    lo_ask = min(ob.sell_orders)
    mp = (hi_bid + lo_ask) / 2

    a_f = _alpha(rule["hf"])
    a_s = _alpha(rule["hs"])
    kf = f"{sym}_rt_ema_f"
    ks = f"{sym}_rt_ema_s"
    ef = mem.get(kf, mp)
    es = mem.get(ks, mp)
    ef += a_f * (mp - ef)
    es += a_s * (mp - es)
    mem[kf] = ef
    mem[ks] = es

    reading = rule["sign"] * (ef - es)
    ceiling = min(POS_CAP, int(rule["bound"]))

    if reading > rule["enter"]:
        goal = ceiling
    elif reading < -rule["enter"]:
        goal = -ceiling
    elif abs(reading) < rule["leave"]:
        goal = 0
    else:
        goal = inv

    gap = goal - inv
    out = []

    if gap > 0:
        left = min(gap, POS_CAP - inv)
        for px in sorted(ob.sell_orders):
            if left <= 0:
                break
            n = min(-ob.sell_orders[px], left)
            if n > 0:
                out.append(Order(sym, px, n))
                left -= n
    elif gap < 0:
        left = min(-gap, POS_CAP + inv)
        for px in sorted(ob.buy_orders, reverse=True):
            if left <= 0:
                break
            n = min(ob.buy_orders[px], left)
            if n > 0:
                out.append(Order(sym, px, -n))
                left -= n

    return out


class Trader:

    def run(self, state: TradingState):
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}

        book = state.order_depths
        fills: Dict[str, List[Order]] = {}

        rg = _rock_gap(book)
        fg = _food_gap(book)

        for sym, ob in book.items():
            inv = state.position.get(sym, 0)

            try:
                if sym in BLACKLIST:
                    fills[sym] = []
                    continue

                if sym in SIGNAL_TABLE:
                    fills[sym] = _chase_trend(sym, ob, inv, SIGNAL_TABLE[sym], mem)
                    continue

                if sym in TUNED_OVERRIDES:
                    p = dict(TUNED_OVERRIDES[sym])
                else:
                    p = dict(QUOTING_PARAMS.get(sym, FALLBACK_PARAMS))

                ref = None
                bias = 0.0

                if sym in ROCK_GROUP and rg is not None:
                    bias -= rg / 5.0
                if sym in FOOD_GROUP and fg is not None:
                    bias -= fg / 2.0

                if sym in TUNED_OVERRIDES:
                    tc = TUNED_OVERRIDES[sym]
                    if ob.buy_orders and ob.sell_orders:
                        sp = min(ob.sell_orders) - max(ob.buy_orders)
                        if sp < tc.get("gap_floor", 0):
                            fills[sym] = []
                            continue
                    else:
                        fills[sym] = []
                        continue

                hc = ROCK_INV_MAX if sym in ROCK_GROUP else None

                fills[sym] = _build_quotes(
                    sym, ob, inv, p, mem,
                    ref_px=ref, extra_bias=bias, hard_cap=hc,
                )

            except Exception as e:
                print(f"Error trading {sym}: {e}")
                fills[sym] = []

        try:
            td = json.dumps(mem)
        except Exception:
            td = ""

        return fills, 0, td
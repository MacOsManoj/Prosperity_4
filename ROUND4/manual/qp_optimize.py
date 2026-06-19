"""
Round 4 portfolio QP.

Maximize  E[PnL]  -  0.5 * lambda * Var[PnL_score]
subject to per-contract caps |q_i| <= cap_i.

Key design choices motivated by the upstream analysis:

  1. Bid-ask is real. Long leg pays the ASK, short leg receives the BID.
     We split each instrument into two non-negative legs (q_long, q_short)
     with q_i = q_long_i - q_short_i. The optimizer never picks both >0
     simultaneously because that's a strictly dominated round-trip cost.

  2. Score = mean PnL over 100 fresh paths (manual rule), so the relevant
     risk is Var[mean] = Var[PnL] / N_score. Scoring N is constant, so it
     does not change argmax for fixed lambda, but we report the score-level
     std so lambda has interpretable units.

  3. Variance is computed on a SINGLE shared MC sample (paths). All legs
     use the same noise so cross-leg covariance (chooser-vs-replication,
     KO-vs-vanilla put, etc) is captured — that's the whole reason the
     QP is preferable to per-leg sizing.

  4. Cap on AC_45_KO is 500 vs 50 elsewhere (manual table).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from ROUND4.manual.mc_engine import (
    simulate_paths, T1_STEPS, T2_STEPS, S0, SIGMA, DT,
    call_payoff, put_payoff,
    binary_put_payoff, knockout_put_payoff, chooser_payoff,
    bs_call, bs_put,
)

# ----- world / contracts (single source of truth) ---------------------------
SIZE = 3_000              # contract multiplier
N_SCORE = 100             # manual scoring sample size
BIN_AMOUNT = 10.0
N_PATHS = 1_000_000         # MC paths for the QP itself

MARKET = {
    # vanillas (T2 = 21d)
    'AC_50_P':   dict(kind='put',         strike=50, expiry=T2_STEPS, bid=12.00, ask=12.05, cap=50),
    'AC_50_C':   dict(kind='call',        strike=50, expiry=T2_STEPS, bid=12.00, ask=12.05, cap=50),
    'AC_35_P':   dict(kind='put',         strike=35, expiry=T2_STEPS, bid= 4.33, ask= 4.35, cap=50),
    'AC_40_P':   dict(kind='put',         strike=40, expiry=T2_STEPS, bid= 6.50, ask= 6.55, cap=50),
    'AC_45_P':   dict(kind='put',         strike=45, expiry=T2_STEPS, bid= 9.05, ask= 9.10, cap=50),
    'AC_60_C':   dict(kind='call',        strike=60, expiry=T2_STEPS, bid= 8.80, ask= 8.85, cap=50),
    # vanillas (T1 = 14d)
    'AC_50_P_2': dict(kind='put',         strike=50, expiry=T1_STEPS, bid= 9.70, ask= 9.75, cap=50),
    'AC_50_C_2': dict(kind='call',        strike=50, expiry=T1_STEPS, bid= 9.70, ask= 9.75, cap=50),
    # exotics
    'AC_50_CO':  dict(kind='chooser',     strike=50, expiry=T2_STEPS, decision=T1_STEPS,
                                                                    bid=22.20, ask=22.30, cap=50),
    'AC_40_BP':  dict(kind='binary_put',  strike=40, expiry=T2_STEPS, payout=BIN_AMOUNT,
                                                                    bid= 5.00, ask= 5.10, cap=50),
    'AC_45_KO':  dict(kind='ko_put',      strike=45, expiry=T2_STEPS, barrier=35,
                                                                    bid= 0.15, ask= 0.175, cap=500),
    'AC':        dict(kind='underlying',  strike=None, expiry=T2_STEPS,
                                                                    bid=49.975, ask=50.025, cap=200),
}


def per_unit_payoff(name, paths):
    """Per-path terminal payoff (no premium, no multiplier) for one unit long."""
    s = MARKET[name]
    k = s['kind']
    if k == 'call':        return call_payoff(paths[:, s['expiry']], s['strike'])
    if k == 'put':         return put_payoff (paths[:, s['expiry']], s['strike'])
    if k == 'binary_put':  return binary_put_payoff(paths[:, s['expiry']], s['strike'], s['payout'])
    if k == 'ko_put':      return knockout_put_payoff(paths, K=s['strike'], barrier=s['barrier'])
    if k == 'chooser':     return chooser_payoff(paths, s['strike'],
                                                  t1_step=s['decision'], T_step=s['expiry'])
    if k == 'underlying':  return paths[:, s['expiry']]
    raise ValueError(k)


def true_E_payoff(name):
    """
    Closed-form fair value of one unit long, under r=0 GBM with module SIGMA.
    Returns None for path-dependent products (KO put) — those need MC.
    Chooser uses the Rubinstein r=0 decomposition: chooser = call(K,T) + put(K,t1).
    """
    s = MARKET[name]
    k = s['kind']
    if k == 'underlying':  return S0
    tau_T = s['expiry'] * DT
    if k == 'call':        return bs_call(S0, s['strike'], tau_T)
    if k == 'put':         return bs_put (S0, s['strike'], tau_T)
    if k == 'binary_put':
        d2 = (np.log(S0 / s['strike']) - 0.5 * SIGMA**2 * tau_T) / (SIGMA * np.sqrt(tau_T))
        return s['payout'] * norm.cdf(-d2)
    if k == 'chooser':
        tau_t1 = s['decision'] * DT
        return bs_call(S0, s['strike'], tau_T) + bs_put(S0, s['strike'], tau_t1)
    if k == 'ko_put':      return None      # path-dependent → MC
    raise ValueError(k)


# ----- build payoff matrices (long / short share path noise) ---------------
def build(seed=2):
    """
    Returns names, caps, X_long, X_short, mu_long, mu_short, fv.
    X_*  : per-path PnL (used only for variance / covariance).
    mu_* : deterministic linear means using closed-form FV where available,
           MC sample mean otherwise (only KO put). Splitting mean (analytical)
           from variance (MC) prevents MC noise from creating phantom 'edge'
           on legs that are actually fairly priced.
    """
    paths = simulate_paths(N_PATHS, n_steps=T2_STEPS, seed=seed)
    names = list(MARKET)
    caps  = np.array([MARKET[n]['cap'] for n in names], dtype=float)
    payoffs = [per_unit_payoff(n, paths) for n in names]
    X_long  = np.column_stack([SIZE * (p - MARKET[n]['ask']) for n, p in zip(names, payoffs)])
    X_short = np.column_stack([SIZE * (MARKET[n]['bid'] - p) for n, p in zip(names, payoffs)])
    fv = np.array([
        p.mean() if true_E_payoff(n) is None else true_E_payoff(n)
        for n, p in zip(names, payoffs)
    ])
    asks = np.array([MARKET[n]['ask'] for n in names])
    bids = np.array([MARKET[n]['bid'] for n in names])
    mu_long  = SIZE * (fv  - asks)
    mu_short = SIZE * (bids - fv)
    return names, caps, X_long, X_short, mu_long, mu_short, fv


# ----- QP solver -----------------------------------------------------------
# Objective options (selectable via `objective=` arg):
#
#   'var'     mean - 0.5*lam*Var(PnL)             ← classic mean-variance.
#   'linear'  mean                                 ← lam ignored. Pure max-EV
#                                                   subject only to caps. Use when
#                                                   you trust mu and the
#                                                   leaderboard rewards mean PnL.
#   'cvar'    mean - lam * tail_loss_alpha         ← penalizes only left tail.
#                                                   tail_loss = -E[PnL | PnL<=VaR_a]
#                                                   Doesn't punish skew/upside.
#                                                   (CVaR_ALPHA below sets a.)
#   'semivar' mean - 0.5*lam*SemiVar(PnL)          ← downside-only second moment.
#                                                   SemiVar = E[(PnL-mu)^2 ; PnL<mu]
#                                                   Smoother than CVaR; ignores
#                                                   right-tail spread (e.g. KO long).
#
# All four use the SAME analytical mean (mu_long, mu_short) so vanilla legs with
# no real edge don't get phantom MC-noise loadings. Only the risk term differs.

CVAR_ALPHA = 0.05    # tail probability for 'cvar' objective. 0.05 = bottom 5%.


def solve(lam, names, caps, X_long, X_short, mu_long, mu_short,
          pin=None, override_caps=None, objective='var'):
    """
    Variables: z = [q_long, q_short], each in [0, cap].
    pin: dict {name: signed_qty} forces q_i = qty.
    override_caps: dict {name: cap} per-leg tighter cap.
    objective: one of 'var', 'linear', 'cvar', 'semivar'. See module header.
    """
    n = len(names)
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    pin = pin or {}
    override_caps = override_caps or {}

    eff_cap = caps.copy()
    for nm, c in override_caps.items():
        eff_cap[name_to_idx[nm]] = c

    bounds = [(0.0, c) for c in eff_cap] + [(0.0, c) for c in eff_cap]
    for nm, qty in pin.items():
        i = name_to_idx[nm]
        if qty >= 0:
            bounds[i]      = (qty, qty)
            bounds[n + i]  = (0.0, 0.0)
        else:
            bounds[i]      = (0.0, 0.0)
            bounds[n + i]  = (-qty, -qty)

    N = X_long.shape[0]
    k_tail = max(1, int(round(CVAR_ALPHA * N)))   # # paths in CVaR tail

    def split(z):
        return z[:n], z[n:]

    def pnl_and_mean(z):
        ql, qs = split(z)
        pnl  = X_long @ ql + X_short @ qs
        mean = mu_long @ ql + mu_short @ qs
        return pnl, mean

    # --- objective dispatch --------------------------------------------------
    if objective == 'linear':
        _g_lin = -np.concatenate([mu_long, mu_short])
        def neg_obj(z):
            _, mean = pnl_and_mean(z)
            return -mean
        grad = lambda _z: _g_lin

    elif objective == 'var':
        # classic mean-variance: max mean - 0.5 * lam * Var(PnL).
        def neg_obj(z):
            pnl, mean = pnl_and_mean(z)
            return -(mean - 0.5 * lam * pnl.var(ddof=1))
        def grad(z):
            pnl, _ = pnl_and_mean(z)
            c = pnl - pnl.mean()
            var_grad_l = 2 * (X_long  - X_long.mean(0)) .T @ c / (N - 1)
            var_grad_s = 2 * (X_short - X_short.mean(0)).T @ c / (N - 1)
            gl = -(mu_long  - 0.5 * lam * var_grad_l)
            gs = -(mu_short - 0.5 * lam * var_grad_s)
            return np.concatenate([gl, gs])

    elif objective == 'cvar':
        # mean - lam * tail_loss_alpha; tail_loss = -mean_of_bottom_alpha
        def neg_obj(z):
            pnl, mean = pnl_and_mean(z)
            tail = np.partition(pnl, k_tail)[:k_tail]    # k smallest values
            tail_loss = -tail.mean()
            return -(mean - lam * tail_loss)
        def grad(z):
            pnl, _ = pnl_and_mean(z)
            idx = np.argpartition(pnl, k_tail)[:k_tail]   # tail indices
            # d(tail_loss)/dq_l = -(1/k) sum_{i in tail} X_long[i]
            tl_grad_l = -X_long[idx].mean(axis=0)
            tl_grad_s = -X_short[idx].mean(axis=0)
            gl = -(mu_long  - lam * tl_grad_l)
            gs = -(mu_short - lam * tl_grad_s)
            return np.concatenate([gl, gs])

    elif objective == 'semivar':
        # mean - 0.5*lam*semi_var. semi_var penalizes only deviations below mean.
        def neg_obj(z):
            pnl, mean = pnl_and_mean(z)
            dev = pnl - pnl.mean()
            semi = (dev**2 * (dev < 0)).sum() / (N - 1)
            return -(mean - 0.5 * lam * semi)
        def grad(z):
            pnl, _ = pnl_and_mean(z)
            mu = pnl.mean()
            dev = pnl - mu
            mask = (dev < 0).astype(float)
            # treat indicator as fixed: d(semi_var)/dq_l = 2/(N-1) * sum(mask*dev * (X_long - X_long.mean()))
            cw = mask * dev
            sv_grad_l = 2 * (X_long  - X_long.mean(0)) .T @ cw / (N - 1)
            sv_grad_s = 2 * (X_short - X_short.mean(0)).T @ cw / (N - 1)
            gl = -(mu_long  - 0.5 * lam * sv_grad_l)
            gs = -(mu_short - 0.5 * lam * sv_grad_s)
            return np.concatenate([gl, gs])

    else:
        raise ValueError(f"unknown objective: {objective!r}. "
                         "use 'var' | 'linear' | 'cvar' | 'semivar'.")

    z0 = np.array([b[0] for b in bounds])    # respects pinned legs
    res = minimize(neg_obj, z0, jac=grad, bounds=bounds, method='L-BFGS-B',
                   options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 5000})
    q = res.x[:n] - res.x[n:]
    pnl = X_long @ res.x[:n] + X_short @ res.x[n:]
    mean = mu_long @ res.x[:n] + mu_short @ res.x[n:]
    return q, pnl, mean


def solve_baskets(lam, baskets, names, caps,
                  X_long, X_short, mu_long, mu_short,
                  pin=None, override_caps=None, objective='var'):
    """
    Optimize over fixed-ratio baskets. Same objective dispatch as solve()
    ('var' | 'linear' | 'cvar' | 'semivar' — see module header).

      baskets: dict {basket_name: {leg_name: coeff}}
        Each basket is one signed scalar y_b. q_leg = sum_b coeff(b,leg) * y_b.
      pin: dict {basket_name: scalar_value} to force a basket.
      override_caps: dict {leg_name: cap} (per-leg, propagates to basket bounds).

    Returns (basket_q dict, leg_q dict, pnl, mean).
    """
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    pin = pin or {}
    override_caps = override_caps or {}
    eff_cap = caps.copy()
    for nm, c in override_caps.items():
        eff_cap[name_to_idx[nm]] = c

    B = len(baskets)
    bnames = list(baskets)

    # Build mapping matrices: M_pos[i,b] = c if c>0 else 0; M_neg[i,b] = -c if c<0 else 0.
    n = len(names)
    M_pos = np.zeros((n, B))
    M_neg = np.zeros((n, B))
    for b_idx, bn in enumerate(bnames):
        for leg, c in baskets[bn].items():
            if c == 0: continue
            i = name_to_idx[leg]
            if c > 0: M_pos[i, b_idx] = c
            else:     M_neg[i, b_idx] = -c

    # Per-basket per-direction matrices
    X_p = X_long  @ M_pos + X_short @ M_neg          # dir + (y_b+)
    X_n = X_short @ M_pos + X_long  @ M_neg          # dir - (y_b-)
    mu_p = mu_long  @ M_pos + mu_short @ M_neg
    mu_n = mu_short @ M_pos + mu_long  @ M_neg

    # Per-basket scalar cap = min over legs of cap_leg / |coeff|.
    abs_M = np.abs(M_pos + M_neg)                    # |coeff| per leg per basket
    with np.errstate(divide='ignore'):
        per_leg_caps = np.where(abs_M > 0, eff_cap[:, None] / np.where(abs_M>0, abs_M, 1), np.inf)
        per_leg_caps = np.where(abs_M > 0, per_leg_caps, np.inf)
    cap_b = per_leg_caps.min(axis=0)                 # (B,)

    bounds = [(0.0, c) for c in cap_b] + [(0.0, c) for c in cap_b]
    for bn, qty in pin.items():
        idx = bnames.index(bn)
        if qty >= 0:
            bounds[idx]     = (qty, qty)
            bounds[B + idx] = (0.0, 0.0)
        else:
            bounds[idx]     = (0.0, 0.0)
            bounds[B + idx] = (-qty, -qty)

    N = X_p.shape[0]
    k_tail = max(1, int(round(CVAR_ALPHA * N)))

    def pm(z):
        yp, yn = z[:B], z[B:]
        return X_p @ yp + X_n @ yn, mu_p @ yp + mu_n @ yn

    if objective == 'linear':
        _g = -np.concatenate([mu_p, mu_n])
        def neg_obj(z):
            _, mean = pm(z); return -mean
        grad = lambda _z: _g

    elif objective == 'var':
        def neg_obj(z):
            pnl, mean = pm(z)
            return -(mean - 0.5 * lam * pnl.var(ddof=1))
        def grad(z):
            pnl, _ = pm(z)
            c = pnl - pnl.mean()
            vp = 2 * (X_p - X_p.mean(0)).T @ c / (N - 1)
            vn = 2 * (X_n - X_n.mean(0)).T @ c / (N - 1)
            gp = -(mu_p - 0.5 * lam * vp)
            gn = -(mu_n - 0.5 * lam * vn)
            return np.concatenate([gp, gn])

    elif objective == 'cvar':
        def neg_obj(z):
            pnl, mean = pm(z)
            tail = np.partition(pnl, k_tail)[:k_tail]
            return -(mean - lam * (-tail.mean()))
        def grad(z):
            pnl, _ = pm(z)
            idx = np.argpartition(pnl, k_tail)[:k_tail]
            tl_p = -X_p[idx].mean(axis=0)
            tl_n = -X_n[idx].mean(axis=0)
            gp = -(mu_p - lam * tl_p)
            gn = -(mu_n - lam * tl_n)
            return np.concatenate([gp, gn])

    elif objective == 'semivar':
        def neg_obj(z):
            pnl, mean = pm(z)
            dev = pnl - pnl.mean()
            semi = (dev**2 * (dev < 0)).sum() / (N - 1)
            return -(mean - 0.5 * lam * semi)
        def grad(z):
            pnl, _ = pm(z)
            dev = pnl - pnl.mean()
            cw = dev * (dev < 0)
            vp = 2 * (X_p - X_p.mean(0)).T @ cw / (N - 1)
            vn = 2 * (X_n - X_n.mean(0)).T @ cw / (N - 1)
            gp = -(mu_p - 0.5 * lam * vp)
            gn = -(mu_n - 0.5 * lam * vn)
            return np.concatenate([gp, gn])
    else:
        raise ValueError(f"unknown objective: {objective!r}")

    z0 = np.array([b[0] for b in bounds])
    res = minimize(neg_obj, z0, jac=grad, bounds=bounds, method='L-BFGS-B',
                   options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 5000})
    yp, yn = res.x[:B], res.x[B:]
    y_signed = yp - yn
    pnl  = X_p @ yp + X_n @ yn
    mean = mu_p @ yp + mu_n @ yn

    basket_q = dict(zip(bnames, y_signed))
    leg_q = {nm: 0.0 for nm in names}
    for bn, coeffs in baskets.items():
        for leg, c in coeffs.items():
            leg_q[leg] += c * basket_q[bn]
    return basket_q, leg_q, pnl, mean


def report(q, pnl, names, mean=None):
    mu = pnl.mean() if mean is None else mean
    sd = pnl.std(ddof=1)
    sd_score = sd / np.sqrt(N_SCORE)        # std of mean of 100 paths
    print(f"  E[PnL]   = {mu:>12,.0f}")
    print(f"  std PnL  = {sd:>12,.0f}   (single path)")
    print(f"  std mean = {sd_score:>12,.0f}   (over {N_SCORE} scoring paths)")
    print(f"  Sharpe (score) = {mu/sd_score:.2f}")
    print(f"  P(score>0) over 100 trials = {norm.cdf(mu/sd_score):.4f}")
    print(f"  positions (rounded):")
    for name, qi in zip(names, q):
        if abs(qi) >= 0.5:
            print(f"    '{name:10s}': {int(round(qi)):+5d},")


# ----- frontier sweep ------------------------------------------------------
if __name__ == "__main__":
    # ===== per-run config: edit these to constrain the solver =====
    # tighter |q_i| <= cap. set to 0 to disable a leg entirely.
    # any leg not listed keeps its MARKET[...]['cap'] default.
    CAP_OVERRIDE = {
        # 'AC_45_KO': 100,        # uncomment to forbid KO trading
    }
    # exact qty: q_i = qty (overrides bounds, skips optimization for that leg).
    PIN = {
        # 'AC_50_P':   +9,
        # 'AC_50_C': +41,
        # 'AC_35_P': -46,
        # 'AC_45_P': +46,
        # 'AC_50_P_2': 41,
        # 'AC_50_C_2': +9,
        # 'AC_50_CO': -50,
        # 'AC_40_BP':-50,
    }
    # ----- basket mode: optimize over fixed-ratio scalars ------------
    # Each basket = one decision variable y_b. Per-leg q = sum_b coeff*y_b.
    # Cap per basket auto-derived: |y_b| <= min_leg cap_leg / |coeff_leg|.
    # Set USE_BASKETS = True to switch the sweep into basket mode.
    USE_BASKETS = False
    BASKETS = {}
    #     'y':  {'AC_50_P': 9,  'AC_50_C': 41,  'AC_50_P_2': 41, 'AC_50_C_2': 9, 'AC_50_CO': -50},
    #     'x':  {'AC_35_P': -46, 'AC_45_P': 46, 'AC_40_BP': -50},
    #     'k':  {'AC_60_C': 1},
    #     'w':  {'AC_40_P':  1},
    #     'c':  {'AC_45_KO': 1},
    #     'z':  {'AC':      1},
    # }
    BASKET_PIN = {}        # e.g. {'y': 1.0} to force the y-basket scalar

    # ----- objective: pick risk model ---------------------------------
    # 'var'     : E - 0.5*lam*Var(PnL)**2     (project's classic mean-variance)
    # 'linear'  : E only.    lam ignored.     Pure max-EV under caps.
    # 'cvar'    : E - lam * tail_loss_alpha   (penalize bottom CVAR_ALPHA % only)
    # 'semivar' : E - 0.5*lam*SemiVar(PnL)    (downside-only second moment)
    OBJECTIVE = 'var'
    # ==============================================================

    names, caps, X_long, X_short, mu_long, mu_short, fv = build(seed=51)

    print("=== per-leg fair value vs market (analytical, KO=MC) ===")
    print(f"{'leg':12s} {'FV':>10s} {'bid':>8s} {'ask':>8s} "
          f"{'edge_long':>10s} {'edge_short':>11s} {'best':>6s}")
    for i, n in enumerate(names):
        bid, ask = MARKET[n]['bid'], MARKET[n]['ask']
        e_long  = fv[i] - ask
        e_short = bid - fv[i]
        side = 'skip' if max(e_long, e_short) <= 0 else ('long' if e_long > e_short else 'short')
        print(f"{n:12s} {fv[i]:>10.4f} {bid:>8.3f} {ask:>8.3f} "
              f"{e_long:>+10.4f} {e_short:>+11.4f} {side:>6s}")

    if CAP_OVERRIDE or PIN:
        print("\n=== solver overrides ===")
        for nm, c in CAP_OVERRIDE.items():
            print(f"  cap[{nm}] = {c}")
        for nm, q_ in PIN.items():
            print(f"  pin[{nm}] = {q_:+d}")

    # Linear objective ignores lam, so only run once.
    LAMBDAS = ([0.0] if OBJECTIVE == 'linear'
            else [k * 10**p for p in range(-9, -7) for k in range(1, 10)])
    # LAMBDAS = ([0.0] if OBJECTIVE == 'linear'
    #         else [1e-9,1e-8,1e-7,1e-6,1e-5])

    print(f"\n=== lambda frontier (objective='{OBJECTIVE}'"
          f"{f', alpha={CVAR_ALPHA}' if OBJECTIVE == 'cvar' else ''}) ===")
    print(f"{'lambda':>10s}  {'E[PnL]':>12s}  {'std':>12s}  {'std/√100':>10s}  "
          f"{'Sharpe':>7s}  {'P(score>0)':>10s}  {'CVaR_5%':>12s}")
    cached = {}
    basket_solutions = {}    # only populated when USE_BASKETS
    for lam in LAMBDAS:
        if USE_BASKETS:
            basket_q, leg_q, pnl, mean = solve_baskets(
                lam, BASKETS, names, caps,
                X_long, X_short, mu_long, mu_short,
                pin=BASKET_PIN, override_caps=CAP_OVERRIDE,
                objective=OBJECTIVE,
            )
            q = np.array([leg_q[n] for n in names])
            basket_solutions[lam] = basket_q
        else:
            q, pnl, mean = solve(lam, names, caps, X_long, X_short, mu_long, mu_short,
                                 pin=PIN, override_caps=CAP_OVERRIDE,
                                 objective=OBJECTIVE)
        cached[lam] = (q, pnl, mean)
        sd = pnl.std(ddof=1)
        sd_s = sd / np.sqrt(N_SCORE)
        p_pos = norm.cdf(mean / sd_s)
        k_t = max(1, int(round(0.05 * len(pnl))))
        cvar_5 = np.partition(pnl, k_t)[:k_t].mean()    # E[PnL | bottom 5%]
        print(f"{lam:>10.1e}  {mean:>12,.0f}  {sd:>12,.0f}  {sd_s:>10,.0f}  "
              f"{mean/sd_s:>7.2f}  {p_pos:>10.4f}  {cvar_5:>12,.0f}")

    print("\n=== per-lambda detail ===")
    for lam in LAMBDAS:
        q, pnl, mean = cached[lam]
        print(f"\n--- lambda = {lam:.0e} ---")
        report(q, pnl, names, mean=mean)
        if USE_BASKETS:
            print("  basket scalars:")
            for bn, val in basket_solutions[lam].items():
                if abs(val) >= 1e-3:
                    print(f"    {bn:>4s} = {val:>+9.4f}")

    # --- PnL distribution grid, one subplot per lambda ---
    n_lam = len(LAMBDAS)
    ncols = 3
    nrows = int(np.ceil(n_lam / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, lam in enumerate(LAMBDAS):
        _, pnl, mu = cached[lam]
        sd = pnl.std(ddof=1)
        p05, p95 = np.percentile(pnl, [5, 95])
        x_lo, x_hi = np.percentile(pnl, [0.01, 99.99])
        sharpe_score = mu / (sd / np.sqrt(N_SCORE))
        ax = axes[i]
        ax.hist(pnl, bins=80, range=(x_lo, x_hi), alpha=0.85, color="C0", edgecolor="white")
        ax.set_xlim(x_lo, x_hi)
        ax.axvline(0, color="k", lw=1)
        ax.axvline(mu, color="C1", lw=2, label=f"mean = {mu:,.0f}")
        ax.axvline(p05, color="C3", ls="--", lw=1, label=f"5% = {p05:,.0f}")
        ax.axvline(p95, color="C2", ls="--", lw=1, label=f"95% = {p95:,.0f}")
        ax.set_title(f"lambda = {lam:.0e}   Sharpe(score) = {sharpe_score:.2f}")
        ax.set_xlabel("terminal PnL per path")
        ax.set_ylabel("paths")
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
        ax.legend(fontsize=8)
    for j in range(n_lam, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("PnL distribution per lambda (QP optimum)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "manual/pnl_per_lambda.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  saved {out}")

    # --- Bootstrap score distribution per lambda ---
    # Each draw = mean PnL over N_SCORE paths sampled with replacement.
    # Decouples #draws from #paths so tail probability is well-resolved.
    B = 100_000
    rng_bs = np.random.default_rng(124)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    print("\n=== bootstrap score (B = {:,} draws of {}) ===".format(B, N_SCORE))
    print(f"{'lambda':>10s}  {'mean':>12s}  {'std':>12s}  {'min':>12s}  "
          f"{'0.1%':>12s}  {'1%':>12s}  {'5%':>12s}  {'50%':>12s}  {'95%':>12s}  "
          f"{'max':>12s}  {'P>0':>7s}  {'Pgauss':>7s}  {'max_dd':>12s}  {'worst_path':>14s}")
    for i, lam in enumerate(LAMBDAS):
        _, pnl, mean = cached[lam]
        idx = rng_bs.integers(0, len(pnl), size=(B, N_SCORE))
        score = pnl[idx].mean(axis=1)
        s_mu  = score.mean()
        s_sd  = score.std(ddof=1)
        s_min = score.min()
        s_max = score.max()
        s_q01, s_q1, s_p05, s_med, s_p95 = np.percentile(score, [0.1, 1, 5, 50, 95])
        p_boot  = (score > 0).mean()
        p_gauss = norm.cdf(mean / (pnl.std(ddof=1) / np.sqrt(N_SCORE)))
        max_dd  = max(0.0, -s_min)
        worst_path = pnl.min()
        print(f"{lam:>10.1e}  {s_mu:>12,.0f}  {s_sd:>12,.0f}  {s_min:>12,.0f}  "
              f"{s_q01:>12,.0f}  {s_q1:>12,.0f}  {s_p05:>12,.0f}  {s_med:>12,.0f}  {s_p95:>12,.0f}  "
              f"{s_max:>12,.0f}  {p_boot:>7.4f}  {p_gauss:>7.4f}  {max_dd:>12,.0f}  {worst_path:>14,.0f}")

        ax = axes[i]
        ax.hist(score, bins=80, alpha=0.85, color="C4", edgecolor="white")
        ax.axvline(0, color="k", lw=1)
        ax.axvline(s_mu, color="C1", lw=2, label=f"mean = {s_mu:,.0f}")
        ax.axvline(s_p05, color="C3", ls="--", lw=1, label=f"5% = {s_p05:,.0f}")
        ax.axvline(s_p95, color="C2", ls="--", lw=1, label=f"95% = {s_p95:,.0f}")
        ax.set_title(f"lambda = {lam:.0e}   P>0 boot={p_boot:.4f}  gauss={p_gauss:.4f}")
        ax.set_xlabel(f"score = mean PnL over {N_SCORE} paths")
        ax.set_ylabel(f"draws (B={B:,})")
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
        ax.legend(fontsize=8)
    for j in range(n_lam, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("bootstrap score distribution per lambda", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "manual/score_per_lambda.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  saved {out}")

    # ---- log(lambda) vs Sharpe (and EV / P>0) -------------------------------
    lam_arr     = np.array([lam for lam in LAMBDAS if lam > 0])
    sharpe_arr  = np.array([cached[lam][2] / (cached[lam][1].std(ddof=1) / np.sqrt(N_SCORE))
                            for lam in lam_arr])
    ev_arr      = np.array([cached[lam][2] for lam in lam_arr])
    pgauss_arr  = norm.cdf(sharpe_arr)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(lam_arr, sharpe_arr, 'o-', color='C0', label='Sharpe (score)')
    ax1.set_xscale('log')
    ax1.set_xlabel("lambda (log scale)")
    ax1.set_ylabel("Sharpe (score)", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.grid(True, which='both', ls=':', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(lam_arr, ev_arr, 's-', color='C3', label='E[PnL]')
    ax2.set_ylabel("E[PnL]", color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    fig.suptitle(f"objective='{OBJECTIVE}'  —  Sharpe & EV vs lambda", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "manual/sharpe_vs_lambda.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved {out}")

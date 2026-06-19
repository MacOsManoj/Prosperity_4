"""
Monte Carlo engine for ROUND4 manual.
Underlying AETHER_CRYSTAL: GBM, sigma=251% annualized, r=0.
Time grid: 4 ticks per trading day, 252 trading days/year.
"""
import numpy as np
from scipy.stats import norm

# --- world constants from manual ---
S0 = 49.975
SIGMA = 2.51
R = 0.0
STEPS_PER_DAY = 4
DAYS_PER_YEAR = 252
DT = 1.0 / (DAYS_PER_YEAR * STEPS_PER_DAY)  # year fraction per tick

# Manual quotes 14 / 21 *Solvinarian* (calendar) days. A "week" = 5 trading days
# (mod confirmation), so 2/3 weeks = 10/15 trading days, NOT 14/21.
# DT is per-trading-tick, so steps must use trading-day count.
TRADING_DAYS_PER_WEEK = 5
T1_DAYS = 2 * TRADING_DAYS_PER_WEEK         # 10 trading days (2 weeks / chooser decision)
T2_DAYS = 3 * TRADING_DAYS_PER_WEEK         # 15 trading days (3 weeks / final expiry)
T1_STEPS = T1_DAYS * STEPS_PER_DAY          # 40 ticks
T2_STEPS = T2_DAYS * STEPS_PER_DAY          # 60 ticks


def simulate_paths(n_paths, n_steps=T2_STEPS, s0=S0, sigma=SIGMA, dt=DT, seed=None):
    """
    Risk-neutral GBM under r=0:
        d log S = -0.5 sigma^2 dt + sigma dW
    Each row is one path. Column 0 = S0; columns 1..n_steps = subsequent ticks,
    so shape is (n_paths, n_steps+1). Including S0 makes path-dependent
    statistics (min over path, value at any tick t) index by tick number directly.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    increments = (-0.5 * sigma**2 * dt) + sigma * np.sqrt(dt) * Z   # log-returns per tick
    log_paths = np.log(s0) + np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return np.exp(log_paths)


# --- vanilla terminal payoffs ---
def call_payoff(S_T, K): return np.maximum(S_T - K, 0.0)
def put_payoff(S_T, K):  return np.maximum(K - S_T, 0.0)


# --- exotic payoffs (all evaluated on a full simulated path) ---
def binary_put_payoff(S_T, K, amount=1.0):
    """Pays `amount` iff S_T < K (digital, no smoothing)."""
    return amount * (S_T < K).astype(float)


def knockout_put_payoff(paths, K, barrier):
    """
    Down-and-out put: vanilla put unless underlying *ever* touches the barrier
    over the life of the option. If barrier=K, payoff is identically 0 because
    finishing in the money (S_T<K) requires the path to have crossed barrier=K.
    """
    knocked = (paths < barrier).any(axis=1)             # path-dependent kill switch
    return np.where(knocked, 0.0, put_payoff(paths[:, -1], K))


def chooser_payoff(paths, K, t1_step=T1_STEPS, T_step=T2_STEPS, dt=DT):
    """
    At decision time t1, holder picks call or put based on which is worth more
    (BS value of remaining time to T). After t1, payoff is just that vanilla.
    Equivalent to max(C(S_t1, K, T-t1), P(S_t1, K, T-t1)) carried to expiry.
    Under r=0 this satisfies Rubinstein decomposition:
        Chooser = Call(K, T) + Put(K, t1)
    which is the replication target our advisor pointed at.
    """
    S_t1, S_T = paths[:, t1_step], paths[:, T_step]
    tau = (T_step - t1_step) * dt
    pick_call = bs_call(S_t1, K, tau) >= bs_put(S_t1, K, tau)
    return np.where(pick_call, call_payoff(S_T, K), put_payoff(S_T, K))


# --- Black-Scholes closed form (r defaults to 0 from world constants) ---
def bs_call(S, K, tau=T2_STEPS * DT, r=R, sigma=SIGMA):
    """Standard BS call. Assumes tau > 0; we never price at expiry."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_put(S, K, tau=T2_STEPS * DT, r=R, sigma=SIGMA):
    """Put-call parity: P = C - S + K e^{-r tau}."""
    return bs_call(S, K, tau, r, sigma) - S + K * np.exp(-r * tau)


def mc_price(payoff, r=R, tau=T2_STEPS * DT):
    """Discounted MC mean + standard error of the mean."""
    disc = np.exp(-r * tau)
    return disc * payoff.mean(), disc * payoff.std(ddof=1) / np.sqrt(len(payoff))


if __name__ == "__main__":
    # sanity: MC vanilla price should match BS within ~1 stderr
    paths = simulate_paths(1000000, T2_STEPS, seed=0)
    K, tau = 50, T2_STEPS * DT
    mc_c, se_c = mc_price(call_payoff(paths[:, -1], K), tau=tau)
    mc_p, se_p = mc_price(put_payoff(paths[:, -1], K), tau=tau)
    print(f"Call(50, 21d): MC={mc_c:.4f} +/- {se_c:.4f}, BS={bs_call(S0, K, tau):.4f}")
    print(f"Put (50, 21d): MC={mc_p:.4f} +/- {se_p:.4f}, BS={bs_put(S0, K, tau):.4f}")

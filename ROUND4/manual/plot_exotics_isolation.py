"""
Advisor step 1: look at exotic payoffs in isolation.
For each exotic, plot terminal payoff vs S_T, colored by the
path-dependent state that vanilla payoffs do not see.
"""
import numpy as np
import matplotlib.pyplot as plt
from ROUND4.manual.mc_engine import (
    simulate_paths, S0, SIGMA, DT, T1_STEPS, T2_STEPS,
    call_payoff, put_payoff,
    binary_put_payoff, knockout_put_payoff, chooser_payoff,
    bs_call, bs_put, mc_price,
)

N = 30_000
paths = simulate_paths(N, T2_STEPS, seed=7)
S_T = paths[:, -1]
S_min = paths.min(axis=1)
S_t1 = paths[:, T1_STEPS]

# strikes / barriers
K = 50
K_BIN = 45
BIN_AMOUNT = 1.0    # plot per-unit; scale for actual payout amount
K_KO = 45
BARRIER_KO = 45     # manual: knock out below 45 (assumed = strike)
TAU_T2 = T2_STEPS * DT
TAU_T1 = T1_STEPS * DT

# ---- payoffs ----
pay_call50 = call_payoff(S_T, 50)
pay_put50 = put_payoff(S_T, 50)
pay_chooser = chooser_payoff(paths, K=50, t1_step=T1_STEPS, T_step=T2_STEPS)
pay_binput = binary_put_payoff(S_T, K_BIN, BIN_AMOUNT)
pay_koput = knockout_put_payoff(paths, K=K_KO, barrier=BARRIER_KO)

# fair MC prices
prices = {
    "Call(50, 21d)":   mc_price(pay_call50,  tau=TAU_T2),
    "Put(50, 21d)":    mc_price(pay_put50,   tau=TAU_T2),
    "Chooser(50)":     mc_price(pay_chooser, tau=TAU_T2),
    "BinPut(45,A=1)":  mc_price(pay_binput,  tau=TAU_T2),
    "KOPut(45,B=45)":  mc_price(pay_koput,   tau=TAU_T2),
}
print("Fair MC prices (per unit):")
for name, (p, se) in prices.items():
    print(f"  {name:20s} = {p:8.4f} +/- {se:.4f}")

# knockout stats
knocked = (paths.min(axis=1) < BARRIER_KO)
print(f"\nKO put: knocked fraction = {knocked.mean():.3f}")
print(f"  ITM at expiry: {(S_T < K_KO).mean():.3f}")
print(f"  ITM AND survived: {((S_T < K_KO) & ~knocked).mean():.3f}")

# chooser branch stats
call_val_t1 = bs_call(S_t1, 50, (T2_STEPS - T1_STEPS) * DT)
put_val_t1 = bs_put(S_t1, 50, (T2_STEPS - T1_STEPS) * DT)
pick_call = call_val_t1 >= put_val_t1
print(f"\nChooser: pick call at t1 fraction = {pick_call.mean():.3f}")

# ===== plots =====
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. Chooser: payoff vs S_T, colored by branch picked at t1
ax = axes[0, 0]
ax.scatter(S_T[pick_call], pay_chooser[pick_call], s=2, alpha=0.4,
           label=f"picked CALL ({pick_call.mean():.0%})", color="C0")
ax.scatter(S_T[~pick_call], pay_chooser[~pick_call], s=2, alpha=0.4,
           label=f"picked PUT ({(~pick_call).mean():.0%})", color="C3")
xs = np.linspace(0.1, S_T.max(), 200)
ax.plot(xs, np.maximum(xs - 50, 0), "k--", lw=1, label="vanilla call(50)")
ax.plot(xs, np.maximum(50 - xs, 0), "k:",  lw=1, label="vanilla put(50)")
ax.set(xlabel="S_T", ylabel="payoff", title="Chooser(50): terminal payoff vs S_T")
ax.legend(fontsize=8); ax.set_xlim(0, 200)

# 2. Chooser payoff vs S_t1 (decision-time underlying)
ax = axes[0, 1]
ax.scatter(S_t1, pay_chooser, s=2, alpha=0.3, c=np.where(pick_call, "C0", "C3"))
ax.axvline(50, color="k", ls="--", lw=1, label="K=50")
ax.set(xlabel="S at t1 (decision time, day 14)", ylabel="terminal payoff",
       title="Chooser: payoff vs decision-time spot\n(blue=picked call, red=picked put)")
ax.legend(); ax.set_xlim(0, 200)

# 3. Binary put: step function
ax = axes[1, 0]
ax.scatter(S_T, pay_binput, s=2, alpha=0.3)
ax.axvline(K_BIN, color="r", ls="--", label=f"K={K_BIN}")
ax.set(xlabel="S_T", ylabel="payoff", xlim=(0, 100),
       title=f"Binary put(K={K_BIN}): all-or-nothing")
ax.legend()

# 4. KO put: payoff vs S_T, colored by knocked/survived
ax = axes[1, 1]
ax.scatter(S_T[knocked], pay_koput[knocked], s=2, alpha=0.3, color="C3",
           label=f"knocked ({knocked.mean():.0%}): payoff=0")
ax.scatter(S_T[~knocked], pay_koput[~knocked], s=2, alpha=0.5, color="C2",
           label=f"survived ({(~knocked).mean():.0%})")
xs2 = np.linspace(0.1, 100, 200)
ax.plot(xs2, np.maximum(K_KO - xs2, 0), "k--", lw=1, label=f"vanilla put({K_KO})")
ax.set(xlabel="S_T", ylabel="payoff", xlim=(0, 100),
       title=f"KO put(K={K_KO}, barrier={BARRIER_KO})")
ax.legend(fontsize=8)

plt.tight_layout()
out = "exotics_isolation.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved: {out}")

# ---- bonus: payoff vs (S_T, S_min) heatmap for KO put ----
# this shows path dependency that pure-S_T plots can't reveal
fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(S_T, S_min, c=pay_koput, s=3, alpha=0.6,
                cmap="viridis", vmin=0, vmax=pay_koput.max())
ax.axhline(BARRIER_KO, color="r", ls="--", label=f"barrier={BARRIER_KO}")
ax.set(xlabel="S_T", ylabel="min S over path",
       title="KO put payoff: depends on (S_T, min S)\nvanilla would only depend on S_T",
       xlim=(0, 100), ylim=(0, max(60, S_min.max())))
plt.colorbar(sc, label="payoff")
ax.legend()
plt.tight_layout()
plt.savefig("ko_path_dependency.png", dpi=130, bbox_inches="tight")
print("saved: ko_path_dependency.png")

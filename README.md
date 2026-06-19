🏝️ IMC Prosperity 4 — Trading Strategies
My algorithmic trading solutions for IMC Prosperity 4

📁 Repository Structure
├── ROUND1/          # Stationary & drifting products
├── ROUND2/          # Mean reversion & buy-and-hold
├── ROUND3/          # Options trading with volatility smile
├── ROUND4/          # Advanced MM with counterparty flow
├── ROUND5/          # Multi-product universe (50+ products)
├── datamodel.py     # IMC-provided data model (Order, TradingState, etc.)
├── backtest_viz.py  # Dash-based backtest visualization tool
├── backtest_viz.ipynb
└── insider_trader.ipynb
Each round folder contains:

round_X_final.py — The submitted strategy
v1.py, v2.py, … — Iterative strategy versions
*_analysis.ipynb — Data analysis & research notebooks
manual/ — Manual trading round analysis (R4, R5)
Historical data — prices_round_X_day_*.csv, trades_round_X_day_*.csv
🧠 Round-by-Round Summary
Round 1 — Stationary & Trending Products
Products: EMERALDS, ASH_COATED_OSMIUM, TOMATOES, INTARIAN_PEPPER_ROOT

Product	Strategy
EMERALDS	Three-phase MM (Take → Clear → Make) at fixed fair = 10,000
ASH_COATED_OSMIUM	Cycle-aware MM — multi-timescale EMAs detect ~91s/~333s/~500s FFT cycles, mean reversion on OU process (half-life ~3 ticks)
TOMATOES	Volume-filtered mid-price drifter with adaptive fair value
INTARIAN_PEPPER_ROOT	Trend-following buy & hold — deterministic +0.001/tick drift, aggressive accumulation to max long
Round 2 — Bayesian Mean Reversion & Momentum
Products: ASH_COATED_OSMIUM, INTARIAN_PEPPER_ROOT

Product	Strategy
ASH_COATED_OSMIUM	Bayesian mean reversion MM — prior μ=10,000 blended with rolling window mean, inventory-adjusted quoting
INTARIAN_PEPPER_ROOT	Momentum buy-and-hold — tracks 100-tick Δ(mid), dumps holdings on negative delta
Built an OOP trader framework (BaseTrader) with EWM helpers, market order execution, and persistent state management.

Round 3 — Options on Velvetfruit Extract
Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_4000 … VEV_6500 (10 call options)

Component	Strategy
HYDROGEL_PACK	Fixed fair-value (10,000) MM with inventory skew
VELVETFRUIT_EXTRACT	Random-walk MM with EMA fair value
Inner options (K ≤ 5300)	EMA-based fair value with dynamic edge taking
Outer options (K ≥ 5400)	Quadratic volatility smile fit → Black-Scholes fair value, z-score taking at ±3.5σ
Implemented from scratch: Black-Scholes pricing, implied vol solver (bisection), 3×3 linear system solver for smile regression, EMA-tracked bias/variance for adaptive thresholds.

Round 4 — Counterparty Flow & Avellaneda-Stoikov
Products: Same as R3 (HYDROGEL, VELVET, VEV options)

Counterparty intelligence — Identified "losing" traders (Mark 38, Mark 55, etc.) and traded against their flow with exponential decay tracking
Anchor-based mean reversion — Pre-computed anchors (HYDRO: 9990.95, VELVET: 5250.71) with rolling z-score signals
Options: Directional delta-scaled trades when underlying z-score > 1.1, Avellaneda-Stoikov MM otherwise
Tape imitation — Replicated profitable participant trades on VEV_4000 by improving the touch by 1 tick


Round 5 — Large Multi-Product Universe
Products: 50+ products across categories (Snackpacks, Panels, Pebbles, Microchips, Robots, Galaxy Sounds, Sleep Pods, Translators, UV Visors, Oxygen Shakes)

Strategy	Products	Logic
EMA crossover trend following	24 products	Dual EMA (fast/slow) with tuned enter/leave thresholds per product
Inventory-tilted quoting	Snackpacks, Panels, Pebbles, etc.	Mid-price + drift adjustment + inventory lean
Basket arbitrage	Pebbles (XS–XL), Foods	Sum-of-parts vs. anchor (50k / 19.88k), bias correction
Blacklisted	~25 products	Empirically unprofitable — no orders sent

Research Notebooks
insider_trader.ipynb — Counterparty trade flow analysis
ROUND3/options.ipynb — Implied volatility surface and smile analysis
ROUND4/analysis.ipynb — Comprehensive round 4 strategy analysis
ROUND5/alpha_price_research.ipynb — Price signal discovery across 50+ products
⚙️ Tech Stack
Language: Python 3
Visualization: Plotly, Dash, Matplotlib
Analysis: Pandas, NumPy
Pricing: Custom Black-Scholes, implied vol solver
🚀 Running
Strategies are self-contained Python files conforming to IMC's Trader class interface. Each round_X_final.py can be submitted directly to the Prosperity platform or run through a local backtester.

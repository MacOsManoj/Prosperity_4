import json
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
from pathlib import Path
from dash import Dash, dcc, html, Input, Output

RUST_RUNS_DIR = Path('prosperity_rust_backtester/runs')
OLD_LOG_FILES = sorted(glob.glob('imc-prosperity-4-backtester/backtests/*.log'))

THEMES = {
    'light': dict(bg='white', paper='white', grid='#e0e0e0', text='#333', mid='#333333', annotation='#333'),
    'dark':  dict(bg='#111111', paper='#111111', grid='#2a2a2a', text='#ccc', mid='#eeeeee', annotation='#ccc'),
}


def get_rust_runs():
    if not RUST_RUNS_DIR.exists():
        return []
    runs = sorted([
        d for d in RUST_RUNS_DIR.iterdir()
        if d.is_dir() and (d / 'activity.csv').exists()
    ], key=lambda d: d.name, reverse=True)
    return runs


def load_rust_run(run_dir):
    run_dir = Path(run_dir)
    activity = pd.read_csv(run_dir / 'activity.csv', sep=';')
    activity['timestamp'] = activity['timestamp'].astype(int)
    for col in ['bid_price_1', 'ask_price_1', 'mid_price']:
        activity[col] = activity[col].replace(0, float('nan'))

    trades = pd.read_csv(run_dir / 'trades.csv', sep=';')
    fills = []
    for _, r in trades.iterrows():
        if r['buyer'] == 'SUBMISSION':
            fills.append({'timestamp': r['timestamp'], 'symbol': r['symbol'], 'price': r['price'], 'qty': r['quantity']})
        elif r['seller'] == 'SUBMISSION':
            fills.append({'timestamp': r['timestamp'], 'symbol': r['symbol'], 'price': r['price'], 'qty': -r['quantity']})
    fills_df = pd.DataFrame(fills) if fills else pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'qty'])

    # Parse own orders from submission.log (same format as old backtester)
    own_orders = []
    log_path = run_dir / 'submission.log'
    if log_path.exists():
        with open(log_path) as f:
            data = json.load(f)
        for entry in data.get('logs', []):
            raw = entry.get('lambdaLog', '')
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
                ts = parsed[0][0]
                for symbol, price, qty in parsed[1]:
                    own_orders.append({'timestamp': ts, 'symbol': symbol, 'price': price, 'qty': qty})
            except Exception:
                pass
    orders_df = pd.DataFrame(own_orders) if own_orders else pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'qty'])

    pnl_df = pd.read_csv(run_dir / 'pnl_by_product.csv', sep=';')

    return activity, fills_df, pnl_df, orders_df


def load_old_log(path):
    with open(path) as f:
        data = json.load(f)

    activities = pd.read_csv(StringIO(data['activitiesLog']), sep=';')
    activities['timestamp'] = activities['timestamp'].astype(int)
    for col in ['bid_price_1', 'ask_price_1', 'mid_price']:
        activities[col] = activities[col].replace(0, float('nan'))

    own_orders = []
    for entry in data['logs']:
        raw = entry['lambdaLog']
        if not raw:
            continue
        parsed = json.loads(raw)
        ts = parsed[0][0]
        for symbol, price, qty in parsed[1]:
            own_orders.append({'timestamp': ts, 'symbol': symbol, 'price': price, 'qty': qty})
    orders_df = pd.DataFrame(own_orders) if own_orders else pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'qty'])

    fills = []
    for trade in data.get('tradeHistory', []):
        if trade['buyer'] == 'SUBMISSION':
            fills.append({'timestamp': trade['timestamp'], 'symbol': trade['symbol'], 'price': trade['price'], 'qty': trade['quantity']})
        elif trade['seller'] == 'SUBMISSION':
            fills.append({'timestamp': trade['timestamp'], 'symbol': trade['symbol'], 'price': trade['price'], 'qty': -trade['quantity']})
    fills_df = pd.DataFrame(fills) if fills else pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'qty'])

    return activities, orders_df, fills_df, None


def build_figure(activities, fills_df, pnl_df=None, orders_df=None, sample=1, theme='light'):
    t = THEMES[theme]
    products = list(activities['product'].unique())
    n_rows = len(products) + (1 if pnl_df is not None else 0)
    row_heights = [3] * len(products) + ([2] if pnl_df is not None else [])
    titles = products + (['PnL over time'] if pnl_df is not None else [])

    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles,
                        shared_xaxes=False, vertical_spacing=0.06,
                        row_heights=row_heights)

    for i, product in enumerate(products, start=1):
        mkt = activities[activities['product'] == product].copy()
        mkt_plot = mkt.iloc[::sample]

        def fmt(row):
            parts = [f"<b>tick</b>: {int(row['timestamp'])}"]
            if pd.notna(row['mid_price']): parts.append(f"<b>mid</b>: {row['mid_price']:.1f}")
            for lvl in [1, 2, 3]:
                bp, bv = row.get(f'bid_price_{lvl}'), row.get(f'bid_volume_{lvl}')
                if pd.notna(bp) and bp != 0:
                    parts.append(f"<b>bid{lvl}</b>: {bp:.0f} x{bv:.0f}")
            for lvl in [1, 2, 3]:
                ap, av = row.get(f'ask_price_{lvl}'), row.get(f'ask_volume_{lvl}')
                if pd.notna(ap) and ap != 0:
                    parts.append(f"<b>ask{lvl}</b>: {ap:.0f} x{av:.0f}")
            return "<br>".join(parts)

        hover = mkt_plot.apply(fmt, axis=1)
        show = (i == 1)

        fig.add_trace(go.Scatter(x=mkt_plot['timestamp'], y=mkt_plot['mid_price'],   mode='lines', name='mid',     line=dict(color=t['mid'],  width=1),   legendgroup='mid',     showlegend=show, hovertext=hover, hoverinfo='x+text'), row=i, col=1)
        fig.add_trace(go.Scatter(x=mkt_plot['timestamp'], y=mkt_plot['bid_price_1'], mode='lines', name='mkt bid', line=dict(color='#00cc44', width=1.5), legendgroup='mkt bid', showlegend=show, hoverinfo='skip'), row=i, col=1)
        fig.add_trace(go.Scatter(x=mkt_plot['timestamp'], y=mkt_plot['ask_price_1'], mode='lines', name='mkt ask', line=dict(color='#ff3333', width=1.5), legendgroup='mkt ask', showlegend=show, hoverinfo='skip'), row=i, col=1)

        # Own orders (old backtester only)
        if orders_df is not None and len(orders_df):
            prod_orders = orders_df[orders_df['symbol'] == product]
            buys  = prod_orders[prod_orders['qty'] > 0]
            sells = prod_orders[prod_orders['qty'] < 0]
            fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['price'], mode='markers', name='our bid', marker=dict(symbol='triangle-up', size=7, color='#4488ff', opacity=0.35), legendgroup='our bid', showlegend=show, hovertemplate='<b>our bid</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>', customdata=buys['qty']), row=i, col=1)
            fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['price'], mode='markers', name='our ask', marker=dict(symbol='triangle-down', size=7, color='#ffaa00', opacity=0.35), legendgroup='our ask', showlegend=show, hovertemplate='<b>our ask</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>', customdata=sells['qty'].abs()), row=i, col=1)

        # Fills
        prod_fills = fills_df[fills_df['symbol'] == product] if len(fills_df) else fills_df
        buy_fills  = prod_fills[prod_fills['qty'] > 0]
        sell_fills = prod_fills[prod_fills['qty'] < 0]
        fig.add_trace(go.Scatter(x=buy_fills['timestamp'], y=buy_fills['price'], mode='markers', name='fill buy', marker=dict(symbol='circle', size=9, color='#4488ff', line=dict(color='white', width=1.5)), legendgroup='fill buy', showlegend=show, hovertemplate='<b>fill buy</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>', customdata=buy_fills['qty']), row=i, col=1)
        fig.add_trace(go.Scatter(x=sell_fills['timestamp'], y=sell_fills['price'], mode='markers', name='fill sell', marker=dict(symbol='circle', size=9, color='#ffaa00', line=dict(color='white', width=1.5)), legendgroup='fill sell', showlegend=show, hovertemplate='<b>fill sell</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>', customdata=sell_fills['qty'].abs()), row=i, col=1)

        fig.update_yaxes(title_text='price', row=i, col=1, gridcolor=t['grid'], color=t['text'])
        fig.update_xaxes(title_text='timestamp', row=i, col=1, gridcolor=t['grid'], color=t['text'])

    # PnL panel
    if pnl_df is not None:
        pnl_row = len(products) + 1
        pnl_cols = [c for c in pnl_df.columns if c not in ('timestamp', 'total')]
        colors = ['#4488ff', '#ff9900', '#00cc88', '#ff4466', '#aa44ff']
        for j, col in enumerate(pnl_cols):
            fig.add_trace(go.Scatter(x=pnl_df['timestamp'], y=pnl_df[col], mode='lines', name=col, line=dict(color=colors[j % len(colors)], width=1.5), legendgroup=col, showlegend=True), row=pnl_row, col=1)
        fig.add_trace(go.Scatter(x=pnl_df['timestamp'], y=pnl_df['total'], mode='lines', name='TOTAL', line=dict(color=t['mid'], width=2, dash='dot'), legendgroup='TOTAL', showlegend=True), row=pnl_row, col=1)
        fig.update_yaxes(title_text='PnL', row=pnl_row, col=1, gridcolor=t['grid'], color=t['text'])
        fig.update_xaxes(title_text='timestamp', row=pnl_row, col=1, gridcolor=t['grid'], color=t['text'])

    for ann in fig.layout.annotations:
        ann.font.color = t['annotation']

    fig.update_layout(
        height=500 * len(products) + (350 if pnl_df is not None else 0),
        hovermode='x',
        margin=dict(t=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                    font=dict(color=t['text']), bgcolor='rgba(0,0,0,0)'),
        plot_bgcolor=t['bg'],
        paper_bgcolor=t['paper'],
        font=dict(color=t['text']),
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

rust_runs = get_rust_runs()
rust_options = [{'label': r.name, 'value': str(r)} for r in rust_runs]
old_options  = [{'label': Path(f).name, 'value': f} for f in OLD_LOG_FILES]

app = Dash(__name__)
app.layout = html.Div(id='page', children=[
    html.H3('Backtest Visualizer', style={'fontFamily': 'monospace', 'margin': '16px'}),
    html.Div([
        html.Label('Backend:', style={'fontFamily': 'monospace', 'marginRight': '8px'}),
        dcc.RadioItems(
            id='backend',
            options=[{'label': 'Rust', 'value': 'rust'}, {'label': 'Old', 'value': 'old'}],
            value='rust',
            inline=True,
            style={'fontFamily': 'monospace'},
            labelStyle={'marginRight': '12px'},
        ),
        html.Label('Run:', style={'fontFamily': 'monospace', 'marginLeft': '24px', 'marginRight': '8px'}),
        dcc.Dropdown(
            id='run-picker',
            options=rust_options,
            value=str(rust_runs[0]) if rust_runs else None,
            clearable=False,
            style={'width': '520px', 'fontFamily': 'monospace'},
        ),
        html.Label('Resolution:', style={'fontFamily': 'monospace', 'marginLeft': '24px', 'marginRight': '8px'}),
        dcc.RadioItems(
            id='sample-rate',
            options=[{'label': 'Full', 'value': '1'}, {'label': '1/5', 'value': '5'}, {'label': '1/10', 'value': '10'}, {'label': '1/25', 'value': '25'}],
            value='1',
            inline=True,
            style={'fontFamily': 'monospace'},
            labelStyle={'marginRight': '12px'},
        ),
        html.Label('Theme:', style={'fontFamily': 'monospace', 'marginLeft': '24px', 'marginRight': '8px'}),
        dcc.RadioItems(
            id='theme',
            options=[{'label': 'Light', 'value': 'light'}, {'label': 'Dark', 'value': 'dark'}],
            value='light',
            inline=True,
            style={'fontFamily': 'monospace'},
            labelStyle={'marginRight': '12px'},
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '4px', 'margin': '0 16px 16px'}),
    dcc.Graph(id='chart', config={'scrollZoom': True}),
])


@app.callback(
    Output('run-picker', 'options'),
    Output('run-picker', 'value'),
    Input('backend', 'value'),
)
def update_picker_options(backend):
    if backend == 'rust':
        runs = get_rust_runs()
        opts = [{'label': r.name, 'value': str(r)} for r in runs]
        val = str(runs[0]) if runs else None
    else:
        opts = old_options
        val = OLD_LOG_FILES[-1] if OLD_LOG_FILES else None
    return opts, val


@app.callback(
    Output('chart', 'figure'),
    Output('page', 'style'),
    Input('backend', 'value'),
    Input('run-picker', 'value'),
    Input('sample-rate', 'value'),
    Input('theme', 'value'),
)
def update(backend, run_path, sample_rate, theme):
    if not run_path:
        return go.Figure(), {}
    if backend == 'rust':
        activities, fills_df, pnl_df, orders_df = load_rust_run(run_path)
        fig = build_figure(activities, fills_df, pnl_df=pnl_df, orders_df=orders_df, sample=int(sample_rate), theme=theme)
    else:
        activities, orders_df, fills_df, _ = load_old_log(run_path)
        fig = build_figure(activities, fills_df, pnl_df=None, orders_df=orders_df, sample=int(sample_rate), theme=theme)
    bg = '#111111' if theme == 'dark' else 'white'
    color = '#cccccc' if theme == 'dark' else '#333333'
    return fig, {'backgroundColor': bg, 'color': color, 'minHeight': '100vh'}


if __name__ == '__main__':
    app.run(debug=False, port=8050)
    print('Open http://localhost:8050')

"""
FNCL 3 vs 6 spread / divergence (two-panel chart; Plotly if available else matplotlib).
Run from repo root: python scripts/01b_spread_divergence.py [--plotly|-p|--matplotlib|-m]
"""
import sys
import pandas as pd
import numpy as np

BACKEND = "plotly"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from _paths import DATA_DIR, IMG_DIR


def _normalize_price_col(df: pd.DataFrame):
    for col in ["PX_LAST", "PX_BID", "mid", "bid"]:
        if col in df.columns:
            return col
    return None


def _get_spread_col(df: pd.DataFrame):
    if "spread" in df.columns:
        return "spread"
    if "bid" in df.columns and "ask" in df.columns:
        df["spread"] = (df["ask"] - df["bid"]) * 32
        return "spread"
    return None


def load_fncl_historical_prices():
    path = DATA_DIR / "fncl_historical_prices.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    f3 = df[["date", "bid_3", "ask_3"]].rename(columns={"bid_3": "bid", "ask_3": "ask"}).copy()
    f6 = df[["date", "bid_6", "ask_6"]].rename(columns={"bid_6": "bid", "ask_6": "ask"}).copy()
    for d in (f3, f6):
        d["mid"] = (d["bid"] + d["ask"]) / 2
        _get_spread_col(d)
        d.set_index("date", inplace=True)
    return f3, f6


def load_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
    base_3 = 98.5
    f3 = pd.DataFrame({
        "date": dates,
        "bid": base_3 + np.cumsum(np.random.randn(150) * 0.015),
        "ask": base_3 + np.cumsum(np.random.randn(150) * 0.015) + 0.02,
    })
    f3["mid"] = (f3["bid"] + f3["ask"]) / 2
    f3["spread"] = (f3["ask"] - f3["bid"]) * 32
    f3["CPR"] = 4 + np.abs(np.random.randn(150) * 0.5)
    f3.set_index("date", inplace=True)
    base_6 = 95.2
    f6 = pd.DataFrame({
        "date": dates,
        "bid": base_6 + np.cumsum(np.random.randn(150) * 0.05),
        "ask": base_6 + np.cumsum(np.random.randn(150) * 0.05) + 0.05,
    })
    f6["mid"] = (f6["bid"] + f6["ask"]) / 2
    f6["spread"] = (f6["ask"] - f6["bid"]) * 32
    f6["CPR"] = 12 + np.abs(np.random.randn(150) * 4)
    f6.set_index("date", inplace=True)
    return f3, f6


def _align_data(f3, f6):
    common = f3.index.intersection(f6.index)
    if len(common) == 0:
        common = f3.index
        f6 = f6.reindex(common).ffill().bfill()
    return f3.loc[common], f6.loc[common]


def create_plotly_figure(f3, f6):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    f3, f6 = _align_data(f3, f6)
    price_col_3 = _normalize_price_col(f3)
    price_col_6 = _normalize_price_col(f6)
    spread_col = _get_spread_col(f3) or "spread"
    use_cpr = "CPR" in f3.columns and "CPR" in f6.columns

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Price & Prepayment Divergence (Sticky 3s vs Volatile 6s)",
            "Liquidity / Bid-Ask Spread",
        ),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )
    c3, c6 = "#2563EB", "#EA580C"

    if use_cpr:
        fig.add_trace(go.Scatter(x=f3.index, y=f3["CPR"], name="FNCL 3s CPR", line=dict(color=c3, width=2.5, shape="spline")), row=1, col=1)
        fig.add_trace(go.Scatter(x=f6.index, y=f6["CPR"], name="FNCL 6s CPR", line=dict(color=c6, width=2.5, shape="spline")), row=1, col=1)
        yaxis_title = "CPR Speed"
    elif price_col_3 and price_col_6:
        fig.add_trace(go.Scatter(x=f3.index, y=f3[price_col_3], name="FNCL 3s Price", line=dict(color=c3, width=2.5, shape="spline")), row=1, col=1)
        fig.add_trace(go.Scatter(x=f6.index, y=f6[price_col_6], name="FNCL 6s Price", line=dict(color=c6, width=2.5, shape="spline")), row=1, col=1)
        yaxis_title = "Price"
    else:
        raise ValueError("No price or CPR column found")

    fig.add_trace(go.Scatter(x=f3.index, y=f3[spread_col], name="FNCL 3s Spread", line=dict(color=c3, width=2, shape="spline")), row=2, col=1)
    fig.add_trace(go.Scatter(x=f6.index, y=f6[spread_col], name="FNCL 6s Spread", line=dict(color=c6, width=2, shape="spline")), row=2, col=1)

    fig.update_layout(
        title=dict(text="Fannie 3.0s vs Fannie 6.0s Divergence", font=dict(size=20, family="Georgia"), x=0.5, xanchor="center"),
        template="plotly_white",
        paper_bgcolor="rgba(250,250,252,1)",
        plot_bgcolor="rgba(248,249,251,1)",
        font=dict(family="Lato, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=100, b=60),
        height=700,
    )
    fig.update_xaxes(gridcolor="rgba(200,200,210,0.5)", zeroline=False, tickformat="%Y-%m-%d")
    fig.update_yaxes(gridcolor="rgba(200,200,210,0.5)", zeroline=False)
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
    fig.update_yaxes(title_text="Bid-Ask Spread", row=2, col=1)
    return fig


def create_matplotlib_figure(f3, f6):
    f3, f6 = _align_data(f3, f6)
    price_col_3 = _normalize_price_col(f3)
    price_col_6 = _normalize_price_col(f6)
    spread_col = _get_spread_col(f3) or "spread"
    use_cpr = "CPR" in f3.columns and "CPR" in f6.columns

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Fannie 3.0s vs Fannie 6.0s Divergence\nSticky 3s vs Volatile 6s", fontsize=16, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("#FAFAFC")
    c3, c6 = "#2563EB", "#EA580C"

    ax1.set_facecolor("#F8F9FB")
    if use_cpr:
        ax1.plot(f3.index, f3["CPR"], label="FNCL 3s CPR", color=c3, linewidth=2.5)
        ax1.plot(f6.index, f6["CPR"], label="FNCL 6s CPR", color=c6, linewidth=2.5)
        ax1.set_ylabel("CPR Speed", fontsize=11)
    elif price_col_3 and price_col_6:
        ax1.plot(f3.index, f3[price_col_3], label="FNCL 3s Price", color=c3, linewidth=2.5)
        ax1.plot(f6.index, f6[price_col_6], label="FNCL 6s Price", color=c6, linewidth=2.5)
        ax1.set_ylabel("Price", fontsize=11)
    ax1.set_title("Price & Prepayment Divergence", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", framealpha=0.95)
    ax1.grid(True, alpha=0.4, linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    ax2.set_facecolor("#F8F9FB")
    ax2.plot(f3.index, f3[spread_col], label="FNCL 3s Spread", color=c3, linewidth=2)
    ax2.plot(f6.index, f6[spread_col], label="FNCL 6s Spread", color=c6, linewidth=2)
    ax2.set_title("Liquidity / Bid-Ask Spread", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Bid-Ask Spread", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.grid(True, alpha=0.4, linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def main():
    global BACKEND
    if len(sys.argv) > 1 and sys.argv[1] in ("--matplotlib", "-m"):
        BACKEND = "matplotlib"
    elif len(sys.argv) > 1 and sys.argv[1] in ("--plotly", "-p"):
        BACKEND = "plotly"

    f3, f6 = load_fncl_historical_prices()
    if f3 is None or f6 is None:
        print("No data/fncl_historical_prices.csv. Using synthetic sample data.")
        f3, f6 = load_sample_data()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = IMG_DIR / "3s_vs_6s_divergence.png"

    if BACKEND == "plotly":
        try:
            fig = create_plotly_figure(f3, f6)
            out_html = IMG_DIR / "3s_vs_6s_divergence.html"
            fig.write_html(str(out_html))
            print(f"Interactive chart saved: {out_html}")
        except ImportError:
            print("Plotly not installed. Using matplotlib.")
            fig = create_matplotlib_figure(f3, f6)
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            print(f"Static chart saved: {out_png}")
    else:
        fig = create_matplotlib_figure(f3, f6)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Static chart saved: {out_png}")


if __name__ == "__main__":
    main()

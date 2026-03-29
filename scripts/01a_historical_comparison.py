"""
FNCL 3% vs 6% historical comparison (four-panel matplotlib chart).
Run from repo root: python scripts/01a_historical_comparison.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from _paths import DATA_DIR, IMG_DIR


def load_fncl_historical_prices():
    """Load data/fncl_historical_prices.csv -> two DataFrames (bid, ask, mid, spread)."""
    path = DATA_DIR / "fncl_historical_prices.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    f3 = df[["date", "bid_3", "ask_3"]].rename(columns={"bid_3": "bid", "ask_3": "ask"}).copy()
    f6 = df[["date", "bid_6", "ask_6"]].rename(columns={"bid_6": "bid", "ask_6": "ask"}).copy()
    for d in (f3, f6):
        d["spread"] = (d["ask"] - d["bid"]) * 32
        d["mid"] = (d["bid"] + d["ask"]) / 2
        d.set_index("date", inplace=True)
    return f3, f6


def load_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D")
    base_price_3 = 98.5
    fannie_3 = pd.DataFrame({
        "date": dates,
        "bid": base_price_3 + np.cumsum(np.random.randn(120) * 0.02),
        "ask": base_price_3 + np.cumsum(np.random.randn(120) * 0.02) + 0.02,
        "spread": np.abs(np.random.randn(120) * 2 + 2),
    })
    fannie_3["mid"] = (fannie_3["bid"] + fannie_3["ask"]) / 2
    fannie_3.set_index("date", inplace=True)
    base_price_6 = 95.2
    fannie_6 = pd.DataFrame({
        "date": dates,
        "bid": base_price_6 + np.cumsum(np.random.randn(120) * 0.03),
        "ask": base_price_6 + np.cumsum(np.random.randn(120) * 0.03) + 0.04,
        "spread": np.abs(np.random.randn(120) * 3 + 4),
    })
    fannie_6["mid"] = (fannie_6["bid"] + fannie_6["ask"]) / 2
    fannie_6.set_index("date", inplace=True)
    return fannie_3, fannie_6


def create_comparison_charts(fannie_3: pd.DataFrame, fannie_6: pd.DataFrame):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("FNCL 3% vs FNCL 6% Mortgage Bond Comparison", fontsize=14, fontweight="bold")

    common_idx = fannie_3.index.intersection(fannie_6.index)
    if len(common_idx) == 0:
        common_idx = fannie_3.index
        f6_aligned = fannie_6.reindex(common_idx).ffill().bfill()
        f3_aligned = fannie_3
    else:
        f3_aligned = fannie_3.loc[common_idx]
        f6_aligned = fannie_6.loc[common_idx]

    ax1 = fig.add_subplot(2, 2, 1)
    if "mid" in f3_aligned.columns:
        ax1.plot(f3_aligned.index, f3_aligned["mid"], label="FNCL 3% (3% Mortgages)", color="#1f77b4", linewidth=2)
    if "mid" in f6_aligned.columns:
        ax1.plot(f6_aligned.index, f6_aligned["mid"], label="FNCL 6% (6% Mortgages)", color="#ff7f0e", linewidth=2)
    ax1.set_title("Mid Price Over Time")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax2 = fig.add_subplot(2, 2, 2)
    if "spread" in f3_aligned.columns:
        ax2.plot(f3_aligned.index, f3_aligned["spread"], label="FNCL 3% Spread", color="#1f77b4", alpha=0.8)
    if "spread" in f6_aligned.columns:
        ax2.plot(f6_aligned.index, f6_aligned["spread"], label="FNCL 6% Spread", color="#ff7f0e", alpha=0.8)
    ax2.set_title("Bid-Ask Spread Comparison")
    ax2.set_ylabel("Spread")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax3 = fig.add_subplot(2, 2, 3)
    if "spread" in f3_aligned.columns and "spread" in f6_aligned.columns:
        spread_diff = f6_aligned["spread"] - f3_aligned["spread"]
        ax3.fill_between(spread_diff.index, 0, spread_diff, alpha=0.5, color="#2ca02c")
        ax3.axhline(y=0, color="black", linewidth=0.8)
        ax3.set_title("Spread Difference (FNCL 6 - FNCL 3)")
        ax3.set_ylabel("Spread Diff")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax4 = fig.add_subplot(2, 2, 4)
    stats = []
    labels = []
    if "spread" in f3_aligned.columns:
        stats.append(f3_aligned["spread"].mean())
        labels.append("FNCL 3\nAvg Spread")
    if "spread" in f6_aligned.columns:
        stats.append(f6_aligned["spread"].mean())
        labels.append("FNCL 6\nAvg Spread")
    if "mid" in f3_aligned.columns:
        stats.append(f3_aligned["mid"].mean())
        labels.append("FNCL 3\nAvg Mid")
    if "mid" in f6_aligned.columns:
        stats.append(f6_aligned["mid"].mean())
        labels.append("FNCL 6\nAvg Mid")
    colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"][: len(stats)]
    bars = ax4.bar(labels, stats, color=colors, alpha=0.8)
    ax4.set_title("Summary Statistics")
    ax4.set_ylabel("Value")
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def main():
    fannie_3, fannie_6 = load_fncl_historical_prices()
    if fannie_3 is None or fannie_6 is None:
        print("No data/fncl_historical_prices.csv. Using synthetic sample data.")
        fannie_3, fannie_6 = load_sample_data()

    print("\n--- FNCL 3% Summary ---")
    if "spread" in fannie_3.columns:
        print(f"  Avg Bid-Ask Spread: {fannie_3['spread'].mean():.2f}")
    if "mid" in fannie_3.columns:
        print(f"  Avg Mid Price: {fannie_3['mid'].mean():.2f}")
    print("\n--- FNCL 6% Summary ---")
    if "spread" in fannie_6.columns:
        print(f"  Avg Bid-Ask Spread: {fannie_6['spread'].mean():.2f}")
    if "mid" in fannie_6.columns:
        print(f"  Avg Mid Price: {fannie_6['mid'].mean():.2f}")

    fig = create_comparison_charts(fannie_3, fannie_6)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "historical_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved: {out}")


if __name__ == "__main__":
    main()

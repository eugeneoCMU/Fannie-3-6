"""
Fannie 3 vs Fannie 6 Comparison Tool
Parses and compares FNCL 3% mortgage bonds vs FNCL 6% mortgages using matplotlib.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving without display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def load_sample_data():
    """Generate sample data for demonstration when no CSV files exist."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D")

    # Fannie 3 - typically lower yield, tighter spread (3% mortgages)
    base_price_3 = 98.5
    fannie_3 = pd.DataFrame({
        "date": dates,
        "bid": base_price_3 + np.cumsum(np.random.randn(120) * 0.02),
        "ask": base_price_3 + np.cumsum(np.random.randn(120) * 0.02) + 0.02,
        "spread": np.abs(np.random.randn(120) * 2 + 2),  # ticks/bps
    })
    fannie_3["mid"] = (fannie_3["bid"] + fannie_3["ask"]) / 2
    fannie_3.set_index("date", inplace=True)

    # Fannie 6 - typically higher yield, different spread dynamics (6% mortgages)
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


def load_csv_data(fncl3_path: str = None, fncl6_path: str = None):
    """
    Load Fannie 3 and Fannie 6 data from CSV files.
    Expected columns: date, bid, ask (and optionally spread, mid).
    If spread/mid are missing, they will be computed.
    """
    base = Path(__file__).parent
    fncl3_path = fncl3_path or base / "fncl3_data.csv"
    fncl6_path = fncl6_path or base / "fncl6_data.csv"

    fannie_3, fannie_6 = None, None

    if fncl3_path.exists():
        df3 = pd.read_csv(fncl3_path)
        df3["date"] = pd.to_datetime(df3["date"])
        if "spread" not in df3.columns and "bid" in df3.columns and "ask" in df3.columns:
            df3["spread"] = (df3["ask"] - df3["bid"]) * 32  # convert to 32nds
        if "mid" not in df3.columns and "bid" in df3.columns and "ask" in df3.columns:
            df3["mid"] = (df3["bid"] + df3["ask"]) / 2
        df3.set_index("date", inplace=True)
        fannie_3 = df3

    if fncl6_path.exists():
        df6 = pd.read_csv(fncl6_path)
        df6["date"] = pd.to_datetime(df6["date"])
        if "spread" not in df6.columns and "bid" in df6.columns and "ask" in df6.columns:
            df6["spread"] = (df6["ask"] - df6["bid"]) * 32
        if "mid" not in df6.columns and "bid" in df6.columns and "ask" in df6.columns:
            df6["mid"] = (df6["bid"] + df6["ask"]) / 2
        df6.set_index("date", inplace=True)
        fannie_6 = df6

    return fannie_3, fannie_6


def create_comparison_charts(fannie_3: pd.DataFrame, fannie_6: pd.DataFrame):
    """Create matplotlib figures comparing Fannie 3 vs Fannie 6."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("FNCL 3% vs FNCL 6% Mortgage Bond Comparison", fontsize=14, fontweight="bold")

    # Align indices for comparison (use common dates)
    common_idx = fannie_3.index.intersection(fannie_6.index)
    if len(common_idx) == 0:
        common_idx = fannie_3.index
        f6_aligned = fannie_6.reindex(common_idx).ffill().bfill()
        f3_aligned = fannie_3
    else:
        f3_aligned = fannie_3.loc[common_idx]
        f6_aligned = fannie_6.loc[common_idx]

    # 1. Mid price comparison (overlay)
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

    # 2. Bid-Ask spread comparison
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

    # 3. Spread difference (6 minus 3)
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

    # 4. Summary statistics bar chart
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
    base = Path(__file__).parent

    # Try to load from CSV first
    fannie_3, fannie_6 = load_csv_data()

    if fannie_3 is None or fannie_6 is None:
        print("No fncl3_data.csv or fncl6_data.csv found. Using sample data.")
        print("To use your own data, create CSV files with columns: date, bid, ask")
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
    output_path = base / "fannie_3_vs_6_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")

    # Uncomment below to display interactively (requires display):
    # matplotlib.use("TkAgg")
    # plt.show()


if __name__ == "__main__":
    main()

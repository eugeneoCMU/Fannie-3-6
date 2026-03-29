"""
MBS Price-Yield Forecast: FNMA 3.0% MBS
Integrates Modified Duration curve to build Price-Yield curve, then maps
Fed Funds implied rate changes to forecast bond prices at FOMC dates.

Math: P(y) = P_0 * exp(-∫ D_mod dy), with dy in decimals (bps/10000)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_duration_curve(base: Path):
    """
    Load Modified Duration vs Rate Shock (bps).
    Expected CSV columns: rate_shock_bps, modified_duration
    """
    path = base / "duration_curve.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Accept various column names
        rate_col = next((c for c in df.columns if "rate" in c.lower() or "shock" in c.lower()), df.columns[0])
        dur_col = next((c for c in df.columns if "duration" in c.lower() or "mod" in c.lower()), df.columns[1])
        return df[rate_col].values.astype(float), df[dur_col].values.astype(float)
    return None, None


def load_fed_funds(base: Path) -> pd.DataFrame:
    """
    Load Fed Funds Futures / FOMC implied rate changes.
    Expected CSV columns: date, rate_change_bps (incremental) or cumulative_rate_bps
    """
    path = base / "fed_funds_futures.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        # If incremental changes, compute cumulative
        if "rate_change_bps" in df.columns and "cumulative_rate_bps" not in df.columns:
            df["cumulative_rate_bps"] = df["rate_change_bps"].cumsum()
        elif "cumulative_rate_bps" not in df.columns and "implied_rate_bps" in df.columns:
            # Assume implied_rate is vs current; treat as cumulative change
            df["cumulative_rate_bps"] = df["implied_rate_bps"]
        return df
    return None


def load_sample_data():
    """Generate sample duration S-curve and Fed Funds data."""
    # FNMA 3.0% MBS: typical modified duration S-curve
    # At negative shocks: prepays accelerate -> duration shortens
    # At zero: peak duration
    # At positive shocks: extension -> duration lengthens then shortens
    rate_shock_bps = np.array([-300, -200, -100, -50, 0, 50, 100, 150, 200, 250, 300])
    # S-curve shape
    mod_duration = np.array([
        4.2, 5.8, 7.2, 7.8, 8.1, 8.0, 7.5, 6.8, 5.9, 5.0, 4.2
    ])

    # Fed Funds: FOMC meetings with implied cumulative rate cuts (bps)
    fed_data = pd.DataFrame({
        "date": pd.to_datetime([
            "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
            "2025-09-17", "2025-11-05", "2025-12-17",
            "2026-01-28", "2026-03-18", "2026-05-06",
        ]),
        "rate_change_bps": [-25, -25, -25, 0, -25, -25, 0, -25, -25, -25],
    })
    fed_data["cumulative_rate_bps"] = fed_data["rate_change_bps"].cumsum()

    return rate_shock_bps, mod_duration, fed_data


def build_price_yield_curve(
    rate_shock_bps: np.ndarray,
    mod_duration: np.ndarray,
    p0: float = 98.5,
    n_points: int = 601,
) -> tuple[np.ndarray, np.ndarray, interp1d]:
    """
    Build Price-Yield curve from duration via integration.
    P(y) = P_0 * exp(-∫ D_mod dy), dy in decimals (bps/10000).
    """
    # 1. Cubic interpolation of Modified Duration vs rate shock
    dur_interp = interp1d(
        rate_shock_bps, mod_duration,
        kind="cubic",
        bounds_error=False,
        fill_value=(mod_duration[0], mod_duration[-1]),
    )

    # 2. Fine grid for integration (-300 to +300 bps)
    y_bps = np.linspace(rate_shock_bps.min(), rate_shock_bps.max(), n_points)
    y_decimal = y_bps / 10000  # rate shock in decimal (bps/10000)

    # 3. Evaluate duration on fine grid
    D_mod = dur_interp(y_bps)

    # 4. Cumulative trapezoid: ∫ D_mod dy from y_0 to y
    integral = cumulative_trapezoid(D_mod, y_decimal, initial=0)

    # 5. P(y) = P_0 * exp(-∫ D_mod dy)
    # Reference: price at shock=0 should be P_0
    idx_0 = np.argmin(np.abs(y_bps))
    integral -= integral[idx_0]  # center so P(y_0)=P_0
    price = p0 * np.exp(-integral)

    # 6. Interpolation function for price
    price_interp = interp1d(
        y_bps, price,
        kind="cubic",
        bounds_error=False,
        fill_value=(price[0], price[-1]),
    )

    return y_bps, price, price_interp


def project_prices(
    fed_df: pd.DataFrame,
    price_fn: interp1d,
    base_rate_bps: float = 0,
) -> pd.DataFrame:
    """
    Map cumulative Fed rate changes onto Price-Yield curve.
    cumulative_rate_bps = total bps change from today; maps to rate shock.
    """
    df = fed_df.copy()
    df["projected_price"] = price_fn(df["cumulative_rate_bps"].values + base_rate_bps)
    return df


def create_figure(
    y_bps: np.ndarray,
    price: np.ndarray,
    projected_df: pd.DataFrame,
    terminal_price: float,
) -> plt.Figure:
    """Side-by-side: (1) Price vs Rate Shock, (2) Projected Price over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Modeled Price vs Rate Shock
    ax1.plot(y_bps, price, color="#2563EB", linewidth=2.5, label="Modeled Price")
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Rate Shock (bps)", fontsize=11)
    ax1.set_ylabel("Price", fontsize=11)
    ax1.set_title("Price vs. Rate Shock Curve\n(FNMA 3.0% MBS)", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(y_bps.min(), y_bps.max())

    # Subplot 2: Projected Price Trajectory
    ax2.plot(
        projected_df["date"],
        projected_df["projected_price"],
        color="#059669",
        marker="o",
        markersize=6,
        linewidth=2,
        label="Projected Price",
    )
    # Terminal Price Ceiling (Fed Dot Plot: -50 bps by end 2027)
    ax2.axhline(
        y=terminal_price,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.9,
        label="Terminal Price Ceiling",
    )
    ax2.annotate(
        f"Cycle Terminal Price: ${terminal_price:.2f}",
        xy=(projected_df["date"].iloc[len(projected_df) // 2], terminal_price),
        xytext=(15, 0),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color="red",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="red", alpha=0.95),
    )
    ax2.set_xlabel("FOMC Meeting Date", fontsize=11)
    ax2.set_ylabel("Projected Price", fontsize=11)
    ax2.set_title("Projected Price Trajectory\n(Based on Fed Funds Implied Path)", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle(
        "FNMA 3.0% MBS: Price-Yield Curve & Fed Rate Cut Forecast",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


# Fed Dot Plot: cumulative rate drop to terminal (3.00%-3.25% by end 2027)
FED_TERMINAL_RATE_BPS = -50


def main():
    base = Path(__file__).parent
    p0 = 98.5  # Current price (par-adj)

    # Load or use sample data
    rate_bps, mod_dur = load_duration_curve(base)
    fed_df = load_fed_funds(base)

    if rate_bps is None or mod_dur is None:
        print("No duration_curve.csv found. Using sample duration S-curve.")
        rate_bps, mod_dur, fed_sample = load_sample_data()
        if fed_df is None:
            fed_df = fed_sample
            print("No fed_funds_futures.csv found. Using sample Fed path.")
    if fed_df is None:
        print("No fed_funds_futures.csv found. Using sample Fed path.")
        _, _, fed_df = load_sample_data()

    # Build Price-Yield curve
    y_bps, price, price_fn = build_price_yield_curve(rate_bps, mod_dur, p0=p0)

    # Project prices at FOMC dates
    projected = project_prices(fed_df, price_fn)

    # Terminal price from Fed Dot Plot (cycle ceiling at -50 bps cumulative)
    terminal_price = float(price_fn(FED_TERMINAL_RATE_BPS))

    print("\n--- Projected Prices at FOMC Dates ---")
    print(projected[["date", "cumulative_rate_bps", "projected_price"]].to_string(index=False))
    print(f"\n--- Cycle Terminal Price (Fed Dot Plot: {FED_TERMINAL_RATE_BPS} bps) ---")
    print(f"  ${terminal_price:.2f}")

    # Plot
    fig = create_figure(y_bps, price, projected, terminal_price)
    out_path = base / "mbs_price_yield_forecast.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()

"""
FNMA 3.0% MBS: integrate duration S-curve for price-yield, map Fed path + terminal ceiling.
Run from repo root: python scripts/02_price_projection.py
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from _paths import DATA_DIR, IMG_DIR


def load_duration_curve():
    path = DATA_DIR / "scurve_rate_shocks.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    rate_col = next((c for c in df.columns if "rate" in c.lower() or "shock" in c.lower()), df.columns[0])
    dur_col = next((c for c in df.columns if "duration" in c.lower() or "mod" in c.lower()), df.columns[1])
    return df[rate_col].values.astype(float), df[dur_col].values.astype(float)


def load_fed_funds():
    path = DATA_DIR / "fed_funds_futures.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "rate_change_bps" in df.columns and "cumulative_rate_bps" not in df.columns:
        df["cumulative_rate_bps"] = df["rate_change_bps"].cumsum()
    elif "cumulative_rate_bps" not in df.columns and "implied_rate_bps" in df.columns:
        df["cumulative_rate_bps"] = df["implied_rate_bps"]
    return df


def load_sample_data():
    rate_shock_bps = np.array([-300, -200, -100, -50, 0, 50, 100, 150, 200, 250, 300])
    mod_duration = np.array([4.2, 5.8, 7.2, 7.8, 8.1, 8.0, 7.5, 6.8, 5.9, 5.0, 4.2])
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


def build_price_yield_curve(rate_shock_bps, mod_duration, p0=98.5, n_points=601):
    dur_interp = interp1d(
        rate_shock_bps, mod_duration,
        kind="cubic",
        bounds_error=False,
        fill_value=(mod_duration[0], mod_duration[-1]),
    )
    y_bps = np.linspace(rate_shock_bps.min(), rate_shock_bps.max(), n_points)
    y_decimal = y_bps / 10000
    D_mod = dur_interp(y_bps)
    integral = cumulative_trapezoid(D_mod, y_decimal, initial=0)
    idx_0 = np.argmin(np.abs(y_bps))
    integral -= integral[idx_0]
    price = p0 * np.exp(-integral)
    price_interp = interp1d(
        y_bps, price,
        kind="cubic",
        bounds_error=False,
        fill_value=(price[0], price[-1]),
    )
    return y_bps, price, price_interp


def project_prices(fed_df, price_fn, base_rate_bps=0):
    df = fed_df.copy()
    df["projected_price"] = price_fn(df["cumulative_rate_bps"].values + base_rate_bps)
    return df


def create_figure(y_bps, price, projected_df, terminal_price):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(y_bps, price, color="#2563EB", linewidth=2.5, label="Modeled Price")
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Rate Shock (bps)", fontsize=11)
    ax1.set_ylabel("Price", fontsize=11)
    ax1.set_title("Price vs. Rate Shock Curve\n(FNMA 3.0% MBS)", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(y_bps.min(), y_bps.max())

    ax2.plot(
        projected_df["date"],
        projected_df["projected_price"],
        color="#059669",
        marker="o",
        markersize=6,
        linewidth=2,
        label="Projected Price",
    )
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


FED_TERMINAL_RATE_BPS = -50


def main():
    p0 = 98.5
    rate_bps, mod_dur = load_duration_curve()
    fed_df = load_fed_funds()

    if rate_bps is None or mod_dur is None:
        print("No data/scurve_rate_shocks.csv. Using sample duration S-curve.")
        rate_bps, mod_dur, fed_sample = load_sample_data()
        if fed_df is None:
            fed_df = fed_sample
            print("No data/fed_funds_futures.csv. Using sample Fed path.")
    if fed_df is None:
        print("No data/fed_funds_futures.csv. Using sample Fed path.")
        _, _, fed_df = load_sample_data()

    y_bps, price, price_fn = build_price_yield_curve(rate_bps, mod_dur, p0=p0)
    projected = project_prices(fed_df, price_fn)
    terminal_price = float(price_fn(FED_TERMINAL_RATE_BPS))

    print("\n--- Projected Prices at FOMC Dates ---")
    print(projected[["date", "cumulative_rate_bps", "projected_price"]].to_string(index=False))
    print(f"\n--- Cycle Terminal Price (Fed Dot Plot: {FED_TERMINAL_RATE_BPS} bps) ---")
    print(f"  ${terminal_price:.2f}")

    fig = create_figure(y_bps, price, projected, terminal_price)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMG_DIR / "terminal_price_model.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()

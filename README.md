# MBS lock-in analysis (Fannie-3-6)

Personal project: compare FNCL 3% vs 6% mortgage bonds, and model FNMA 3.0% MBS price vs rate shocks using an integrated duration S-curve and Fed Funds / Dot Plot paths.

**GitHub:** [eugeneoCMU/Fannie-3-6](https://github.com/eugeneoCMU/Fannie-3-6)

## Layout

| Path | Purpose |
|------|---------|
| `data/scurve_rate_shocks.csv` | Modified duration vs parallel rate shock (bps), roughly -300 to +300 |
| `data/fed_funds_futures.csv` | FOMC dates and incremental `rate_change_bps` (cumulative built in code) |
| `data/fncl_historical_prices.csv` | Wide format: `date`, `bid_3`, `ask_3`, `bid_6`, `ask_6` |
| `scripts/01a_historical_comparison.py` | Four-panel matplotlib comparison (mid, spread, diff, stats) |
| `scripts/01b_spread_divergence.py` | Two-panel divergence / spread (Plotly HTML if available, else PNG) |
| `scripts/02_price_projection.py` | Duration integration → price-yield; Fed path + terminal ceiling (-50 bps) |
| `images/` | Generated charts (`historical_comparison.png`, `3s_vs_6s_divergence.png`, `terminal_price_model.png`) |
| `images/source/` | Reference exports (e.g. SVG charts) |
| `extra-cool-information/` | Optional reference: older PNG exports, Excel grids, shortcuts (not used by scripts; see folder README) |

## Setup

```bash
pip install -r requirements.txt
```

## Run (from repository root)

```bash
python scripts/01a_historical_comparison.py
python scripts/01b_spread_divergence.py          # add --matplotlib or -m for PNG-only
python scripts/02_price_projection.py
```

Outputs are written under `images/`.

## Data notes

- Replace CSVs in `data/` with your own Bloomberg or internal pulls; keep column names as documented above.
- If `fncl_historical_prices.csv` is missing, scripts fall back to short synthetic series for demos.
- `.lnk` shortcut files are ignored by git (see `.gitignore`). Grids and shortcuts live under `extra-cool-information/` if you keep them in the repo tree.

## Model (script 02)

Price vs yield shock uses \(P(y) = P_0 \exp(-\int D_{\mathrm{mod}}\,dy)\) with shocks in decimals (bps/10000). **FED_TERMINAL_RATE_BPS** in `02_price_projection.py` encodes the Dot Plot terminal (-50 bps cumulative in the sample).

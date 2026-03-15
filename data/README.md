# Data Directory

Place your input CSV files here before running the notebook.

---

## `inputs/geographic_setup.csv`

One row per project site.

| Column | Type | Description |
|---|---|---|
| `Name` | string | Project site name |
| `Latitude` | float | Decimal degrees (WGS84) |
| `Longitude` | float | Decimal degrees (WGS84) |
| `Max_Height_m` | float | Maximum elevation (m) — used for PHS feasibility check |

---

## `inputs/resource_assessment.csv`

One row per candidate technology. The `Sources` column must exactly match
the technology names selected in the notebook configuration step.

| Column | Type | Unit | Description |
|---|---|---|---|
| `Sources` | string | — | Technology name |
| `Max_Capacity_MW` | float | MW | Maximum installable capacity |
| `Investment_per_MW` | float | €/MW | Overnight capital cost |
| `O&M_per_MW_yr` | float | €/MW/yr | Annual fixed O&M cost |
| `Lifetime` | float | years | Economic lifetime for CRF calculation |
| `CO2_per_MWh` | float | tCO₂/MWh | Direct emission intensity |
| `Fuel_Cost` | float | €/MWh_th | Fuel cost per thermal MWh (0 for renewables) |
| `Efficiency` | float | 0–1 | Conversion efficiency |
| `Merit_Order` | int | — | Dispatch priority (lower = first) |
| `Storage_MWh` | float | €/MWh | Energy component CAPEX (0 for non-storage) |

---

## `time_series/`

Each file is a **single-column CSV** with exactly **8 760 rows**.

| File | Unit | Description |
|---|---|---|
| `demand.csv` | MW | Hourly electricity demand |
| `wind_prod.csv` | p.u. (0–1) | Hourly wind capacity factor |
| `solar_prod.csv` | p.u. (0–1) | Hourly solar capacity factor |
| `<tech>_prod.csv` | p.u. (0–1) | Additional generation profiles as needed |

**Validation rules:** exactly 8 760 values, no NaN, no negatives, capacity
factors in [0, 1].

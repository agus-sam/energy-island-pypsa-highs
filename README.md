# Energy Island System Optimisation (PyPSA · HiGHS)

A techno-economic **Linear Programming (LP) capacity-expansion and dispatch** model for optimising the generation mix and storage capacity of a renewable-based **energy island** or **isolated microgrid** system.

Built with [**PyPSA**](https://pypsa.org/) (Python for Power System Analysis) and solved with [**HiGHS**](https://highs.dev/) — a high-performance open-source LP solver accessed through PyPSA's [linopy](https://linopy.readthedocs.io/) backend. No external binary installation is required; HiGHS is loaded automatically when `highspy` is installed.

The model simultaneously determines optimal **investment decisions** (installed capacity of each technology) and **hourly operational dispatch** to meet system electricity demand at minimum total system cost, minimum CO₂ emissions, or with a guaranteed presence of every selected technology.

---

## Interactive Dashboard

Results from the notebook can be explored in the standalone interactive dashboard:

> **[🔗 Energy Island Dashboard](https://agus-sam.github.io/energy_island_dashboard/)**

---

## Repository Structure

```
energy-island-pypsa/
├── Energy_Island_PyPSA_HiGHS.ipynb   ← main notebook (run this)
├── index.html                         ← interactive dashboard (7 sections)
├── README.md
├── requirements.txt
├── LICENSE                            ← MIT
├── .gitignore
├── src/                               ← all Python code
│   ├── __init__.py
│   ├── constants.py                   ← model assumptions (single source of truth)
│   ├── style.py                       ← matplotlib global theme
│   ├── geographic.py                  ← site coordinate & elevation loader
│   ├── setup_options.py               ← interactive system configuration
│   ├── resources.py                   ← techno-economic parameter loader
│   ├── timeseries.py                  ← 8760-hour demand & generation profiles
│   ├── model.py                       ← PyPSA network builder, solver, exporters
│   └── visualization.py              ← 16 publication-quality chart methods
├── data/
│   ├── inputs/                        ← geographic_setup.csv, resource_assessment.csv
│   ├── time_series/                   ← demand.csv, wind_prod.csv, solar_prod.csv, ...
│   └── README.md                      ← column specifications
├── results/                           ← solver outputs (.xlsx, .json)
│   └── README.md
└── .github/workflows/ci.yml          ← syntax check CI
```

---

## System Technologies

### Variable Renewable Generation

| Technology | Description |
|---|---|
| **Wind** | Hourly generation profile scaled to 1 MW installed capacity |
| **Solar PV** | Hourly generation profile scaled to 1 MW installed capacity |

### Dispatchable Generation

| Technology | Dispatch mode | Description |
|---|---|---|
| **Biomass** | Non-flexible (must-run) | Combustion of solid biomass fuels; dispatch pinned to hourly profile × installed capacity |
| **Biogas** | Non-flexible (must-run) | Anaerobic digestion of organic waste; dispatch pinned to hourly profile × installed capacity |
| **Waste-to-Energy (WTE)** | Non-flexible (must-run) | Energy recovery from municipal solid waste; dispatch pinned to hourly profile × installed capacity |
| **Hydro** | Flexible dispatch | Run-of-river or reservoir hydropower; free to dispatch 0 → profile × capacity |
| **Geothermal** | Non-flexible (must-run) | Baseload generation; dispatch pinned to hourly profile × installed capacity |
| **Natural Gas turbine** | Flexible backup | Fast-response peaking unit; high variable cost and emissions |
| **Biodiesel** | Flexible backup | Renewable liquid fuel backup; similar dispatch role to natural gas |

> **Non-flexible dispatch:** Biomass, Biogas, Geothermal, and WTE are constrained via a custom linopy lower-bound to dispatch exactly at `profile(t) × p_nom_opt` each hour. This prevents the solver from under-utilising committed plant. Unused capacity in flexible dispatchable or balancing technologies is standby reserve — not curtailment.

### Energy Storage

| Technology | Typical Duration | Description |
|---|---|---|
| **BESS** | 2–6 hours | Battery Energy Storage System; fast response, flexible siting |
| **PHS** | 6–12 hours | Pumped Hydro Storage; requires ≥ 300 m head difference |
| **Hydrogen** | 12–48+ hours | Long-duration storage via electrolysis and fuel cell |

---

## Optimisation Objectives

Three objectives are available and selectable from the interactive setup widget:

| Objective | Description |
|---|---|
| **Lowest LCOE** | Minimises total annualised system cost (capital + O&M + fuel + penalties) |
| **Lowest CO₂** | Adds a carbon shadow price of 10,000 €/tCO₂ to marginal costs, heavily penalising emitting technologies |
| **Most Diversified** | Minimises total system cost while forcing every checked technology to install at least `DIVERSIFIED_MIN_MW` (default 2 MW) — ensures a technology-diverse solution |

---

## Model Features

- **LP capacity expansion** — simultaneous investment sizing across all technologies via PyPSA + linopy + HiGHS
- **Hourly dispatch modelling** — full 8,760-hour annual resolution
- **Non-flexible dispatch constraint** — Biomass, Biogas, Geothermal, and WTE dispatch is pinned to `profile(t) × p_nom_opt` via a custom linopy lower-bound injected through PyPSA's `extra_functionality` hook
- **Grid loss factor** — raw demand inflated by a configurable distribution loss factor (default 4%)
- **Capital Recovery Factor** — CAPEX annualised over asset lifetime at user-defined discount rate
- **Storage SoC tracking** — cyclic state-of-charge with round-trip efficiency losses; minimum SoC floor enforced via extra constraint hook
- **Storage duration limits** — user-defined maximum energy-to-power ratio per technology
- **Soft curtailment penalty** — small €/MWh penalty on VRE output to discourage oversizing with curtailment
- **Storage cycling cost** — proxy cost per MWh charged to prevent unnecessary cycling
- **PHS feasibility check** — project location elevation validated against 300 m minimum head requirement
- **Grid congestion analysis** — hourly bus utilisation ratio computed as gross demand / available supply capacity; hours above configurable threshold flagged as congested; monthly breakdown and duration curve
- **Three-objective optimisation** — Lowest LCOE, Lowest CO₂, or Most Diversified
- **Interactive setup widget** — technology selection, storage duration, discount rate, currency symbol, and demand scaling configurable via ipywidgets
- **Configurable currency symbol** — accepts any string (€, $, £, RM, Rp, etc.); applied to all cost labels and exports
- **Demand scaling** — multiply demand CSV by a constant factor directly from the widget, without editing source data
- **Dual export** — hourly results to Excel (`.xlsx`) and structured JSON (`.json`) for dashboard consumption; both include grid congestion data

---

## Optimisation Formulation

### Objective — Lowest LCOE

Minimise total annualised system cost:

$$
\begin{aligned}
\min \quad & \sum_{g} p_g^{nom} \cdot c_g^{cap} \\ & + \sum_{s} p_s^{nom} \cdot c_s^{cap} \\ & + \sum_{g,t} p_{g,t} \cdot c_g^{mc} \\ & + \sum_{s,t} p_{s,t}^{+} \cdot c^{stor}
\end{aligned}
$$

where $c^{cap}$ is the annualised capital cost computed via the Capital Recovery Factor:

$$CRF = \frac{r(1+r)^n}{(1+r)^n - 1}$$

and storage capital cost per MW aggregates both power and energy components:

$$c_s^{cap} = K_s^{MW} \cdot CRF_s + OM_s + K_s^{MWh} \cdot CRF_s \cdot MaxHours_s$$


### Objective — Lowest CO₂

Marginal cost is augmented with a carbon shadow price (CSP = 10,000 €/tCO₂):

$$c_g^{mc} \leftarrow c_g^{mc} + CO_{2,g} \times CSP$$


### Objective — Most Diversified

Identical marginal costs to Lowest LCOE, with a mandatory minimum capacity applied to every selected technology:

$$
\begin{aligned}
p_g^{nom} &\geq DIVERSIFIED\_MIN\_MW && \forall g \in \text{selected generation} \\
p_s^{nom} &\geq DIVERSIFIED\_MIN\_MW && \forall s \in \text{selected storage}
\end{aligned}
$$

### Key Constraints

| Constraint | Description |
|---|---|
| Power balance | Sum of generation + storage dispatch = gross demand (every hour) |
| Capacity limits | Dispatch ≤ `p_max_pu(t)` × installed capacity |
| Max capacity bounds | Installed capacity ≤ `Max_Capacity_MW` (resource constraint) |
| Storage power limits | Charge / discharge ≤ installed power capacity |
| SoC dynamics | $\text{SoC}_{t} = \text{SoC}_{t-1} + \eta \cdot \text{charge}_{t} - \text{discharge}_{t} / \eta$ |
| SoC minimum | $\text{SoC}_{t} \geq 0.20 \times p^{\text{nom}} \times \text{MaxHours}$ |
| SoC maximum | $\text{SoC}_{t} \leq p^{\text{nom}} \times \text{MaxHours}$ |
| Cyclic SoC | Initial SoC = final SoC (enforced via `cyclic_state_of_charge = True`) |
| Non-flexible floor | $p_{g,t} \geq \text{profile}(t) \times p^{\text{nom}}_g \;\; \forall g \in \{$Biomass, Biogas, Geothermal, WTE$\}$ |
| Grid loss factor | $P_{\text{load}}(t) = P_{\text{demand}}(t) \times (1 + \text{GLF})$ |

---

## Model Constants

All hard-coded assumptions are defined in `src/constants.py` — the **single source of truth** for global model parameters.

| Constant | Default | Description |
|---|---|---|
| `HOURS_PER_YEAR` | 8760 | Full-year hourly resolution |
| `GRID_LOSS_FACTOR` | 0.04 | Distribution loss factor — 4% of delivered energy |
| `SOC_MIN_FRACTION` | 0.20 | Minimum state-of-charge — 20% depth of discharge limit |
| `SOC_INIT_FRACTION` | 0.50 | Initial SoC for non-cyclic mode (reserved) |
| `CURTAILMENT_PENALTY` | 1 | €/MWh soft penalty on curtailed VRE |
| `STORAGE_CHARGE_COST` | 5 | €/MWh proxy cost to prevent unnecessary storage cycling |
| `CARBON_SHADOW_PRICE` | 10,000 | €/tCO₂ shadow price in Lowest CO₂ mode |
| `DIVERSIFIED_MIN_MW` | 2 | MW minimum installed capacity per technology in Most Diversified mode |
| `CONGESTION_THRESHOLD` | 0.80 | Bus utilisation ratio above which hours are flagged as congested |

### Notes

The **grid loss factor** inflates raw demand to gross generation requirement: $P_{\text{gross}}(t) = P_{\text{demand}}(t) \times (1 + \text{GLF})$. A 4% loss factor is typical for small isolated island networks.

The **curtailment penalty** discourages oversizing with curtailment without distorting investment decisions. The **storage charge cost** prevents the solver from cycling storage unnecessarily.

The **carbon shadow price** in Lowest CO₂ mode forces the solver to minimise emissions while respecting investment feasibility. The **diversification minimum** ensures no technology is excluded from the portfolio in Most Diversified mode.

The **congestion threshold** defines the utilisation level above which the single-bus island system is considered under stress — available headroom is thin and vulnerable to forecast errors or unplanned outages.

---

## Data Requirements

### `data/inputs/geographic_setup.csv`

| Column | Type | Description |
|---|---|---|
| `Name` | string | Location name |
| `Latitude` | float | Decimal degrees (WGS84) |
| `Longitude` | float | Decimal degrees (WGS84) |
| `Max_Height_m` | float | Maximum elevation (m) — PHS feasibility check |

### `data/inputs/resource_assessment.csv`

| Column | Unit | Description |
|---|---|---|
| `Sources` | — | Technology name (must match setup selections exactly) |
| `Max_Capacity_MW` | MW | Maximum installable capacity (resource or land constraint) |
| `Investment_per_MW` | currency/MW | Capital cost per MW of installed power capacity |
| `O&M_per_MW_yr` | currency/MW/yr | Annual fixed operation and maintenance cost |
| `Lifetime` | years | Economic asset lifetime (used in CRF) |
| `CO2_per_MWh` | tCO₂/MWh | Direct emissions per MWh of electricity generated |
| `Fuel_Cost` | currency/MWh_th | Variable fuel cost per MWh of thermal fuel consumed (0 for renewables) |
| `Efficiency` | 0–1 | Generator thermal efficiency or storage one-way efficiency |
| `Merit_Order` | integer | Dispatch priority (lower = dispatched first) |
| `Storage_MWh` | currency/MWh | Capital cost per MWh of energy storage capacity (0 for generators) |

**Note:** `Fuel_Cost` is per MWh of **thermal input**. The model computes the marginal cost per MWh electric as `Fuel_Cost / Efficiency`. For storage, `Efficiency` represents the one-way efficiency; round-trip efficiency = efficiency².

### `data/time_series/`

Each file is a single-column CSV with exactly **8,760 rows** (one per hour). No header row required but accepted.

| File | Unit | Description |
|---|---|---|
| `demand.csv` | MW | Hourly electricity demand |
| `wind_prod.csv` | p.u. (0–1) | Hourly wind capacity factor |
| `solar_prod.csv` | p.u. (0–1) | Hourly solar capacity factor |
| `<tech>_prod.csv` | p.u. (0–1) | Additional generation profiles as needed |

---

## How to Run

### 1. Install dependencies

```bash
pip install pypsa highspy linopy pandas numpy matplotlib plotly openpyxl ipywidgets jupyterlab
```

### 2. Prepare data

Place CSV files in `data/inputs/` and `data/time_series/` following the specifications above.

### 3. Run the notebook

```bash
jupyter lab Energy_Island_PyPSA_HiGHS.ipynb
```

Execute cells top-to-bottom:

| Step | Action |
|:---:|---|
| **1** | Import libraries and apply chart style |
| **2** | Load geographic project information — enter file path and click Load |
| **3** | Configure technologies, objective, discount rate, currency, and demand scale — click Confirm |
| **4** | Load resource assessment CSV — enter file path and click Load |
| **5** | Load hourly demand and generation time series — enter file paths and click Load |
| **6** | Build and solve the PyPSA optimisation model |
| **7** | Export results to Excel (`.xlsx`) and JSON (`.json`) |
| **8** | Analyse and visualise results |

**Demand Scaling**

The **Demand Scale %** widget in Step 3 multiplies every hourly value of the loaded demand CSV by a constant factor:

| Setting | Scenario |
|---|---|
| `100%` | Baseline — demand loaded as-is (default) |
| `110%` | +10% demand growth |
| `90%` | −10% efficiency / conservation |
| `150%` | +50% electrification or EV uptake |

The loader prints the raw and scaled peak (MW) and annual total (GWh) when the demand file is loaded in Step 5.

**Generation profile data sources:**
- [Renewables.ninja](https://www.renewables.ninja) — Wind and solar PV hourly profiles
- [Global Solar Atlas](https://globalsolaratlas.info) — Solar irradiance data
- [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/) — European Commission PV estimation tool
- [ERA5 / Copernicus](https://cds.climate.copernicus.eu) — Global climate reanalysis data
- [IRENA Resource Map](https://resourceirena.irena.org) — Renewable resource atlases

---

## Key Outputs

| Output | Description |
|---|---|
| Installed capacity | Optimal MW per generation technology |
| Storage sizing | Power (MW) and energy (MWh) capacity per storage technology |
| System LCOE | Levelised cost of electricity (currency/MWh) |
| Technology LCOE / LCOS | Per-technology levelised cost of generation and storage |
| LCOE cost breakdown | Stacked CAPEX / O&M / fuel decomposition per technology |
| Annual energy mix | Generation breakdown as share of total demand (donut chart with callout leader lines) |
| Dispatch charts | Stacked hourly generation in merit order |
| Capacity factors | Annual generation ÷ (installed capacity × 8,760 h) |
| Monthly capacity factor grid | Technology × month CF heatmap with annotated cell values |
| Demand heatmap | Average gross demand by hour-of-day × month (24 × 12 grid) |
| State-of-charge | Storage SoC profiles with capacity and minimum SoC reference bands |
| Residual load | Demand minus renewables — identifies storage and backup needs |
| Load duration curve | Sorted annual dispatch showing peak-to-valley coverage |
| Worst renewable week | Auto-detected most critical 168-hour window for system reliability |
| Energy Sankey | Interactive annual energy flow diagram (generation → storage → load → losses) |
| CO₂ summary | Per-technology and total annual emissions with intensity (gCO₂/kWh) |
| Grid congestion | Bus utilisation ratio timeseries, duration curve, monthly congested hours breakdown |
| Excel export | Full 8,760-hour hourly dispatch including utilisation ratio and congestion flag |
| JSON export | Structured results file consumed by the interactive dashboard; includes `grid_congestion` key |

### Visualisation Methods (`ResultsVisualization`)

| Method | Chart type | Description |
|---|---|---|
| `summary()` | Text | Installed capacity of all generation and storage technologies |
| `calculate_lcoe()` | Text | Per-technology LCOE/LCOS, system LCOE, and annual CO₂ summary |
| `plot_installed_capacity()` | Bar chart | Installed MW per generation technology with value labels |
| `plot_storage_capacity()` | Side-by-side bars | Storage power (MW) and energy (MWh) capacity |
| `plot_energy_mix()` | Donut chart | Annual generation mix as % of net demand — callout leader lines with collision avoidance |
| `plot_capacity_factors()` | Horizontal bars | Achieved capacity factor; technology name printed inside bar, CF % to the right |
| `plot_lcoe_breakdown()` | Stacked horizontal bars | Annualised cost decomposed into CAPEX / O&M / fuel per technology |
| `plot_dispatch(hours)` | Stacked area | Hourly dispatch by technology with net and gross demand overlay lines |
| `plot_soc(hours)` | Line chart | Storage state of charge with capacity and 20% minimum SoC reference bands |
| `plot_residual(hours)` | Area chart | Residual load after renewables (deficit/surplus) |
| `plot_load_duration()` | Step chart | Sorted load duration curve with stacked generation breakdown |
| `plot_worst_residual_week()` | Combined | Auto-detects and plots dispatch, SoC, and residual for the hardest 168-hour window |
| `plot_demand_heatmap()` | Heatmap | Average gross demand by hour-of-day (rows 0–23) × month (columns Jan–Dec) |
| `plot_monthly_cf()` | Heatmap grid | Monthly capacity factor per technology, with CF % annotated in each cell |
| `plot_energy_sankey()` | Sankey (Plotly) | Interactive annual energy flow: generation → storage → Grid Bus → Load + Losses + Curtailment; height scales with technology count |
| `plot_grid_congestion()` | Multi-panel | Bus utilisation timeseries with congested hours shaded, utilisation duration curve, monthly congested hours bar chart |

Pass `hours` as a `(start, end)` tuple or a single integer (168-hour window from that hour):

```python
viz.plot_dispatch((1000, 1168))   # explicit window
viz.plot_dispatch(1000)           # 168-hour window from hour 1000
```

### Excel Export Columns (`optimisation_results.xlsx`)

| Column pattern | Unit | Description |
|---|---|---|
| `Prod_{tech}_MW` | MW | Hourly generation dispatch per technology |
| `Discharge_{storage}_MW` | MW | Storage discharge (positive flow to grid) |
| `Charge_{storage}_MW` | MW | Storage charge (positive flow from grid) |
| `SOC_{storage}_MWh` | MWh | State of charge at end of each hour |
| `GrossDemand_MW` | MW | Demand including grid losses |
| `NetDemand_MW` | MW | Raw consumer demand |
| `Utilisation_Ratio` | 0–1 | Hourly bus utilisation ratio (gross demand / available supply) |
| `Congested` | 0/1 | 1 if utilisation exceeds congestion threshold |

### JSON Export Structure (`dashboard_results.json`)

Produced by `model.export_dashboard_json(filepath, geo=geo)`. Pass `geo=geo` to embed geographic data in the output.

| Key | Contents |
|---|---|
| `meta` | Objective, discount rate, currency, grid loss factor, technology lists |
| `geographic` | Project name, latitude, longitude, max height — only present when `geo` is passed |
| `capacities` | Installed MW (and MWh for storage) + `potential_mw` (from `Max_Capacity_MW`) per technology |
| `energy_mix` | Annual GWh, demand share (%), capacity factor, per-technology LCOE, CO₂, curtailment GWh |
| `lcoe_summary` | Per-technology LCOE/LCOS, annualised cost, `annualised_capex`, `annual_opex`, `annual_fuel_cost`, `total_capex` |
| `lcoe_summary._system` | System LCOE, total annualised cost, total demand, `total_annualised_capex`, `total_annual_opex`, `total_annual_fuel`, `total_capex` |
| `co2_summary` | Per-technology tCO₂/yr, total CO₂, emission intensity (gCO₂/kWh) |
| `grid_congestion` | Threshold, congested hours, peak/mean utilisation, monthly breakdown (12 values), hourly utilisation array (8,760 values) |
| `hourly` | Full 8,760-row array with generation, storage charge/discharge/SoC, gross demand, and net demand |

### Dashboard (`index.html`)

The interactive dashboard consumes the JSON output and renders seven sections:

| Section | Contents |
|---|---|
| **A — Installed Capacity Mix** | Capacity bar chart, potential vs installed comparison |
| **B — Hourly Dispatch** | Technology-filterable hourly dispatch with month/week selectors |
| **C — Demand vs Generation** | Energy balance, storage operation, Sankey flow diagram |
| **D — Storage & Load** | Load duration curve, demand heatmap, state-of-charge |
| **E — System Cost Breakdown** | LCOE/LCOS bars, cost composition (CAPEX/OPEX/fuel), cost view selector |
| **F — Performance & Emissions** | Capacity factors, diurnal profile, CO₂ emissions, curtailment, monthly CF grid |
| **G — Grid Congestion** | Bus utilisation timeseries, utilisation duration curve, monthly congested hours |

The dashboard supports multiple scenarios (Lowest LCOE, Lowest CO₂, Most Diversified) via a dropdown selector, with keyboard shortcuts (`1`–`7` for sections, `←`/`→` for scenarios).

---

## Potential Extensions

| Extension | Description |
|---|---|
| Multi-bus network | Line-constrained dispatch for multi-zone island systems |
| Stochastic optimisation | Uncertainty in demand, wind, and solar resources |
| Multi-year investment | Phased capacity expansion over a planning horizon |
| CO₂ budget constraint | Hard emission cap via `n.add("GlobalConstraint", ...)` |
| Demand response | Flexible loads as a system balancing resource |
| EV fleet integration | Electric vehicles as distributed storage |
| Line loss modelling | Explicit resistive losses on internal distribution lines |

---

## Disclaimer

Developed for educational and research purposes. Results should not be used as the sole basis for investment or planning decisions. The author makes no guarantees regarding the completeness or accuracy of the model outputs.

---

## References

- Brown, T. et al. (2018). PyPSA: Python for Power System Analysis. *Journal of Open Research Software*, 6(1), 4.
- Huangfu, Q. & Hall, J.A.J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10, 119–142. [HiGHS]
- IRENA (2024). *Renewable Power Generation Costs in 2024*. International Renewable Energy Agency.
- IEA (2024). *World Energy Outlook 2024*. International Energy Agency.
- NREL (2024). *Annual Technology Baseline 2024*. National Renewable Energy Laboratory.
- Pfenninger, S. & Staffell, I. (2016). Long-term patterns of European PV output using 30 years of validated hourly reanalysis and satellite data. *Energy*, 114, 1251–1265.
- Staffell, I. & Pfenninger, S. (2016). Using bias-corrected reanalysis to simulate current and future wind power output. *Energy*, 114, 1224–1239.
- Dhungel, G. (2022). Modelling the Energy System of Ambon City, Indonesia. Master Thesis, Hochschule Nordhausen / Reiner Lemoine Institut.
- Tumiran et al. (2022). Generation Expansion Planning Based on Local Renewable Energy Resources: Ambon-Seram. *Sustainability*, 14(5), 3032.
- Anthropic (2025). *Claude AI Assistant* (claude.ai). Used to support model development, code review, and documentation.

---

*Author: Agus Samsudin — Energy Systems Modelling · Optimisation · Renewable Energy*

# ── Model Constants ───────────────────────────────────────────────────────────
#
# All hard-coded assumptions are defined here as named constants.
# This is the ONLY file that needs to be edited to change model assumptions —
# no values are duplicated elsewhere in the codebase.

HOURS_PER_YEAR       = 8760     # full year
GRID_LOSS_FACTOR     = 0.04     # distribution loss fraction
SOC_MIN_FRACTION     = 0.20     # minimum SoC (20% DoD)
SOC_INIT_FRACTION    = 0.50     # reserved: initial SoC for non-cyclic mode
CURTAILMENT_PENALTY  = 1        # €/MWh soft penalty on curtailed VRE
STORAGE_CHARGE_COST  = 5        # €/MWh proxy cost per MWh charged
CARBON_SHADOW_PRICE  = 10_000   # €/tCO2 — penalises emissions in Lowest CO2 mode
DIVERSIFIED_MIN_MW   = 2        # minimum installed capacity per technology in Most Diversified mode

# ── Grid Congestion Thresholds ────────────────────────────────────────────────
CONGESTION_THRESHOLD = 0.80     # utilisation ratio above which the bus is "congested"

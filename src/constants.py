# ── Model Constants ───────────────────────────────────────────────────────────
#
# All hard-coded assumptions are defined here as named constants.
# This is the only file that needs to be edited to change model assumptions —
# no values are duplicated elsewhere in the codebase.
#
# Currency: IDR

HOURS_PER_YEAR       = 8760     # full year
GRID_LOSS_FACTOR     = 0.04     # distribution loss fraction
SOC_MIN_FRACTION     = 0.20     # minimum state of charge (20 % depth of discharge)
SOC_INIT_FRACTION    = 0.50     # reserved: initial SoC for non-cyclic mode
CURTAILMENT_PENALTY  = 10       # currency/MWh soft penalty on curtailed VRE
STORAGE_CHARGE_COST  = 2        # currency/MWh proxy cost per MWh charged
CARBON_SHADOW_PRICE  = 20000    # currency/tCO₂ — penalises emissions in Lowest CO₂ mode
DIVERSIFIED_MIN_MW   = 1        # minimum installed capacity per technology in Most Diversified mode
CARBON_TAX           = 55       # currency/tCO₂

# ── Grid Congestion Thresholds ────────────────────────────────────────────────
CONGESTION_THRESHOLD = 0.80     # utilisation ratio above which the bus is considered congested

#Currency used in this scenario: EUR

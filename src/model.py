import os
import json as _json
import numpy as np
import pandas as pd
import pypsa

from src.constants import (
    HOURS_PER_YEAR, GRID_LOSS_FACTOR, SOC_MIN_FRACTION,
    CURTAILMENT_PENALTY, STORAGE_CHARGE_COST,
    CARBON_SHADOW_PRICE, DIVERSIFIED_MIN_MW, CONGESTION_THRESHOLD,
)


class IslandEnergyPyPSA:
    """
    LP investment + dispatch model for an island / microgrid energy system.

    Built using PyPSA with the HiGHS solver (via linopy).
    """

    def __init__(self, setup, resources, ts):
        self.setup     = setup
        self.resources = resources
        self.ts        = ts
        self.network   = None

    @staticmethod
    def _crf(rate, years):
        """Capital Recovery Factor: annualises CAPEX over asset lifetime."""
        r, n = rate, years
        return (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    def build(self):
        """Construct the PyPSA network with all generators, storage, and load."""
        r             = self.setup.discount_rate
        gen_techs     = self.setup.selected_gen + self.setup.selected_balancing
        storage_types = self.setup.selected_storage

        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(HOURS_PER_YEAR, name="hour"))

        n.add("Bus", "Island", carrier="AC")

        # This ensures generators are sized to cover both consumer demand
        # and the technical losses in the island distribution network.
        effective_demand = self.ts.demand * (1.0 + GRID_LOSS_FACTOR)
        n.add("Load", "Demand", bus="Island",
              p_set=pd.Series(effective_demand, index=n.snapshots))

        NON_FLEXIBLE_TECHS = {"Biomass", "Biogas", "Geothermal", "WTE"}
        self._non_flex_profiles = {}   # tech → numpy array or scalar

        for tech in gen_techs:
            p   = self.resources.get(tech)
            crf = self._crf(r, p["Lifetime"])

            # Annualised capital cost (CAPEX × CRF + fixed O&M) — €/MW/yr
            ann_capital = p["Investment_per_MW"] * crf + p["O&M_per_MW_yr"]

            # Availability profile (p_max_pu):
            if tech in self.ts.generation:
                raw_profile = self.ts.generation[tech]
                p_max_pu    = pd.Series(
                    np.clip(raw_profile, 0.0, 1.0), index=n.snapshots
                )
            else:
                p_max_pu = 1.0

            # Variable marginal cost — €/MWh of electrical output:
            #   · Fuel cost normalised by generator efficiency
            #   · For VRE (zero fuel cost): add soft curtailment penalty
            #   · For CO2 mode: add carbon shadow price × CO2 intensity
            real_mc = p["Fuel_Cost"] / p["Efficiency"] if p["Efficiency"] > 0 else 0.0
            if p["Fuel_Cost"] == 0:
                mc = real_mc + CURTAILMENT_PENALTY
            else:
                mc = real_mc

            if self.setup.objective == "Lowest CO2":
                mc += p["CO2_per_MWh"] * CARBON_SHADOW_PRICE

            p_nom_min_val = DIVERSIFIED_MIN_MW if self.setup.objective == "Most Diversified" else 0.0

            n.add("Generator", tech,
                  bus              = "Island",
                  carrier          = tech,
                  p_nom_extendable = True,
                  p_nom_min        = p_nom_min_val,
                  p_nom_max        = p["Max_Capacity_MW"],
                  p_max_pu         = p_max_pu,
                  capital_cost     = ann_capital,
                  marginal_cost    = mc)

            # Store the profile so _add_non_flexible_floor() can build the
            # linopy lower-bound constraint after PyPSA constructs the model.
            if tech in NON_FLEXIBLE_TECHS:
                if isinstance(p_max_pu, pd.Series):
                    self._non_flex_profiles[tech] = p_max_pu.values  # numpy (8760,)
                else:
                    self._non_flex_profiles[tech] = float(p_max_pu)  # scalar

        for s in storage_types:
            p    = self.resources.get(s)
            crf  = self._crf(r, p["Lifetime"])
            mhrs = self.setup.max_storage_hours.get(s, 4)

            # Combined annualised capital cost per MW of power capacity:
            #   power CAPEX × CRF + fixed O&M + energy CAPEX × CRF × max_hours
            ann_capital = (
                p["Investment_per_MW"] * crf
                + p["O&M_per_MW_yr"]
                + p["Storage_MWh"] * crf * mhrs
            )

            s_nom_min_val = DIVERSIFIED_MIN_MW if self.setup.objective == "Most Diversified" else 0.0

            n.add("StorageUnit", s,
                  bus                    = "Island",
                  carrier                = s,
                  p_nom_extendable       = True,
                  p_nom_min              = s_nom_min_val,
                  p_nom_max              = p["Max_Capacity_MW"],
                  capital_cost           = ann_capital,
                  marginal_cost          = STORAGE_CHARGE_COST,
                  efficiency_store       = p["Efficiency"],
                  efficiency_dispatch    = p["Efficiency"],
                  max_hours              = mhrs,
                  cyclic_state_of_charge = True)

        self.network = n

    def _add_soc_floor(self, n, snapshots):
        """
        Minimum state-of-charge constraint (extra_functionality hook).

        Adds the linear LP constraint:
          SoC(s, t)  >=  SOC_MIN_FRACTION × max_hours(s) × p_nom(s)
          .
        """
        if not list(n.storage_units.index):
            return
        m    = n.model
        soc  = m["StorageUnit-state_of_charge"]   # (snapshots × storage_units)
        pnom = m["StorageUnit-p_nom"]              # (extendable_storage_units,)
        for s in n.storage_units.index:
            mhrs = n.storage_units.at[s, "max_hours"]
            try:
                m.add_constraints(
                    soc.sel({"StorageUnit-ext": s}) >=
                    SOC_MIN_FRACTION * mhrs * pnom.sel({"StorageUnit-ext": s}),
                    name=f"min_soc_{s}"
                )
            except Exception as e:
                print(f"  WARNING: min-SoC constraint for {s} skipped: {e}")

    def _add_non_flexible_floor(self, n, snapshots):
        """
        Non-flexible dispatch constraint (extra_functionality hook).

        Adds  p[g,t] >= profile[t] * p_nom[g]  directly to the linopy model
        for Biomass, Biogas, Geothermal, and WTE so HiGHS pins dispatch to
        exactly  profile(t) * p_nom_opt  at every hour.

        Critical: the linopy Variable must be on the LEFT of *
        (i.e. pnom_var * coeff, not coeff * pnom_var).
        xr.DataArray.__mul__(linopy.Variable) does NOT produce a linopy
        LinearExpression — it silently wraps the Variable as a Python object.
        Only linopy.Variable.__mul__(xr.DataArray) produces a proper expression.
        """
        profiles = getattr(self, "_non_flex_profiles", {})
        if not profiles:
            return
        import xarray as xr
        m = n.model

        # Snapshot coordinate values — must match the linopy model's internal coords.
        snap_vals = n.snapshots.values

        print("  Non-flexible floor constraints:")
        for tech, profile in profiles.items():
            try:
                gen_p    = m["Generator-p"]      # dims: (snapshot, name)
                gen_pnom = m["Generator-p_nom"]  # dims: (name,)  — extendable only

                # Select this technology
                p_var    = gen_p.sel(name=tech)     # → Variable(snapshot,)
                pnom_var = gen_pnom.sel(name=tech)  # → Variable(scalar)

                # Build per-snapshot coefficient DataArray with EXPLICIT coords
                if isinstance(profile, np.ndarray):
                    coeff = xr.DataArray(
                        profile,
                        dims=["snapshot"],
                        coords={"snapshot": snap_vals}
                    )
                else:
                    coeff = float(profile)

                # CORRECT order: linopy.Variable * xr.DataArray
                rhs = pnom_var * coeff

                m.add_constraints(
                    p_var >= rhs,
                    name=f"non_flex_floor_{tech}"
                )
                print(f"    {tech}: OK")

            except Exception as e:
                print(f"    {tech}: FAILED — {e!r}")
                try:
                    print(f"    Available model variables: {list(m.variables)}")
                except Exception:
                    pass

    def solve(self, solver_options=None):
        """
        Solve the PyPSA network using HiGHS.

        The minimum-SoC constraint is injected via PyPSA's
        extra_functionality hook before the model is passed to HiGHS.
        """
        opts = solver_options or {}
        n    = self.network

        def _extra_functionality(n, snapshots):
            self._add_soc_floor(n, snapshots)
            self._add_non_flexible_floor(n, snapshots)

        status, condition = n.optimize(
            solver_name          = "highs",
            solver_options       = opts,
            extra_functionality  = _extra_functionality
        )

        if condition.lower() != "optimal":
            raise RuntimeError(
                f"Solver did not find an optimal solution. "
                f"Status: {status} | Condition: {condition}\n"
                "Check that demand is feasible for the selected technologies."
            )

        obj_label = {
            "Lowest LCOE":      "Lowest LCOE",
            "Lowest CO2":       "Lowest CO₂ Emissions",
            "Most Diversified": f"Most Diversified (min {DIVERSIFIED_MIN_MW} MW per technology)",
        }.get(self.setup.objective, self.setup.objective)
        print(f"\nOptimisation complete  [{obj_label}].")
        print("\n  Generation capacities:")
        for gen in n.generators.index:
            cap = n.generators.at[gen, "p_nom_opt"]
            print(f"    {gen:<14}: {cap:>8.2f}  MW")
        if list(n.storage_units.index):
            print("\n  Storage power capacities:")
            for s in n.storage_units.index:
                cap  = n.storage_units.at[s, "p_nom_opt"]
                mhrs = n.storage_units.at[s, "max_hours"]
                print(f"    {s:<14}: {cap:>8.2f}  MW  |  {cap*mhrs:>8.2f}  MWh")

    def _stor_discharge(self, s):
        """Return discharge timeseries (positive values only) for storage unit s."""
        return self.network.storage_units_t.p[s].clip(lower=0)

    def _stor_charge(self, s):
        """Return charge timeseries (positive values only) for storage unit s."""
        return (-self.network.storage_units_t.p[s]).clip(lower=0)

    # ── Grid Congestion Analysis ──────────────────────────────────────────

    def compute_congestion(self):
        """
        Compute hourly grid congestion metrics for the single-bus island system.

        In a single-bus model without explicit line limits, "congestion" is
        measured as the bus-level utilisation ratio:

            utilisation(t) = gross_demand(t) / total_available_supply(t)

        where total_available_supply is the sum of all generator available
        capacity (p_nom_opt × p_max_pu) plus storage discharge headroom
        at each hour.

        A utilisation ratio above CONGESTION_THRESHOLD (default 0.80) flags
        hours where the system is under stress — available headroom is thin,
        and any unplanned outage or forecast error could cause load shedding.

        Returns a dict with:
            utilisation   : np.array(8760,) — hourly utilisation ratio [0..1+]
            congested     : np.array(8760,) — boolean, True when above threshold
            hours_above   : int  — total congested hours
            peak_util     : float — maximum utilisation observed
            mean_util     : float — annual average utilisation
            threshold     : float — the threshold used
            monthly_hours : list[int] — congested hours per month (12 values)
        """
        n = self.network
        gross_demand = n.loads_t.p_set["Demand"].values  # (8760,)

        # Available supply = sum of available generation + storage discharge headroom
        available = np.zeros(HOURS_PER_YEAR)
        for gen in n.generators.index:
            cap = n.generators.at[gen, "p_nom_opt"]
            if gen in n.generators_t.p_max_pu.columns:
                available += cap * n.generators_t.p_max_pu[gen].values
            else:
                available += cap

        for s in n.storage_units.index:
            pwr = n.storage_units.at[s, "p_nom_opt"]
            available += pwr  # max discharge power headroom

        # Avoid division by zero in edge cases
        available = np.maximum(available, 1e-6)
        utilisation = gross_demand / available
        congested   = utilisation >= CONGESTION_THRESHOLD

        # Monthly breakdown
        months_idx = np.searchsorted(
            np.cumsum([744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]),
            np.arange(HOURS_PER_YEAR), side='right'
        )
        monthly_hours = [0] * 12
        for h in range(HOURS_PER_YEAR):
            if congested[h]:
                monthly_hours[min(months_idx[h], 11)] += 1

        return {
            "utilisation":   utilisation,
            "congested":     congested,
            "hours_above":   int(congested.sum()),
            "peak_util":     float(utilisation.max()),
            "mean_util":     float(utilisation.mean()),
            "threshold":     CONGESTION_THRESHOLD,
            "monthly_hours": monthly_hours,
        }

    # ── Export: Excel ─────────────────────────────────────────────────────

    def export_results(self, filepath="results/optimisation_results.xlsx"):
        """
        Export hourly dispatch results to Excel.
        Usage: model.export_results('results/scenario_A.xlsx')
        """
        n  = self.network
        df = pd.DataFrame(index=n.snapshots)

        for gen in n.generators.index:
            df[f"Prod_{gen}_MW"] = n.generators_t.p[gen].values

        for s in n.storage_units.index:
            net_p = n.storage_units_t.p[s].values
            df[f"Discharge_{s}_MW"] = np.maximum(net_p, 0.0)
            df[f"Charge_{s}_MW"]    = np.maximum(-net_p, 0.0)
            df[f"SOC_{s}_MWh"]      = n.storage_units_t.state_of_charge[s].values

        df["GrossDemand_MW"] = n.loads_t.p_set["Demand"].values
        df["NetDemand_MW"]   = self.ts.demand

        # Grid congestion column
        cong = self.compute_congestion()
        df["Utilisation_Ratio"] = cong["utilisation"]
        df["Congested"]         = cong["congested"].astype(int)

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True
        )
        df.to_excel(filepath, index_label="Hour")
        print(f"Results exported to: {filepath}")

    # ── Export: JSON ───────────────────────────────────────────────────────

    def export_json(self, filepath="results/optimisation_results.json"):
        """
        Export optimisation results as a JSON file for dashboard consumption.

        Produces a structured dict with:
          - meta          : run metadata (objective, discount rate, currency)
          - capacities    : installed generation and storage capacities
          - annual_energy : annual generation, capacity factors, LCOE per tech
          - storage       : storage power, energy, LCOS per unit
          - kpis          : system LCOE, total CO2, emission intensity
          - dispatch      : hourly timeseries (generation, storage, demand)
          - grid_congestion : congestion summary + hourly utilisation
        """
        n   = self.network
        r   = self.setup.discount_rate
        cur = self.setup.currency

        meta = {
            "objective":     self.setup.objective,
            "discount_rate": r,
            "currency":      cur,
            "grid_loss_pct": round(GRID_LOSS_FACTOR * 100, 2),
            "technologies":  {
                "generation": self.setup.selected_gen,
                "storage":    self.setup.selected_storage,
                "balancing":  self.setup.selected_balancing,
            },
        }

        capacities = {}
        for gen in n.generators.index:
            capacities[gen] = {
                "type":        "generator",
                "capacity_mw": round(float(n.generators.at[gen, "p_nom_opt"]), 4),
            }
        for s in n.storage_units.index:
            pwr  = float(n.storage_units.at[s, "p_nom_opt"])
            mhrs = float(n.storage_units.at[s, "max_hours"])
            capacities[s] = {
                "type":        "storage",
                "power_mw":    round(pwr, 4),
                "energy_mwh":  round(pwr * mhrs, 4),
                "max_hours":   mhrs,
            }

        annual_energy = {}
        total_ac      = 0.0
        for gen in n.generators.index:
            cap = float(n.generators.at[gen, "p_nom_opt"])
            if cap < 1e-6:
                annual_energy[gen] = {"capacity_mw": 0, "generation_mwh": 0}
                continue
            p         = self.resources.get(gen)
            crf       = IslandEnergyPyPSA._crf(r, p["Lifetime"])
            gen_mwh   = float(n.generators_t.p[gen].sum())
            ann_capex = cap * p["Investment_per_MW"] * crf
            ann_om    = cap * p["O&M_per_MW_yr"]
            real_mc   = p["Fuel_Cost"] / p["Efficiency"] if p["Efficiency"] > 0 else 0.0
            fuel_cost = real_mc * gen_mwh
            ann_cost  = ann_capex + ann_om + fuel_cost
            total_ac += ann_cost
            cf        = gen_mwh / (cap * HOURS_PER_YEAR) if cap > 0 else 0.0
            lcoe      = ann_cost / gen_mwh if gen_mwh > 0 else None
            co2_tech  = gen_mwh * p["CO2_per_MWh"]
            annual_energy[gen] = {
                "capacity_mw":    round(cap, 4),
                "generation_mwh": round(gen_mwh, 2),
                "capacity_factor": round(cf, 4),
                "lcoe":           round(lcoe, 4) if lcoe is not None else None,
                "ann_cost":       round(ann_cost, 2),
                "co2_t":          round(co2_tech, 2),
            }

        storage_out = {}
        for s in n.storage_units.index:
            p    = self.resources.get(s)
            crf  = IslandEnergyPyPSA._crf(r, p["Lifetime"])
            pwr  = float(n.storage_units.at[s, "p_nom_opt"])
            mhrs = float(n.storage_units.at[s, "max_hours"])
            ene  = pwr * mhrs
            ac   = (pwr * p["Investment_per_MW"] * crf
                    + pwr * p["O&M_per_MW_yr"]
                    + ene * p["Storage_MWh"] * crf)
            total_ac += ac
            dis  = float(self._stor_discharge(s).sum())
            chg  = float(self._stor_charge(s).sum())
            rte  = dis / chg if chg > 0 else 0.0
            lcos = ac / dis if dis > 0 else None
            storage_out[s] = {
                "power_mw":      round(pwr, 4),
                "energy_mwh":    round(ene, 4),
                "discharge_mwh": round(dis, 2),
                "charge_mwh":    round(chg, 2),
                "rte":           round(rte, 4),
                "ann_cost":      round(ac, 2),
                "lcos":          round(lcos, 4) if lcos is not None else None,
            }

        total_dem   = float(sum(self.ts.demand))
        system_lcoe = total_ac / total_dem if total_dem > 0 else 0.0
        total_co2   = sum(
            v["co2_t"] for v in annual_energy.values() if "co2_t" in v
        )
        kpis = {
            "total_ann_cost":              round(total_ac, 2),
            "total_demand_mwh":            round(total_dem, 2),
            "system_lcoe":                 round(system_lcoe, 4),
            "total_co2_t":                 round(total_co2, 2),
            "emission_intensity_gco2_kwh": round(
                total_co2 / total_dem * 1000 if total_dem > 0 else 0.0, 4
            ),
            "grid_loss_factor":            GRID_LOSS_FACTOR,
            "currency":                    cur,
        }

        dispatch = {"hour": list(range(HOURS_PER_YEAR))}
        for gen in n.generators.index:
            dispatch[f"gen_{gen}_mw"] = [
                round(float(v), 4) for v in n.generators_t.p[gen].values
            ]
        for s in n.storage_units.index:
            net_p = n.storage_units_t.p[s].values
            dispatch[f"discharge_{s}_mw"] = [round(float(max(v, 0.0)), 4) for v in net_p]
            dispatch[f"charge_{s}_mw"]    = [round(float(max(-v, 0.0)), 4) for v in net_p]
            dispatch[f"soc_{s}_mwh"]      = [
                round(float(v), 4)
                for v in n.storage_units_t.state_of_charge[s].values
            ]
        dispatch["gross_demand_mw"] = [
            round(float(v), 4) for v in n.loads_t.p_set["Demand"].values
        ]
        dispatch["net_demand_mw"] = [round(float(v), 4) for v in self.ts.demand]

        # Grid congestion
        cong = self.compute_congestion()
        grid_congestion = {
            "threshold":     cong["threshold"],
            "hours_above":   cong["hours_above"],
            "peak_util":     round(cong["peak_util"], 4),
            "mean_util":     round(cong["mean_util"], 4),
            "monthly_hours": cong["monthly_hours"],
            "hourly_utilisation": [round(float(v), 4) for v in cong["utilisation"]],
        }

        result = {
            "meta":            meta,
            "capacities":      capacities,
            "annual_energy":   annual_energy,
            "storage":         storage_out,
            "kpis":            kpis,
            "dispatch":        dispatch,
            "grid_congestion": grid_congestion,
        }

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True
        )
        with open(filepath, "w", encoding="utf-8") as fh:
            _json.dump(result, fh, indent=2)
        print(f"JSON results exported to: {filepath}")
        return filepath


    def export_dashboard_json(self, filepath="results/dashboard_results.json", geo=None):
        """
        Export a structured JSON file ready for the HTML dashboard.

        Schema additions (v2) vs v1
        ----------------------------
        capacities[tech]
            potential_mw      Max_Capacity_MW from resource assessment (MW)
        lcoe_summary[tech]
            annualised_capex  Annual capital cost component (EUR/yr)
            annual_opex       Annual fixed O&M cost (EUR/yr)
            annual_fuel_cost  Annual variable fuel cost (EUR/yr)
            total_capex       Overnight capital cost, non-annualised (EUR)
        lcoe_summary[_system]
            total_annualised_capex  System total annualised CAPEX (EUR/yr)
            total_annual_opex       System total fixed OPEX (EUR/yr)
            total_annual_fuel       System total fuel cost (EUR/yr)
            total_capex             System overnight CAPEX (EUR)
        grid_congestion
            threshold         Utilisation ratio threshold
            hours_above       Number of hours above threshold
            peak_util         Peak hourly utilisation ratio
            mean_util         Annual mean utilisation ratio
            monthly_hours     Congested hours per month (12 values)
            hourly_utilisation  Full 8760 hourly utilisation array
        """
        n   = self.network
        r   = self.setup.discount_rate
        cur = getattr(self.setup, "currency", "EUR")

        gross_demand_mwh = float(n.loads_t.p_set["Demand"].sum())
        net_demand_mwh   = float(sum(self.ts.demand))

        meta = {
            "objective":        self.setup.objective,
            "discount_rate":    r,
            "currency":         cur,
            "grid_loss_factor": GRID_LOSS_FACTOR,
            "technologies": {
                "generation": self.setup.selected_gen,
                "balancing":  self.setup.selected_balancing,
                "storage":    self.setup.selected_storage,
            },
        }

        # ── Capacities (+ potential from resource assessment) ─────────────
        capacities = {}
        for g in n.generators.index:
            p_res = self.resources.get(g)
            capacities[g] = {
                "type":         "generator",
                "capacity_mw":  round(float(n.generators.at[g, "p_nom_opt"]), 3),
                "potential_mw": round(max(float(p_res["Max_Capacity_MW"]), 0.0), 3),
            }
        for s in n.storage_units.index:
            p_nom = float(n.storage_units.at[s, "p_nom_opt"])
            mhrs  = self.setup.max_storage_hours.get(s, 4)
            p_res_s = self.resources.get(s)
            capacities[s] = {
                "type":         "storage",
                "power_mw":     round(p_nom, 3),
                "energy_mwh":   round(p_nom * mhrs, 3),
                "max_hours":    mhrs,
                "potential_mw": round(max(float(p_res_s["Max_Capacity_MW"]), 0.0), 3),
            }

        # ── Energy mix + cost component tracking ─────────────────────────
        energy_mix     = {}
        total_ann_cost = 0.0
        total_co2      = 0.0
        _cc            = {}   # cost components per tech

        for g in n.generators.index:
            p_nom   = float(n.generators.at[g, "p_nom_opt"])
            gen_mwh = float(n.generators_t.p[g].sum())
            gen_gwh = gen_mwh / 1e3
            p       = self.resources.get(g)
            crf     = IslandEnergyPyPSA._crf(r, p["Lifetime"])
            real_mc = p["Fuel_Cost"] / p["Efficiency"] if p["Efficiency"] > 0 else 0.0

            # Decomposed cost components
            ann_capex_g   = p_nom * p["Investment_per_MW"] * crf
            ann_opex_g    = p_nom * p["O&M_per_MW_yr"]
            ann_fuel_g    = gen_mwh * real_mc
            total_capex_g = p_nom * p["Investment_per_MW"]
            ann_c         = ann_capex_g + ann_opex_g + ann_fuel_g

            total_ann_cost += ann_c
            cf    = gen_mwh / (p_nom * HOURS_PER_YEAR) if p_nom > 0 else 0.0
            lcoe  = ann_c / gen_mwh if gen_mwh > 0 else None
            co2_t = gen_mwh * float(p.get("CO2_per_MWh", 0.0))
            total_co2 += co2_t

            if g in n.generators_t.p_max_pu.columns:
                avail_mwh = float((p_nom * n.generators_t.p_max_pu[g]).sum())
                curt_gwh  = round(max(0.0, avail_mwh - gen_mwh) / 1e3, 3)
            else:
                curt_gwh  = 0.0

            energy_mix[g] = {
                "annual_gwh":      round(gen_gwh, 3),
                "share_pct":       round(100 * gen_mwh / gross_demand_mwh, 2) if gross_demand_mwh else 0,
                "capacity_factor": round(cf, 4),
                "lcoe_per_mwh":    round(lcoe, 2) if lcoe is not None else None,
                "annualised_cost": round(ann_c, 0),
                "co2_tco2":        round(co2_t, 1),
                "curtailment_gwh": curt_gwh,
            }
            _cc[g] = {
                "annualised_capex": round(ann_capex_g, 0),
                "annual_opex":      round(ann_opex_g, 0),
                "annual_fuel_cost": round(ann_fuel_g, 0),
                "total_capex":      round(total_capex_g, 0),
            }

        for s in n.storage_units.index:
            p_nom   = float(n.storage_units.at[s, "p_nom_opt"])
            mhrs    = self.setup.max_storage_hours.get(s, 4)
            dis     = float(self._stor_discharge(s).sum())
            chg     = float(self._stor_charge(s).sum())
            dis_gwh = dis / 1e3
            chg_gwh = chg / 1e3
            p       = self.resources.get(s)
            crf     = IslandEnergyPyPSA._crf(r, p["Lifetime"])

            # Decomposed cost components for storage
            ann_capex_s   = (p_nom * p["Investment_per_MW"] * crf
                             + p_nom * mhrs * p["Storage_MWh"] * crf)
            ann_opex_s    = p_nom * p["O&M_per_MW_yr"]
            total_capex_s = p_nom * p["Investment_per_MW"] + p_nom * mhrs * p["Storage_MWh"]
            ann_c         = ann_capex_s + ann_opex_s   # no fuel for storage

            total_ann_cost += ann_c
            lcos = ann_c / dis if dis > 0 else None
            rte  = dis / chg  if chg > 0 else 0.0

            energy_mix[s] = {
                "annual_gwh":      round(dis_gwh, 3),
                "share_pct":       round(100 * dis / gross_demand_mwh, 2) if gross_demand_mwh else 0,
                "discharge_gwh":   round(dis_gwh, 3),
                "charge_gwh":      round(chg_gwh, 3),
                "rte":             round(rte, 4),
                "lcos_per_mwh":    round(lcos, 2) if lcos is not None else None,
                "annualised_cost": round(ann_c, 0),
            }
            _cc[s] = {
                "annualised_capex": round(ann_capex_s, 0),
                "annual_opex":      round(ann_opex_s, 0),
                "annual_fuel_cost": 0,
                "total_capex":      round(total_capex_s, 0),
            }

        # ── LCOE summary (with cost breakdown) ───────────────────────────
        system_lcoe_per_mwh = total_ann_cost / gross_demand_mwh if gross_demand_mwh else 0.0
        lcoe_summary = {}
        for g in n.generators.index:
            cc = _cc.get(g, {})
            lcoe_summary[g] = {
                "lcoe_per_mwh":     energy_mix[g]["lcoe_per_mwh"],
                "annualised_cost":  energy_mix[g]["annualised_cost"],
                "annualised_capex": cc.get("annualised_capex", 0),
                "annual_opex":      cc.get("annual_opex", 0),
                "annual_fuel_cost": cc.get("annual_fuel_cost", 0),
                "total_capex":      cc.get("total_capex", 0),
            }
        for s in n.storage_units.index:
            cc = _cc.get(s, {})
            lcoe_summary[s] = {
                "lcos_per_mwh":     energy_mix[s]["lcos_per_mwh"],
                "annualised_cost":  energy_mix[s]["annualised_cost"],
                "annualised_capex": cc.get("annualised_capex", 0),
                "annual_opex":      cc.get("annual_opex", 0),
                "annual_fuel_cost": cc.get("annual_fuel_cost", 0),
                "total_capex":      cc.get("total_capex", 0),
            }

        # System-level cost aggregates
        sys_ann_capex   = sum(v["annualised_capex"] for v in _cc.values())
        sys_ann_opex    = sum(v["annual_opex"]      for v in _cc.values())
        sys_ann_fuel    = sum(v["annual_fuel_cost"]  for v in _cc.values())
        sys_total_capex = sum(v["total_capex"]       for v in _cc.values())

        lcoe_summary["_system"] = {
            "system_lcoe_per_mwh":    round(system_lcoe_per_mwh, 2),
            "total_annualised_cost":  round(total_ann_cost, 0),
            "total_demand_mwh":       round(gross_demand_mwh, 1),
            "total_annualised_capex": round(sys_ann_capex, 0),
            "total_annual_opex":      round(sys_ann_opex, 0),
            "total_annual_fuel":      round(sys_ann_fuel, 0),
            "total_capex":            round(sys_total_capex, 0),
        }

        # ── CO₂ summary ──────────────────────────────────────────────────
        co2_summary = {}
        for g in n.generators.index:
            co2_summary[g] = {"annual_tco2": energy_mix[g]["co2_tco2"]}
        co2_summary["_system"] = {
            "total_tco2": round(total_co2, 1),
            "emission_intensity_gco2_per_kwh": round(
                total_co2 / net_demand_mwh * 1000 if net_demand_mwh else 0.0, 4
            ),
        }

        # ── Hourly timeseries ─────────────────────────────────────────────
        gross_arr = n.loads_t.p_set["Demand"].values
        net_arr   = self.ts.demand

        hourly = []
        for h in range(len(n.snapshots)):
            row = {
                "hour":            h,
                "gross_demand_mw": round(float(gross_arr[h]), 3),
                "net_demand_mw":   round(float(net_arr[h]),   3),
            }
            for g in n.generators.index:
                row[f"gen_{g}_mw"] = round(float(n.generators_t.p[g].iloc[h]), 3)
            for s in n.storage_units.index:
                p_val = float(n.storage_units_t.p[s].iloc[h])
                row[f"dis_{s}_mw"]  = round(max(p_val,  0.0), 3)
                row[f"chg_{s}_mw"]  = round(max(-p_val, 0.0), 3)
                row[f"soc_{s}_mwh"] = round(
                    float(n.storage_units_t.state_of_charge[s].iloc[h]), 3
                )
            hourly.append(row)

        # ── Geographic (optional) ─────────────────────────────────────────
        geographic = None
        if geo is not None and geo.data is not None:
            row_g = geo.data.iloc[0]
            geographic = {
                "name":         str(row_g.get("Name", "")),
                "latitude":     round(float(row_g["Latitude"]),  6),
                "longitude":    round(float(row_g["Longitude"]), 6),
                "max_height_m": float(row_g["Max_Height_m"]),
            }

        # ── Grid Congestion ───────────────────────────────────────────────
        cong = self.compute_congestion()
        grid_congestion = {
            "threshold":           cong["threshold"],
            "hours_above":         cong["hours_above"],
            "peak_util":           round(cong["peak_util"], 4),
            "mean_util":           round(cong["mean_util"], 4),
            "monthly_hours":       cong["monthly_hours"],
            "hourly_utilisation":  [round(float(v), 4) for v in cong["utilisation"]],
        }

        output = {
            "meta":            meta,
            "capacities":      capacities,
            "energy_mix":      energy_mix,
            "lcoe_summary":    lcoe_summary,
            "co2_summary":     co2_summary,
            "hourly":          hourly,
            "grid_congestion": grid_congestion,
        }
        if geographic is not None:
            output["geographic"] = geographic

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True
        )
        with open(filepath, "w", encoding="utf-8") as fh:
            _json.dump(output, fh, indent=2)
        print(f"\u2705 Dashboard JSON exported \u2192 {filepath}")
        return filepath

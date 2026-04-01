import os
import json as _json
import math
import numpy as np
import pandas as pd
import pypsa

from src.constants import (
    HOURS_PER_YEAR, GRID_LOSS_FACTOR, SOC_MIN_FRACTION,
    CURTAILMENT_PENALTY, STORAGE_CHARGE_COST,
    CARBON_SHADOW_PRICE, DIVERSIFIED_MIN_MW, CONGESTION_THRESHOLD,
    CARBON_TAX,
)


class IslandEnergyPyPSA:
    """
    LP investment + dispatch model for an island / microgrid energy system.

    Built using PyPSA with the HiGHS solver (via linopy).
    """

    def __init__(self, setup, resources, ts, geo=None):
        self.setup     = setup
        self.resources = resources
        self.ts        = ts
        self.geo       = geo
        self.network   = None

    @staticmethod
    def _crf(rate, years):
        """Capital Recovery Factor: annualises CAPEX over asset lifetime."""
        if years <= 0:
            raise ValueError(f"Asset lifetime must be positive, got {years}")
        if rate <= -1.0:
            raise ValueError(f"Discount rate must be > -1, got {rate}")

        r, n = float(rate), int(years)

        if abs(r) < 1e-10:
            return 1.0 / n

        factor = (1.0 + r) ** n
        return (r * factor) / (factor - 1.0)

    @staticmethod
    def _effective_ann_capex(investment_per_mw, rate, asset_lifetime, project_lifetime):
        """
        Annualised CAPEX per MW that accounts for reinvestment when an asset's
        lifetime is shorter than the project horizon.

        

        Parameters
        ----------
        investment_per_mw  : float  — overnight CAPEX per MW (or per MWh for energy)
        rate               : float  — annual discount rate (e.g. 0.08)
        asset_lifetime     : int    — asset economic lifetime in years
        project_lifetime   : int    — planning horizon in years

        Returns
        -------
        float  — effective annualised CAPEX per MW per year
        """
        r = float(rate)
        L = int(asset_lifetime)
        T = int(project_lifetime)

        if L <= 0 or T <= 0:
            raise ValueError("Lifetimes must be positive integers.")

        # Sum present value of each reinvestment, discounted to year 0.
        # Purchase at year 0, L, 2L, …  (stop once we have passed T–1).
        pv_total = 0.0
        purchase_year = 0
        while purchase_year < T:
            if r < 1e-10:
                discount = 1.0
            else:
                discount = 1.0 / (1.0 + r) ** purchase_year
            pv_total += investment_per_mw * discount
            purchase_year += L

        # Annualise the total PV over the full project horizon
        project_crf = IslandEnergyPyPSA._crf(r, T)
        return pv_total * project_crf

    def build(self):
        """Construct the PyPSA network with all generators, storage, and load."""
        r             = self.setup.discount_rate
        proj_life     = getattr(self.setup, "project_lifetime", 25)
        gen_techs     = self.setup.selected_gen + self.setup.selected_balancing
        storage_types = self.setup.selected_storage

        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(HOURS_PER_YEAR, name="hour"))

        n.add("Carrier", "AC")
        for tech in gen_techs:
            n.add("Carrier", tech)
        for s in storage_types:
            n.add("Carrier", s)

        n.add("Bus", "Island", carrier="AC")

        effective_demand = self.ts.demand * (1.0 + GRID_LOSS_FACTOR)
        n.add("Load", "Demand", bus="Island",
              p_set=pd.Series(effective_demand, index=n.snapshots))

        NON_FLEXIBLE_TECHS = {"Biomass", "Biogas", "Geothermal", "WTE"}
        self._non_flex_profiles = {}   # tech → numpy array or scalar

        for tech in gen_techs:
            p   = self.resources.get(tech)
            crf = self._crf(r, p["Lifetime"])

            ann_capital = (
                self._effective_ann_capex(
                    p["Investment_per_MW"], r, p["Lifetime"], proj_life
                )
                + p["O&M_per_MW_yr"]
            )

            # Availability profile (p_max_pu):
            if tech in self.ts.generation:
                raw_profile = self.ts.generation[tech]
                p_max_pu    = pd.Series(
                    np.clip(raw_profile, 0.0, 1.0), index=n.snapshots
                )
            else:
                p_max_pu = 1.0

            real_mc = p["Fuel_Cost"] / p["Efficiency"] if p["Efficiency"] > 0 else 0.0
            if p["Fuel_Cost"] == 0:
                mc = real_mc + CURTAILMENT_PENALTY
            else:
                mc = real_mc
            if tech.lower() not in {"grid", "import", "electricity grid"}:
                mc += float(p.get("CO2_per_MWh", 0.0)) * float(CARBON_TAX)

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

            if tech in NON_FLEXIBLE_TECHS:
                if isinstance(p_max_pu, pd.Series):
                    self._non_flex_profiles[tech] = p_max_pu.values  # numpy (8760,)
                else:
                    self._non_flex_profiles[tech] = float(p_max_pu)  # scalar

        for s in storage_types:
            p    = self.resources.get(s)
            mhrs = self.setup.max_storage_hours.get(s, 4)

            # ── Independent power / energy optimisation via Bus+Store+2Links ──
            #
            # We model each storage technology as:
            #   Island ──[charge Link]──► s_Bus ──[Store]
            #   Island ◄─[discharge Link]── s_Bus
            #
            # This gives the LP two independent extendable variables:
            #   • charge/discharge Link  →  p_nom_opt  (MW, power capacity)
            #   • Store                  →  e_nom_opt  (MWh, energy capacity)
            #
            # The ceiling  e_nom ≤ p_nom_charge × max_hours  is enforced via
            # extra_functionality so the solver is free to choose any E/P
            # ratio up to the user-specified ceiling.

            s_bus = f"{s}_bus"
            n.add("Bus", s_bus, carrier=s)

            # Efficiency: resource assessment gives round-trip efficiency.
            # Each Link applies sqrt(RTE) so end-to-end RTE = target.
            rte_target        = float(p["Efficiency"])
            eff_per_direction = math.sqrt(rte_target)

            p_nom_max_val = p["Max_Capacity_MW"]
            p_nom_min_val = DIVERSIFIED_MIN_MW if self.setup.objective == "Most Diversified" else 0.0

            # Power CAPEX + O&M annualised with reinvestment → charge Link.
            ann_power_cost = (
                self._effective_ann_capex(
                    p["Investment_per_MW"], r, p["Lifetime"], proj_life
                )
                + p["O&M_per_MW_yr"]
            )

            # Energy CAPEX annualised with reinvestment → Store (per MWh).
            ann_energy_capex = self._effective_ann_capex(
                p["Storage_MWh"], r, p["Lifetime"], proj_life
            )

            # Charge Link: Island → s_Bus
            n.add("Link", f"{s}_charge",
                  bus0             = "Island",
                  bus1             = s_bus,
                  carrier          = s,
                  p_nom_extendable = True,
                  p_nom_min        = p_nom_min_val,
                  p_nom_max        = p_nom_max_val,
                  efficiency       = eff_per_direction,
                  capital_cost     = ann_power_cost,
                  marginal_cost    = STORAGE_CHARGE_COST)

            # Discharge Link: s_Bus → Island (no additional capital cost)
            n.add("Link", f"{s}_discharge",
                  bus0             = s_bus,
                  bus1             = "Island",
                  carrier          = s,
                  p_nom_extendable = True,
                  p_nom_min        = 0.0,
                  p_nom_max        = p_nom_max_val,
                  efficiency       = eff_per_direction,
                  capital_cost     = 0.0,
                  marginal_cost    = 0.0)

            # Store: energy reservoir on s_Bus.
            # e_nom_max = p_nom_max × max_hours is a hard ceiling on energy;
            # the LP will optimise e_nom freely between 0 and this value.
            e_nom_max_val = p_nom_max_val * mhrs
            n.add("Store", s,
                  bus              = s_bus,
                  carrier          = s,
                  e_nom_extendable = True,
                  e_nom_min        = 0.0,
                  e_nom_max        = e_nom_max_val,
                  e_cyclic         = True,
                  capital_cost     = ann_energy_capex,
                  e_min_pu         = SOC_MIN_FRACTION)

        self.network = n

    def _add_soc_floor(self, n, snapshots):
        """
        Minimum state-of-charge is handled declaratively via e_min_pu on
        each Store component (set to SOC_MIN_FRACTION in build()). This method
        is retained as a no-op so the hook call in solve() remains unchanged.
        """
        pass

    def _add_ep_ceiling(self, n, snapshots):
        """
        E/P ceiling constraint (extra_functionality hook).

        Enforces  e_nom(s) ≤ p_nom_charge(s) × max_hours  so that the
        optimised energy capacity cannot exceed the power-based ceiling even
        though power and energy are sized independently.
        """
        import xarray as xr
        m = n.model
        for s in self.setup.selected_storage:
            mhrs = self.setup.max_storage_hours.get(s, 4)
            try:
                e_nom     = m["Store-e_nom"].sel(name=s)
                p_nom_c   = m["Link-p_nom"].sel(name=f"{s}_charge")
                p_nom_d   = m["Link-p_nom"].sel(name=f"{s}_discharge")

                # E/P ceiling: energy capacity ≤ power capacity × max_hours
                m.add_constraints(
                    e_nom <= p_nom_c * mhrs,
                    name=f"ep_ceiling_{s}"
                )
                # Symmetry: discharge power capacity = charge power capacity
                # (same physical inverter/converter limits both directions)
                m.add_constraints(
                    p_nom_d == p_nom_c,
                    name=f"pnom_sym_{s}"
                )
            except Exception as e:
                print(f"  WARNING: E/P ceiling constraint for {s} skipped: {e}")

    def _add_non_flexible_floor(self, n, snapshots):
        """
        Non-flexible dispatch constraint (extra_functionality hook).

        Adds  p[g,t] >= profile[t] * p_nom[g]  directly to the linopy model
        for Biomass, Biogas, Geothermal, and WTE so HiGHS pins dispatch to
        exactly  profile(t) * p_nom_opt  at every hour.

        """
        profiles = getattr(self, "_non_flex_profiles", {})
        if not profiles:
            return
        import xarray as xr
        m = n.model

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
            self._add_ep_ceiling(n, snapshots)       # e_nom ≤ p_nom_charge × max_hours
            self._add_non_flexible_floor(n, snapshots)

        status, condition = n.optimize(
            solver_name              = "highs",
            solver_options           = opts,
            extra_functionality      = _extra_functionality,
            include_objective_constant = False
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
        if self.setup.selected_storage:
            print("\n  Storage capacities:")
            for s in self.setup.selected_storage:
                pwr = float(n.links.at[f"{s}_charge", "p_nom_opt"]) if f"{s}_charge" in n.links.index else 0.0
                ene = float(n.stores.at[s, "e_nom_opt"])            if s in n.stores.index else 0.0
                ep  = ene / pwr if pwr > 1e-6 else 0.0
                print(f"    {s:<14}: {pwr:>8.2f}  MW  |  {ene:>8.2f}  MWh  (E/P = {ep:.2f} h)")

    def _stor_discharge(self, s):
        """Return discharge power delivered to Island (MW, positive) for storage s.

        PyPSA Link sign convention for {s}_discharge (bus0=s_Bus, bus1=Island):
          p0 > 0  →  withdrawing from s_Bus   (storage emptying)
          p1 < 0  →  injecting into Island    (load served)
        Delivered MW = -p1  (efficiency loss already applied by PyPSA).

        Fallback: if p1 is not stored, reconstruct from p0 × efficiency.
        p0 is always stored as the LP decision variable.
        """
        n   = self.network
        key = f"{s}_discharge"
        if key not in n.links.index:
            return pd.Series(0.0, index=n.snapshots)
        # Prefer p1 (already accounts for efficiency)
        if key in n.links_t.p1.columns:
            return (-n.links_t.p1[key]).clip(lower=0)
        # Fallback: p0 is the flow variable; multiply by efficiency to get delivered power
        if key in n.links_t.p0.columns:
            eff = float(n.links.at[key, "efficiency"])
            return (n.links_t.p0[key] * eff).clip(lower=0)
        return pd.Series(0.0, index=n.snapshots)

    def _stor_charge(self, s):
        """Return charge power drawn from Island (MW, positive) for storage s.

        PyPSA Link sign convention for {s}_charge (bus0=Island, bus1=s_Bus):
          p0 > 0  →  withdrawing from Island  (charging the store)
        Charge MW = p0  (always positive for a unidirectional link).
        """
        n   = self.network
        key = f"{s}_charge"
        if key not in n.links.index:
            return pd.Series(0.0, index=n.snapshots)
        if key in n.links_t.p0.columns:
            return n.links_t.p0[key].clip(lower=0)
        return pd.Series(0.0, index=n.snapshots)

    def _stor_soc(self, s):
        """Return state-of-charge timeseries (MWh) for storage technology s."""
        n = self.network
        if s in n.stores.index and s in n.stores_t.e.columns:
            return n.stores_t.e[s]
        return pd.Series(0.0, index=n.snapshots)

    def _stor_power_mw(self, s):
        """Return optimised power capacity (MW) for storage technology s.

        Power capacity is stored on the charge Link p_nom_opt.
        The discharge Link is sized identically via the ep_ceiling constraint.
        """
        n   = self.network
        key = f"{s}_charge"
        return float(n.links.at[key, "p_nom_opt"]) if key in n.links.index else 0.0

    def _stor_energy_mwh(self, s):
        """Return optimised energy capacity (MWh) for storage technology s."""
        n = self.network
        return float(n.stores.at[s, "e_nom_opt"]) if s in n.stores.index else 0.0

    # ── Storage charging source attribution ──────────────────────────────
    def _stor_charge_by_source(self, s):
        """
        Return a DataFrame (8760 rows × n_generators columns) with the MW
        contribution of each generator to charging storage technology s.

        Method
        ------
        At every hour t, the total charge drawn from the Island bus equals
        _stor_charge(s)[t].  Each active generator's share of that charge is
        proportional to its instantaneous dispatch:

            charge_from_g(t) = charge_total(t) × p_g(t) / Σ p_g(t)

        If total dispatch is zero the charge is split equally (edge case only).
        This is the pro-rata energy attribution used in electricity market
        accounting and endorsed by the GHG Protocol Scope 2 guidance.

        Returns
        -------
        pd.DataFrame  indexed by snapshot, columns = generator names (MW)
        """
        n        = self.network
        charge_s = self._stor_charge(s).values   # shape (8760,)

        gen_dispatch = {}
        for g in n.generators.index:
            if g in n.generators_t.p.columns:
                gen_dispatch[g] = n.generators_t.p[g].values
            else:
                gen_dispatch[g] = np.zeros(HOURS_PER_YEAR)

        total_dispatch = sum(gen_dispatch.values())                 # (8760,)
        total_dispatch = np.maximum(total_dispatch, 1e-9)           # avoid /0

        result = {}
        for g, disp in gen_dispatch.items():
            share            = disp / total_dispatch                # (8760,)
            result[g]        = charge_s * share

        return pd.DataFrame(result, index=n.snapshots)

    def _stor_charge_source_annual(self, s):
        """
        Return a dict {generator_name: annual_MWh_charged} for storage s.

        Used in export_economic_csv() to populate the per-source charging
        columns and in the dashboard JSON for the storage charging breakdown chart.
        """
        df = self._stor_charge_by_source(s)
        return {col: float(df[col].sum()) for col in df.columns}

    # ── Battery cycle counting ────────────────────────────────────────────
    def _stor_annual_cycles(self, s):
        """
        Return the number of full equivalent cycles per year for storage s.

        Definition (IEC 62660 / NREL):
            annual_cycles = total_annual_discharge_MWh / e_nom_usable_MWh

        where e_nom_usable = e_nom_opt × (1 - SOC_MIN_FRACTION).

        This gives a physically meaningful count: 365 cycles/yr ≈ one full
        charge-discharge per day; typical Li-ion warranty is 3000–6000 cycles.
        """
        e_nom = self._stor_energy_mwh(s)
        usable = e_nom * (1.0 - SOC_MIN_FRACTION)
        if usable < 1e-6:
            return 0.0
        annual_discharge = float(self._stor_discharge(s).sum())
        return round(annual_discharge / usable, 1)

    # ── Grid Congestion Analysis ──────────────────────────────────────

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

        for s in self.setup.selected_storage:
            pwr = self._stor_power_mw(s)
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

        for s in self.setup.selected_storage:
            df[f"Discharge_{s}_MW"] = self._stor_discharge(s).values
            df[f"Charge_{s}_MW"]    = self._stor_charge(s).values
            df[f"SOC_{s}_MWh"]      = self._stor_soc(s).values
            charge_src = self._stor_charge_by_source(s)
            for g in charge_src.columns:
                df[f"Charge_{s}_from_{g}_MW"] = charge_src[g].values
            cycles = self._stor_annual_cycles(s)
            print(f"  {s} annual equivalent cycles: {cycles:.1f}")

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
        r         = self.setup.discount_rate
        proj_life = getattr(self.setup, "project_lifetime", 25)
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
        for s in self.setup.selected_storage:
            pwr  = self._stor_power_mw(s)
            ene  = self._stor_energy_mwh(s)
            mhrs = self.setup.max_storage_hours.get(s, 4)
            capacities[s] = {
                "type":        "storage",
                "power_mw":    round(pwr, 4),
                "energy_mwh":  round(ene, 4),
                "ep_ratio_h":  round(ene / pwr, 3) if pwr > 1e-6 else 0.0,
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
        for s in self.setup.selected_storage:
            p    = self.resources.get(s)
            pwr  = self._stor_power_mw(s)
            ene  = self._stor_energy_mwh(s)
            mhrs = self.setup.max_storage_hours.get(s, 4)
            dis  = float(self._stor_discharge(s).sum())
            chg  = float(self._stor_charge(s).sum())
            ann_power_cost = (
                self._effective_ann_capex(p["Investment_per_MW"], r, p["Lifetime"], proj_life)
                + p["O&M_per_MW_yr"]
            )
            ann_energy_cost = self._effective_ann_capex(p["Storage_MWh"], r, p["Lifetime"], proj_life)
            ac   = pwr * ann_power_cost + ene * ann_energy_cost
            total_ac += ac
            rte  = dis / chg if chg > 0 else 0.0
            lcos = ac / dis  if dis > 0 else None
            charge_sources = self._stor_charge_source_annual(s)
            cycles = self._stor_annual_cycles(s)
            storage_out[s] = {
                "power_mw":      round(pwr, 4),
                "energy_mwh":    round(ene, 4),
                "ep_ratio_h":    round(ene / pwr, 3) if pwr > 1e-6 else 0.0,
                "discharge_mwh": round(dis, 2),
                "charge_mwh":    round(chg, 2),
                "charge_by_source_mwh": {k: round(v, 2) for k, v in charge_sources.items()},
                "annual_equivalent_cycles": cycles,
                "rte":           round(rte, 4),
                "ann_cost":      round(ac, 2),
                "lcos":          round(lcos, 4) if lcos is not None else None,
            }

        total_dem   = float(sum(self.ts.demand))
        gross_dem   = total_dem * (1.0 + GRID_LOSS_FACTOR)
        system_lcoe = total_ac / gross_dem if gross_dem > 0 else 0.0
        total_co2   = sum(v["co2_t"] for v in annual_energy.values() if "co2_t" in v)
        total_carbon_tax = total_co2 * float(CARBON_TAX)
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
        for s in self.setup.selected_storage:
            dispatch[f"discharge_{s}_mw"] = [round(float(v), 4) for v in self._stor_discharge(s).values]
            dispatch[f"charge_{s}_mw"]    = [round(float(v), 4) for v in self._stor_charge(s).values]
            dispatch[f"soc_{s}_mwh"]      = [round(float(v), 4) for v in self._stor_soc(s).values]
            charge_src_df = self._stor_charge_by_source(s)
            for g in charge_src_df.columns:
                dispatch[f"charge_{s}_from_{g}_mw"] = [
                    round(float(v), 4) for v in charge_src_df[g].values
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
        storage[s]
            charge_by_source_mwh    Annual MWh charged per generator source
            annual_equivalent_cycles  Full equivalent cycles per year
        grid_congestion
            threshold         Utilisation ratio threshold
            hours_above       Number of hours above threshold
            peak_util         Peak hourly utilisation ratio
            mean_util         Annual mean utilisation ratio
            monthly_hours     Congested hours per month (12 values)
            hourly_utilisation  Full 8760 hourly utilisation array
        geographic
            name, latitude, longitude, max_height_m, project_description
        """
        geo = geo if geo is not None else self.geo

        n   = self.network
        r         = self.setup.discount_rate
        proj_life = getattr(self.setup, "project_lifetime", 25)
        cur = getattr(self.setup, "currency", "EUR")

        gross_demand_mwh = float(n.loads_t.p_set["Demand"].sum())
        net_demand_mwh   = float(sum(self.ts.demand))

        # ── Selected (non-zero capacity) technologies from optimisation ──────
        selected_gen_opt = [
            g for g in n.generators.index
            if float(n.generators.at[g, "p_nom_opt"]) > 1e-3
        ]
        selected_stor_opt = [
            s for s in self.setup.selected_storage
            if self._stor_power_mw(s) > 1e-3
        ]

        # ── Brand / technology specification from resource assessment ─────────
        brand_technologies = {}
        for tech in list(n.generators.index) + list(self.setup.selected_storage):
            p_res = self.resources.get(tech)
            brand = (
                p_res.get("brand_technologies")
                or p_res.get("Brand")
                or p_res.get("Model")
                or p_res.get("Specification")
                or p_res.get("Technology")
                or None
            )
            if brand:
                brand_technologies[tech] = str(brand)

        meta = {
            "objective":           self.setup.objective,
            "discount_rate":       r,
            "currency":            cur,
            "grid_loss_factor":    GRID_LOSS_FACTOR,
            "carbon_tax_rate":     float(CARBON_TAX),
            "technologies": {
                "generation": self.setup.selected_gen,
                "balancing":  self.setup.selected_balancing,
                "storage":    self.setup.selected_storage,
                "selected_gen":  selected_gen_opt,
                "selected_stor": selected_stor_opt,
            },
        }

        # ── Capacities ────────────────────────────────────────────────
        capacities = {}
        for g in n.generators.index:
            p_res = self.resources.get(g)
            capacities[g] = {
                "type":         "generator",
                "capacity_mw":  round(float(n.generators.at[g, "p_nom_opt"]), 3),
                "potential_mw": round(max(float(p_res["Max_Capacity_MW"]), 0.0), 3),
            }
        for s in self.setup.selected_storage:
            pwr   = self._stor_power_mw(s)
            ene   = self._stor_energy_mwh(s)
            mhrs  = self.setup.max_storage_hours.get(s, 4)
            p_res_s = self.resources.get(s)
            capacities[s] = {
                "type":         "storage",
                "power_mw":     round(pwr, 3),
                "energy_mwh":   round(ene, 3),
                "ep_ratio_h":   round(ene / pwr, 3) if pwr > 1e-6 else 0.0,
                "max_hours_ceiling": mhrs,
                "potential_mw": round(max(float(p_res_s["Max_Capacity_MW"]), 0.0), 3),
                "potential_energy_mwh": round(max(float(p_res_s["Max_Capacity_MW"]), 0.0) * mhrs, 3),
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
                "curtailment_gwh": curt_gwh,
            }
            _cc[g] = {
                "annualised_capex": round(ann_capex_g, 0),
                "annual_opex":      round(ann_opex_g, 0),
                "annual_fuel_cost": round(ann_fuel_g, 0),
                "total_capex":      round(total_capex_g, 0),
                "annualised_cost":  round(ann_c, 0),
                "lcoe_per_mwh":     round(lcoe, 2) if lcoe is not None else None,
                "co2_tco2":         round(co2_t, 1),
            }

        for s in self.setup.selected_storage:
            pwr     = self._stor_power_mw(s)
            ene     = self._stor_energy_mwh(s)
            dis     = float(self._stor_discharge(s).sum())
            chg     = float(self._stor_charge(s).sum())
            dis_gwh = dis / 1e3
            chg_gwh = chg / 1e3
            p       = self.resources.get(s)

            ann_power_unit  = (
                self._effective_ann_capex(p["Investment_per_MW"], r, p["Lifetime"], proj_life)
                + p["O&M_per_MW_yr"]
            )
            ann_energy_unit = self._effective_ann_capex(p["Storage_MWh"], r, p["Lifetime"], proj_life)
            ann_capex_s     = (pwr * self._effective_ann_capex(p["Investment_per_MW"], r, p["Lifetime"], proj_life)
                               + ene * ann_energy_unit)
            ann_opex_s      = pwr * p["O&M_per_MW_yr"]
            ann_c           = ann_capex_s + ann_opex_s
            total_capex_s   = pwr * p["Investment_per_MW"] + ene * p["Storage_MWh"]

            total_ann_cost += ann_c
            lcos = ann_c / dis if dis > 0 else None
            rte  = dis / chg   if chg > 0 else 0.0
            cf_s = dis / (pwr * HOURS_PER_YEAR) if pwr > 1e-6 else 0.0

            charge_sources = self._stor_charge_source_annual(s)
            cycles = self._stor_annual_cycles(s)

            energy_mix[s] = {
                "annual_gwh":      round(dis_gwh, 3),
                "share_pct":       round(100 * dis / gross_demand_mwh, 2) if gross_demand_mwh else 0,
                "discharge_gwh":   round(dis_gwh, 3),
                "charge_gwh":      round(chg_gwh, 3),
                "rte":             round(rte, 4),
                "capacity_factor": round(cf_s, 4),
                "charge_by_source_gwh": {k: round(v / 1e3, 3) for k, v in charge_sources.items()},
                "annual_equivalent_cycles": cycles,
            }
            _cc[s] = {
                "annualised_capex": round(ann_capex_s, 0),
                "annual_opex":      round(ann_opex_s, 0),
                "annual_fuel_cost": 0,
                "total_capex":      round(total_capex_s, 0),
                "annualised_cost":  round(ann_c, 0),
                "lcos_per_mwh":     round(lcos, 2) if lcos is not None else None,
            }

        # ── LCOE summary (with cost breakdown) ───────────────────────────
        system_lcoe_per_mwh = total_ann_cost / gross_demand_mwh if gross_demand_mwh else 0.0
        lcoe_summary = {}
        for g in n.generators.index:
            cc = _cc.get(g, {})
            lcoe_summary[g] = {
                "lcoe_per_mwh":     cc.get("lcoe_per_mwh"),
                "annualised_cost":  cc.get("annualised_cost", 0),
                "annualised_capex": cc.get("annualised_capex", 0),
                "annual_opex":      cc.get("annual_opex", 0),
                "annual_fuel_cost": cc.get("annual_fuel_cost", 0),
                "total_capex":      cc.get("total_capex", 0),
            }
        for s in self.setup.selected_storage:
            cc = _cc.get(s, {})
            lcoe_summary[s] = {
                "lcos_per_mwh":     cc.get("lcos_per_mwh"),
                "annualised_cost":  cc.get("annualised_cost", 0),
                "annualised_capex": cc.get("annualised_capex", 0),
                "annual_opex":      cc.get("annual_opex", 0),
                "annual_fuel_cost": cc.get("annual_fuel_cost", 0),
                "total_capex":      cc.get("total_capex", 0),
            }

        sys_ann_capex   = sum(v["annualised_capex"] for v in _cc.values())
        sys_ann_opex    = sum(v["annual_opex"]      for v in _cc.values())
        sys_ann_fuel    = sum(v["annual_fuel_cost"]  for v in _cc.values())
        sys_total_capex = sum(v["total_capex"]       for v in _cc.values())

        lcoe_summary["_system"] = {
            "system_lcoe_per_mwh":    round(system_lcoe_per_mwh, 2),
            "total_annualised_cost":  round(total_ann_cost, 0),
            "total_demand_mwh":       round(gross_demand_mwh, 1),
            "net_demand_mwh":         round(net_demand_mwh, 1),
            "peak_demand_mw":         round(float(n.loads_t.p_set["Demand"].max()), 3),
            "total_annualised_capex": round(sys_ann_capex, 0),
            "total_annual_opex":      round(sys_ann_opex, 0),
            "total_annual_fuel":      round(sys_ann_fuel, 0),
            "total_capex":            round(sys_total_capex, 0),
        }

        # ── CO₂ summary ──────────────────────────────────────────────────
        total_carbon_tax = 0.0
        for g in n.generators.index:
            if g.lower() not in {"grid", "import", "electricity grid"}:
                total_carbon_tax += _cc.get(g, {}).get("co2_tco2", 0.0) * float(CARBON_TAX)

        co2_summary = {}
        for g in n.generators.index:
            co2_summary[g] = {"annual_tco2": _cc[g].get("co2_tco2", 0.0)}
        co2_summary["_system"] = {
            "total_tco2": round(total_co2, 1),
            "total_carbon_tax_eur": round(total_carbon_tax, 0),
            "carbon_tax_rate_eur_per_tco2": CARBON_TAX,
            "emission_intensity_gco2_per_kwh": round(
                total_co2 / net_demand_mwh * 1000 if net_demand_mwh else 0.0, 4
            ),
        }

        # ── Hourly timeseries ─────────────────────────────────────────────
        gross_arr = n.loads_t.p_set["Demand"].values
        net_arr   = self.ts.demand

        VARIABLE_RE        = {"Wind", "Solar"}
        NON_FLEXIBLE_TECHS = {"Biomass", "Biogas", "Geothermal", "WTE"}

        hourly = []
        for h in range(len(n.snapshots)):
            row = {
                "hour":            h,
                "gross_demand_mw": round(float(gross_arr[h]), 3),
                "net_demand_mw":   round(float(net_arr[h]),   3),
            }

            for g in n.generators.index:
                if g in VARIABLE_RE or g in NON_FLEXIBLE_TECHS:
                    p_nom = float(n.generators.at[g, "p_nom_opt"])
                    if g in n.generators_t.p_max_pu.columns:
                        prod = p_nom * float(n.generators_t.p_max_pu[g].iloc[h])
                    else:
                        prod = p_nom
                else:
                    prod = float(n.generators_t.p[g].iloc[h])
                row[f"gen_{g}_mw"] = round(float(prod), 3)

            for s in self.setup.selected_storage:
                row[f"dis_{s}_mw"]  = round(float(self._stor_discharge(s).iloc[h]), 3)
                row[f"chg_{s}_mw"]  = round(float(self._stor_charge(s).iloc[h]),    3)
                row[f"soc_{s}_mwh"] = round(float(self._stor_soc(s).iloc[h]),       3)
            hourly.append(row)

        # ── Geographic ───────────────────────────────────────────────────────
        geographic = None
        if geo is not None and hasattr(geo, 'data') and geo.data is not None:
            if len(geo.data) > 0:
                try:
                    row_g = geo.data.iloc[0]
                    required_cols = {"Latitude", "Longitude", "Max_Height_m"}
                    missing = required_cols - set(row_g.index)

                    if missing:
                        print(f"  WARNING: Geographic data missing columns: {missing}")
                    else:
                        geographic = {
                            "name":                str(row_g.get("Name", "Unknown")),
                            "latitude":            round(float(row_g["Latitude"]),  6),
                            "longitude":           round(float(row_g["Longitude"]), 6),
                            "max_height_m":        float(row_g["Max_Height_m"]),
                            "project_description": str(row_g.get("Project_Description", "")),
                        }
                except (KeyError, ValueError, IndexError) as e:
                    print(f"  WARNING: Failed to parse geographic data: {e}")

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
            "meta":               meta,
            "capacities":         capacities,
            "energy_mix":         energy_mix,
            "lcoe_summary":       lcoe_summary,
            "co2_summary":        co2_summary,
            "brand_technologies": brand_technologies,
            "hourly":             hourly,
            "grid_congestion":    grid_congestion,
        }
        if geographic is not None:
            output["geographic"] = geographic

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True
        )
        with open(filepath, "w", encoding="utf-8") as fh:
            _json.dump(output, fh, indent=2)
        print(f"✓ Dashboard JSON exported → {filepath}")
        return filepath

    # ── Export: Economic CSV ──────────────────────────────────────────────

    def export_economic_csv(self, filepath="results/result_optimisation_economic.csv", geo=None):
        """
        Export a flat CSV for downstream economic valuation analysis.

        Column order and semantics:
          Project_name, Project_Description, Currency, Discount_rate, Lifetime,
          Technology, Capacity, Energy_MWh, EP_ratio_h, Capacity_Factor,
          Energy_production, Delivered_energy, Charging_MWh,
          Charging_from_<gen>_MWh  (one column per generator, storage rows only),
          Curtailed_energy, CAPEX, OPEX, Fuel_cost,
          Degradation_rate, Annual_cycles (storage only),
          PPA_price, Price_post_PPA, latitude, longitude, CO2_emitted_ton
        """
        geo = geo if geo is not None else self.geo

        n         = self.network
        r         = self.setup.discount_rate
        cur       = getattr(self.setup, "currency", "EUR")
        proj      = getattr(self.setup, "project_name", "")
        proj_desc = getattr(self.setup, "project_description", "")

        # Pull from geo if setup doesn't have them
        if geo is not None and hasattr(geo, "data") and geo.data is not None and len(geo.data) > 0:
            try:
                row_g = geo.data.iloc[0]
                if not proj:
                    proj = str(row_g.get("Name", ""))
                if not proj_desc:
                    proj_desc = str(row_g.get("Project_Description", ""))
            except Exception:
                pass

        # ── Resolve project latitude / longitude from geographic data ─────
        proj_lat = ""
        proj_lon = ""
        if geo is not None and hasattr(geo, "data") and geo.data is not None:
            if len(geo.data) > 0:
                try:
                    row_g    = geo.data.iloc[0]
                    proj_lat = round(float(row_g["Latitude"]),  6)
                    proj_lon = round(float(row_g["Longitude"]), 6)
                except (KeyError, ValueError, TypeError):
                    proj_lat = ""
                    proj_lon = ""

        # Collect all generator names for per-source charging columns
        all_gens = list(n.generators.index)

        rows = []

        # ── Generators ───────────────────────────────────────────────────
        for g in n.generators.index:
            p_nom = float(n.generators.at[g, "p_nom_opt"])
            if p_nom < 1e-6:
                continue

            p       = self.resources.get(g)
            gen_mwh = float(n.generators_t.p[g].sum())
            cf      = gen_mwh / (p_nom * HOURS_PER_YEAR) if p_nom > 0 else 0.0

            if g in n.generators_t.p_max_pu.columns:
                avail_mwh = float((p_nom * n.generators_t.p_max_pu[g]).sum())
                curt_mwh  = max(0.0, avail_mwh - gen_mwh)
            else:
                curt_mwh = 0.0

            total_capex = p_nom * p["Investment_per_MW"]
            ann_opex    = p_nom * p["O&M_per_MW_yr"]
            real_mc     = p["Fuel_Cost"] / p["Efficiency"] if p["Efficiency"] > 0 else 0.0
            ann_fuel    = gen_mwh * real_mc
            delivered_mwh = gen_mwh / (1.0 + GRID_LOSS_FACTOR)
            co2_ton = round(gen_mwh * float(p.get("CO2_per_MWh", 0.0)), 1)

            row = {
                "Technology":        g,
                "Lifetime":          int(p["Lifetime"]),
                "Capacity":          round(p_nom, 2),
                "Energy_MWh":        "",      # generators have no energy reservoir
                "EP_ratio_h":        "",
                "Capacity_Factor":   round(cf, 4),
                "Energy_production": round(gen_mwh, 2),
                "Delivered_energy":  round(delivered_mwh, 2),
                "Charging_MWh":      0.0,
                "Curtailed_energy":  round(curt_mwh, 2),
                "CAPEX":             round(total_capex, 0),
                "OPEX":              round(ann_opex, 0),
                "Fuel_cost":         round(ann_fuel, 0),
                "Degradation_rate":  1,
                "Annual_cycles":     "",     # not applicable for generators
                "PPA_price":         "",
                "Price_post_PPA":    "",
                "CO2_emitted_ton":   co2_ton,
            }
            for gen_name in all_gens:
                row[f"Charging_from_{gen_name}_MWh"] = 0.0
            rows.append(row)

        # ── Storage ──────────────────────────────────────────────────────
        for s in self.setup.selected_storage:
            pwr = self._stor_power_mw(s)
            ene = self._stor_energy_mwh(s)
            if pwr < 1e-6:
                continue

            p    = self.resources.get(s)
            dis  = float(self._stor_discharge(s).sum())
            chg  = float(self._stor_charge(s).sum())
            cf_s = dis / (pwr * HOURS_PER_YEAR) if pwr > 1e-6 else 0.0

            total_capex_s = pwr * p["Investment_per_MW"] + ene * p["Storage_MWh"]
            ann_opex_s    = pwr * p["O&M_per_MW_yr"]

            cycles = self._stor_annual_cycles(s)

            charge_sources = self._stor_charge_source_annual(s)

            delivered_dis = dis / (1.0 + GRID_LOSS_FACTOR)

            row = {
                "Technology":        s,
                "Lifetime":          int(p["Lifetime"]),
                "Capacity":          round(pwr, 2),
                "Energy_MWh":        round(ene, 2),
                "EP_ratio_h":        round(ene / pwr, 3) if pwr > 1e-6 else 0.0,
                "Capacity_Factor":   round(cf_s, 4),
                "Energy_production": round(dis, 2),
                "Delivered_energy":  round(delivered_dis, 2),
                "Charging_MWh":      round(chg, 2),
                "Curtailed_energy":  0.0,
                "CAPEX":             round(total_capex_s, 0),
                "OPEX":              round(ann_opex_s, 0),
                "Fuel_cost":         0.0,
                "Degradation_rate":  2,
                "Annual_cycles":     cycles,
                "PPA_price":         "",
                "Price_post_PPA":    "",
                "CO2_emitted_ton":   0.0,
            }
            for gen_name in all_gens:
                row[f"Charging_from_{gen_name}_MWh"] = round(
                    charge_sources.get(gen_name, 0.0), 2
                )
            rows.append(row)

        # ── Portfolio summary row ─────────────────────────────────────────
        sys_capex   = sum(float(row["CAPEX"])             for row in rows)
        sys_opex    = sum(float(row["OPEX"])              for row in rows)
        sys_fuel    = sum(float(row["Fuel_cost"])         for row in rows)
        sys_energy  = sum(float(row["Energy_production"]) for row in rows)
        sys_deliv   = sum(float(row["Delivered_energy"])  for row in rows)
        sys_chg     = sum(float(row["Charging_MWh"])      for row in rows)
        sys_curt    = sum(float(row["Curtailed_energy"])  for row in rows)
        sys_co2     = sum(float(row["CO2_emitted_ton"])   for row in rows)

        portfolio_row = {
            "Technology":        "Portofolio",
            "Lifetime":          "",
            "Capacity":          "",
            "Energy_MWh":        "",
            "EP_ratio_h":        "",
            "Capacity_Factor":   "",
            "Energy_production": round(sys_energy, 2),
            "Delivered_energy":  round(sys_deliv,  2),
            "Charging_MWh":      round(sys_chg,    2),
            "Curtailed_energy":  round(sys_curt,   2),
            "CAPEX":             round(sys_capex,  0),
            "OPEX":              round(sys_opex,   0),
            "Fuel_cost":         round(sys_fuel,   0),
            "Degradation_rate":  0,
            "Annual_cycles":     "",
            "PPA_price":         "",
            "Price_post_PPA":    "",
            "CO2_emitted_ton":   round(sys_co2,    1),
        }
        for gen_name in all_gens:
            col = f"Charging_from_{gen_name}_MWh"
            portfolio_row[col] = round(
                sum(float(row.get(col, 0.0)) for row in rows), 2
            )
        rows.append(portfolio_row)

        # ── Assemble DataFrame with exact column order ────────────────────
        n_tech         = len(rows)
        project_names  = [proj]      + [""] * (n_tech - 1)
        descriptions   = [proj_desc] + [""] * (n_tech - 1)
        currencies     = [cur]       + [""] * (n_tech - 1)
        discount_rates = [round(r, 6)] * n_tech
        latitudes      = [proj_lat]  + [""] * (n_tech - 1)
        longitudes     = [proj_lon]  + [""] * (n_tech - 1)

        df = pd.DataFrame(rows)
        df.insert(0, "Discount_rate",       discount_rates)
        df.insert(0, "Currency",            currencies)
        df.insert(0, "Project_Description", descriptions)
        df.insert(0, "Project_name",        project_names)

        df["latitude"]  = latitudes
        df["longitude"] = longitudes

        # Build the exact column order dynamically so per-source columns are
        # sandwiched between Charging_MWh and Curtailed_energy
        source_cols = [f"Charging_from_{g}_MWh" for g in all_gens]

        col_order = (
            ["Project_name", "Project_Description", "Currency", "Discount_rate",
             "Lifetime", "Technology",
             "Capacity", "Energy_MWh", "EP_ratio_h",
             "Capacity_Factor",
             "Energy_production", "Delivered_energy",
             "Charging_MWh"]
            + source_cols
            + ["Curtailed_energy",
               "CAPEX", "OPEX", "Fuel_cost",
               "Degradation_rate", "Annual_cycles",
               "PPA_price", "Price_post_PPA",
               "latitude", "longitude", "CO2_emitted_ton"]
        )

        # Ensure all expected columns exist (fillna for any that weren't set)
        for col in col_order:
            if col not in df.columns:
                df[col] = ""

        df = df[col_order]

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True
        )
        df.to_csv(filepath, index=False)
        n_techs = len(rows) - 1
        print(f"✓ Economic CSV exported → {filepath}  ({n_techs} technologies + portfolio row)")
        return filepath

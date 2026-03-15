import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go

from src.constants import (
    HOURS_PER_YEAR, GRID_LOSS_FACTOR, SOC_MIN_FRACTION, CONGESTION_THRESHOLD,
)
from src.model import IslandEnergyPyPSA

# ── CELL: Step 8 — Results Visualisation ─────────────────────────────────────

class ResultsVisualization:
    """
    Professional visualisation of island energy PyPSA optimisation results.

    All charts share the global rcParams style from Step 1.
    The _style_ax() helper applies consistent finishing to every axes.
    Results are read directly from the solved pypsa.Network object.
    The currency symbol is taken from setup.currency and applied to all
    cost labels, axis titles, and summary print statements automatically.
    """

    # ── Color palette ──────────────────────────────────────────────────────
    TECH_COLORS = {
        "Wind":           "#2563EB",   # saturated blue
        "Solar":          "#F59E0B",   # amber
        "Biomass":        "#16A34A",   # forest green
        "Biogas":         "#84CC16",   # lime
        "Geothermal":     "#DC2626",   # warm red
        "Hydro":          "#0891B2",   # deep cyan
        "WTE":            "#57534E",   # stone grey
        "PHS":            "#7C3AED",   # violet
        "BESS":           "#059669",   # emerald
        "Natural Gas":    "#9CA3AF",   # neutral grey
        "Biodiesel":      "#D97706",   # warm amber
        "Hydrogen":       "#06B6D4",   # sky cyan
        "Load":           "#1E293B",   # slate 800
        "Curtailment":    "#EF4444",   # red
        "Losses":         "#CBD5E1",   # slate 300
    }

    STACK_ORDER = [
        "WTE", "Geothermal", "Hydro", "Biomass", "Biogas",
        "Wind", "Solar", "PHS", "BESS", "Hydrogen", "Biodiesel", "Natural Gas"
    ]

    # ── Dispatchable technologies — never treated as VRE ──────────────────
    DISPATCHABLE = {"Geothermal", "Hydro", "Biomass", "Biogas", "WTE", "Natural Gas", "Biodiesel"}

    def __init__(self, model, setup, resources, ts):
        self.network   = model.network
        self.setup     = setup
        self.resources = resources
        self.ts        = ts
        self.cur       = setup.currency   # shorthand used throughout

    # ── Helpers ───────────────────────────────────────────────────────────

    def _parse_hours(self, hours):
        """Accept (start, end) tuple or single int (returns 168-hour window)."""
        return hours if isinstance(hours, tuple) else (hours, hours + 168)

    def _color(self, name):
        return self.TECH_COLORS.get(name, "#aaaaaa")

    def _gen_dispatch(self, tech, start=None, end=None):
        """Return generation dispatch array for a generator, optionally sliced."""
        vals = self.network.generators_t.p[tech].values
        return vals[start:end] if start is not None else vals

    def _stor_discharge(self, s, start=None, end=None):
        """Return storage discharge (positive) array, optionally sliced."""
        net       = self.network.storage_units_t.p[s].values
        discharge = np.maximum(net, 0.0)
        return discharge[start:end] if start is not None else discharge

    def _stor_charge(self, s, start=None, end=None):
        """Return storage charge (positive) array, optionally sliced."""
        net    = self.network.storage_units_t.p[s].values
        charge = np.maximum(-net, 0.0)
        return charge[start:end] if start is not None else charge

    def _is_vre(self, tech):
        """
        Return True if tech is a variable renewable with a resource profile.

        Technologies in DISPATCHABLE are always excluded regardless of whether
        PyPSA registered a p_max_pu profile for them — a flat or near-flat
        profile does not make Geothermal, Hydro, or Biomass a variable
        renewable. Unused dispatchable capacity is standby reserve, not
        curtailment.
        """
        if tech in self.DISPATCHABLE:
            return False
        return tech in self.network.generators_t.p_max_pu.columns

    def _curtailment(self, tech):
        """
        Compute implicit curtailment for VRE technologies only.

        Curtailment = available generation (profile × capacity) minus
        dispatched generation. Only meaningful for variable renewables
        with a time-varying p_max_pu profile. Dispatchable technologies
        return a zero array — their unused capacity is standby reserve,
        not wasted energy.
        """
        n = self.network
        if not self._is_vre(tech):
            return np.zeros(HOURS_PER_YEAR)
        cap        = n.generators.at[tech, "p_nom_opt"]
        pu         = n.generators_t.p_max_pu[tech].values
        available  = cap * pu
        dispatched = n.generators_t.p[tech].values
        return np.maximum(available - dispatched, 0.0)

    @staticmethod
    def _style_ax(ax, title, xlabel="", ylabel="", legend=True):
        """Apply consistent finishing style to an axes object."""
        ax.set_title(title, pad=12, loc="left")
        if xlabel: ax.set_xlabel(xlabel, color="#666666")
        if ylabel: ax.set_ylabel(ylabel, color="#666666")
        if legend:
            ax.legend(framealpha=0.95, edgecolor="#e2e8f0",
                      fancybox=False, loc="upper right", ncol=2,
                      fontsize=8.5, labelspacing=0.4)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")

    # ── Text summaries ────────────────────────────────────────────────────

    def summary(self):
        n   = self.network
        cur = self.cur
        print("\n" + "─"*52)
        print("  INSTALLED GENERATION CAPACITY")
        print("─"*52)
        for gen in n.generators.index:
            cap = n.generators.at[gen, "p_nom_opt"]
            print(f"  {gen:<14}: {cap:>8.2f}  MW")
        if list(n.storage_units.index):
            print("\n" + "─"*52)
            print("  STORAGE CAPACITY")
            print("─"*52)
            for s in n.storage_units.index:
                pwr  = n.storage_units.at[s, "p_nom_opt"]
                mhrs = n.storage_units.at[s, "max_hours"]
                ene  = pwr * mhrs
                print(f"  {s}:")
                print(f"    Power capacity  : {pwr:>8.2f}  MW")
                print(f"    Energy capacity : {ene:>8.2f}  MWh")
                print(f"    Duration        : {mhrs:>8.2f}  h")
        print("─"*52)
        print(f"  Grid loss factor: {GRID_LOSS_FACTOR:.1%}  "
              f"(gross demand = {sum(self.ts.demand)*(1+GRID_LOSS_FACTOR)/1e3:,.1f} GWh/yr)")

    def calculate_lcoe(self):
        n        = self.network
        r        = self.setup.discount_rate
        cur      = self.cur
        total_ac = 0.0

        print("\n" + "─"*62)
        print("  TECHNOLOGY LCOE")
        print("─"*62)
        for gen in n.generators.index:
            cap = n.generators.at[gen, "p_nom_opt"]
            if cap < 1e-6:
                continue
            p         = self.resources.get(gen)
            crf       = IslandEnergyPyPSA._crf(r, p["Lifetime"])
            ann_capex = cap * p["Investment_per_MW"] * crf
            ann_om    = cap * p["O&M_per_MW_yr"]
            gen_mwh   = n.generators_t.p[gen].sum()
            real_mc   = p["Fuel_Cost"] / p["Efficiency"] if p["Efficiency"] > 0 else 0.0
            fuel_cost = real_mc * gen_mwh
            ann_cost  = ann_capex + ann_om + fuel_cost
            total_ac += ann_cost
            if gen_mwh > 0:
                lcoe = ann_cost / gen_mwh
                cf   = gen_mwh / (cap * HOURS_PER_YEAR) if cap > 0 else 0
                print(f"  {gen:<14}: {lcoe:>8.2f}  {cur}/MWh  "
                      f"(gen {gen_mwh/1e3:,.1f} GWh/yr, CF {cf:.1%}, "
                      f"cost {cur}{ann_cost:,.0f}/yr)")

        print("\n" + "─"*62)
        print("  STORAGE LCOS")
        print("─"*62)
        for s in n.storage_units.index:
            p    = self.resources.get(s)
            crf  = IslandEnergyPyPSA._crf(r, p["Lifetime"])
            pwr  = n.storage_units.at[s, "p_nom_opt"]
            mhrs = n.storage_units.at[s, "max_hours"]
            ene  = pwr * mhrs
            ac   = (pwr * p["Investment_per_MW"] * crf
                    + pwr * p["O&M_per_MW_yr"]
                    + ene * p["Storage_MWh"] * crf)
            dis  = self._stor_discharge(s).sum()
            chg  = self._stor_charge(s).sum()
            rte  = dis / chg if chg > 0 else 0
            total_ac += ac
            if dis > 0:
                print(f"  {s:<14}: {ac/dis:>8.2f}  {cur}/MWh  "
                      f"(discharge {dis/1e3:,.1f} GWh/yr, "
                      f"RTE {rte:.1%})")

        total_dem   = sum(self.ts.demand)
        system_lcoe = total_ac / total_dem if total_dem > 0 else 0
        print("\n" + "─"*62)
        print("  SYSTEM LCOE")
        print("─"*62)
        print(f"  Total annualised cost : {cur}{total_ac:>12,.0f}")
        print(f"  Total annual demand   : {total_dem/1e3:>12,.1f}  GWh")
        print(f"  Grid loss factor      : {GRID_LOSS_FACTOR:>11.1%}")
        print(f"  System LCOE           : {system_lcoe:>12.2f}  {cur}/MWh")
        print("─"*62)

        # ── Annual CO2 summary ─────────────────────────────────────────────
        total_co2 = 0.0
        gen_co2   = {}
        for gen in n.generators.index:
            p        = self.resources.get(gen)
            gen_mwh  = n.generators_t.p[gen].sum()
            co2_tech = gen_mwh * p["CO2_per_MWh"]
            if co2_tech > 0:
                gen_co2[gen] = co2_tech
                total_co2   += co2_tech
        if total_co2 > 0:
            print("\n" + "="*60)
            print(" ANNUAL CO2 EMISSIONS")
            print("─"*62)
            for gen, co2 in gen_co2.items():
                print(f"  {gen:<14}: {co2:>10,.1f}  tCO2/yr")
            print(f"  {'TOTAL':<14}: {total_co2:>10,.1f}  tCO2/yr")
            print(f"  Emission intensity: {total_co2/total_dem*1000:>8.1f}  gCO2/kWh")
            print("─"*62)

        return system_lcoe

    # ── Charts ────────────────────────────────────────────────────────────

    def plot_installed_capacity(self):
        """Bar chart: installed generation capacity by technology."""
        n      = self.network
        techs  = list(n.generators.index)
        caps   = [n.generators.at[t, "p_nom_opt"] for t in techs]
        colors = [self._color(t) for t in techs]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(techs, caps, color=colors, width=0.65,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, val in zip(bars, caps):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + max(caps) * 0.015,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=8.5, color="#374151")
        plt.xticks(rotation=30, ha="right")
        ax.set_ylim(0, max(caps) * 1.12)
        self._style_ax(ax, "Installed Generation Capacity", ylabel="MW", legend=False)
        plt.tight_layout()
        plt.show()

    def plot_storage_capacity(self):
        """Side-by-side bars: storage power and energy capacity."""
        n = self.network
        if not list(n.storage_units.index):
            print("No storage technologies selected.")
            return
        storage = list(n.storage_units.index)
        power   = [n.storage_units.at[s, "p_nom_opt"] for s in storage]
        energy  = [n.storage_units.at[s, "p_nom_opt"] * n.storage_units.at[s, "max_hours"]
                   for s in storage]
        colors  = [self._color(s) for s in storage]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, vals, unit, title in [
            (axes[0], power,  "MW",  "Storage Power Capacity"),
            (axes[1], energy, "MWh", "Storage Energy Capacity"),
        ]:
            bars = ax.bar(storage, vals, color=colors, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=5)
            self._style_ax(ax, title, ylabel=unit, legend=False)
        plt.tight_layout()
        plt.show()

    def plot_energy_mix(self):
        """
        Donut chart: annual generation mix as share of total demand.
        Arrow tips are anchored to exact wedge midpoints read from wedge.theta1/theta2;
        label positions are nudged separately for collision avoidance.
        """
        n            = self.network
        total_demand = sum(self.ts.demand)
        supply       = {}
        for gen in n.generators.index:
            used = n.generators_t.p[gen].sum()
            if used > 1:
                supply[gen] = used
        for s in n.storage_units.index:
            dis = self._stor_discharge(s).sum()
            if dis > 1:
                supply[s] = dis

        labels = list(supply.keys())
        values = [supply[k] / total_demand * 100 for k in labels]
        colors = [self._color(l) for l in labels]

        fig, ax = plt.subplots(figsize=(7, 6.5))

        wedges, _ = ax.pie(
            values,
            labels=None,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=0.8)
        )

        SMALL_SLICE = 5.0
        OUTER_R     = 1.22   # label radius for normal slices
        OUTER_R_SM  = 1.48   # label radius for small slices
        TIP_R       = 0.97   # arrow tip sits just inside the outer rim (radius=1.0)

        # Read exact wedge midpoint from matplotlib geometry — never compute
        # cumulatively, which drifts. Handle wrap-around across 0°/360°.
        label_data = []
        for wedge, lab, val in zip(wedges, labels, values):
            t1, t2 = wedge.theta1, wedge.theta2
            if t2 < t1:
                t2 += 360                    # unwrap slice that crosses 0°
            tip_angle = (t1 + t2) / 2       # exact midpoint — never nudged
            label_data.append([tip_angle, lab, val, tip_angle])

        label_data.sort(key=lambda x: x[0])
        MIN_SEP = 9.0
        for i in range(1, len(label_data)):
            if label_data[i][0] - label_data[i - 1][0] < MIN_SEP:
                label_data[i][0] = label_data[i - 1][0] + MIN_SEP

        for label_ang, lab, val, tip_ang in label_data:
            tl = np.deg2rad(label_ang)
            cos_l, sin_l = np.cos(tl), np.sin(tl)

            tt = np.deg2rad(tip_ang)
            cos_t, sin_t = np.cos(tt), np.sin(tt)

            r_text = OUTER_R_SM if val < SMALL_SLICE else OUTER_R

            x_text, y_text = r_text * cos_l, r_text * sin_l
            x_tip,  y_tip  = TIP_R  * cos_t, TIP_R  * sin_t

            ha = "left" if cos_l >= 0 else "right"

            ax.annotate(
                f"{lab}\n({val:.1f}%)",
                xy=(x_tip, y_tip),
                xytext=(x_text, y_text),
                ha=ha, va="center", fontsize=9,
                arrowprops=dict(
                    arrowstyle="-",
                    color="#555555",
                    lw=1.0,
                    connectionstyle="arc3,rad=0.0"
                ),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e5e7eb", alpha=0.92, lw=0.5)
            )

        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-1.9, 1.9)
        ax.set_title("Annual Energy Mix (% of net demand)", pad=16, loc="center")
        plt.tight_layout()
        plt.show()

    def plot_capacity_factors(self):
        """
        Horizontal bar chart of achieved capacity factors by technology.

        The capacity factor is computed as actual annual generation divided
        by the theoretical maximum (installed capacity × 8760 h/yr).
        Technology names are printed inside each bar; the CF percentage
        is printed to the right. The y-axis tick labels are hidden since
        names appear directly on the bars.
        """
        n     = self.network
        techs = [g for g in n.generators.index
                 if n.generators.at[g, "p_nom_opt"] > 1e-3]
        cfs   = []
        for tech in techs:
            cap     = n.generators.at[tech, "p_nom_opt"]
            gen_mwh = n.generators_t.p[tech].sum()
            cfs.append(gen_mwh / (cap * HOURS_PER_YEAR) if cap > 0 else 0)

        colors = [self._color(t) for t in techs]
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(techs) * 0.72)))
        bars = ax.barh(techs, [cf * 100 for cf in cfs],
                       color=colors, edgecolor="white", linewidth=0.5)

        for bar, tech, cf in zip(bars, techs, cfs):
            bar_mid = bar.get_y() + bar.get_height() / 2
            bar_w   = bar.get_width()

            ax.text(0.5, bar_mid, tech,
                    ha="left", va="center", fontsize=9,
                    color="white", fontweight="bold")

            ax.text(bar_w + 0.8, bar_mid,
                    f"{cf:.1%}", va="center", fontsize=9, color="#374151",
                    fontweight=500)

        ax.set_xlim(0, 115)
        ax.axvline(100, color="#d1d5db", linewidth=0.7, linestyle=":", zorder=1)
        ax.set_yticks([])
        self._style_ax(ax, "Achieved Capacity Factor by Technology",
                       xlabel="Capacity Factor (%)", legend=False)
        plt.tight_layout()
        plt.show()

    def plot_lcoe_breakdown(self):
        """
        Stacked horizontal bar chart of annualised cost breakdown by technology.

        Bars are segmented into three cost components:
          · CAPEX  — annualised capital cost (Investment × CRF)
          · O&M    — annual fixed operation and maintenance cost
          · Fuel   — variable fuel cost based on actual generation
        """
        n   = self.network
        r   = self.setup.discount_rate
        cur = self.cur

        techs      = [g for g in n.generators.index
                      if n.generators.at[g, "p_nom_opt"] > 1e-6]
        capex_vals = []
        om_vals    = []
        fuel_vals  = []

        for tech in techs:
            cap     = n.generators.at[tech, "p_nom_opt"]
            p       = self.resources.get(tech)
            crf     = IslandEnergyPyPSA._crf(r, p["Lifetime"])
            gen_mwh = n.generators_t.p[tech].sum()
            real_mc = p["Fuel_Cost"] / p["Efficiency"] if p["Efficiency"] > 0 else 0.0
            capex_vals.append(cap * p["Investment_per_MW"] * crf / 1e6)
            om_vals.append(cap * p["O&M_per_MW_yr"] / 1e6)
            fuel_vals.append(real_mc * gen_mwh / 1e6)

        y      = np.arange(len(techs))
        colors = [self._color(t) for t in techs]
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(techs) * 0.68)))

        ax.barh(y, capex_vals, color=colors, edgecolor="white",
                linewidth=0.8, label="CAPEX (annualised)", alpha=0.95)
        ax.barh(y, om_vals, left=capex_vals, color=colors,
                edgecolor="white", linewidth=0.5, label="O&M", alpha=0.60)
        ax.barh(y, fuel_vals,
                left=[c + o for c, o in zip(capex_vals, om_vals)],
                color=colors, edgecolor="white", linewidth=0.5,
                label="Fuel", alpha=0.30)

        for i, tech in enumerate(techs):
            ax.text(0.005, i, tech,
                    ha="left", va="center", fontsize=9,
                    color="white", fontweight="bold")

        ax.set_yticks([])
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.2f}")
        )
        self._style_ax(ax,
                       title=f"Annualised Cost Breakdown by Technology (M{cur}/yr)",
                       xlabel=f"M{cur}/yr", legend=False)
        ax.legend(["CAPEX (annualised)", "O&M", "Fuel"],
                  framealpha=0.9, edgecolor="#cccccc",
                  fontsize=9, loc="lower right")
        plt.tight_layout()
        plt.show()

    def plot_dispatch(self, hours=168):
        """Stacked area dispatch chart in technology merit order."""
        n          = self.network
        start, end = self._parse_hours(hours)
        T          = range(start, end)
        gen_stack  = []
        labels     = []
        colors     = []
        for tech in self.STACK_ORDER:
            if tech in list(n.generators.index):
                gen_stack.append(self._gen_dispatch(tech, start, end))
                labels.append(tech)
                colors.append(self._color(tech))
            if tech in list(n.storage_units.index):
                gen_stack.append(self._stor_discharge(tech, start, end))
                labels.append(f"{tech} (discharge)")
                colors.append(self._color(tech))
        if not gen_stack:
            print("No generation data to plot.")
            return
        demand = self.ts.demand[start:end]
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.stackplot(T, gen_stack, labels=labels, colors=colors, alpha=0.85)
        ax.plot(T, demand, color="#1E293B", linewidth=2.2, linestyle="--",
                label="Net Demand", zorder=5)
        ax.plot(T, demand * (1 + GRID_LOSS_FACTOR), color="#555555",
                linewidth=1, linestyle=":", alpha=0.6,
                label="Gross Demand (incl. losses)", zorder=4)
        self._style_ax(
            ax,
            title=f"System Dispatch  (hours {start}–{end})",
            xlabel="Hour of year", ylabel="MW"
        )
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        plt.tight_layout()
        plt.show()

    def plot_soc(self, hours=168):
        """Battery state of charge with capacity reference bands."""
        n = self.network
        if not list(n.storage_units.index):
            print("No storage technologies selected.")
            return
        start, end = self._parse_hours(hours)
        T          = range(start, end)
        fig, ax    = plt.subplots(figsize=(14, 4.5))
        for s in n.storage_units.index:
            soc  = n.storage_units_t.state_of_charge[s].values[start:end]
            pwr  = n.storage_units.at[s, "p_nom_opt"]
            mhrs = n.storage_units.at[s, "max_hours"]
            cap  = pwr * mhrs
            ax.fill_between(T, soc, alpha=0.25, color=self._color(s))
            ax.plot(T, soc, color=self._color(s), linewidth=2, label=s)
            ax.axhline(cap,                    color=self._color(s), linestyle=":",  linewidth=1, alpha=0.6)
            ax.axhline(cap * SOC_MIN_FRACTION, color="#E05C5C",      linestyle="--", linewidth=1, alpha=0.7)
        self._style_ax(
            ax,
            title=f"Storage State of Charge  (hours {start}–{end})",
            xlabel="Hour of year", ylabel="MWh"
        )
        plt.tight_layout()
        plt.show()

    def plot_residual(self, hours=168):
        """Residual load chart: demand minus all non-storage generation."""
        n          = self.network
        start, end = self._parse_hours(hours)
        T          = range(start, end)
        ren        = np.zeros(end - start)
        for gen in n.generators.index:
            ren += self._gen_dispatch(gen, start, end)
        residual = self.ts.demand[start:end] - ren
        fig, ax  = plt.subplots(figsize=(14, 4))
        ax.plot(T, residual, color="#1A1A2E", linewidth=1.2)
        ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
        ax.fill_between(T, residual, 0,
                        where=(residual > 0), color="#EF4444", alpha=0.35,
                        label="Deficit (storage / gas needed)")
        ax.fill_between(T, residual, 0,
                        where=(residual <= 0), color="#22C55E", alpha=0.25,
                        label="Surplus (curtailment / storage charge)")
        self._style_ax(
            ax,
            title=f"Residual Load  (hours {start}–{end})",
            xlabel="Hour of year", ylabel="MW",
            legend=True
        )
        ax.legend(loc="upper right", ncol=1, fontsize=9)
        plt.tight_layout()
        plt.show()

    def plot_load_duration(self):
        """
        Load duration curve (LDC): sorted hourly gross demand with generation
        coverage breakdown.
        """
        n   = self.network
        idx = np.argsort(self.ts.demand)[::-1]
        x   = np.arange(1, HOURS_PER_YEAR + 1)

        stack, labels, colors = [], [], []
        for tech in self.STACK_ORDER:
            if tech in list(n.generators.index):
                stack.append(n.generators_t.p[tech].values[idx])
                labels.append(tech)
                colors.append(self._color(tech))
            if tech in list(n.storage_units.index):
                stack.append(self._stor_discharge(tech)[idx])
                labels.append(f"{tech} (discharge)")
                colors.append(self._color(tech))

        fig, ax = plt.subplots(figsize=(14, 5))
        if stack:
            ax.stackplot(x, stack, labels=labels, colors=colors, alpha=0.82)
        ax.plot(x, self.ts.demand[idx], color="#1E293B",
                linewidth=2.2, linestyle="--", label="Net Demand", zorder=5)
        ax.plot(x, self.ts.demand[idx] * (1 + GRID_LOSS_FACTOR),
                color="#555555", linewidth=1, linestyle=":",
                alpha=0.6, label="Gross Demand", zorder=4)
        self._style_ax(
            ax,
            title="Load Duration Curve — Annual Dispatch (sorted by demand)",
            xlabel="Hours (ranked, peak to valley)", ylabel="MW"
        )
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        plt.tight_layout()
        plt.show()

    def plot_worst_residual_week(self):
        """
        Identify and plot the week with the highest cumulative residual load.
        This is the most critical week for storage and gas backup adequacy.
        """
        n        = self.network
        demand   = np.array(self.ts.demand)
        ren      = np.zeros(HOURS_PER_YEAR)
        for gen in n.generators.index:
            ren += n.generators_t.p[gen].values
        residual    = demand - ren
        WEEK        = 168
        week_sums   = np.array([residual[i:i+WEEK].sum() for i in range(HOURS_PER_YEAR - WEEK)])
        worst_start = int(np.argmax(week_sums))
        print(f"\n  Worst renewable week: hours {worst_start:,}–{worst_start + WEEK:,}")
        print(f"  Cumulative residual : {week_sums[worst_start]:,.0f} MWh")
        print(f"  Calendar week ≈ {worst_start // 168 + 1}")
        self.plot_dispatch((worst_start, worst_start + WEEK))
        self.plot_soc((worst_start, worst_start + WEEK))
        self.plot_residual((worst_start, worst_start + WEEK))

    def plot_demand_heatmap(self):
        """
        24h × 12mo heatmap of average hourly demand.

        Rows are hour-of-day (0–23), columns are months (Jan–Dec).
        Colour intensity maps demand magnitude — reveals diurnal and
        seasonal load structure at a glance.
        """
        demand = self.ts.demand * (1 + GRID_LOSS_FACTOR)
        hm = np.zeros((24, 12))
        cnt = np.zeros((24, 12))
        months_idx = np.searchsorted(
            np.cumsum([744,672,744,720,744,720,744,744,720,744,720,744]),
            np.arange(HOURS_PER_YEAR), side='right'
        )
        for h in range(HOURS_PER_YEAR):
            hod, mo = h % 24, min(months_idx[h], 11)
            hm[hod, mo] += demand[h]
            cnt[hod, mo] += 1
        cnt[cnt == 0] = 1
        hm /= cnt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        im = ax.imshow(hm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                            'Jul','Aug','Sep','Oct','Nov','Dec'])
        ax.set_yticks(range(0, 24, 3))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
        ax.set_xlabel("Month")
        ax.set_ylabel("Hour of day")
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("MW", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        self._style_ax(ax, "Demand Heatmap — Hour of Day × Month", legend=False)
        ax.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_monthly_cf(self):
        """
        Monthly capacity factor grid: technology × month.

        Each cell shows the achieved CF for that technology in that month,
        with colour intensity mapped to utilisation level. Gives a compact
        view of seasonal performance across the entire technology portfolio.
        """
        n = self.network
        months_idx = np.searchsorted(
            np.cumsum([744,672,744,720,744,720,744,744,720,744,720,744]),
            np.arange(HOURS_PER_YEAR), side='right'
        )
        hrs_per_month = np.array([744,672,744,720,744,720,744,744,720,744,720,744])

        techs = [g for g in n.generators.index
                 if n.generators.at[g, "p_nom_opt"] > 1e-3]
        if not techs:
            print("No generation technologies to plot.")
            return

        cf_matrix = np.zeros((len(techs), 12))
        for i, tech in enumerate(techs):
            cap = n.generators.at[tech, "p_nom_opt"]
            dispatch = n.generators_t.p[tech].values
            for h in range(HOURS_PER_YEAR):
                mo = min(months_idx[h], 11)
                cf_matrix[i, mo] += dispatch[h]
            for m in range(12):
                cf_matrix[i, m] = cf_matrix[i, m] / (cap * hrs_per_month[m]) * 100

        fig, ax = plt.subplots(figsize=(9, max(2.5, len(techs) * 0.55 + 1)))
        im = ax.imshow(cf_matrix, aspect='auto', cmap='GnBu',
                       interpolation='nearest', vmin=0, vmax=100)
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                            'Jul','Aug','Sep','Oct','Nov','Dec'])
        ax.set_yticks(range(len(techs)))
        ax.set_yticklabels(techs)

        # Annotate cells with CF %
        for i in range(len(techs)):
            for j in range(12):
                val = cf_matrix[i, j]
                color = "white" if val > 50 else "#374151"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight=500)

        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("CF %", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        self._style_ax(ax, "Monthly Capacity Factor by Technology", legend=False)
        ax.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_energy_sankey(self, title="Annual Energy Flow (GWh/yr)"):
        """
        Interactive Sankey diagram of annual energy flows.

        Column layout (left → right)
        ─────────────────────────────
        Col 1  Generation nodes  (one per active generator)
        Col 2  Storage nodes     (one per active storage unit, if any)
        Col 3  Grid Bus          (single balancing node)
        Col 4  Sinks             Load · Losses · Curtailment

        Flow architecture
        ─────────────────
        1. Generator  →  Storage       pro-rata share of total charging
        2. Generator  →  Curtailment   VRE only; available potential − dispatched
        3. Generator  →  Grid Bus      remainder (dispatched − storage allocation)
        4. Storage    →  Grid Bus      discharge
        5. Storage    →  Losses        round-trip loss (charge − discharge)
        6. Grid Bus   →  Load          net consumer demand
        7. Grid Bus   →  Losses        distribution loss (GLF × gross demand)

        Grid and storage losses are merged into a single "Losses" sink node.
        Chart height scales automatically with number of generator technologies.
        """
        import plotly.graph_objects as go

        n             = self.network
        gen_techs     = list(n.generators.index)
        storage_types = list(n.storage_units.index)
        has_storage   = len(storage_types) > 0

        # ── Annual totals ────────────────────────────────────────────────────
        gen_dispatched = {g: float(n.generators_t.p[g].sum())            for g in gen_techs}
        gen_curtailed  = {g: float(self._curtailment(g).sum())            for g in gen_techs}
        stor_charge    = {s: float(self._stor_charge(s).sum())            for s in storage_types}
        stor_discharge = {s: float(self._stor_discharge(s).sum())         for s in storage_types}
        stor_loss      = {s: max(stor_charge[s] - stor_discharge[s], 0.)  for s in storage_types}

        total_dispatched = sum(gen_dispatched.values())
        total_charge     = sum(stor_charge.values())

        gross_demand_mwh = float(n.loads_t.p_set["Demand"].sum())
        net_demand_mwh   = float(sum(self.ts.demand))
        grid_loss_mwh    = gross_demand_mwh - net_demand_mwh

        # ── Node registry ────────────────────────────────────────────────────
        node_labels, node_colors, node_x, node_y = [], [], [], []

        def add_node(label, color, x, y=0.5):
            node_labels.append(label)
            node_colors.append(color)
            node_x.append(x)
            node_y.append(y)
            return len(node_labels) - 1

        # Column x-positions
        X_GEN  = 0.02
        X_STOR = 0.36
        X_GRID = 0.62 if has_storage else 0.48
        X_SINK = 0.98

        # Generator nodes — evenly spread vertically to prevent clipping
        n_gen = len(gen_techs)
        gen_idx = {}
        for i, g in enumerate(gen_techs):
            y_pos = (i + 0.5) / n_gen if n_gen > 1 else 0.5
            gen_idx[g] = add_node(g, self._color(g), X_GEN, y_pos)

        # Storage nodes — evenly spread vertically
        stor_idx = {}
        if has_storage:
            n_stor = len(storage_types)
            for i, s in enumerate(storage_types):
                y_pos = (i + 0.5) / n_stor if n_stor > 1 else 0.5
                stor_idx[s] = add_node(s, self._color(s), X_STOR, y_pos)

        grid_idx = add_node("Grid Bus",    "#94A3B8", X_GRID, 0.45)
        load_idx = add_node("Load",        "#2563EB", X_SINK, 0.28)
        loss_idx = add_node("Losses",      "#CBD5E1", X_SINK, 0.72)
        curt_idx = add_node("Curtailment", "#EF4444", X_SINK, 0.92)

        # ── Link registry ────────────────────────────────────────────────────
        link_src, link_tgt, link_val, link_col, link_lbl = [], [], [], [], []

        def add_link(src, tgt, mwh, color_name, label):
            if mwh < 1.0:
                return
            link_src.append(src)
            link_tgt.append(tgt)
            link_val.append(round(mwh, 1))
            link_lbl.append(label)
            base = self._color(color_name).lstrip("#")
            r, g, b = int(base[0:2], 16), int(base[2:4], 16), int(base[4:6], 16)
            link_col.append(f"rgba({r},{g},{b},0.35)")

        # ── Generator flows ──────────────────────────────────────────────────
        for g in gen_techs:
            disp  = gen_dispatched[g]
            curt  = gen_curtailed[g]
            share = disp / total_dispatched if total_dispatched > 0 else 0.0

            # → each storage unit (pro-rata by dispatch share)
            if has_storage:
                for s in storage_types:
                    alloc = share * stor_charge[s]
                    add_link(gen_idx[g], stor_idx[s], alloc, g,
                             f"{g} → {s} charging  {alloc/1e3:.2f} GWh")

            stor_alloc_total = share * total_charge

            # → Grid Bus (dispatched minus what goes to storage)
            to_grid = max(disp - stor_alloc_total, 0.0)
            add_link(gen_idx[g], grid_idx, to_grid, g,
                     f"{g} → Grid Bus  {to_grid/1e3:.2f} GWh")

            # → Curtailment (VRE only; zero for dispatchable)
            add_link(gen_idx[g], curt_idx, curt, g,
                     f"{g} → Curtailment  {curt/1e3:.2f} GWh")

        # ── Storage flows ────────────────────────────────────────────────────
        for s in storage_types:
            dis  = stor_discharge[s]
            loss = stor_loss[s]
            rte  = dis / stor_charge[s] * 100 if stor_charge[s] > 0 else 0.0

            add_link(stor_idx[s], grid_idx, dis, s,
                     f"{s} discharge → Grid Bus  {dis/1e3:.2f} GWh  (η {rte:.0f}%)")
            # round-trip loss → merged Losses node
            add_link(stor_idx[s], loss_idx, loss, s,
                     f"{s} round-trip loss  {loss/1e3:.2f} GWh")

        # ── Grid Bus → sinks ─────────────────────────────────────────────────
        add_link(grid_idx, load_idx, net_demand_mwh, "Load",
                 f"Grid Bus → Load  {net_demand_mwh/1e3:.2f} GWh")
        # grid distribution loss → merged Losses node
        add_link(grid_idx, loss_idx, grid_loss_mwh, "Losses",
                 f"Grid distribution loss  {grid_loss_mwh/1e3:.2f} GWh  "
                 f"({GRID_LOSS_FACTOR*100:.0f}% of gross demand)")

        # ── Summary subtitle ─────────────────────────────────────────────────
        total_curt_gwh = sum(gen_curtailed.values()) / 1e3
        total_gen_gwh  = total_dispatched / 1e3
        curt_pct       = (total_curt_gwh / (total_gen_gwh + total_curt_gwh) * 100
                          if (total_gen_gwh + total_curt_gwh) > 0 else 0.0)
        total_dis_gwh  = sum(stor_discharge.values()) / 1e3
        total_sl_gwh   = sum(stor_loss.values()) / 1e3
        total_loss_gwh = grid_loss_mwh / 1e3 + total_sl_gwh

        subtitle = (
            f"Dispatched: {total_gen_gwh:.1f} GWh  │  "
            f"Net demand: {net_demand_mwh/1e3:.1f} GWh  │  "
            f"Total losses: {total_loss_gwh:.1f} GWh  │  "
            f"Storage discharge: {total_dis_gwh:.1f} GWh  │  "
            f"Curtailment: {total_curt_gwh:.1f} GWh ({curt_pct:.1f}%)"
        )

        # ── Plotly Sankey ─────────────────────────────────────────────────────
        # Height scales with generator count so nodes never get clipped
        dynamic_height = max(520, 90 * max(n_gen, len(storage_types) if has_storage else 1) + 140)

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                label     = node_labels,
                color     = node_colors,
                x         = node_x,
                y         = node_y,
                pad       = 18,
                thickness = 20,
                line      = dict(color="rgba(255,255,255,0.6)", width=0.5),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Total flow: %{value:,.0f} MWh"
                    "<extra></extra>"
                ),
            ),
            link=dict(
                source        = link_src,
                target        = link_tgt,
                value         = link_val,
                label         = link_lbl,
                color         = link_col,
                hovertemplate = "%{label}<extra></extra>",
            ),
        ))

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{title}</b><br>"
                    f"<span style='font-size:12px;color:#888'>{subtitle}</span>"
                ),
                font=dict(size=16),
                x=0.01,
                xanchor="left",
            ),
            font          = dict(family="Georgia, serif", size=11.5, color="#374151"),
            paper_bgcolor = "white",
            plot_bgcolor  = "white",
            height        = dynamic_height,
            margin        = dict(l=20, r=20, t=110, b=30),
        )

        fig.show()

    # ── Grid Congestion ───────────────────────────────────────────────────

    def plot_grid_congestion(self, hours=None):
        """
        Grid congestion analysis: bus utilisation ratio over time.

        In a single-bus island model there are no transmission lines to
        congest, so "congestion" is measured as the ratio of gross demand
        to total available supply capacity at each hour:

            utilisation(t) = gross_demand(t) / available_supply(t)

        When utilisation approaches 1.0 the system has no headroom — any
        forecast error, unplanned outage, or demand spike could trigger
        load shedding.  Hours above the configurable threshold (default
        80 %) are highlighted as "congested".

        The chart has three panels:

        1. **Timeseries** — hourly utilisation with congested hours shaded
        2. **Duration curve** — utilisation sorted from highest to lowest
        3. **Monthly bar** — congested hours per calendar month

        Parameters
        ----------
        hours : tuple(int, int) or None
            Optional (start, end) slice for the timeseries panel.
            If None, the full 8 760-hour year is shown.
        """
        from src.model import IslandEnergyPyPSA  # already imported, but explicit

        # Use the model's compute_congestion — keeps logic in one place
        model = type('_Proxy', (), {'network': self.network, 'ts': self.ts,
                                     'setup': self.setup, 'resources': self.resources,
                                     '_stor_discharge': lambda s, name: self.network.storage_units_t.p[name].clip(lower=0),
                                     '_stor_charge': lambda s, name: (-self.network.storage_units_t.p[name]).clip(lower=0)})()

        # Recompute inline to avoid circular dependency issues
        n = self.network
        gross_demand = n.loads_t.p_set["Demand"].values

        available = np.zeros(HOURS_PER_YEAR)
        for gen in n.generators.index:
            cap = n.generators.at[gen, "p_nom_opt"]
            if gen in n.generators_t.p_max_pu.columns:
                available += cap * n.generators_t.p_max_pu[gen].values
            else:
                available += cap
        for s in n.storage_units.index:
            pwr = n.storage_units.at[s, "p_nom_opt"]
            available += pwr

        available    = np.maximum(available, 1e-6)
        utilisation  = gross_demand / available
        congested    = utilisation >= CONGESTION_THRESHOLD
        total_cong_h = int(congested.sum())

        # Monthly breakdown
        months_idx = np.searchsorted(
            np.cumsum([744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]),
            np.arange(HOURS_PER_YEAR), side='right'
        )
        monthly_hours = [0] * 12
        for h in range(HOURS_PER_YEAR):
            if congested[h]:
                monthly_hours[min(months_idx[h], 11)] += 1

        # ── Text summary ──────────────────────────────────────────────────
        print("\n" + "─" * 62)
        print("  GRID CONGESTION ANALYSIS")
        print("─" * 62)
        print(f"  Utilisation threshold : {CONGESTION_THRESHOLD:.0%}")
        print(f"  Congested hours       : {total_cong_h:,} / {HOURS_PER_YEAR:,}  "
              f"({total_cong_h / HOURS_PER_YEAR:.1%})")
        print(f"  Peak utilisation      : {utilisation.max():.1%}")
        print(f"  Mean utilisation      : {utilisation.mean():.1%}")
        print("─" * 62)

        # ── Panel 1: Timeseries ───────────────────────────────────────────
        if hours is not None:
            start, end = hours if isinstance(hours, tuple) else (hours, hours + 168)
        else:
            start, end = 0, HOURS_PER_YEAR
        T   = range(start, end)
        u_s = utilisation[start:end]
        c_s = congested[start:end]

        fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                                 gridspec_kw={"height_ratios": [3, 2, 2]})

        ax1 = axes[0]
        ax1.plot(T, u_s * 100, color="#2563EB", linewidth=1.0, alpha=0.85)
        ax1.fill_between(T, u_s * 100, 0,
                         where=c_s, color="#EF4444", alpha=0.25,
                         label=f"Congested (≥ {CONGESTION_THRESHOLD:.0%})")
        ax1.fill_between(T, u_s * 100, 0,
                         where=~c_s, color="#2563EB", alpha=0.08)
        ax1.axhline(CONGESTION_THRESHOLD * 100, color="#DC2626",
                    linewidth=1.2, linestyle="--", alpha=0.7,
                    label=f"Threshold ({CONGESTION_THRESHOLD:.0%})")
        ax1.set_ylim(0, max(105, utilisation.max() * 105))
        self._style_ax(ax1,
                       title=f"Bus Utilisation Ratio  (hours {start}–{end})",
                       xlabel="Hour of year", ylabel="Utilisation (%)")
        ax1.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        # ── Panel 2: Duration curve ───────────────────────────────────────
        ax2 = axes[1]
        sorted_u = np.sort(utilisation)[::-1] * 100
        x_dur    = np.arange(1, HOURS_PER_YEAR + 1)
        ax2.fill_between(x_dur, sorted_u, 0, color="#2563EB", alpha=0.15)
        ax2.plot(x_dur, sorted_u, color="#2563EB", linewidth=1.2)
        ax2.axhline(CONGESTION_THRESHOLD * 100, color="#DC2626",
                    linewidth=1.2, linestyle="--", alpha=0.7)
        # Mark intersection
        cross_idx = np.searchsorted(-sorted_u, -CONGESTION_THRESHOLD * 100)
        if 0 < cross_idx < HOURS_PER_YEAR:
            ax2.axvline(cross_idx, color="#DC2626", linewidth=0.8,
                        linestyle=":", alpha=0.5)
            ax2.text(cross_idx + 80, CONGESTION_THRESHOLD * 100 + 2,
                     f"{cross_idx:,} h", fontsize=8.5, color="#DC2626")
        self._style_ax(ax2,
                       title="Utilisation Duration Curve",
                       xlabel="Hours (ranked, highest to lowest)",
                       ylabel="Utilisation (%)", legend=False)
        ax2.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        # ── Panel 3: Monthly congested hours ──────────────────────────────
        ax3    = axes[2]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        colors = ["#EF4444" if h > 0 else "#cbd5e1" for h in monthly_hours]
        bars   = ax3.bar(months, monthly_hours, color=colors,
                         edgecolor="white", linewidth=0.5, zorder=3)
        for bar, val in zip(bars, monthly_hours):
            if val > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         val + max(monthly_hours) * 0.02,
                         str(val), ha="center", va="bottom",
                         fontsize=8.5, color="#374151")
        self._style_ax(ax3,
                       title="Congested Hours by Month",
                       xlabel="", ylabel="Hours", legend=False)

        plt.tight_layout()
        plt.show()
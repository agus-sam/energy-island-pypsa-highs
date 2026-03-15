from src.constants import GRID_LOSS_FACTOR

try:
    import ipywidgets as widgets
    from IPython.display import display
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


class SetupOptions:
    """Interactive system configuration widget."""

    STORAGE_DEFAULTS = {"BESS": 4, "PHS": 8, "Hydrogen": 24}

    def __init__(self):
        self.selected_gen       = []
        self.selected_storage   = []
        self.selected_balancing = []
        self.max_storage_hours  = {}
        self.objective          = "Lowest LCOE"
        self.discount_rate      = 0.08
        self.currency           = "€"
        self.demand_scale_pct   = 100.0

    def display(self):
        if not HAS_WIDGETS:
            print("ipywidgets not available — set attributes directly.")
            return

        # ── Objective ─────────────────────────────────────────────────────
        self.objective_dd = widgets.Dropdown(
            options=["Lowest LCOE", "Lowest CO2", "Most Diversified"],
            description="Objective"
        )

        # ── Currency symbol text input ─────────────────────────────────────
        self.currency_input = widgets.Text(
            value="€",
            description="Currency:",
            placeholder="e.g. € $ £ RM Rp",
            layout=widgets.Layout(width="280px")
        )

        # ── Generation technologies ────────────────────────────────────────
        self.gen_boxes = [
            widgets.Checkbox(description=t)
            for t in ["Wind", "Solar", "Biomass", "Biogas", "Geothermal", "Hydro", "WTE"]
        ]

        # ── Storage technologies ───────────────────────────────────────────
        self.storage_items = []
        for name, default_hrs in self.STORAGE_DEFAULTS.items():
            cb  = widgets.Checkbox(description=name)
            hrs = widgets.FloatText(value=default_hrs, description="Max Hours",
                                    layout=widgets.Layout(width="200px"))
            self.storage_items.append({"name": name, "checkbox": cb,
                                        "hours_input": hrs, "row": widgets.HBox([cb, hrs])})

        # ── Balancing and financial parameters ────────────────────────────
        self.balancing_boxes = [
            widgets.Checkbox(description="Natural Gas"),
            widgets.Checkbox(description="Biodiesel"),
        ]
        self.discount_input  = widgets.FloatText(value=0.08, description="Discount Rate")
        self.demand_scale_input = widgets.FloatText(
            value=100.0,
            description="Demand Scale %:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="280px"),
            tooltip="100 = baseline, 110 = +10% growth, 90 = -10% reduction"
        )

        btn    = widgets.Button(description="Confirm Setup", button_style="success")
        output = widgets.Output()

        display(self.objective_dd)
        display(self.currency_input)
        print("\nVariable Renewable Generation:")
        for cb in self.gen_boxes:
            display(cb)
        print("\nEnergy Storage (set max duration hours):")
        for item in self.storage_items:
            display(item["row"])
        print("\nBalancing Technology:")
        for cb in self.balancing_boxes:
            display(cb)
        display(self.discount_input)
        display(self.demand_scale_input)
        display(btn)
        display(output)

        def confirm(_):
            self.selected_gen = [cb.description for cb in self.gen_boxes if cb.value]
            self.selected_storage  = []
            self.max_storage_hours = {}
            for item in self.storage_items:
                if item["checkbox"].value:
                    self.selected_storage.append(item["name"])
                    self.max_storage_hours[item["name"]] = item["hours_input"].value
            self.selected_balancing = [cb.description for cb in self.balancing_boxes if cb.value]
            self.objective          = self.objective_dd.value
            self.discount_rate      = self.discount_input.value
            self.currency           = self.currency_input.value.strip() or "€"
            self.demand_scale_pct   = self.demand_scale_input.value
            with output:
                output.clear_output()
                print("Setup confirmed.")
                print(f"  Objective      : {self.objective}")
                print(f"  Currency       : {self.currency}")
                print(f"  Generation     : {self.selected_gen}")
                print(f"  Storage        : {self.selected_storage}")
                print(f"  Max hours      : {self.max_storage_hours}")
                print(f"  Balancing      : {self.selected_balancing}")
                print(f"  Discount rate  : {self.discount_rate:.1%}")
                print(f"  Demand scale   : {self.demand_scale_pct:.1f}%")
                print(f"  Grid loss      : {GRID_LOSS_FACTOR:.1%}")

        btn.on_click(confirm)

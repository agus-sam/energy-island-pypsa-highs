import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import display
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


class ResourceAssessment:
    """Load and validate technology and resource parameter CSV."""

    REQUIRED_COLUMNS = {
        "Sources", "Max_Capacity_MW", "Investment_per_MW",
        "O&M_per_MW_yr", "Lifetime", "CO2_per_MWh",
        "Fuel_Cost", "Efficiency", "Merit_Order", "Storage_MWh"
    }

    def __init__(self):
        self.data   = None
        self._cache = {}   # pre-built dict for O(1) parameter lookups

    def _load(self, filepath):
        """Load CSV from filepath, validate, and build lookup cache."""
        df = pd.read_csv(filepath)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"ERROR: Missing columns: {missing}")
            return

        if df["Sources"].isnull().any():
            print("ERROR: 'Sources' column contains blank values.")
            return

        numeric_cols = [
            "Max_Capacity_MW", "Investment_per_MW", "O&M_per_MW_yr",
            "Lifetime", "CO2_per_MWh", "Fuel_Cost", "Efficiency", "Storage_MWh"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                print(f"ERROR: Non-numeric values in column '{col}'.")
                return

        eff_bad = df[(df["Efficiency"] <= 0) | (df["Efficiency"] > 1)]
        if not eff_bad.empty:
            print(f"WARNING: Efficiency outside (0, 1]: {eff_bad['Sources'].tolist()}")

        self.data   = df
        self._cache = {row["Sources"]: row for _, row in df.iterrows()}
        print("Resource data loaded.")
        if HAS_WIDGETS:
            display(self.data)
        else:
            print(self.data.to_string())

    def upload(self):
        if not HAS_WIDGETS:
            print("ipywidgets not available — use _load(filepath) directly.")
            return

        style      = {"description_width": "140px"}
        path_input = widgets.Text(
            value="data/inputs/resource_assessment.csv",
            description="File path:",
            layout=widgets.Layout(width="500px"),
            style=style
        )
        btn    = widgets.Button(description="Load File", button_style="primary")
        output = widgets.Output()

        print("Resource Assessment File Loader")
        print("Expected columns: Sources, Max_Capacity_MW, Investment_per_MW, ")
        print("  O&M_per_MW_yr, Lifetime, CO2_per_MWh, Fuel_Cost, Efficiency, Merit_Order, Storage_MWh")
        display(path_input, btn, output)

        def on_click(_):
            with output:
                output.clear_output()
                try:
                    self._load(path_input.value)
                except FileNotFoundError:
                    print(f"ERROR: File not found: {path_input.value}")
                except Exception as e:
                    print(f"ERROR: {e}")

        btn.on_click(on_click)

    def get(self, tech):
        """Return parameter row for a technology. Raises KeyError if not found."""
        if tech not in self._cache:
            raise KeyError(f"Technology '{tech}' not found in resource data.")
        return self._cache[tech]

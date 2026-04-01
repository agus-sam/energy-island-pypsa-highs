import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import display
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


class GeographicLoader:
    """Load and display geographic project information."""

    REQUIRED_COLUMNS = {"Name", "Latitude", "Longitude", "Max_Height_m", "Project_Description"}
    PHS_MIN_HEIGHT_M = 300

    def __init__(self):
        self.data = None

    def _load(self, filepath):
        """Load geographic CSV, validate columns, and display summary."""
        df = pd.read_csv(filepath)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            core_required = {"Name", "Latitude", "Longitude", "Max_Height_m"}
            missing_core = core_required - set(df.columns)
            if missing_core:
                print(f"ERROR: Missing required geographic columns: {missing_core}")
                return
            for col in (self.REQUIRED_COLUMNS - core_required - set(df.columns)):
                df[col] = ""
                print(f"  WARNING: Optional column '{col}' not found — added as blank.")

        for col in ("Latitude", "Longitude", "Max_Height_m"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                print(f"  WARNING: Non-numeric values in '{col}' — affected rows set to NaN.")

        self.data = df
        print("Geographic data loaded.")
        if HAS_WIDGETS:
            display(df)
        else:
            print(df.to_string())

        row = df.iloc[0]
        print(f"\n  Project name : {row.get('Name', 'N/A')}")
        desc = str(row.get("Project_Description", "")).strip()
        if desc:
            print(f"  Description  : {desc[:120]}{'…' if len(desc) > 120 else ''}")

        height = row["Max_Height_m"]
        if pd.isna(height):
            print("  WARNING: Max_Height_m is missing — PHS feasibility cannot be assessed.")
        elif height < self.PHS_MIN_HEIGHT_M:
            print(
                f"  WARNING: Max_Height_m = {height} m is below the {self.PHS_MIN_HEIGHT_M} m"
                " threshold recommended for Pumped Hydro Storage feasibility."
            )
        else:
            print(f"  PHS feasibility: height = {height} m — threshold met.")

    def upload(self):
        if not HAS_WIDGETS:
            print("ipywidgets not available — use _load(filepath) directly.")
            return

        style      = {"description_width": "140px"}
        path_input = widgets.Text(
            value="data/inputs/geographic_setup.csv",
            description="File path:",
            layout=widgets.Layout(width="500px"),
            style=style
        )
        btn    = widgets.Button(description="Load File", button_style="primary")
        output = widgets.Output()

        print("Geographic File Loader")
        print("Expected columns: Name, Latitude, Longitude, Max_Height_m, Project_Description")
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

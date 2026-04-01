import numpy as np
import pandas as pd
from src.constants import HOURS_PER_YEAR

try:
    import ipywidgets as widgets
    from IPython.display import display
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


class TimeSeriesData:
    """Load and validate hourly time series for demand and generation."""

    def __init__(self, setup):
        self.setup      = setup
        self.generation = {}    # dict: tech_name -> np.array(8760,)
        self.demand     = None  # np.array(8760,)

    @staticmethod
    def _validate(arr, label):
        """Validate length, NaN, and sign. Raises ValueError on failure."""
        if len(arr) != HOURS_PER_YEAR:
            raise ValueError(
                f"{label}: expected {HOURS_PER_YEAR} rows, got {len(arr)}."
            )
        if np.isnan(arr).any():
            raise ValueError(f"{label}: contains NaN values.")
        if (arr < 0).any():
            raise ValueError(f"{label}: contains negative values.")

    @staticmethod
    def _load_csv(filepath, label):
        """Load a single-column CSV from filepath and return numpy array."""
        arr = pd.read_csv(filepath).iloc[:, 0].astype(float).values
        TimeSeriesData._validate(arr, label)
        print(f"  {label}: {len(arr)} values loaded.")
        return arr

    def upload_generation(self):
        """Display one file path input per selected generation technology."""
        if not HAS_WIDGETS:
            print("ipywidgets not available — use _load_csv() directly.")
            return

        style   = {"description_width": "160px"}
        output  = widgets.Output()
        inputs  = {}

        for tech in self.setup.selected_gen:
            default = f"data/time_series/{tech.lower()}_prod.csv"
            inp = widgets.Text(
                value=default,
                description=f"{tech} profile:",
                layout=widgets.Layout(width="500px"),
                style=style
            )
            inputs[tech] = inp
            display(inp)

        btn = widgets.Button(description="Load All Generation Profiles", button_style="primary")
        display(btn, output)

        def on_click(_):
            with output:
                output.clear_output()
                for tech_name, inp in inputs.items():
                    try:
                        self.generation[tech_name] = self._load_csv(inp.value, tech_name)
                    except FileNotFoundError:
                        print(f"ERROR: File not found: {inp.value}")
                    except ValueError as e:
                        print(f"ERROR: {e}")
                    except Exception as e:
                        print(f"ERROR ({tech_name}): {e}")
                loaded = list(self.generation.keys())
                if loaded:
                    print(f"\nLoaded: {loaded}")

        btn.on_click(on_click)

    def upload_demand(self):
        """Display a file path input for the demand profile."""
        if not HAS_WIDGETS:
            print("ipywidgets not available — use _load_csv() directly.")
            return

        style      = {"description_width": "140px"}
        path_input = widgets.Text(
            value="data/time_series/demand.csv",
            description="Demand profile:",
            layout=widgets.Layout(width="500px"),
            style=style
        )
        btn    = widgets.Button(description="Load Demand", button_style="primary")
        output = widgets.Output()

        display(path_input, btn, output)

        def on_click(_):
            with output:
                output.clear_output()
                try:
                    raw = self._load_csv(path_input.value, "Demand")
                    scale = self.setup.demand_scale_pct / 100.0
                    self.demand = raw * scale
                    if scale == 1.0:
                        print(f"  Demand scaling  : 100.0% — no adjustment")
                    else:
                        print(f"  Demand scaling  : {self.setup.demand_scale_pct:.1f}% (×{scale:.4f})")
                        print(f"  Peak   (raw → scaled) : {raw.max():.1f} MW  →  {self.demand.max():.1f} MW")
                        print(f"  Annual (raw → scaled) : {raw.sum()/1e3:.1f} GWh  →  {self.demand.sum()/1e3:.1f} GWh")
                except FileNotFoundError:
                    print(f"ERROR: File not found: {path_input.value}")
                except ValueError as e:
                    print(f"ERROR: {e}")
                except Exception as e:
                    print(f"ERROR: {e}")

        btn.on_click(on_click)

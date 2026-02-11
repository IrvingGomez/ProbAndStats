import pandas as pd

class AppState:
    """
    Global application state.
    """

    def __init__(self):
        # Raw and filtered data
        self.df: pd.DataFrame | None = None
        self.filtered_df: pd.DataFrame | None = None

        # Authoritative column classification
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []

        # Active filters
        self.active_filters: dict[str, list] = {}

        # Table and figure to export
        self.export_table = None
        self.export_figure = None

        # Inference cache
        self.mean_ci: tuple[float, float] | None = None
        self.sigma_ci: tuple[float, float] | None = None

        self.selected_column = None


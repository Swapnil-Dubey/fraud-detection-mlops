import pandas as pd


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: str):
    """Compare reference (training) data against current (production) data for drift."""
    pass


def generate_performance_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: str):
    """Generate classification performance report if labels are available."""
    pass

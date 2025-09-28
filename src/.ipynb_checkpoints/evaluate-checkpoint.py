import pandas as pd
from pathlib import Path

# Resolve project root
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
    if ROOT.name == "src":
        ROOT = ROOT.parent

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

reg_txt = REPORTS / "regression_sklearn_mse.txt"
cls_one = REPORTS / "classification_metrics_logit_only.csv"

# ---- Build regression summary ----
rows = []
if reg_txt.exists():
    mse = float(reg_txt.read_text(encoding="utf-8").strip().split("=")[-1])
    rows.append({"Model": "LinearRegression (sklearn)", "MSE": mse})
reg_summary = pd.DataFrame(rows)
reg_summary.to_csv(REPORTS / "regression_summary.csv", index=False)

# ---- Build classification summary ----
if cls_one.exists():
    pd.read_csv(cls_one).to_csv(REPORTS / "classification_metrics.csv", index=False)

print("Wrote:", REPORTS / "regression_summary.csv")
print("Wrote:", REPORTS / "classification_metrics.csv")

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support

# ---------------- Paths ----------------
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
    if ROOT.name == "src": ROOT = ROOT.parent

INP = ROOT / "data" / "processed" / "features.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP)

# ---------------- Regression ----------------
y_col = next((c for c in ["rating_number","aggregate_rating","rating"] if c in df.columns), None)
assert y_col is not None, "No numeric rating column found!"

y_all = pd.to_numeric(df[y_col], errors="coerce")
X_all = df.drop(columns=[y_col])
mask = ~y_all.isna()
Xr, yr = X_all.loc[mask], y_all.loc[mask]

num_cols = [c for c in Xr.columns if pd.api.types.is_numeric_dtype(Xr[c])]
cat_cols = [c for c in Xr.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
pipe_lr = Pipeline([("pre", pre), ("lr", LinearRegression())]).fit(Xtr, ytr)
mse = mean_squared_error(yte, pipe_lr.predict(Xte))
(REPORTS / "regression_sklearn_mse.txt").write_text(f"MSE={mse:.6f}\n", encoding="utf-8")

print(f"[Regression] MSE saved to reports/regression_sklearn_mse.txt: {mse:.6f}")

# ---------------- Classification ----------------
if "rating_text" in df.columns:
    pos = {"good","very good","excellent"}
    neg = {"poor","average"}
    y_bin = df["rating_text"].astype(str).str.lower().map(lambda s: 1 if s in pos else (0 if s in neg else None))

    mask = y_bin.notna()
    Xc, yc = df.loc[mask].drop(columns=["rating_text"]), y_bin.loc[mask].astype(int)

    numc = [c for c in Xc.columns if pd.api.types.is_numeric_dtype(Xc[c])]
    catc = [c for c in Xc.columns if c not in numc]

    pre_c = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))]), numc),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), catc)
    ])

    Xtrc, Xtec, ytrc, ytec = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=42)
    pipe_cls = Pipeline([("pre", pre_c), ("clf", LogisticRegression(max_iter=1000))]).fit(Xtrc, ytrc)
    yp = pipe_cls.predict(Xtec)

    pr, rc, f1, _ = precision_recall_fscore_support(ytec, yp, average="binary", zero_division=0)
    out_df = pd.DataFrame([{"Model":"LogisticRegression","Precision":pr,"Recall":rc,"F1":f1}])
    out_df.to_csv(REPORTS / "classification_metrics_logit_only.csv", index=False)

    print(f"[Classification] Metrics saved to reports/classification_metrics_logit_only.csv")

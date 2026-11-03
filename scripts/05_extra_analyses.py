"""Supporting numbers used in the reports: pilot feasibility, split-design leakage demo, missingness mechanism."""
import json, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")
out = {}
d = pd.read_csv("data/clean_imputed.csv")
u = pd.read_csv("data/clean_unimputed.csv")

# (a) pilot feasibility: one-feature regression and mean baseline
y = d["co2_gkm"]; X = d[["weight_kg"]]
lr = LinearRegression().fit(X, y); p = lr.predict(X)
out["pilot"] = dict(n=len(d), baseline_rmse=float(y.std(ddof=0)), r_weight=float(np.corrcoef(d.weight_kg, y)[0, 1]),
                    slope_g_per_100kg=float(lr.coef_[0] * 100), intercept=float(lr.intercept_), r2=float(r2_score(y, p)),
                    rmse=float(np.sqrt(mean_squared_error(y, p))))
X2 = d[["weight_kg", "engine_size_l", "horsepower"]]
lr2 = LinearRegression().fit(X2, y); p2 = lr2.predict(X2)
out["pilot3"] = dict(r2=float(r2_score(y, p2)), rmse=float(np.sqrt(mean_squared_error(y, p2))))

# (b) leakage demo: random K-fold vs family-grouped K-fold (train-only data, simple imputation for speed)
NUM = ["model_year", "engine_size_l", "cylinders", "horsepower", "weight_kg", "turbo"]
CAT = ["make", "vehicle_class", "transmission", "drivetrain", "fuel_type", "powertrain"]
tr = pd.read_csv("results/train_indices.csv")["idx"].values
T = u.loc[tr]
cat = Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
ct = ColumnTransformer([("c", cat, CAT), ("n", SimpleImputer(strategy="median"), NUM)])
res = {}
for name, est in [("Random forest (min_samples_leaf=1)", RandomForestRegressor(n_estimators=100, min_samples_leaf=1, random_state=0)),
                  ("Gradient boosting (default)", HistGradientBoostingRegressor(random_state=0))]:
    pipe = Pipeline([("p", ct), ("m", est)])
    r1 = -cross_val_score(pipe, T[NUM + CAT], T.co2_gkm, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_root_mean_squared_error")
    r2 = -cross_val_score(pipe, T[NUM + CAT], T.co2_gkm, cv=GroupKFold(5), groups=T.model_family, scoring="neg_root_mean_squared_error")
    res[name] = dict(random_kfold_rmse=float(r1.mean()), group_kfold_rmse=float(r2.mean()))
    print(name, res[name])
out["split_demo"] = res

# (c) missingness mechanism for weight (after cleaning, before imputation)
u["band"] = pd.cut(u.model_year, [2014, 2017, 2020, 2024], labels=["2015-17", "2018-20", "2021-24"])
out["weight_missing_by_band_pct"] = (u.groupby("band")["weight_kg"].apply(lambda s: s.isna().mean() * 100)).round(1).to_dict()
out["hp_missing_by_band_pct"] = (u.groupby("band")["horsepower"].apply(lambda s: s.isna().mean() * 100)).round(1).to_dict()
# class balance + first rows of raw
raw = pd.read_csv("data/raw_vehicles.csv")
out["raw_head"] = raw.head(5).to_dict("records")
out["engine_iqr_example"] = dict(q1=float(d.engine_size_l.quantile(.25)), q3=float(d.engine_size_l.quantile(.75)))
json.dump(out, open("results/extra_analyses.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: v for k, v in out.items() if k != "raw_head"}, indent=1))

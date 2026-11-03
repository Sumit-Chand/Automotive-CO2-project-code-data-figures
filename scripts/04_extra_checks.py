"""Extra evaluation check: does imputation of weight/horsepower/engine hurt accuracy?"""
import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
df = pd.read_csv("data/clean_unimputed.csv")
NUM = ["model_year","engine_size_l","cylinders","horsepower","weight_kg","turbo"]
CAT = ["make","vehicle_class","transmission","drivetrain","fuel_type","powertrain"]
te = pd.read_csv("results/test_indices.csv")["idx"].values
T = df.loc[te].copy()
model = joblib.load("models/best_co2_regressor.joblib")
T["pred"] = model.predict(T[NUM+CAT])
T["res"] = T.co2_gkm - T.pred
miss_num = T[["weight_kg","horsepower","engine_size_l","cylinders"]].isna().any(axis=1)
miss_w = T["weight_kg"].isna()
out = {}
for name, m in [("any_numeric_imputed", miss_num), ("weight_imputed", miss_w)]:
    out[name] = dict(n_with=int(m.sum()), rmse_with=float(np.sqrt((T.res[m]**2).mean())),
                     n_without=int((~m).sum()), rmse_without=float(np.sqrt((T.res[~m]**2).mean())))
json.dump(out, open("results/extra_checks.json","w"), indent=1)
print(json.dumps(out, indent=1))

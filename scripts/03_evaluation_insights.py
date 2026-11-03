"""
03_evaluation_insights.py  (Week 4)
-----------------------------------
Critical evaluation of the Week-3 model and translation of results into business insight.

  1. Segment-level error analysis (where is the model weak?)
  2. Prediction intervals (quantile gradient boosting vs. conformal-style residual interval)
  3. Forward-looking validation (train on model years <= 2021, test on 2022-2024)
  4. What-if scenario analysis (engineering levers)
  5. Fleet-average simulation (engineering + product-mix levers)
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
PAL = ["#1F4E79", "#2E86AB", "#F18F01", "#C73E1D", "#6A994E", "#7B2D8E", "#8D99AE"]
SEED = 42
R = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


df = pd.read_csv("data/clean_unimputed.csv")
NUM = ["model_year", "engine_size_l", "cylinders", "horsepower", "weight_kg", "turbo"]
CAT = ["make", "vehicle_class", "transmission", "drivetrain", "fuel_type", "powertrain"]
FEATS = NUM + CAT
tr = pd.read_csv("results/train_indices.csv")["idx"].values
te = pd.read_csv("results/test_indices.csv")["idx"].values
Xtr, ytr, gtr = df.loc[tr, FEATS], df.loc[tr, "co2_gkm"], df.loc[tr, "model_family"]
Xte, yte = df.loc[te, FEATS].copy(), df.loc[te, "co2_gkm"]
meta_te = df.loc[te].copy()

model = joblib.load("models/best_co2_regressor.joblib")
pred = model.predict(Xte)
meta_te["pred"] = pred
meta_te["res"] = meta_te["co2_gkm"] - pred
meta_te["abs_err"] = meta_te["res"].abs()
meta_te["pct_err"] = meta_te["abs_err"] / meta_te["co2_gkm"] * 100
R["overall"] = dict(rmse=rmse(yte, pred), mae=float(mean_absolute_error(yte, pred)), r2=float(r2_score(yte, pred)), bias=float(meta_te["res"].mean()),
                    share_err_gt15=float((meta_te["abs_err"] > 15).mean() * 100), share_err_gt20=float((meta_te["abs_err"] > 20).mean() * 100))

# -----------------------------------------------------------------------------
# 1. SEGMENT ERROR ANALYSIS
# -----------------------------------------------------------------------------
def seg(col, order=None):
    g = meta_te.groupby(col).apply(lambda s: pd.Series(dict(n=len(s), rmse=rmse(s["co2_gkm"], s["pred"]), mae=s["abs_err"].mean(),
                                                           mape=s["pct_err"].mean(), bias=s["res"].mean(), mean_co2=s["co2_gkm"].mean())))
    return g.loc[order] if order is not None else g


seg_class = seg("vehicle_class")
seg_pt = seg("powertrain", ["ICE", "Mild hybrid", "Full hybrid"])
seg_fuel = seg("fuel_type")
seg_dt = seg("drivetrain")
meta_te["year_band"] = pd.cut(meta_te["model_year"], [2014, 2017, 2020, 2024], labels=["2015-17", "2018-20", "2021-24"])
seg_year = seg("year_band")
meta_te["make_group"] = meta_te["make"]
seg_make = seg("make")
R["seg_class"] = seg_class.round(2).to_dict("index")
R["seg_powertrain"] = seg_pt.round(2).to_dict("index")
R["seg_fuel"] = seg_fuel.round(2).to_dict("index")
R["seg_drivetrain"] = seg_dt.round(2).to_dict("index")
R["seg_year"] = seg_year.round(2).to_dict("index")
R["seg_make_rmse_range"] = [float(seg_make["rmse"].min()), float(seg_make["rmse"].max()), seg_make["rmse"].idxmin(), seg_make["rmse"].idxmax()]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
sc = seg_class.sort_values("rmse")
axes[0].barh(sc.index, sc["rmse"], color=PAL[1])
axes[0].scatter(sc["bias"], range(len(sc)), color=PAL[3], zorder=3, label="mean bias (actual - predicted)")
axes[0].axvline(R["overall"]["rmse"], color="black", ls="--", lw=0.9); axes[0].text(R["overall"]["rmse"] + 0.2, -0.45, "overall RMSE", fontsize=8)
axes[0].axvline(0, color="grey", lw=0.6)
axes[0].set_xlabel("RMSE (g/km)"); axes[0].set_title("Error by vehicle class"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
sp = seg_pt
axes[1].bar(sp.index, sp["rmse"], color=PAL[0], width=0.55)
axes[1].scatter(range(len(sp)), sp["bias"], color=PAL[3], zorder=3)
axes[1].axhline(0, color="grey", lw=0.6)
for i, (r_, b_) in enumerate(zip(sp["rmse"], sp["mape"])):
    axes[1].text(i, r_ + 0.2, f"MAPE {b_:.1f}%", ha="center", fontsize=8)
axes[1].set_ylabel("RMSE (g/km)"); axes[1].set_title("Error by powertrain")
save(fig, "fig20_error_by_segment")

worst = meta_te.sort_values("abs_err", ascending=False).head(8)[["vehicle_class", "make", "model_year", "engine_size_l", "weight_kg", "powertrain", "fuel_type", "co2_gkm", "pred", "res"]]
worst["pred"] = worst["pred"].round(0); worst["res"] = worst["res"].round(0)
R["worst_cases"] = json.loads(worst.to_json(orient="records"))
R["worst_cases_share_hybrid_or_diesel"] = float(((meta_te.sort_values("abs_err", ascending=False).head(50)["powertrain"] != "ICE") | (meta_te.sort_values("abs_err", ascending=False).head(50)["fuel_type"] == "Diesel")).mean() * 100)

# -----------------------------------------------------------------------------
# 2. PREDICTION INTERVALS
# -----------------------------------------------------------------------------
def prep_gb():
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("cat", cat_pipe, CAT), ("num", "passthrough", NUM)])
    return Pipeline([("ct", ct), ("imp", IterativeImputer(max_iter=10, random_state=SEED, skip_complete=True))])


q = {}
for a in (0.05, 0.95):
    m = Pipeline([("prep", prep_gb()), ("model", HistGradientBoostingRegressor(loss="quantile", quantile=a, learning_rate=0.05, max_iter=300,
                                                                          max_leaf_nodes=15, min_samples_leaf=20, random_state=SEED))])
    m.fit(Xtr, ytr)
    q[a] = m.predict(Xte)
lo_q, hi_q = np.minimum(q[0.05], q[0.95]), np.maximum(q[0.05], q[0.95])
cov_q = ((yte.values >= lo_q) & (yte.values <= hi_q))

oof = cross_val_predict(model, Xtr, ytr, groups=gtr, cv=GroupKFold(n_splits=5))
q90 = float(np.quantile(np.abs(ytr.values - oof), 0.90))
lo_c, hi_c = pred - q90, pred + q90
cov_c = ((yte.values >= lo_c) & (yte.values <= hi_c))

# relative (multiplicative) conformal interval: |res|/pred, matches the log-linear structure
rel = np.abs(ytr.values - oof) / oof
q90r = float(np.quantile(rel, 0.90))
lo_r, hi_r = pred * (1 - q90r), pred * (1 + q90r)
cov_r = ((yte.values >= lo_r) & (yte.values <= hi_r))
meta_te["cov_q"], meta_te["cov_c"], meta_te["cov_r"] = cov_q, cov_c, cov_r
R["intervals"] = dict(
    nominal=90.0,
    quantile_gb=dict(coverage=float(cov_q.mean() * 100), mean_width=float((hi_q - lo_q).mean())),
    conformal_abs=dict(coverage=float(cov_c.mean() * 100), mean_width=float(2 * q90), q=q90),
    conformal_rel=dict(coverage=float(cov_r.mean() * 100), mean_width=float((hi_r - lo_r).mean()), q_pct=q90r * 100),
    coverage_by_class_conf_abs=meta_te.groupby("vehicle_class")["cov_c"].mean().mul(100).round(1).to_dict(),
    coverage_by_class_conf_rel=meta_te.groupby("vehicle_class")["cov_r"].mean().mul(100).round(1).to_dict(),
    coverage_by_class_quantile=meta_te.groupby("vehicle_class")["cov_q"].mean().mul(100).round(1).to_dict(),
)
print("Intervals:", json.dumps({k: v for k, v in R["intervals"].items() if not k.startswith("coverage_by")}, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
samp = meta_te.sample(70, random_state=3).sort_values("pred")
xs = np.arange(len(samp))
ix = samp.index.map(lambda i: list(meta_te.index).index(i))
axes[0].fill_between(xs, lo_r[ix], hi_r[ix], color=PAL[1], alpha=0.25, label="90% interval (relative conformal)")
axes[0].plot(xs, samp["pred"], color=PAL[0], lw=1.2, label="prediction")
axes[0].scatter(xs, samp["co2_gkm"], s=14, color=PAL[3], zorder=3, label="actual")
axes[0].set_xlabel("70 random test vehicles (sorted by prediction)"); axes[0].set_ylabel("CO2 (g/km)"); axes[0].set_title("Prediction intervals on test vehicles"); axes[0].legend(frameon=False, fontsize=8)
cc = pd.DataFrame({"Quantile GB": meta_te.groupby("vehicle_class")["cov_q"].mean() * 100,
                   "Absolute conformal": meta_te.groupby("vehicle_class")["cov_c"].mean() * 100,
                   "Relative conformal": meta_te.groupby("vehicle_class")["cov_r"].mean() * 100})
cc.plot.bar(ax=axes[1], color=[PAL[6], PAL[2], PAL[0]], width=0.8)
axes[1].axhline(90, color="black", ls="--", lw=0.9); axes[1].set_ylim(60, 101)
axes[1].set_ylabel("Empirical coverage (%)"); axes[1].set_title("Coverage by vehicle class (target 90%)"); axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
save(fig, "fig21_prediction_intervals")

# -----------------------------------------------------------------------------
# 3. FORWARD-LOOKING (TEMPORAL) VALIDATION
# -----------------------------------------------------------------------------
past = df["model_year"] <= 2021
Xp, yp = df.loc[past, FEATS], df.loc[past, "co2_gkm"]
Xf, yf = df.loc[~past, FEATS], df.loc[~past, "co2_gkm"]
tm = clone(model).fit(Xp, yp)
pf = tm.predict(Xf)
# same rows scored by group-split model for reference (only rows that are in the group-split test set)
R["temporal"] = dict(n_train=int(past.sum()), n_test=int((~past).sum()), rmse=rmse(yf, pf), mae=float(mean_absolute_error(yf, pf)), r2=float(r2_score(yf, pf)),
                     bias=float((yf - pf).mean()), by_year={int(y): dict(rmse=rmse(yf[Xf.model_year == y], pf[(Xf.model_year == y).values]), bias=float((yf[Xf.model_year == y] - pf[(Xf.model_year == y).values]).mean()))
                                                          for y in sorted(Xf.model_year.unique())})
# per-component temporal check: ridge-log only and GB only
for name in ("ridge_log", "gbm"):
    comp = clone(dict(model.named_estimators_ if hasattr(model, "named_estimators_") else model.estimators)[name]) if False else None
ridge_only = clone(model.estimators[0][1]).fit(Xp, yp)
gb_only = clone(model.estimators[1][1]).fit(Xp, yp)
R["temporal"]["ridge_log_rmse"] = rmse(yf, ridge_only.predict(Xf)); R["temporal"]["ridge_log_bias"] = float((yf - ridge_only.predict(Xf)).mean())
R["temporal"]["gb_rmse"] = rmse(yf, gb_only.predict(Xf)); R["temporal"]["gb_bias"] = float((yf - gb_only.predict(Xf)).mean())
print("temporal", {k: v for k, v in R["temporal"].items() if k != "by_year"})

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
axes[0].scatter(yf, pf, s=9, alpha=0.4, color=PAL[0])
lim = [yf.min() - 5, yf.max() + 5]; axes[0].plot(lim, lim, color=PAL[3]); axes[0].set_xlim(lim); axes[0].set_ylim(lim)
axes[0].set_xlabel("Actual CO2 (g/km)"); axes[0].set_ylabel("Predicted (g/km)"); axes[0].set_title(f"Train <= 2021, test 2022-24\nR2={R['temporal']['r2']:.3f}  RMSE={R['temporal']['rmse']:.1f}")
yrs = sorted(R["temporal"]["by_year"])
axes[1].bar([str(y) for y in yrs], [R["temporal"]["by_year"][y]["bias"] for y in yrs], color=PAL[2])
axes[1].axhline(0, color="grey"); axes[1].set_ylabel("Mean bias (actual - predicted, g/km)"); axes[1].set_title("Bias by forecast year")
cmp = pd.DataFrame({"Random group split": [R["overall"]["rmse"]], "Forward-looking (temporal)": [R["temporal"]["rmse"]],
                    "Ridge-log only (temporal)": [R["temporal"]["ridge_log_rmse"]], "GB only (temporal)": [R["temporal"]["gb_rmse"]]}).T
axes[2].barh(cmp.index, cmp[0], color=[PAL[1], PAL[0], PAL[4], PAL[6]]); axes[2].invert_yaxis()
for i, v in enumerate(cmp[0]):
    axes[2].text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8)
axes[2].set_xlabel("RMSE (g/km)"); axes[2].set_title("Validation design comparison")
save(fig, "fig22_temporal_validation")

# -----------------------------------------------------------------------------
# 4. WHAT-IF SCENARIOS (engineering levers)
# -----------------------------------------------------------------------------
base = pred.copy()
rng = np.random.default_rng(1)


def scenario(name, mask, edit):
    Xs = Xte.copy()
    Xs = edit(Xs, mask)
    new = model.predict(Xs)
    d = (new - base)[mask]
    pct = d / base[mask] * 100
    bs = [pct[rng.integers(0, len(pct), len(pct))].mean() for _ in range(500)]
    return dict(name=name, n=int(mask.sum()), mean_delta=float(d.mean()), mean_pct=float(pct.mean()), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                new_pred=new)


allm = np.ones(len(Xte), dtype=bool)
ice = (Xte["powertrain"] == "ICE").values
awd = (Xte["drivetrain"] == "AWD").values
sc_defs = [
    ("Cut curb weight by 100 kg", allm, lambda X, m: X.assign(weight_kg=X["weight_kg"] - 100)),
    ("Cut curb weight by 5 %", allm, lambda X, m: X.assign(weight_kg=X["weight_kg"] * 0.95)),
    ("Downsize engine by 0.5 L (hp unchanged)", (Xte["engine_size_l"] >= 1.8).values, lambda X, m: X.assign(engine_size_l=X["engine_size_l"] - 0.5)),
    ("ICE -> mild hybrid", ice, lambda X, m: X.assign(powertrain="Mild hybrid")),
    ("ICE -> full hybrid", ice, lambda X, m: X.assign(powertrain="Full hybrid")),
    ("AWD -> FWD (where AWD not essential)", awd, lambda X, m: X.assign(drivetrain="FWD")),
    ("Switch to CVT (non-CVT, non-manual)", ((Xte["transmission"] != "CVT") & (Xte["transmission"] != "Manual")).values, lambda X, m: X.assign(transmission="CVT")),
    ("Combined: -5% weight, ICE->mild hybrid, -0.3 L", allm, lambda X, m: X.assign(weight_kg=X["weight_kg"] * 0.95, engine_size_l=(X["engine_size_l"] - 0.3).clip(lower=1.0),
                                                                                  powertrain=np.where(X["powertrain"] == "ICE", "Mild hybrid", X["powertrain"]))),
]
scen = [scenario(*s) for s in sc_defs]
R["scenarios"] = [{k: v for k, v in s.items() if k != "new_pred"} for s in scen]
for s in R["scenarios"]:
    print(f"{s['name']:55s} n={s['n']:4d}  {s['mean_delta']:7.1f} g/km  {s['mean_pct']:6.1f}% [{s['ci_lo']:.1f},{s['ci_hi']:.1f}]")

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ss = sorted(R["scenarios"], key=lambda s: s["mean_pct"])
ax.barh([s["name"] for s in ss], [s["mean_pct"] for s in ss], xerr=[[s["mean_pct"] - s["ci_lo"] for s in ss], [s["ci_hi"] - s["mean_pct"] for s in ss]],
        color=[PAL[4] if "Combined" not in s["name"] else PAL[0] for s in ss], ecolor="grey", capsize=3)
for i, s in enumerate(ss):
    ax.text(s["mean_pct"] - 0.3 if s["mean_pct"] < 0 else s["mean_pct"] + 0.3, i, f"{s['mean_delta']:.1f} g/km", va="center", ha="right" if s["mean_pct"] < 0 else "left", fontsize=8, color="white" if s["mean_pct"] < -6 else "black")
ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("Average change in predicted CO2 (%) for eligible vehicles, 95% bootstrap CI")
ax.set_title("What-if analysis on 1,159 held-out vehicles")
save(fig, "fig23_scenarios")

# -----------------------------------------------------------------------------
# 5. FLEET SIMULATION
# -----------------------------------------------------------------------------
fleet_base = float(base.mean())
ice_share = ice.mean()
hyb_delta_ice = (scen[4]["new_pred"] - base)[ice].mean()      # average change when an ICE car becomes a full hybrid
wt_delta = (scen[1]["new_pred"] - base).mean()                 # fleet-wide -5% weight
combo_delta = (scen[7]["new_pred"] - base).mean()

# product-mix lever: move 10 points of fleet share from heavy SUV/pickup to small SUV
cls_mean = meta_te.groupby("vehicle_class")["pred"].mean()
cls_share = meta_te["vehicle_class"].value_counts(normalize=True)
heavy = ["SUV - Standard", "Pickup Truck"]
heavy_mean = float((cls_mean[heavy] * cls_share[heavy]).sum() / cls_share[heavy].sum())
mix_delta = 0.10 * (cls_mean["SUV - Small"] - heavy_mean)

hist_slope = -1.9737      # g/km per model year from Week-2 EDA trend (results/eda_results.json)
eda = json.load(open("results/eda_results.json"))
hist_slope = eda["trend"]["slope_g_per_year"]
fl = [
    ("Baseline (test portfolio)", 0.0),
    ("+10 pts of sales to full hybrid", 0.10 * hyb_delta_ice),
    ("+20 pts of sales to full hybrid", 0.20 * hyb_delta_ice),
    ("-5 % curb weight, all vehicles", wt_delta),
    ("Mix: 10 pts heavy SUV/pickup -> small SUV", mix_delta),
    ("Combined package (weight, mild hybrids, downsizing)", combo_delta),
    ("Combined + 10 pts full hybrid + mix shift", combo_delta + 0.10 * hyb_delta_ice + mix_delta),
]
fl_rows = [dict(scenario=n, delta=float(d), fleet_co2=float(fleet_base + d), pct=float(d / fleet_base * 100), years_equiv=float(abs(d / hist_slope))) for n, d in fl]
R["fleet"] = dict(baseline=fleet_base, ice_share=float(ice_share), hyb_delta_per_ice_vehicle=float(hyb_delta_ice), heavy_mean=heavy_mean, small_suv_mean=float(cls_mean["SUV - Small"]),
                  hist_slope=float(hist_slope), rows=fl_rows, cls_mean={k: float(v) for k, v in cls_mean.items()})
for r_ in fl_rows:
    print(f"{r_['scenario']:55s} {r_['fleet_co2']:7.1f} g/km  {r_['pct']:6.1f}%  = {r_['years_equiv']:.1f} model-years of historical progress")

fig, ax = plt.subplots(figsize=(9.5, 4.3))
names = [r_["scenario"] for r_ in fl_rows][::-1]
vals = [r_["fleet_co2"] for r_ in fl_rows][::-1]
ax.barh(names, vals, color=[PAL[0] if "Combined +" in n else (PAL[6] if "Baseline" in n else PAL[1]) for n in names])
for i, (v, r_) in enumerate(zip(vals, fl_rows[::-1])):
    ax.text(v + 0.5, i, f"{v:.0f} g/km ({r_['pct']:.1f}%)", va="center", fontsize=8)
ax.set_xlim(min(vals) - 15, max(vals) + 28); ax.set_xlabel("Predicted fleet-average CO2 (g/km)")
ax.set_title("Fleet-average CO2 under alternative strategies (axis truncated)")
save(fig, "fig24_fleet_scenarios")

# -----------------------------------------------------------------------------
# 6. Feature-importance recap for the narrative
# -----------------------------------------------------------------------------
json.dump(R, open("results/evaluation_results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("done")

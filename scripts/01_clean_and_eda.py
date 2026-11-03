"""
01_clean_and_eda.py  (Week 2)
-----------------------------
1. Audit the raw file
2. Clean it (labels, duplicates, impossible values, unit errors, missing values)
3. Benchmark imputation strategies with a masked-value experiment
4. Exploratory data analysis + figures
5. Multicollinearity diagnostics (correlation, VIF, condition number, bootstrap coefficients)

Outputs
  data/clean_unimputed.csv  -> used for modelling (imputation happens INSIDE the CV pipeline)
  data/clean_imputed.csv    -> used for EDA / plotting
  figures/fig*.png, results/eda_results.json
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
PAL = ["#1F4E79", "#2E86AB", "#F18F01", "#C73E1D", "#6A994E", "#7B2D8E", "#8D99AE", "#BC4B51", "#5B8E7D"]
R = {}  # results container


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 1. RAW AUDIT
# =============================================================================
raw = pd.read_csv("data/raw_vehicles.csv")
R["raw_shape"] = list(raw.shape)
R["raw_missing"] = raw.isna().sum()[raw.isna().sum() > 0].to_dict()
R["raw_dtypes"] = raw.dtypes.astype(str).to_dict()
log = []  # cleaning log: (step, rule, rows affected)

df = raw.copy()

# =============================================================================
# 2. CLEANING
# =============================================================================
# 2.1 duplicates
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
log.append(("Remove exact duplicate rows", "df.duplicated()", n_dup))

# 2.2 label standardisation
def norm_label(s):
    return s.str.strip().str.lower().str.replace(r"[\s:\-]+", " ", regex=True)

class_map = {
    "sub compact": "Sub-compact", "compact": "Compact", "mid size": "Mid-size", "full size": "Full-size",
    "suv small": "SUV - Small", "suv standard": "SUV - Standard", "pickup truck": "Pickup Truck",
    "pickup": "Pickup Truck", "minivan": "Minivan", "sports/performance": "Sports/Performance",
}
before = df["vehicle_class"].nunique()
df["vehicle_class"] = norm_label(df["vehicle_class"]).map(class_map)
log.append(("Standardise vehicle_class labels", f"{before} raw spellings -> {df['vehicle_class'].nunique()} classes", int(before - df['vehicle_class'].nunique())))

tmap = {"automatic": "Automatic", "auto": "Automatic", "manual": "Manual", "mt": "Manual", "cvt": "CVT", "dct": "DCT"}
before = df["transmission"].nunique()
df["transmission"] = norm_label(df["transmission"]).map(tmap)
log.append(("Standardise transmission labels", f"{before} raw spellings -> {df['transmission'].nunique()} types", int(before - df['transmission'].nunique())))

fmap = {"regular": "Regular", "premium": "Premium", "diesel": "Diesel"}
before = df["fuel_type"].nunique()
df["fuel_type"] = norm_label(df["fuel_type"]).map(fmap)
log.append(("Standardise fuel_type labels", f"{before} raw spellings -> {df['fuel_type'].nunique()} types", int(before - df['fuel_type'].nunique())))

# 2.3 typo in model year (2105 -> 2015, transposed digits)
m = df["model_year"] > 2030
log.append(("Fix impossible model_year", "2105 -> 2015 (transposed digits); range check 2010-2030", int(m.sum())))
df.loc[m, "model_year"] = df.loc[m, "model_year"].astype(str).str.replace("2105", "2015").astype(int)

# 2.4 weight unit error (pounds entered as kg).  Real curb weights in this segment never exceed ~3,300 kg.
m = df["weight_kg"] > 3300
log.append(("Convert weight entered in pounds", "weight_kg > 3300  ->  value / 2.2046", int(m.sum())))
df.loc[m, "weight_kg"] = (df.loc[m, "weight_kg"] / 2.2046).round()

# 2.5 impossible values -> NaN (kept as missing, treated in imputation step)
rules = {
    "engine_size_l": (0.6, 8.5),
    "horsepower": (30, 1500),
    "weight_kg": (600, 3500),
}
for col, (lo, hi) in rules.items():
    m = (df[col] < lo) | (df[col] > hi)
    log.append((f"Invalid {col} set to missing", f"outside [{lo}, {hi}]", int(m.sum())))
    df.loc[m, col] = np.nan
m = ~df["cylinders"].isin([3, 4, 5, 6, 8, 10, 12])
log.append(("Invalid cylinders set to missing", "not in {3,4,5,6,8,10,12}", int(m.sum())))
df.loc[m, "cylinders"] = np.nan

# 2.6 target sanity: drop rows whose target is impossible (cannot impute the label)
m = (df["co2_gkm"] <= 0) | (df["co2_gkm"] >= 900)
log.append(("Drop rows with invalid target co2_gkm", "<= 0 or sentinel 999", int(m.sum())))
df = df.loc[~m].reset_index(drop=True)

# 2.7 comb_l100km can be reconstructed exactly from CO2 and fuel type (physical identity)
factor = np.where(df["fuel_type"] == "Diesel", 26.5, 23.2)
m = df["comb_l100km"].isna()
log.append(("Rebuild missing comb_l100km", "co2_gkm / emission factor (23.2 gasoline, 26.5 diesel)", int(m.sum())))
df.loc[m, "comb_l100km"] = (df.loc[m, "co2_gkm"] / factor[m]).round(1)

df["cylinders"] = df["cylinders"].astype("float")
df["turbo"] = df["turbo"].astype(int)

# snapshot BEFORE imputation (used for modelling and for the missingness plot)
unimputed = df.copy()
R["missing_after_invalid_to_nan"] = unimputed.isna().sum()[unimputed.isna().sum() > 0].to_dict()
R["rows_after_basic_cleaning"] = int(len(unimputed))

# =============================================================================
# 3. IMPUTATION BENCHMARK (mask known values, measure recovery error)
# =============================================================================
num_feats = ["engine_size_l", "cylinders", "horsepower", "weight_kg", "model_year", "turbo"]
dummies_cols = pd.get_dummies(df["vehicle_class"], prefix="cls").columns.tolist()


def design(frame):
    X = frame[num_feats].copy()
    X = pd.concat([X, pd.get_dummies(frame["vehicle_class"], prefix="cls").reindex(columns=dummies_cols, fill_value=0).astype(float)], axis=1)
    return X


complete = df.dropna(subset=num_feats).reset_index(drop=True)


def run_imputers(target, frac=0.10, seeds=(0, 1, 2, 3, 4)):
    out = {k: [] for k in ["Global median", "Class median", "KNN (k=7)", "Iterative (BayesianRidge)", "Family median*"]}
    for s in seeds:
        rng = np.random.default_rng(s)
        mask = rng.random(len(complete)) < frac
        truth = complete.loc[mask, target].values
        work = complete.copy()
        work.loc[mask, target] = np.nan
        # global
        out["Global median"].append(np.abs(work[target].median() - truth).mean())
        # class median
        cm = work.groupby("vehicle_class")[target].transform("median")
        out["Class median"].append(np.abs(cm[mask].values - truth).mean())
        # family median (not available for a brand-new design -> reference only)
        fm = work.groupby("model_family")[target].transform("median").fillna(cm)
        out["Family median*"].append(np.abs(fm[mask].values - truth).mean())
        X = design(work)
        sc = StandardScaler().fit(X)
        Xs = pd.DataFrame(sc.transform(X), columns=X.columns)
        j = list(X.columns).index(target)
        knn = KNNImputer(n_neighbors=7).fit_transform(Xs)
        out["KNN (k=7)"].append(np.abs(knn[mask, j] * sc.scale_[j] + sc.mean_[j] - truth).mean())
        it = IterativeImputer(max_iter=10, random_state=0).fit_transform(X)
        out["Iterative (BayesianRidge)"].append(np.abs(it[mask, j] - truth).mean())
    return {k: float(np.mean(v)) for k, v in out.items()}


imp_res = {"weight_kg": run_imputers("weight_kg"), "horsepower": run_imputers("horsepower")}
R["imputation_benchmark_MAE"] = imp_res
R["imputation_complete_rows"] = int(len(complete))
print("Imputation benchmark:", json.dumps(imp_res, indent=1))

# =============================================================================
# 3b. APPLY chosen imputation (Iterative) -> clean_imputed.csv  (for EDA only)
# =============================================================================
imp = df.copy()
X = design(imp)
filled = IterativeImputer(max_iter=10, random_state=0).fit_transform(X)
filled = pd.DataFrame(filled, columns=X.columns)
for c in ["engine_size_l", "horsepower", "weight_kg"]:
    imp[c] = filled[c].values
imp["engine_size_l"] = imp["engine_size_l"].round(1)
imp["horsepower"] = imp["horsepower"].round()
imp["weight_kg"] = imp["weight_kg"].round()
# cylinders from engine-size band (data-driven mode)
bands = pd.cut(imp["engine_size_l"], [0, 1.4, 2.9, 4.4, 6.4, 99], labels=False)
cyl_mode = imp.groupby(bands)["cylinders"].agg(lambda s: s.mode().iloc[0])
imp["cylinders"] = imp["cylinders"].fillna(bands.map(cyl_mode))
# categorical: mode within vehicle class
for c in ["transmission", "drivetrain", "fuel_type"]:
    mode_by_class = imp.groupby("vehicle_class")[c].agg(lambda s: s.mode().iloc[0])
    imp[c] = imp[c].fillna(imp["vehicle_class"].map(mode_by_class))

log.append(("Impute numeric gaps (engine, hp, weight)", "IterativeImputer (BayesianRidge) on numeric + class dummies", int(unimputed[["engine_size_l", "horsepower", "weight_kg"]].isna().any(axis=1).sum())))
log.append(("Impute categorical gaps", "mode within vehicle_class", int(unimputed[["transmission", "drivetrain", "fuel_type"]].isna().any(axis=1).sum())))
log.append(("Fill cylinders for invalid / missing engine", "mode of cylinders within engine-size band", int(unimputed["cylinders"].isna().sum())))

# engineered features
for frame in (imp, unimputed):
    frame["power_to_weight"] = (frame["horsepower"] / frame["weight_kg"] * 1000).round(1)       # hp per tonne
    frame["hp_per_litre"] = (frame["horsepower"] / frame["engine_size_l"]).round(1)
    frame["high_emitter"] = (frame["co2_gkm"] >= 200).astype(int)
imp.to_csv("data/clean_imputed.csv", index=False)
unimputed.to_csv("data/clean_unimputed.csv", index=False)
assert imp.drop(columns=[]).isna().sum().sum() == 0, "imputed file must be complete"
R["clean_rows"] = int(len(imp))
R["cleaning_log"] = [dict(step=a, rule=b, rows=c) for a, b, c in log]

# =============================================================================
# 4. EDA
# =============================================================================
d = imp
num_cols = ["engine_size_l", "cylinders", "horsepower", "weight_kg", "co2_gkm"]
R["describe"] = d[["model_year", "engine_size_l", "cylinders", "horsepower", "weight_kg", "comb_l100km", "co2_gkm", "power_to_weight"]].describe().round(2).to_dict()
R["class_counts"] = d["vehicle_class"].value_counts().to_dict()
R["high_emitter_share"] = float(d["high_emitter"].mean())
R["skew_kurt"] = {c: [float(stats.skew(d[c])), float(stats.kurtosis(d[c]))] for c in ["co2_gkm", "horsepower", "engine_size_l", "weight_kg"]}
R["n_families"] = int(d["model_family"].nunique())

# ---- Fig 1: missingness -----------------------------------------------------
miss_raw = raw.isna().mean() * 100
miss_after = unimputed.isna().mean() * 100
cols = [c for c in raw.columns if miss_after.get(c, 0) > 0 or miss_raw[c] > 0]
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(cols))
ax.bar(x - 0.2, [miss_raw[c] for c in cols], 0.4, label="Raw file", color=PAL[1])
ax.bar(x + 0.2, [miss_after.get(c, 0) for c in cols], 0.4, label="After invalid values set to missing", color=PAL[2])
ax.set_xticks(x); ax.set_xticklabels(cols, rotation=30, ha="right")
ax.set_ylabel("% of rows missing"); ax.set_title("Missing values per column")
ax.legend(frameon=False)
save(fig, "fig01_missingness")

# ---- Fig 2: outliers before/after -------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(11, 5))
for j, (c, lab) in enumerate([("engine_size_l", "Engine size (L)"), ("horsepower", "Horsepower"), ("weight_kg", "Weight (kg)"), ("co2_gkm", "CO2 (g/km)")]):
    sns.boxplot(y=raw[c], ax=axes[0, j], color="#D9A5A0", fliersize=2)
    axes[0, j].set_title(f"{lab} - RAW"); axes[0, j].set_ylabel("")
    sns.boxplot(y=d[c], ax=axes[1, j], color=PAL[1], fliersize=2)
    axes[1, j].set_title(f"{lab} - CLEANED"); axes[1, j].set_ylabel("")
save(fig, "fig02_outliers_before_after")

# ---- Statistical outlier screening on cleaned data ---------------------------
out_tbl = {}
for c in ["engine_size_l", "horsepower", "weight_kg", "co2_gkm", "power_to_weight"]:
    q1, q3 = d[c].quantile([.25, .75]); iqr = q3 - q1
    flag = (d[c] < q1 - 1.5 * iqr) | (d[c] > q3 + 1.5 * iqr)
    out_tbl[c] = dict(n_flagged=int(flag.sum()), pct=float(flag.mean() * 100), upper_fence=float(q3 + 1.5 * iqr),
                      top_class=d.loc[flag, "vehicle_class"].value_counts().index[0] if flag.any() else "-",
                      top_class_share=float(d.loc[flag, "vehicle_class"].value_counts(normalize=True).iloc[0]) if flag.any() else 0)
R["iqr_outliers"] = out_tbl
iso = IsolationForest(contamination=0.01, random_state=0).fit(d[["engine_size_l", "horsepower", "weight_kg", "co2_gkm"]])
d["iso_flag"] = iso.predict(d[["engine_size_l", "horsepower", "weight_kg", "co2_gkm"]]) == -1
R["isoforest"] = dict(n_flagged=int(d["iso_flag"].sum()), by_class=d.loc[d["iso_flag"], "vehicle_class"].value_counts().to_dict())
# within-class IQR on CO2 (context-aware)
def within_class_flags(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return (s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)
wc = d.groupby("vehicle_class")["co2_gkm"].transform(within_class_flags)
R["within_class_co2_outliers"] = int(wc.sum())
d.drop(columns="iso_flag").to_csv("data/clean_imputed.csv", index=False)

# ---- Fig 3: distributions ---------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, (c, lab, col) in zip(axes.ravel(), [("co2_gkm", "CO2 (g/km)", PAL[0]), ("comb_l100km", "Combined consumption (L/100 km)", PAL[1]),
                                              ("engine_size_l", "Engine size (L)", PAL[2]), ("horsepower", "Horsepower", PAL[3]),
                                              ("weight_kg", "Curb weight (kg)", PAL[4]), ("power_to_weight", "Power-to-weight (hp/tonne)", PAL[5])]):
    sns.histplot(d[c], kde=True, ax=ax, color=col, bins=35)
    ax.set_xlabel(lab); ax.set_ylabel("Count")
    ax.set_title(f"skew = {stats.skew(d[c]):.2f}")
save(fig, "fig03_distributions")

# ---- Fig 4: CO2 by class -----------------------------------------------------
order = d.groupby("vehicle_class")["co2_gkm"].median().sort_values().index
fig, ax = plt.subplots(figsize=(9, 4.6))
sns.boxplot(data=d, y="vehicle_class", x="co2_gkm", order=order, palette="Blues", ax=ax, fliersize=2)
ax.axvline(200, color=PAL[3], ls="--", lw=1.2); ax.text(201, -0.45, "200 g/km 'high emitter' line", color=PAL[3], fontsize=8)
ax.set_xlabel("CO2 emissions (g/km)"); ax.set_ylabel("")
ax.set_title("CO2 emissions by vehicle class")
save(fig, "fig04_co2_by_class")
R["co2_by_class"] = d.groupby("vehicle_class")["co2_gkm"].agg(["count", "mean", "median", "std"]).round(1).sort_values("median").to_dict("index")

# ---- Fig 5: powertrain / fuel / drivetrain -----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for ax, c, o in [(axes[0], "powertrain", ["ICE", "Mild hybrid", "Full hybrid"]), (axes[1], "fuel_type", ["Regular", "Premium", "Diesel"]), (axes[2], "drivetrain", ["FWD", "RWD", "AWD"])]:
    sns.boxplot(data=d, x=c, y="co2_gkm", order=o, palette="Set2", ax=ax, fliersize=2)
    ax.set_xlabel(c.replace("_", " ").title()); ax.set_ylabel("CO2 (g/km)" if c == "powertrain" else "")
save(fig, "fig05_co2_by_powertrain_fuel_drive")
R["co2_by_powertrain"] = d.groupby("powertrain")["co2_gkm"].agg(["count", "mean", "median"]).round(1).to_dict("index")
R["co2_by_fuel"] = d.groupby("fuel_type")["co2_gkm"].agg(["count", "mean", "median"]).round(1).to_dict("index")
R["co2_by_drivetrain"] = d.groupby("drivetrain")["co2_gkm"].agg(["count", "mean", "median"]).round(1).to_dict("index")
R["co2_by_transmission"] = d.groupby("transmission")["co2_gkm"].agg(["count", "mean", "median"]).round(1).to_dict("index")

# within-class hybrid benefit (controls for class mix)
piv = d.groupby(["vehicle_class", "powertrain"])["co2_gkm"].mean().unstack()
benefit = ((piv["Full hybrid"] - piv["ICE"]) / piv["ICE"] * 100).dropna()
R["hybrid_vs_ice_pct_by_class"] = benefit.round(1).to_dict()
R["hybrid_vs_ice_pct_avg"] = float(benefit.mean())
awd = ((d[d.drivetrain == "AWD"].groupby("vehicle_class")["co2_gkm"].mean() / d[d.drivetrain == "FWD"].groupby("vehicle_class")["co2_gkm"].mean() - 1) * 100).dropna()
R["awd_vs_fwd_pct_by_class"] = awd.round(1).to_dict()
R["awd_vs_fwd_pct_avg"] = float(awd.mean())

# significance tests
H, p = stats.kruskal(*[g["co2_gkm"].values for _, g in d.groupby("vehicle_class")])
k = d["vehicle_class"].nunique()
R["kruskal_class"] = dict(H=float(H), p=float(p), eta2=float((H - k + 1) / (len(d) - k)))
H2, p2 = stats.kruskal(*[g["co2_gkm"].values for _, g in d.groupby("powertrain")])
R["kruskal_powertrain"] = dict(H=float(H2), p=float(p2))

# ---- Fig 6: key relationships ------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
rel = [("weight_kg", "Curb weight (kg)"), ("engine_size_l", "Engine size (L)"), ("horsepower", "Horsepower"), ("power_to_weight", "Power-to-weight (hp/tonne)")]
R["pearson_with_co2"] = {}
for ax, (c, lab) in zip(axes.ravel(), rel):
    r = np.corrcoef(d[c], d["co2_gkm"])[0, 1]
    rho = stats.spearmanr(d[c], d["co2_gkm"])[0]
    R["pearson_with_co2"][c] = dict(pearson=float(r), spearman=float(rho))
    sns.scatterplot(data=d, x=c, y="co2_gkm", hue="powertrain", hue_order=["ICE", "Mild hybrid", "Full hybrid"], palette=[PAL[0], PAL[2], PAL[4]], alpha=0.45, s=14, ax=ax, edgecolor=None)
    sns.regplot(data=d, x=c, y="co2_gkm", scatter=False, lowess=False, color="black", line_kws=dict(lw=1.2), ax=ax)
    ax.set_xlabel(lab); ax.set_ylabel("CO2 (g/km)"); ax.set_title(f"Pearson r = {r:.2f}   Spearman = {rho:.2f}")
    if c != "weight_kg":
        ax.get_legend().remove()
save(fig, "fig06_scatter_relationships")

# ---- Fig 7: trends -----------------------------------------------------------
yr = d.groupby("model_year")["co2_gkm"].agg(["mean", "std", "count"])
yr["ci"] = 1.96 * yr["std"] / np.sqrt(yr["count"])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
axes[0].plot(yr.index, yr["mean"], marker="o", color=PAL[0])
axes[0].fill_between(yr.index, yr["mean"] - yr["ci"], yr["mean"] + yr["ci"], alpha=0.2, color=PAL[0])
axes[0].set_title("Mean CO2 by model year (95% CI)"); axes[0].set_ylabel("g/km"); axes[0].set_xlabel("Model year")
pt = pd.crosstab(d["model_year"], d["powertrain"], normalize="index") * 100
pt[["ICE", "Mild hybrid", "Full hybrid"]].plot.bar(stacked=True, ax=axes[1], color=[PAL[0], PAL[2], PAL[4]], width=0.8)
axes[1].set_title("Powertrain mix (%)"); axes[1].set_xlabel("Model year"); axes[1].legend(frameon=False, fontsize=8)
tr = pd.DataFrame({"Turbo": d.groupby("model_year")["turbo"].mean() * 100,
                   "Manual": d.groupby("model_year")["transmission"].apply(lambda s: (s == "Manual").mean() * 100),
                   "CVT": d.groupby("model_year")["transmission"].apply(lambda s: (s == "CVT").mean() * 100)})
tr.plot(marker="o", ax=axes[2], color=[PAL[3], PAL[6], PAL[1]])
axes[2].set_title("Technology share (%)"); axes[2].set_xlabel("Model year"); axes[2].legend(frameon=False, fontsize=8)
save(fig, "fig07_trends")
y15, y24 = yr.loc[2015, "mean"], yr.loc[2024, "mean"]
R["trend"] = dict(co2_2015=float(y15), co2_2024=float(y24), pct_change=float((y24 / y15 - 1) * 100),
                  hybrid_share_2015=float(100 - pt.loc[2015, "ICE"]), hybrid_share_2024=float(100 - pt.loc[2024, "ICE"]),
                  weight_2015=float(d[d.model_year == 2015].weight_kg.mean()), weight_2024=float(d[d.model_year == 2024].weight_kg.mean()),
                  hp_2015=float(d[d.model_year == 2015].horsepower.mean()), hp_2024=float(d[d.model_year == 2024].horsepower.mean()),
                  turbo_2015=float(tr.loc[2015, "Turbo"]), turbo_2024=float(tr.loc[2024, "Turbo"]),
                  manual_2015=float(tr.loc[2015, "Manual"]), manual_2024=float(tr.loc[2024, "Manual"]))
slope = stats.linregress(yr.index, yr["mean"])
R["trend"]["slope_g_per_year"] = float(slope.slope); R["trend"]["slope_p"] = float(slope.pvalue)

# ---- Fig 10: imputation benchmark --------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 3.8))
meth = list(imp_res["weight_kg"].keys())
x = np.arange(len(meth))
ax.bar(x - 0.2, [imp_res["weight_kg"][m_] for m_ in meth], 0.4, color=PAL[0], label="weight_kg (MAE, kg)")
ax.bar(x + 0.2, [imp_res["horsepower"][m_] for m_ in meth], 0.4, color=PAL[2], label="horsepower (MAE, hp)")
for i, m_ in enumerate(meth):
    ax.text(i - 0.2, imp_res["weight_kg"][m_] + 1, f"{imp_res['weight_kg'][m_]:.0f}", ha="center", fontsize=8)
    ax.text(i + 0.2, imp_res["horsepower"][m_] + 1, f"{imp_res['horsepower'][m_]:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(meth, rotation=15, ha="right"); ax.set_ylabel("Mean absolute error")
ax.set_title("Imputation benchmark: error when 10% of known values are hidden"); ax.legend(frameon=False)
save(fig, "fig08_imputation_benchmark")

# =============================================================================
# 5. MULTICOLLINEARITY
# =============================================================================
leak = ["city_l100km", "hwy_l100km", "comb_l100km", "comb_mpg"]
design_feats = ["model_year", "engine_size_l", "cylinders", "horsepower", "weight_kg", "turbo"]
corr_cols = design_feats + leak + ["co2_gkm"]
C = d[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8.2, 6.6))
sns.heatmap(C, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot_kws=dict(size=7.5), cbar_kws=dict(shrink=0.8))
ax.set_title("Pearson correlation matrix (numeric variables)")
save(fig, "fig09_correlation_heatmap")
R["corr_pairs"] = {
    "engine-cylinders": float(C.loc["engine_size_l", "cylinders"]),
    "engine-horsepower": float(C.loc["engine_size_l", "horsepower"]),
    "engine-weight": float(C.loc["engine_size_l", "weight_kg"]),
    "horsepower-weight": float(C.loc["horsepower", "weight_kg"]),
    "cylinders-horsepower": float(C.loc["cylinders", "horsepower"]),
    "comb_l100km-co2": float(C.loc["comb_l100km", "co2_gkm"]),
    "city-co2": float(C.loc["city_l100km", "co2_gkm"]),
    "hwy-co2": float(C.loc["hwy_l100km", "co2_gkm"]),
    "comb_mpg-co2": float(C.loc["comb_mpg", "co2_gkm"]),
}


def vif(frame, cols):
    out = {}
    Z = (frame[cols] - frame[cols].mean()) / frame[cols].std()
    for c in cols:
        y = Z[c].values
        Xo = Z.drop(columns=c).values
        Xo = np.c_[np.ones(len(Xo)), Xo]
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        r2 = 1 - ((y - Xo @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = float(1 / max(1 - r2, 1e-6))
    return out


vif_all = vif(d, design_feats + leak)
vif_design = vif(d, design_feats)
vif_reduced = vif(d, [c for c in design_feats if c != "cylinders"])
R["vif_all_with_leakage"] = vif_all
R["vif_design"] = vif_design
R["vif_reduced_no_cylinders"] = vif_reduced

def cond_number(frame, cols):
    Z = ((frame[cols] - frame[cols].mean()) / frame[cols].std()).values
    s = np.linalg.svd(Z, compute_uv=False)
    return float(s.max() / s.min())

R["condition_number"] = dict(with_leakage=cond_number(d, design_feats + leak), design=cond_number(d, design_feats),
                             reduced=cond_number(d, [c for c in design_feats if c != "cylinders"]))
# PCA (eigenvalues of correlation matrix) of design features
ev = np.sort(np.linalg.eigvalsh(d[design_feats].corr().values))[::-1]
R["pca_explained_ratio_design"] = (ev / ev.sum()).round(3).tolist()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
names_all = list(vif_all.keys())
vals = [min(vif_all[n], 1e4) for n in names_all]
axes[0].barh(names_all[::-1], vals[::-1], color=[PAL[3] if n in leak else PAL[0] for n in names_all][::-1])
axes[0].set_xscale("log"); axes[0].axvline(5, color="grey", ls=":"); axes[0].axvline(10, color="black", ls="--")
axes[0].set_title("VIF - all numeric columns (red = rating columns)"); axes[0].set_xlabel("VIF (log scale)")
names_d = list(vif_design.keys())
yy = np.arange(len(names_d))
axes[1].barh(yy + 0.2, [vif_design[n] for n in names_d], 0.4, color=PAL[0], label="with cylinders")
axes[1].barh(yy - 0.2, [vif_reduced.get(n, 0) for n in names_d], 0.4, color=PAL[4], label="cylinders dropped")
axes[1].set_yticks(yy); axes[1].set_yticklabels(names_d)
axes[1].axvline(5, color="grey", ls=":"); axes[1].axvline(10, color="black", ls="--")
axes[1].set_title("VIF - design features only"); axes[1].set_xlabel("VIF"); axes[1].legend(frameon=False, fontsize=8)
save(fig, "fig10_vif")

# bootstrap stability of OLS coefficients
def boot_coefs(cols, B=400, seed=1):
    rng = np.random.default_rng(seed)
    Z = (d[cols] - d[cols].mean()) / d[cols].std()
    yv = ((d["co2_gkm"] - d["co2_gkm"].mean()) / d["co2_gkm"].std()).values
    Xm = np.c_[np.ones(len(Z)), Z.values]
    res = np.zeros((B, len(cols)))
    for b in range(B):
        ix = rng.integers(0, len(Z), len(Z))
        beta, *_ = np.linalg.lstsq(Xm[ix], yv[ix], rcond=None)
        res[b] = beta[1:]
    return pd.DataFrame(res, columns=cols)

cols_a = ["engine_size_l", "cylinders", "horsepower", "weight_kg", "model_year", "turbo"]
cols_b = [c for c in cols_a if c != "cylinders"]
Ba, Bb = boot_coefs(cols_a), boot_coefs(cols_b)
R["bootstrap_coef"] = {c: dict(with_cyl_mean=float(Ba[c].mean()), with_cyl_sd=float(Ba[c].std()),
                               no_cyl_mean=float(Bb[c].mean()), no_cyl_sd=float(Bb[c].std())) for c in cols_b}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, c in zip(axes, ["engine_size_l", "horsepower", "weight_kg"]):
    ax.boxplot([Ba[c], Bb[c]], widths=0.5, patch_artist=True,
           boxprops=dict(facecolor="#BFD7EA"), medianprops=dict(color="black"))
    ax.set_xticks([1, 2]); ax.set_xticklabels(["with\ncylinders", "cylinders\ndropped"])
    ax.set_title(f"{c}\nSD {Ba[c].std():.3f} -> {Bb[c].std():.3f}", fontsize=10)
    ax.set_ylabel("standardised coefficient")
save(fig, "fig11_bootstrap_coefficients")

# ridge vs OLS shrinkage numbers are produced in Week 3.
json.dump(R, open("results/eda_results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("clean rows:", len(imp), "| cleaning log:")
for a, b, c in log:
    print(f"  {a:50s} {c}")
print("class counts", R["class_counts"])
print("trend", R["trend"])
print("hybrid vs ice", R["hybrid_vs_ice_pct_avg"], "awd vs fwd", R["awd_vs_fwd_pct_avg"])
print("VIF design", {k: round(v, 1) for k, v in vif_design.items()})
print("VIF reduced", {k: round(v, 1) for k, v in vif_reduced.items()})
print("VIF all", {k: round(v, 1) for k, v in vif_all.items()})
print("cond", R["condition_number"])
print("boot", {k: (round(v["with_cyl_sd"], 3), round(v["no_cyl_sd"], 3)) for k, v in R["bootstrap_coef"].items()})

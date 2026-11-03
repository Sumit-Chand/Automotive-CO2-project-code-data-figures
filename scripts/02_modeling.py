"""
02_modeling.py  (Week 3)
------------------------
Predict CO2 emissions (g/km) from vehicle design specifications.

Key design decisions
  * Leakage control : city/hwy/comb consumption and MPG are dropped (they are the CO2 measurement in other units).
  * Group-aware split: all model years of one model line stay on the same side of the split (GroupShuffleSplit /
    GroupKFold) -> avoids optimistic scores caused by near-duplicate siblings.
  * Imputation INSIDE the pipeline so that test data never influences imputation statistics.
  * Models: Mean baseline, OLS, Ridge, Ridge on log-target, Random Forest, Hist. Gradient Boosting (+ tuned versions).
  * Extension: classification of 'high emitter' (CO2 >= 200 g/km).
"""
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV, cross_validate, learning_curve
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
PAL = ["#1F4E79", "#2E86AB", "#F18F01", "#C73E1D", "#6A994E", "#7B2D8E", "#8D99AE"]
SEED = 42
R = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1. DATA & SPLIT
# -----------------------------------------------------------------------------
df = pd.read_csv("data/clean_unimputed.csv")
NUM = ["model_year", "engine_size_l", "cylinders", "horsepower", "weight_kg", "turbo"]
CAT = ["make", "vehicle_class", "transmission", "drivetrain", "fuel_type", "powertrain"]
FEATS = NUM + CAT
TARGET = "co2_gkm"
X, y, groups = df[FEATS], df[TARGET], df["model_family"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
tr_idx, te_idx = next(gss.split(X, y, groups))
Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
gtr = groups.iloc[tr_idx]
assert set(groups.iloc[tr_idx]).isdisjoint(set(groups.iloc[te_idx]))
R["split"] = dict(n_total=len(df), n_train=len(Xtr), n_test=len(Xte), families_train=int(gtr.nunique()),
                  families_test=int(groups.iloc[te_idx].nunique()), train_mean=float(ytr.mean()), test_mean=float(yte.mean()),
                  train_std=float(ytr.std()), test_std=float(yte.std()))
R["features"] = dict(numeric=NUM, categorical=CAT)
cv = GroupKFold(n_splits=5)
print("split", R["split"])


# -----------------------------------------------------------------------------
# 2. PREPROCESSING + MODELS
# -----------------------------------------------------------------------------
def prep(scale=False, iterative=True):
    """Imputation -> one-hot -> (optional) scaling. Fitted on training folds only."""
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("cat", cat_pipe, CAT), ("num", "passthrough", NUM)])
    steps = [("ct", ct)]
    steps.append(("imp", IterativeImputer(max_iter=10, random_state=SEED, skip_complete=True) if iterative else SimpleImputer(strategy="median")))
    if scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps)


MEM = joblib.Memory("cache", verbose=0)   # caches the (expensive) imputation step per CV fold


def make(reg, scale=False):
    return Pipeline([("prep", prep(scale)), ("model", reg)], memory=MEM)


models = {
    "Mean baseline": make(DummyRegressor()),
    "Linear regression (OLS)": make(LinearRegression(), scale=True),
    "Ridge (CV alpha)": make(RidgeCV(alphas=np.logspace(-3, 3, 25)), scale=True),
    "Ridge on log(CO2)": Pipeline([("prep", prep(True)), ("model", TransformedTargetRegressor(RidgeCV(alphas=np.logspace(-3, 3, 25)), func=np.log, inverse_func=np.exp))]),
    "Random forest (default)": make(RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=1, random_state=SEED)),
    "Gradient boosting (default)": make(HistGradientBoostingRegressor(random_state=SEED)),
}

scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
cv_rows = []
for name, mdl in models.items():
    t0 = time.time()
    s = cross_validate(mdl, Xtr, ytr, groups=gtr, cv=cv, scoring=scoring, n_jobs=1)
    cv_rows.append(dict(model=name, cv_rmse=-s["test_rmse"].mean(), cv_rmse_sd=s["test_rmse"].std(), cv_mae=-s["test_mae"].mean(),
                        cv_r2=s["test_r2"].mean(), fit_s=time.time() - t0))
    print(f"{name:32s} CV RMSE {-s['test_rmse'].mean():7.2f} +/- {s['test_rmse'].std():5.2f}   R2 {s['test_r2'].mean():.3f}")
cv_df = pd.DataFrame(cv_rows)

# -----------------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (RandomizedSearchCV + GroupKFold)
# -----------------------------------------------------------------------------
rf_space = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [None, 12, 20],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [0.4, 0.6, 0.8, 1.0],
}
hgb_space = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
    "model__max_iter": [200, 400, 700],
    "model__max_leaf_nodes": [8, 15, 31],
    "model__min_samples_leaf": [10, 20, 40],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}
searches = {}
for tag, est, space, n_iter in [
    ("Random forest (tuned)", make(RandomForestRegressor(n_jobs=1, random_state=SEED)), rf_space, 10),
    ("Gradient boosting (tuned)", make(HistGradientBoostingRegressor(random_state=SEED)), hgb_space, 20),
]:
    t0 = time.time()
    rs = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="neg_root_mean_squared_error", cv=cv, random_state=SEED, n_jobs=1, refit=True)
    rs.fit(Xtr, ytr, groups=gtr)
    searches[tag] = rs
    res = pd.DataFrame(rs.cv_results_)
    print(f"{tag}: best CV RMSE {-rs.best_score_:.2f}  params {rs.best_params_}  ({time.time()-t0:.0f}s)")
    cv_df = pd.concat([cv_df, pd.DataFrame([dict(model=tag, cv_rmse=-rs.best_score_,
                       cv_rmse_sd=res.loc[rs.best_index_, "std_test_score"], cv_mae=np.nan, cv_r2=np.nan, fit_s=time.time() - t0)])], ignore_index=True)
    R.setdefault("tuning", {})[tag] = dict(best_params={k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()},
                                           best_cv_rmse=float(-rs.best_score_), n_iter=n_iter, n_candidates_total=int(np.prod([len(v) for v in space.values()])))
    res_out = res[["rank_test_score", "mean_test_score", "std_test_score"] + [c for c in res.columns if c.startswith("param_")]].sort_values("rank_test_score")
    res_out["mean_test_score"] = -res_out["mean_test_score"]
    res_out.head(5).to_csv(f"results/top5_{tag.split()[0].lower()}_search.csv", index=False)
    R["tuning"][tag]["top5"] = json.loads(res_out.head(5).to_json(orient="records"))
    R["tuning"][tag]["worst_cv_rmse"] = float(res_out["mean_test_score"].max())

# simple ensemble: average of the log-linear Ridge and the tuned gradient-boosting model
from sklearn.ensemble import VotingRegressor
BLEND = "Blend (Ridge-log + tuned GB)"
blend = VotingRegressor([("ridge_log", models["Ridge on log(CO2)"]), ("gbm", searches["Gradient boosting (tuned)"].best_estimator_)])
sb = cross_validate(blend, Xtr, ytr, groups=gtr, cv=cv, scoring=scoring, n_jobs=1)
cv_df = pd.concat([cv_df, pd.DataFrame([dict(model=BLEND, cv_rmse=-sb["test_rmse"].mean(), cv_rmse_sd=sb["test_rmse"].std(),
                   cv_mae=-sb["test_mae"].mean(), cv_r2=sb["test_r2"].mean(), fit_s=0.0)])], ignore_index=True)
print(f"{BLEND}: CV RMSE {-sb['test_rmse'].mean():.2f}")

# refit the default models on the full training set -> TEST evaluation
fitted = {}
rows = []
for name, mdl in models.items():
    mdl.fit(Xtr, ytr)
    fitted[name] = mdl
for tag, rs in searches.items():
    fitted[tag] = rs.best_estimator_
fitted[BLEND] = blend.fit(Xtr, ytr)


def metrics(yt, yp):
    return dict(MAE=mean_absolute_error(yt, yp), RMSE=float(np.sqrt(mean_squared_error(yt, yp))), R2=r2_score(yt, yp),
                MAPE=mean_absolute_percentage_error(yt, yp) * 100)


test_rows = []
for name, mdl in fitted.items():
    t0 = time.time()
    p = mdl.predict(Xte)
    m = metrics(yte, p)
    pt = mdl.predict(Xtr)
    m_tr = metrics(ytr, pt)
    test_rows.append(dict(model=name, **m, train_RMSE=m_tr["RMSE"], train_R2=m_tr["R2"]))
test_df = pd.DataFrame(test_rows).merge(cv_df[["model", "cv_rmse", "cv_rmse_sd", "cv_r2"]], on="model")
test_df.to_csv("results/model_comparison.csv", index=False)
R["model_table"] = json.loads(test_df.to_json(orient="records"))
print(test_df.round(3).to_string())

best_name = test_df.loc[test_df["cv_rmse"].idxmin(), "model"]
best = fitted[best_name]
R["best_model"] = best_name
pred = best.predict(Xte)
R["best_test_metrics"] = metrics(yte, pred)
# bootstrap CI for test RMSE / R2
rng = np.random.default_rng(0)
boots = []
for _ in range(1000):
    ix = rng.integers(0, len(yte), len(yte))
    boots.append((np.sqrt(mean_squared_error(yte.values[ix], pred[ix])), r2_score(yte.values[ix], pred[ix]), mean_absolute_error(yte.values[ix], pred[ix])))
boots = np.array(boots)
R["best_test_ci"] = dict(rmse=np.percentile(boots[:, 0], [2.5, 97.5]).tolist(), r2=np.percentile(boots[:, 1], [2.5, 97.5]).tolist(), mae=np.percentile(boots[:, 2], [2.5, 97.5]).tolist())
joblib.dump(best, "models/best_co2_regressor.joblib")

# paired comparison on the test set: squared errors, Wilcoxon signed-rank
def paired(a, b):
    ea = (yte.values - fitted[a].predict(Xte)) ** 2
    eb = (yte.values - fitted[b].predict(Xte)) ** 2
    return float(stats.wilcoxon(ea, eb).pvalue)

R["paired_tests"] = {"Gradient boosting (tuned) vs Ridge on log(CO2)": paired("Gradient boosting (tuned)", "Ridge on log(CO2)"),
                     "Gradient boosting (tuned) vs Random forest (tuned)": paired("Gradient boosting (tuned)", "Random forest (tuned)"),
                     "Ridge on log(CO2) vs Ridge (CV alpha)": paired("Ridge on log(CO2)", "Ridge (CV alpha)"),
                     "Blend vs Ridge on log(CO2)": paired(BLEND, "Ridge on log(CO2)")}

# ridge alpha chosen
R["ridge_alpha"] = float(fitted["Ridge (CV alpha)"].named_steps["model"].alpha_)
R["ridge_log_alpha"] = float(fitted["Ridge on log(CO2)"].named_steps["model"].regressor_.alpha_)

# -----------------------------------------------------------------------------
# 4. FIGURES
# -----------------------------------------------------------------------------
# 4.1 model comparison
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
order = test_df.sort_values("RMSE")
ax[0].barh(order["model"], order["RMSE"], color=[PAL[3] if m == best_name else PAL[1] for m in order["model"]])
for i, v in enumerate(order["RMSE"]):
    ax[0].text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8)
ax[0].set_xlabel("Test RMSE (g/km)  - lower is better"); ax[0].set_title("Hold-out test error"); ax[0].invert_yaxis()
ax[1].barh(order["model"], order["R2"], color=[PAL[3] if m == best_name else PAL[1] for m in order["model"]])
for i, v in enumerate(order["R2"]):
    ax[1].text(max(v, 0) + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
ax[1].set_xlabel("Test R2"); ax[1].set_yticklabels([]); ax[1].set_title("Variance explained"); ax[1].invert_yaxis()
save(fig, "fig12_model_comparison")

# 4.2 predicted vs actual (best) + residuals
res = yte.values - pred
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].scatter(yte, pred, s=10, alpha=0.4, color=PAL[0])
lim = [yte.min() - 5, yte.max() + 5]
ax[0].plot(lim, lim, color=PAL[3], lw=1.2); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Actual CO2 (g/km)"); ax[0].set_ylabel("Predicted CO2 (g/km)"); ax[0].set_title(f"{best_name}\nR2={R['best_test_metrics']['R2']:.3f}  RMSE={R['best_test_metrics']['RMSE']:.1f}")
ax[1].scatter(pred, res, s=10, alpha=0.4, color=PAL[1]); ax[1].axhline(0, color=PAL[3])
ax[1].set_xlabel("Predicted CO2 (g/km)"); ax[1].set_ylabel("Residual (actual - predicted)"); ax[1].set_title("Residuals vs predicted")
sns.histplot(res, kde=True, ax=ax[2], color=PAL[4], bins=35)
ax[2].set_xlabel("Residual (g/km)"); ax[2].set_title(f"Residual distribution (mean {res.mean():.2f}, SD {res.std():.1f})")
save(fig, "fig13_best_model_diagnostics")
R["residuals"] = dict(mean=float(res.mean()), sd=float(res.std()), skew=float(stats.skew(res)), kurtosis=float(stats.kurtosis(res)),
                      within_5=float((np.abs(res) <= 5).mean() * 100), within_10=float((np.abs(res) <= 10).mean() * 100), max_abs=float(np.abs(res).max()))
# heteroscedasticity quick check: correlation of |res| with prediction
R["residuals"]["abs_res_vs_pred_spearman"] = float(stats.spearmanr(np.abs(res), pred)[0])

# 4.3 hyper-parameter search outcomes
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for a, tag in zip(ax, ["Random forest (tuned)", "Gradient boosting (tuned)"]):
    r_ = pd.DataFrame(searches[tag].cv_results_)
    a.plot(np.sort(-r_["mean_test_score"].values), marker="o", color=PAL[0])
    a.axhline(cv_df.loc[cv_df.model == tag.replace("tuned", "default"), "cv_rmse"].values[0], color=PAL[3], ls="--", label="default settings")
    a.set_title(tag + f"\n{len(r_)} random candidates, 5-fold GroupKFold"); a.set_xlabel("Candidate (sorted)"); a.set_ylabel("CV RMSE (g/km)"); a.legend(frameon=False)
save(fig, "fig14_tuning_results")

# 4.4 permutation importance on raw features (test set)
pi = permutation_importance(best, Xte, yte, n_repeats=15, random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=1)
imp = pd.DataFrame(dict(feature=FEATS, rmse_increase=pi.importances_mean, sd=pi.importances_std)).sort_values("rmse_increase", ascending=False)
imp.to_csv("results/permutation_importance.csv", index=False)
R["perm_importance"] = json.loads(imp.round(3).to_json(orient="records"))
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.barh(imp["feature"][::-1], imp["rmse_increase"][::-1], xerr=imp["sd"][::-1], color=PAL[0], ecolor="grey")
ax.set_xlabel("Increase in test RMSE when feature is shuffled (g/km)"); ax.set_title(f"Permutation importance - {best_name}")
save(fig, "fig15_permutation_importance")

# 4.5 partial dependence
fig, ax = plt.subplots(2, 3, figsize=(12, 6.2))
Xpd = Xtr.copy()
for c_ in CAT:
    Xpd[c_] = Xpd[c_].fillna(Xpd[c_].mode().iloc[0])
PartialDependenceDisplay.from_estimator(best, Xpd, ["weight_kg", "engine_size_l", "horsepower", "model_year", "powertrain", "drivetrain"], ax=ax.ravel(), kind="average",
                                        categorical_features=["powertrain", "drivetrain"], grid_resolution=30, n_jobs=1)
for a in ax.ravel():
    a.set_ylabel("Predicted CO2 (g/km)")
save(fig, "fig16_partial_dependence")

# 4.6 learning curve (best)
sizes = np.linspace(0.15, 1.0, 6)
ts, trn, val = learning_curve(best, Xtr, ytr, groups=gtr, cv=GroupKFold(n_splits=5), train_sizes=sizes, scoring="neg_root_mean_squared_error", n_jobs=1)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ts, -trn.mean(1), marker="o", color=PAL[0], label="Training RMSE")
ax.plot(ts, -val.mean(1), marker="o", color=PAL[3], label="Validation RMSE")
ax.fill_between(ts, -val.mean(1) - val.std(1), -val.mean(1) + val.std(1), alpha=0.15, color=PAL[3])
ax.set_xlabel("Number of training rows"); ax.set_ylabel("RMSE (g/km)"); ax.set_title("Learning curve of the selected model"); ax.legend(frameon=False)
save(fig, "fig17_learning_curve")
R["learning_curve"] = dict(sizes=ts.tolist(), train_rmse=(-trn.mean(1)).tolist(), val_rmse=(-val.mean(1)).tolist())

# 4.7 OLS coefficients vs Ridge coefficients (multicollinearity remedy)
prep_lin = fitted["Linear regression (OLS)"].named_steps["prep"]
names = prep_lin.named_steps["ct"].get_feature_names_out()
ols = fitted["Linear regression (OLS)"].named_steps["model"].coef_
rdg = fitted["Ridge (CV alpha)"].named_steps["model"].coef_
cn = pd.DataFrame(dict(feature=[n.split("__")[1] for n in names], OLS=ols, Ridge=rdg))
cn_num = cn[cn.feature.isin(NUM)].set_index("feature")
R["ols_vs_ridge_numeric"] = cn_num.round(3).to_dict("index")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
cn_num.plot.bar(ax=ax, color=[PAL[3], PAL[1]], width=0.7)
ax.set_ylabel("Standardised coefficient (g/km per SD)"); ax.set_title("OLS vs Ridge: numeric coefficients"); ax.set_xlabel("")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
save(fig, "fig18_ols_vs_ridge")

# -----------------------------------------------------------------------------
# 5. EXTENSION - CLASSIFICATION OF HIGH EMITTERS
# -----------------------------------------------------------------------------
yc = df["high_emitter"]
yctr, ycte = yc.iloc[tr_idx], yc.iloc[te_idx]
clf_models = {
    "Majority baseline": Pipeline([("prep", prep(True)), ("model", DummyClassifier(strategy="most_frequent"))]),
    "Logistic regression": Pipeline([("prep", prep(True)), ("model", LogisticRegression(max_iter=2000, C=1.0))]),
    "Gradient boosting classifier": Pipeline([("prep", prep(False)), ("model", HistGradientBoostingClassifier(random_state=SEED, learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20))]),
}
clf_rows = []
for n_, m_ in clf_models.items():
    cvs = cross_validate(m_, Xtr, yctr, groups=gtr, cv=cv, scoring=["f1", "accuracy", "roc_auc"], n_jobs=1)
    m_.fit(Xtr, yctr)
    pc = m_.predict(Xte)
    proba = m_.predict_proba(Xte)[:, 1] if hasattr(m_, "predict_proba") else pc
    clf_rows.append(dict(model=n_, cv_f1=float(cvs["test_f1"].mean()), accuracy=accuracy_score(ycte, pc), precision=precision_score(ycte, pc, zero_division=0),
                         recall=recall_score(ycte, pc), f1=f1_score(ycte, pc), roc_auc=roc_auc_score(ycte, proba)))
# threshold-the-regressor alternative
pc_reg = (pred >= 200).astype(int)
clf_rows.append(dict(model="Regression model + 200 g/km cut-off", cv_f1=np.nan, accuracy=accuracy_score(ycte, pc_reg), precision=precision_score(ycte, pc_reg),
                     recall=recall_score(ycte, pc_reg), f1=f1_score(ycte, pc_reg), roc_auc=roc_auc_score(ycte, pred)))
clf_df = pd.DataFrame(clf_rows)
clf_df.to_csv("results/classification_comparison.csv", index=False)
R["classification_table"] = json.loads(clf_df.to_json(orient="records"))
R["high_emitter_rate"] = dict(train=float(yctr.mean()), test=float(ycte.mean()))
print(clf_df.round(3).to_string())
gb = clf_models["Gradient boosting classifier"]
pg = gb.predict(Xte); prob = gb.predict_proba(Xte)[:, 1]
cm = confusion_matrix(ycte, pg)
R["confusion_gbc"] = cm.tolist()
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0], xticklabels=["<200", ">=200"], yticklabels=["<200", ">=200"])
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual"); ax[0].set_title("Confusion matrix (test)")
for n_, m_ in [("Logistic regression", clf_models["Logistic regression"]), ("Gradient boosting", gb)]:
    fpr, tpr, _ = roc_curve(ycte, m_.predict_proba(Xte)[:, 1])
    ax[1].plot(fpr, tpr, label=f"{n_} (AUC {roc_auc_score(ycte, m_.predict_proba(Xte)[:, 1]):.3f})")
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].set_xlabel("False-positive rate"); ax[1].set_ylabel("True-positive rate"); ax[1].set_title("ROC curve"); ax[1].legend(frameon=False, fontsize=8)
pp, rr, _ = precision_recall_curve(ycte, prob)
ax[2].plot(rr, pp, color=PAL[0]); ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision"); ax[2].set_title("Precision-recall curve (gradient boosting)")
save(fig, "fig19_classification")

# misclassification distance: how far from 200 are the mistakes?
wrong = pg != ycte.values
R["clf_errors"] = dict(n_wrong=int(wrong.sum()), median_abs_dist_to_200=float(np.median(np.abs(yte.values[wrong] - 200))),
                       share_within_10=float((np.abs(yte.values[wrong] - 200) <= 10).mean() * 100))

json.dump(R, open("results/modeling_results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
# keep split indices for Week 4
pd.DataFrame({"idx": te_idx}).to_csv("results/test_indices.csv", index=False)
pd.DataFrame({"idx": tr_idx}).to_csv("results/train_indices.csv", index=False)
print("BEST:", best_name, R["best_test_metrics"], R["best_test_ci"])
print("paired", R["paired_tests"])
print("resid", R["residuals"])
print("ols vs ridge", R["ols_vs_ridge_numeric"])
print("perm", imp.head(6).round(2).to_string())

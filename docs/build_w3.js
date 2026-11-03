const L = require("./lib.js");
const { p, h1, h2, h3, bullets, numbered, code, table, tcap, figure, callout, spacer, titleBlock, build, f0, f1, f2, f3, load } = L;
const M = load("modeling_results.json"), E = load("eda_results.json"), X = load("extra_analyses.json");
const sp = M.split, bt = M.best_test_metrics, ci = M.best_test_ci, rs = M.residuals;
const mt = Object.fromEntries(M.model_table.map((r) => [r.model, r]));
const tn = M.tuning, pt = M.paired_tests;
const pv = (x) => (x < 0.001 ? "< 0.001" : f3(x));

const c = [];
c.push(...titleBlock("Advanced Modelling and Predictive Analysis",
  "Predicting vehicle CO₂ emissions (g/km): model selection, tuning, validation and diagnostics", "Week 3 · Automotive Data Analysis"));

c.push(callout("Result in brief", [
  `Selected model: **blend of a log-target Ridge regression and a tuned gradient-boosting model**, chosen by lowest group-cross-validated RMSE (${f2(mt["Blend (Ridge-log + tuned GB)"].cv_rmse)} g/km).`,
  `On ${sp.n_test.toLocaleString()} test vehicles from ${sp.families_test} model lines never seen in training: **R² = ${f3(bt.R2)}** (95% CI ${f3(ci.r2[0])}–${f3(ci.r2[1])}), **RMSE = ${f2(bt.RMSE)} g/km** (CI ${f2(ci.rmse[0])}–${f2(ci.rmse[1])}), **MAE = ${f2(bt.MAE)} g/km**, MAPE = ${f1(bt.MAPE)}%. The mean-only baseline has RMSE ${f1(mt["Mean baseline"].RMSE)}.`,
  `A classifier for “high emitters” (≥ 200 g/km) reaches **F1 = ${f3(M.classification_table[1].f1)}** (logistic regression).`,
  "A simple log-linear Ridge model is almost as good as the best blend (difference not significant), and the flexible tree models do *not* beat it. The reasons are explained in Section 6.",
]));
c.push(spacer(60));

// 1 ----------------------------------------------------------------------
c.push(h1("1. Problem formulation"));
c.push(p("**Task:** supervised regression. Given specifications known before a vehicle is sold, predict `co2_gkm`. The Week 2 analysis shaped three modelling decisions:"));
c.push(...bullets([
  "**Leakage control.** `city_l100km`, `hwy_l100km`, `comb_l100km` and `comb_mpg` correlate 0.94–0.997 with CO₂ because they are the same measurement in other units. They are excluded; including them would give R² ≈ 1 and a model useless at design stage.",
  "**Group structure.** The 5,823 rows come from 720 model lines, each appearing in up to ten model years with near-identical specifications. Validation must keep all years of a line on the same side of each split.",
  "**Multiplicative effects.** Hybrids and diesel scale CO₂ by a percentage rather than adding a fixed amount, suggesting a log-transformed target.",
]));

// 2 ----------------------------------------------------------------------
c.push(h1("2. Data preparation"));
c.push(h2("2.1 Features"));
c.push(tcap("Features used (12) and how each is prepared inside the pipeline."));
c.push(table(["Feature", "Type", "Missing values", "Encoding / scaling"], [
  ["model_year, engine_size_l, cylinders, horsepower, weight_kg, turbo", "Numeric", "Iterative regression imputer (fitted on training fold only)", "Standardised for linear models; raw for trees"],
  ["make, vehicle_class, transmission, drivetrain, fuel_type, powertrain", "Categorical", "Most-frequent value", "One-hot, unknown categories ignored"],
], [38, 14, 28, 20]));
c.push(h2("2.2 Feature selection"));
c.push(p(`No feature was removed by hand beyond the leakage columns. Redundant candidates were assessed with evidence: cylinders has a VIF of ${f1(E.vif_design.cylinders)} (Week 2) and a permutation importance of only ${f2(M.perm_importance.find((x) => x.feature === "cylinders").rmse_increase)} g/km (Section 6.4), so it adds essentially nothing and is retained only because Ridge and trees tolerate it at no cost. Engine size and horsepower, although correlated (r = ${f2(E.corr_pairs["engine-horsepower"])}), both remain important (13.4 and 2.9 g/km).`));
c.push(h2("2.3 Train/test split"));
c.push(p(`A **group-aware split** (`+"`GroupShuffleSplit`"+`, 80/20, seed 42) sends whole model lines to either side: ${sp.n_train.toLocaleString()} training rows (${sp.families_train} lines) and ${sp.n_test.toLocaleString()} test rows (${sp.families_test} lines). An assertion confirms no line appears in both. The target is similar in both parts (mean ${f1(sp.train_mean)} vs ${f1(sp.test_mean)} g/km; SD ${f1(sp.train_std)} vs ${f1(sp.test_std)}). The test set is used once, after all choices are made. All tuning uses 5-fold \`GroupKFold\` inside the training set.`));
c.push(h3("Why grouping matters: an experiment"));
const sd = X.split_demo;
c.push(table(["Model (default settings)", "CV RMSE, random folds", "CV RMSE, grouped folds", "Optimism of random folds"],
  Object.entries(sd).map(([k, v]) => [k, f2(v.random_kfold_rmse), f2(v.group_kfold_rmse), `${f0((1 - v.random_kfold_rmse / v.group_kfold_rmse) * 100)}% too low`]), [36, 22, 22, 20], { aligns: [L.L, L.R, L.R, L.R] }));
c.push(p("With random folds, the model sees sibling years of the same line in training and appears 12–15% more accurate than it will be on a genuinely new design. All reported numbers therefore use grouped folds.", { before: 100 }));
c.push(h2("2.4 Preprocessing pipeline"));
c.push(...code(`def prep(scale=False):
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct  = ColumnTransformer([("cat", cat, CAT), ("num", "passthrough", NUM)])
    steps = [("ct", ct), ("imp", IterativeImputer(max_iter=10, random_state=42, skip_complete=True))]
    if scale: steps.append(("sc", StandardScaler()))
    return Pipeline(steps)

def make(reg, scale=False):                      # memory caches the slow imputation per CV fold
    return Pipeline([("prep", prep(scale)), ("model", reg)], memory=joblib.Memory("cache"))`));
c.push(p("Putting imputation, encoding and scaling inside the pipeline means each cross-validation fold learns them from its own training part only, so validation rows never influence preprocessing."));

// 3 ----------------------------------------------------------------------
c.push(h1("3. Model selection and justification"));
c.push(tcap("Candidate models, in increasing flexibility."));
c.push(table(["Model", "Why it is included", "Expected strength", "Expected weakness"], [
  ["Mean baseline", "Floor every model must beat", "None", "Ignores all features"],
  ["OLS regression", "EDA shows near-linear relations (r ≈ 0.87)", "Fast, interpretable", "Sensitive to collinearity; additive only"],
  ["Ridge (CV α)", "Remedy for collinearity found in Week 2", "Stable coefficients", "Still additive"],
  ["Ridge on log(CO₂)", "Hybrid/diesel effects are multiplicative: ln CO₂ = β₀ + Σ βⱼxⱼ makes them additive", "Captures percentage effects; extrapolates", "Interactions only if specified"],
  ["Random forest", "Captures non-linearity and interactions without scaling", "Robust to outliers", "Cannot extrapolate; overfits small data"],
  ["Gradient boosting", "State of the art for tabular data", "Flexible, accurate", "More tuning; cannot extrapolate"],
  ["Blend (Ridge-log + GB)", "Different error patterns may cancel", "Combines smooth trend and non-linear corrections", "Harder to explain"],
], [18, 34, 24, 24], { size: 17 }));
c.push(p("**Why the log target.** If a full hybrid cuts consumption by 30% regardless of size, the effect is about −43 g/km on a compact car (143 g/km) but about −83 g/km on a pickup (278 g/km). A linear model on raw CO₂ needs an interaction between powertrain and size to represent this; a linear model on ln(CO₂) represents it with one coefficient. Predictions are transformed back with exp(·), implemented with `TransformedTargetRegressor`.", { before: 100 }));

// 4 ----------------------------------------------------------------------
c.push(h1("4. Training, validation and hyper-parameter tuning"));
c.push(h2("4.1 Baseline and linear models"));
c.push(p(``+"`RidgeCV`"+` selects the penalty α from 25 values between 10⁻³ and 10³ by efficient leave-one-out validation. It chose α = ${f2(M.ridge_alpha)} for raw CO₂ and ${f1(M.ridge_log_alpha)} for log CO₂. These are tiny relative to ${sp.n_train.toLocaleString()} rows, so shrinkage barely changes the fit (Section 6.5), confirming that collinearity does not hurt predictions.`));
c.push(h2("4.2 Randomised hyper-parameter search with grouped cross-validation"));
c.push(p(`The tree-based models have settings that matter. An exhaustive search would be expensive on the single CPU used here, so **randomised search** samples candidate settings: ${tn["Random forest (tuned)"].n_iter} of ${tn["Random forest (tuned)"].n_candidates_total} possible combinations for the random forest and ${tn["Gradient boosting (tuned)"].n_iter} of ${tn["Gradient boosting (tuned)"].n_candidates_total} for gradient boosting, each scored by 5-fold `+"`GroupKFold`"+` RMSE.`));
c.push(...code(`rf_space = {"model__n_estimators": [150, 250], "model__max_depth": [None, 12, 20],
            "model__min_samples_leaf": [1, 2, 4], "model__max_features": [0.4, 0.6, 0.8, 1.0]}
hgb_space = {"model__learning_rate": [0.03, 0.05, 0.08, 0.12],
             "model__max_iter": [200, 400, 700], "model__max_leaf_nodes": [8, 15, 31],
             "model__min_samples_leaf": [10, 20, 40], "model__l2_regularization": [0.0, 0.1, 1.0]}

search = RandomizedSearchCV(make(HistGradientBoostingRegressor(random_state=42)), hgb_space,
                            n_iter=20, scoring="neg_root_mean_squared_error",
                            cv=GroupKFold(n_splits=5), random_state=42, refit=True)
search.fit(X_train, y_train, groups=family_train)        # groups keep model lines together
print(-search.best_score_, search.best_params_)`));
const rf = tn["Random forest (tuned)"], gb = tn["Gradient boosting (tuned)"];
c.push(tcap("Search outcome."));
c.push(table(["Model", "Best CV RMSE", "Worst candidate", "Default settings", "Best settings found"], [
  ["Random forest", f2(rf.best_cv_rmse), f2(rf.worst_cv_rmse), f2(mt["Random forest (default)"].cv_rmse), `trees ${rf.best_params.n_estimators}, min leaf ${rf.best_params.min_samples_leaf}, max features ${rf.best_params.max_features}, depth ${rf.best_params.max_depth === null ? "None" : rf.best_params.max_depth}`],
  ["Gradient boosting", f2(gb.best_cv_rmse), f2(gb.worst_cv_rmse), f2(mt["Gradient boosting (default)"].cv_rmse), `learning rate ${gb.best_params.learning_rate}, ${gb.best_params.max_iter} iterations, ${gb.best_params.max_leaf_nodes} leaves, min leaf ${gb.best_params.min_samples_leaf}, L2 ${gb.best_params.l2_regularization}`],
], [16, 12, 13, 13, 46], { aligns: [L.L, L.R, L.R, L.R, L.L], size: 17 }));
c.push(...figure("fig14_tuning_results", "Cross-validated RMSE of all sampled candidates (sorted) against the default settings (dashed line).", 540));
c.push(p(`Tuning improved gradient boosting from ${f2(mt["Gradient boosting (default)"].cv_rmse)} to ${f2(gb.best_cv_rmse)} g/km and the random forest from ${f2(mt["Random forest (default)"].cv_rmse)} to ${f2(rf.best_cv_rmse)}. The best gradient-boosting setting favours **strong regularisation** (minimum 40 rows per leaf, L2 penalty 1.0) with only 200 iterations, a sign that the underlying signal is smooth and extra flexibility mostly fits noise. Top-five candidate tables are saved in `+"`results/top5_*_search.csv`"+`.`));
c.push(h2("4.3 Blend"));
c.push(...code(`blend = VotingRegressor([("ridge_log", models["Ridge on log(CO2)"]),
                        ("gbm", searches["Gradient boosting (tuned)"].best_estimator_)])
cross_validate(blend, X_train, y_train, groups=family_train, cv=GroupKFold(5), scoring=scoring)`));
c.push(p("`VotingRegressor` averages the two predictions with equal weight."));

// 5 ----------------------------------------------------------------------
c.push(h1("5. Performance results"));
c.push(...figure("fig12_model_comparison", "Hold-out test RMSE and R² for all models. The selected blend is highlighted.", 580));
const order = ["Mean baseline", "Linear regression (OLS)", "Ridge (CV alpha)", "Ridge on log(CO2)", "Random forest (default)", "Random forest (tuned)", "Gradient boosting (default)", "Gradient boosting (tuned)", "Blend (Ridge-log + tuned GB)"];
c.push(tcap("Cross-validated (training) and hold-out test performance. CV = 5-fold GroupKFold mean ± SD."));
c.push(table(["Model", "CV RMSE", "Test MAE", "Test RMSE", "Test R²", "Test MAPE %", "Train RMSE"],
  order.map((k) => [k, `${f2(mt[k].cv_rmse)} ± ${f2(mt[k].cv_rmse_sd)}`, f2(mt[k].MAE), f2(mt[k].RMSE), f3(mt[k].R2), f1(mt[k].MAPE), f2(mt[k].train_RMSE)]),
  [28, 17, 11, 12, 11, 11, 12], { aligns: [L.L, L.R, L.R, L.R, L.R, L.R, L.R], highlightRow: 8, size: 17 }));
c.push(h3("How to read the table"));
c.push(...bullets([
  `**Every model beats the baseline** by a wide margin (RMSE ${f1(mt["Mean baseline"].RMSE)} → about 9–12).`,
  `**Overfitting is visible in the tree models:** the tuned random forest has training RMSE ${f1(mt["Random forest (tuned)"].train_RMSE)} but test RMSE ${f1(mt["Random forest (tuned)"].RMSE)} (a 3× gap); the linear and blended models have train and test errors within about 1 g/km.`,
  `**Log target helps linear models:** Ridge on log(CO₂) cuts test RMSE from ${f2(mt["Ridge (CV alpha)"].RMSE)} to ${f2(mt["Ridge on log(CO2)"].RMSE)} (paired Wilcoxon test on squared errors p = ${pv(pt["Ridge on log(CO2) vs Ridge (CV alpha)"])}).`,
  `**Tuned gradient boosting is significantly worse than log-Ridge** (p ${pv(pt["Gradient boosting (tuned) vs Ridge on log(CO2)"])}) and than the blend.`,
  `**Blend vs log-Ridge:** the blend has the lowest cross-validated RMSE (${f2(mt["Blend (Ridge-log + tuned GB)"].cv_rmse)} vs ${f2(mt["Ridge on log(CO2)"].cv_rmse)}), so it is selected by the pre-declared rule. On the test set the two are statistically indistinguishable (${f2(bt.RMSE)} vs ${f2(mt["Ridge on log(CO2)"].RMSE)}; p = ${f2(pt["Blend vs Ridge on log(CO2)"])}). The selection rule used cross-validation, **not** test results, to avoid choosing on the test set.`,
]));
c.push(h2("5.1 Diagnostics of the selected model"));
c.push(...figure("fig13_best_model_diagnostics", "Predicted vs actual (left), residuals vs predicted (centre) and residual distribution (right) on the test set.", 610));
c.push(p(`Residuals are centred near zero (mean ${f2(rs.mean)} g/km), roughly symmetric (skew ${f2(rs.skew)}) with light tails (excess kurtosis ${f2(rs.kurtosis)}). **${f0(rs.within_5)}%** of vehicles are predicted within ±5 g/km and **${f0(rs.within_10)}%** within ±10 g/km; the largest miss is ${f1(rs.max_abs)} g/km. Residual spread grows slowly with the prediction (Spearman correlation of |residual| with prediction = ${f2(rs.abs_res_vs_pred_spearman)}), i.e. mild heteroscedasticity: errors are roughly proportional to CO₂ (MAPE ${f1(bt.MAPE)}%), as expected for a model with multiplicative structure. This is one reason relative prediction intervals are used in Week 4.`));
c.push(...figure("fig17_learning_curve", "Learning curve of the selected model (5-fold grouped CV).", 400));
const lc = M.learning_curve;
c.push(p(`Validation RMSE falls from ${f1(lc.val_rmse[0])} (${lc.sizes[0]} rows) to ${f1(lc.val_rmse[5])} (${lc.sizes[5].toLocaleString()} rows), while training RMSE rises from ${f1(lc.train_rmse[0])} to ${f1(lc.train_rmse[5])}. The curves are converging with a remaining gap of about ${f1(lc.val_rmse[5] - lc.train_rmse[5])} g/km: variance is not a serious problem and further data would give only modest gains; the remaining error is largely unexplained model-line variation and imputed values.`));

// 6 ----------------------------------------------------------------------
c.push(h1("6. Interpretation"));
c.push(h2("6.1 Why does simple beat flexible here?"));
c.push(p("Three reasons are visible in the evidence. (1) The target is close to log-linear in the features, so the log-Ridge already represents the structure. (2) Tree ensembles approximate smooth trends with steps and cannot extrapolate beyond the training range, e.g. into later model years. (3) There are only 576 training lines, so the extra flexibility is paid for with variance. **Caveat:** the data are simulated from a smooth consumption formula, so this advantage of linear models is likely larger here than in real registration data, where hidden interactions (aerodynamics, gearing, calibration) favour boosting. The pipeline will show this directly if real data are substituted."));
c.push(h2("6.2 Permutation importance"));
c.push(...figure("fig15_permutation_importance", "Increase in test RMSE when each feature is randomly shuffled (selected model, 15 repeats).", 440));
const pi = M.perm_importance;
c.push(p(`Weight dominates (+${f1(pi[0].rmse_increase)} g/km RMSE when shuffled), followed by powertrain (+${f1(pi[1].rmse_increase)}) and engine size (+${f1(pi[2].rmse_increase)}). Make, drivetrain, vehicle class and horsepower contribute 3–4 g/km each, and cylinders essentially nothing (+${f2(pi.find((x) => x.feature === "cylinders").rmse_increase)}), as predicted by the Week 2 multicollinearity analysis. Because permutation importance is computed on the model’s predictions, it is not distorted by correlated predictors the way raw regression coefficients can be, although correlated features can share importance.`));
c.push(h2("6.3 Partial dependence"));
c.push(...figure("fig16_partial_dependence", "Average predicted CO₂ as one feature varies (categorical panels show the average prediction for each category; their vertical axis starts near 130 g/km).", 610));
c.push(p("CO₂ rises almost linearly with weight (about 7 g/km per 100 kg) and engine size (about 12 g/km per litre), falls slowly with model year (about 1 g/km per year after controlling for everything else), and is far lower for full hybrids than ICE vehicles. Horsepower has a small marginal effect once weight and engine size are known, consistent with the weak power-to-weight correlation in Week 2."));
c.push(h2("6.4 Does collinearity distort the linear model?"));
const oc = M.ols_vs_ridge_numeric;
c.push(table(["Feature (standardised)", "OLS coefficient", "Ridge coefficient"], Object.keys(oc).map((k) => [k, f2(oc[k].OLS), f2(oc[k].Ridge)]), [40, 30, 30], { aligns: [L.L, L.R, L.R] }));
c.push(p(`Ridge and OLS coefficients are almost identical and test RMSE is the same to two decimals (${f2(mt["Ridge (CV alpha)"].RMSE)} vs ${f2(mt["Linear regression (OLS)"].RMSE)}). With VIF values below 22, a condition number near 10 and ${sp.n_train.toLocaleString()} training rows, collinearity affects interpretation of individual coefficients only marginally and does not harm predictions. This closes the question raised in Week 2.`));

// 7 ----------------------------------------------------------------------
c.push(h1("7. Extension: classifying high emitters"));
c.push(p(`Regulators and fleet managers often need a yes/no screen. A binary target `+"`high_emitter`"+` (CO₂ ≥ 200 g/km) was created; ${f0(M.high_emitter_rate.train * 100)}% of training and ${f0(M.high_emitter_rate.test * 100)}% of test vehicles are positive, so classes are only mildly imbalanced and accuracy alone would still hide errors (a majority-class model scores 64.7% accuracy with F1 = 0). The same pipeline, split and grouped CV were used.`));
const ct = M.classification_table;
c.push(tcap("Classification results on the test set."));
c.push(table(["Model", "CV F1", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], ct.map((r) => [r.model, r.cv_f1 === null ? "–" : f3(r.cv_f1), f3(r.accuracy), f3(r.precision), f3(r.recall), f3(r.f1), f3(r.roc_auc)]),
  [32, 11, 12, 12, 11, 11, 12], { aligns: [L.L, L.R, L.R, L.R, L.R, L.R, L.R], highlightRow: 1, size: 17 }));
c.push(...figure("fig19_classification", "Confusion matrix, ROC curves and precision–recall curve for the high-emitter task.", 610));
const cm = M.confusion_gbc, ce = M.clf_errors;
c.push(p(`Logistic regression is best (F1 ${f3(ct[1].f1)}, AUC ${f3(ct[1].roc_auc)}), meeting the F1 ≥ 0.90 target (H5). Simply applying the 200 g/km cut-off to the regression prediction gives almost the same F1 (${f3(ct[3].f1)}), so **one regression model can serve both purposes**. For the gradient-boosting classifier, the confusion matrix is ${cm[0][0]} true negatives, ${cm[0][1]} false positives, ${cm[1][0]} false negatives and ${cm[1][1]} true positives. Its ${ce.n_wrong} errors are not random: the median misclassified vehicle is only ${f0(ce.median_abs_dist_to_200)} g/km away from the threshold and ${f0(ce.share_within_10)}% lie within 10 g/km, so mistakes are borderline cases that a “grey zone” rule could flag for review.`));

// 8 ----------------------------------------------------------------------
c.push(h1("8. Challenges encountered"));
c.push(table(["Challenge", "How it was handled"], [
  ["Target leakage through rating columns", "Excluded all four; documented by correlation and VIF in Week 2"],
  ["Optimistic scores from sibling model years", "GroupShuffleSplit and GroupKFold by model line; experiment shows 12–15% optimism otherwise"],
  ["Test-set leakage through imputation", "Imputation moved inside the pipeline, fitted per fold"],
  ["Slow search on one CPU (IterativeImputer refits per candidate)", "`Pipeline(memory=...)` caches each fold’s imputation; randomised search with 10–20 candidates"],
  ["Tree models overfit (train RMSE 3.6 vs test 11.6)", "Tuning with regularisation; kept as blend component only"],
  ["Choosing between near-tied models", "Rule fixed in advance: lowest CV RMSE; paired Wilcoxon tests reported"],
  ["Partial-dependence plot failed on missing categories", "Filled categorical gaps with the mode for plotting only"],
], [38, 62]));

// 9 ----------------------------------------------------------------------
c.push(h1("9. Conclusion and hand-off to Week 4"));
c.push(p(`The final model predicts CO₂ for unseen model lines with RMSE ${f1(bt.RMSE)} g/km (about ${f1((bt.RMSE / sp.test_mean) * 100)}% of the mean) and R² ${f2(bt.R2)}, meeting every Week 1 target (R² ≥ 0.90, RMSE ≤ 12, F1 ≥ 0.90). The main drivers are mass, powertrain and engine size. The evidence also shows that a transparent log-linear model is nearly as accurate as any ensemble, which matters for explainability. Week 4 will (1) examine where the model is weak, (2) quantify uncertainty with prediction intervals, (3) test it forward in time, and (4) use it for what-if and fleet analysis. The fitted model is saved as `+"`models/best_co2_regressor.joblib`"+`.`));

build("../outputs/Week3_Modeling_and_Prediction_Automotive_CO2.docx", { headerText: "Automotive CO₂ Prediction Project · Week 3 – Advanced Modelling and Predictive Analysis", children: c });

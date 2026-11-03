const L = require("./lib.js");
const { p, h1, h2, h3, bullets, numbered, code, table, tcap, figure, callout, spacer, pageBreak, titleBlock, build, f0, f1, f2, f3, load } = L;
const X = load("extra_analyses.json"), E = load("eda_results.json");
const pil = X.pilot, pil3 = X.pilot3;

const c = [];
c.push(...titleBlock("Predicting Vehicle CO₂ Emissions from Design Specifications",
  "Research plan and strategy for an automotive data-science project in Python", "Week 1 · Research, Planning and Strategy"));

c.push(callout("Summary", [
  "**Problem.** Fleet-average CO₂ limits force manufacturers to estimate, early in design, how many grams of CO₂ per kilometre (g/km) a vehicle will emit and which engineering or product-mix decisions reduce that figure most.",
  "**Approach.** Build a leakage-free regression model that predicts CO₂ (g/km) from specifications known before sale (engine, weight, power, powertrain, drivetrain, transmission, year), then use it for what-if analysis and fleet scenarios.",
  "**Plan.** Four weeks, about 32 hours each: plan → clean and explore → model → evaluate and recommend. Success targets are R² ≥ 0.90, RMSE ≤ 12 g/km and F1 ≥ 0.90 for flagging high emitters.",
]));
c.push(spacer(80));

// 1 ---------------------------------------------------------------------
c.push(h1("1. Industry context and problem identification"));
c.push(p("Three pressures define the current passenger-car landscape, and each can be addressed with data science:"));
c.push(...bullets([
  "**Regulation.** Fleet-average CO₂ or fuel-economy rules apply in the EU (a 95 g/km target phased in from 2020), the United States (EPA/NHTSA standards) and India (Corporate Average Fuel Efficiency norms). A manufacturer is judged on its sales-weighted average, so every model decision moves the fleet number.",
  "**Electrification transition.** Mild and full hybrids are growing as a bridge towards battery-electric vehicles. Planners need to know how much CO₂ each step really saves.",
  "**Vehicle mix and mass.** Buyers keep moving towards heavier SUVs and pickups, which offsets engineering gains.",
]));
c.push(h2("1.1 Candidate problems and selection"));
c.push(p("Four problems suggested in the brief were scored on five criteria (1 = poor, 5 = excellent). Weights reflect what matters for a four-week, self-contained, public-data project."));
const crit = [["Public data availability", 0.30], ["Business relevance", 0.20], ["Fit to 4-week scope", 0.20], ["Breadth of techniques it exercises", 0.20], ["Low risk of target leakage / ambiguity", 0.10]];
const cand = [
  ["Predictive maintenance", [2, 4, 3, 4, 3]],
  ["Customer behaviour profiling", [2, 4, 3, 3, 4]],
  ["Used-car price prediction", [4, 3, 5, 3, 4]],
  ["**CO₂ / fuel-efficiency prediction**", [5, 5, 5, 4, 3]],
];
const tot = (s) => s.reduce((a, v, i) => a + v * crit[i][1], 0);
c.push(tcap("Weighted decision matrix. Weights: " + crit.map((x) => `${x[0]} ${x[1]}`).join("; ") + "."));
c.push(table(["Candidate", "Data", "Relevance", "Scope", "Techniques", "Leakage risk", "Weighted score"],
  cand.map((r) => [r[0], ...r[1].map(String), f2(tot(r[1]))]), [30, 9, 11, 9, 12, 12, 14], { aligns: [L.L, L.C, L.C, L.C, L.C, L.C, L.R], highlightRow: 3 }));
c.push(p(`**Worked example (CO₂ prediction):** 5×0.30 + 5×0.20 + 5×0.20 + 4×0.20 + 3×0.10 = 1.50 + 1.00 + 1.00 + 0.80 + 0.30 = **${f2(tot(cand[3][1]))}**. Predictive maintenance reaches only 2×0.30 + 4×0.20 + 3×0.20 + 4×0.20 + 3×0.10 = ${f2(tot(cand[0][1]))}, mainly because open sensor/failure data is scarce. Leakage risk scored 3 for CO₂ because the dataset contains the same measurement in several units (L/100 km, MPG); this is manageable and becomes a learning point (Section 5.4).`, { before: 80 }));

// 2 ---------------------------------------------------------------------
c.push(h1("2. Problem statement, questions and objectives"));
c.push(callout("Problem statement", "Using only specifications that are known before a vehicle reaches the market (engine size, cylinders, power, curb weight, turbo, transmission, drivetrain, powertrain type, fuel type, make, vehicle class and model year), build and validate a model that predicts the vehicle’s tailpipe CO₂ emissions in g/km, quantify the CO₂ effect of the main engineering and product-mix levers, and translate the findings into actions that reduce sales-weighted fleet-average emissions."));
c.push(h2("2.1 Research questions and testable hypotheses"));
c.push(table(["ID", "Research question", "Hypothesis", "How it will be tested"], [
  ["RQ1", "Which specifications explain most of the variation in CO₂?", "H1: Curb weight and engine size are the strongest predictors (|r| > 0.7 with CO₂).", "Pearson/Spearman correlation (Wk 2); permutation importance (Wk 3)"],
  ["RQ2", "How much does electrification reduce CO₂ at equal vehicle class?", "H2: Full hybrids emit 20–35% less than comparable ICE vehicles.", "Within-class group means (Wk 2); model what-if (Wk 4)"],
  ["RQ3", "Is the fleet getting cleaner over time?", "H3: Mean CO₂ falls with model year (negative slope, p < 0.05).", "Linear trend on yearly means (Wk 2); temporal validation (Wk 4)"],
  ["RQ4", "Does all-wheel drive carry a penalty?", "H4: AWD vehicles emit more than FWD within the same class.", "Within-class comparison (Wk 2); scenario AWD→FWD (Wk 4)"],
  ["RQ5", "Can a model flag high emitters (≥ 200 g/km) reliably?", "H5: F1 ≥ 0.90 on unseen model lines.", "Classification extension (Wk 3)"],
], [6, 24, 36, 34]));
c.push(h2("2.2 SMART objectives and success criteria"));
c.push(table(["Objective", "Measure", "Target", "Baseline to beat"], [
  ["Predict CO₂ accurately on unseen model lines", "Test RMSE / R²", "RMSE ≤ 12 g/km; R² ≥ 0.90", "Mean predictor (RMSE ≈ 50 g/km)"],
  ["Flag high emitters", "F1 on held-out lines", "F1 ≥ 0.90", "Majority class (F1 = 0)"],
  ["Quantify uncertainty", "Empirical coverage of 90% intervals", "88–92%", "No interval"],
  ["Produce actionable strategy", "Scenario table with fleet impact in g/km and %", "≥ 5 levers ranked", "Qualitative advice"],
], [34, 26, 22, 18]));

// 3 ---------------------------------------------------------------------
c.push(h1("3. Data sources"));
c.push(p("The study uses publicly available vehicle data. Candidate sources were reviewed for content, access and suitability:"));
c.push(tcap("Public data sources reviewed."));
c.push(table(["Source", "Content", "Access", "Strengths", "Limitations"], [
  ["Natural Resources Canada – Fuel Consumption Ratings (open.canada.ca)", "Make, model, class, engine size, cylinders, transmission, fuel type, city/highway/combined L/100 km, CO₂ g/km, by model year", "Open Government Licence, CSV download", "Closest match to this project; many model years; includes CO₂", "No curb weight or horsepower; ratings are lab based"],
  ["US EPA / DOE fueleconomy.gov “vehicles.csv”", "Fuel economy, CO₂ g/mile, engine displacement, drive type, hybrid flags since 1984", "Free CSV download", "Very large; includes electrified vehicles", "Different units (MPG, g/mile); no weight or power"],
  ["European Environment Agency – CO₂ monitoring of new passenger cars", "Registered new cars: WLTP CO₂, mass, engine capacity, power, fuel", "Open data, large files", "Has mass and power; real registrations", "Very large; heavy preprocessing"],
  ["UCI “Auto MPG” data set", "398 cars (1970–82): mpg, cylinders, displacement, horsepower, weight", "Free download", "Classic teaching set with weight and power", "Old vehicles; very small"],
], [22, 28, 16, 17, 17], { size: 17 }));
c.push(h2("3.1 Data strategy"));
c.push(p("The preferred route is to combine a public ratings file (CO₂, powertrain, drivetrain) with a source that provides mass and power. Because the development environment has no internet access, the project uses a **simulated development dataset** whose structure and ranges mimic these public sources: 720 model lines, model years 2015–2024, nine vehicle classes and realistic data-quality problems. All brand names are anonymised. Every script reads a single CSV with a fixed column set, so a real public file can be dropped in after a column mapping (Appendix A). All quantitative findings in later weeks must therefore be re-validated on real data before use."));
c.push(h2("3.2 Planned data dictionary"));
c.push(table(["Variable", "Type", "Role", "Notes"], [
  ["model_family", "ID", "Grouping", "All years of one model line; used to prevent leakage in validation"],
  ["make, vehicle_class, transmission, drivetrain, fuel_type, powertrain", "Categorical", "Feature", "Labels need standardising"],
  ["model_year, engine_size_l, cylinders, horsepower, weight_kg, turbo", "Numeric", "Feature", "Impossible values and unit errors expected"],
  ["city_l100km, hwy_l100km, comb_l100km, comb_mpg", "Numeric", "**Excluded**", "Same measurement as the target in other units (leakage)"],
  ["co2_gkm", "Numeric", "**Target**", "Tailpipe CO₂ in g/km"],
  ["high_emitter", "Binary", "Secondary target", "1 if co2_gkm ≥ 200"],
], [34, 12, 14, 40]));

// 4 ---------------------------------------------------------------------
c.push(h1("4. Overall methodology"));
c.push(...figure("fig00_workflow", "Six-stage workflow and the week in which each stage is executed.", 610));

// 5 ---------------------------------------------------------------------
c.push(h1("5. Analysis plan in detail"));
c.push(h2("5.1 Data cleaning and preprocessing"));
c.push(p("Each issue type has a stated detection rule and treatment, so the process can be audited:"));
c.push(table(["Issue", "Detection rule", "Treatment", "Example"], [
  ["Exact duplicates", "`df.duplicated()`", "Drop, keep first", "Same row listed twice"],
  ["Inconsistent labels", "`value_counts()` and string normalisation", "Map to canonical labels", "“suv - small”, “SUV: Small ” → “SUV - Small”"],
  ["Unit errors", "Values outside physical range for the segment", "Convert after verifying the converted value is plausible", "4,850 lb in a kg column → 4,850 / 2.2046 = 2,200 kg"],
  ["Impossible values", "Range rules (engine 0.6–8.5 L, power 30–1,500 hp)", "Set to missing, then impute", "Engine size 99 L"],
  ["Invalid target", "CO₂ ≤ 0 or sentinel (999)", "Drop row; the label cannot be imputed", "CO₂ = −1"],
  ["Missing values", "`isna().mean()`; check pattern vs model year", "Benchmark imputers by hiding known values (Section 5.1.1)", "Weight missing more often in older years"],
], [16, 26, 30, 28], { size: 17 }));
c.push(h3("5.1.1 Imputation benchmark design"));
c.push(p("Rather than assuming a method, 10% of known weights will be hidden, imputed by each method and scored by mean absolute error (MAE), averaged over five random masks. Methods: global median, class median, k-nearest neighbours, iterative regression imputer. The winner is used inside the modelling pipeline so test data never influences imputation statistics."));
c.push(h3("5.1.2 Outlier rule"));
c.push(p(`The interquartile-range (IQR) rule flags a value as an outlier if it lies below Q1 − 1.5·IQR or above Q3 + 1.5·IQR. **Worked example:** for engine size Q1 = ${f1(X.engine_iqr_example.q1)} L and Q3 = ${f1(X.engine_iqr_example.q3)} L, so IQR = ${f1(X.engine_iqr_example.q3 - X.engine_iqr_example.q1)} and the upper fence is ${f2(X.engine_iqr_example.q3 + 1.5 * (X.engine_iqr_example.q3 - X.engine_iqr_example.q1))} L. A 6.2 L pickup is flagged yet perfectly legitimate, so flagged values are checked against vehicle class (and an Isolation Forest) before anything is removed. Only physically impossible values are corrected.`));
c.push(h2("5.2 Exploratory analysis plan"));
c.push(table(["Question", "Technique", "Chart"], [
  ["What do the variables look like?", "Histogram + KDE, skewness", "Histograms"],
  ["Which classes and technologies emit most?", "Group means, Kruskal–Wallis test", "Box plots"],
  ["Which numeric variables drive CO₂?", "Pearson / Spearman correlation", "Scatter with fit line, heat map"],
  ["Is the fleet improving?", "Yearly mean with 95% CI, linear trend", "Line and stacked bars"],
  ["Are predictors redundant?", "VIF, condition number, bootstrap coefficients", "VIF bars, box plots"],
], [38, 38, 24]));
c.push(h2("5.3 Modelling and evaluation plan"));
c.push(p("Models are introduced in order of complexity so each addition has to justify itself: mean baseline → ordinary least squares → Ridge → Ridge on log(CO₂) → random forest → gradient boosting → blend. Validation uses **GroupKFold by model line**, because different years of the same line are almost identical rows; a random split would leak information and overstate accuracy. Hyper-parameters are tuned with `RandomizedSearchCV`. The final test set is touched once."));
c.push(h3("Metrics with a worked example"));
c.push(p("Suppose five vehicles have actual CO₂ of 120, 150, 180, 210, 240 g/km and predictions of 125, 145, 190, 205, 230:"));
c.push(table(["Actual", "Predicted", "Error", "|Error|", "Error²"], [
  ["120", "125", "−5", "5", "25"], ["150", "145", "5", "5", "25"], ["180", "190", "−10", "10", "100"], ["210", "205", "5", "5", "25"], ["240", "230", "10", "10", "100"],
  ["**Mean**", "", "", "**7.0**", "**55.0**"],
], [16, 18, 16, 16, 16], { aligns: [L.R, L.R, L.R, L.R, L.R], size: 18 }));
c.push(p("MAE = 35/5 = **7.0 g/km**; RMSE = √55 = **7.42 g/km** (penalises large misses more); with mean actual 180, SST = Σ(y−ȳ)² = 3,600+900+0+900+3,600 = 9,000 and SSE = 275, so R² = 1 − 275/9,000 = **0.969**. For classification, F1 = 2·precision·recall / (precision + recall).", { before: 100 }));
c.push(h2("5.4 Leakage and multicollinearity plan"));
c.push(p("Fuel consumption in L/100 km is CO₂ divided by a constant (about 23.2 g per L/100 km for petrol), so it can never be a legitimate predictor of CO₂ in a design-stage model; it is excluded. Among legitimate predictors, engine size, cylinders and horsepower are expected to be correlated. The **variance inflation factor** for predictor j is VIF_j = 1 / (1 − R²_j), where R²_j comes from regressing predictor j on all other predictors. **Worked example:** R²_j = 0.95 gives VIF = 1/0.05 = 20, meaning the coefficient’s variance is inflated twentyfold; common rules of thumb treat VIF above 5–10 as a warning. The plan also uses the condition number of the standardised design matrix, principal-component variance shares, and bootstrap coefficient stability, and compares remedies (drop a redundant variable, regularise with Ridge, or use tree models)."));

// 6 ---------------------------------------------------------------------
c.push(h1("6. Python tools and their roles"));
c.push(tcap("Libraries, specific functions used, and the role each plays."));
c.push(table(["Library", "Functions / classes planned", "Role in this project"], [
  ["pandas", "`read_csv`, `drop_duplicates`, `groupby`, `pd.cut`, `get_dummies`", "Loading, cleaning, grouping, summary tables"],
  ["numpy", "`np.linalg.lstsq`, `svd`, `quantile`, `random.default_rng`", "Vectorised maths; VIF and condition number computed from first principles; bootstrap resampling"],
  ["scipy.stats", "`kruskal`, `spearmanr`, `skew`, `linregress`, `wilcoxon`", "Hypothesis tests, trend slope, paired model comparison"],
  ["matplotlib / seaborn", "`histplot`, `boxplot`, `heatmap`, `regplot`, `barh`", "All figures; box plots expose outliers, heat maps expose collinearity"],
  ["scikit-learn – preprocessing", "`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`, `IterativeImputer`, `KNNImputer`", "Leakage-safe preprocessing fitted only on training folds"],
  ["scikit-learn – models", "`LinearRegression`, `RidgeCV`, `TransformedTargetRegressor`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `VotingRegressor`", "Candidate models from simple to flexible"],
  ["scikit-learn – validation", "`GroupShuffleSplit`, `GroupKFold`, `RandomizedSearchCV`, `cross_validate`, `learning_curve`", "Honest evaluation and tuning"],
  ["scikit-learn – inspection", "`permutation_importance`, `PartialDependenceDisplay`", "Explain what the model learned"],
  ["joblib", "`dump`, `load`, `Memory`", "Save the model; cache expensive imputation per fold"],
  ["Optional extensions", "statsmodels (formal VIF/OLS tables), SHAP, LightGBM/XGBoost", "Considered for future work; not required"],
], [20, 42, 38], { size: 17 }));
c.push(h2("6.1 Pipeline skeleton"));
c.push(...code(`cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
ct  = ColumnTransformer([("cat", cat, CAT), ("num", "passthrough", NUM)])
prep = Pipeline([("ct", ct), ("imp", IterativeImputer(max_iter=10))])
model = Pipeline([("prep", prep), ("model", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
scores = cross_validate(model, X_train, y_train, groups=family, cv=GroupKFold(5),
                        scoring="neg_root_mean_squared_error")`));

// 7 ---------------------------------------------------------------------
c.push(h1("7. Pilot feasibility check"));
c.push(p("Before committing to the plan, a quick pilot was run on the development dataset to check that the problem is learnable and that the planned targets are realistic."));
c.push(...code(`X = df[["weight_kg"]]; y = df["co2_gkm"]
lr = LinearRegression().fit(X, y)
print(r2_score(y, lr.predict(X)), np.sqrt(mean_squared_error(y, lr.predict(X))))`));
c.push(table(["Model", "R²", "RMSE (g/km)"], [
  ["Mean predictor", "0.000", f1(pil.baseline_rmse)],
  ["CO₂ ~ weight (one variable)", f3(pil.r2), f1(pil.rmse)],
  ["CO₂ ~ weight + engine size + horsepower", f3(pil3.r2), f1(pil3.rmse)],
], [50, 20, 30], { aligns: [L.L, L.R, L.R] }));
c.push(p(`Weight alone correlates r = ${f2(pil.r_weight)} with CO₂ and explains ${f0(pil.r2 * 100)}% of the variance: each extra 100 kg adds about ${f1(pil.slope_g_per_100kg)} g/km. Adding two more variables lifts R² to ${f2(pil3.r2)}, so a richer model with powertrain and drivetrain should plausibly reach the 0.90 target. This is evidence of feasibility, not a result: it was fitted and scored on the same rows with no validation.`, { before: 100 }));

// 8 ---------------------------------------------------------------------
c.push(h1("8. Timeline and milestones"));
c.push(...figure("fig_gantt", "Four-week schedule. Diamonds mark milestones M1–M4.", 610));
c.push(table(["Week", "Main tasks", "Hours", "Milestone and acceptance criteria"], [
  ["1", "Problem scoping, source survey, pilot, methodology, plan document", "33", "M1: plan approved; problem statement, hypotheses, tool list and timeline complete"],
  ["2", "Data acquisition/simulation, cleaning log, imputation benchmark, EDA, VIF study", "34", "M2: documented cleaning log; ≥ 8 figures; collinearity decision justified"],
  ["3", "Split and pipeline, 6 candidate models, tuning, diagnostics, classification extension", "34", "M3: test metrics with confidence intervals; tuned models beat baselines"],
  ["4", "Error analysis, intervals, temporal validation, scenarios, recommendations", "32", "M4: recommendations traceable to evidence; limitations stated"],
], [8, 44, 8, 40], { aligns: [L.C, L.L, L.C, L.L] }));
c.push(h2("8.1 Risk register"));
c.push(table(["Risk", "Likelihood", "Impact", "Mitigation"], [
  ["Real public data lacks weight or power", "High", "Medium", "Join EEA registrations or use class medians; state limitation"],
  ["Target leakage inflates scores", "Medium", "High", "Exclude consumption columns; group-aware validation"],
  ["Collinearity makes coefficients unstable", "High", "Medium", "VIF study; Ridge; tree models; interpret via permutation importance"],
  ["Simulated data is unrealistically easy", "Medium", "High", "Use real data when available; report as development dataset"],
  ["Small subgroups give noisy estimates", "Medium", "Low", "Report n per segment; bootstrap CIs"],
  ["Compute limits (single CPU)", "Medium", "Low", "Cache imputation; limit search to 10–20 candidates"],
], [34, 14, 12, 40]));

// 9 ---------------------------------------------------------------------
c.push(h1("9. Expected outcomes and deliverables"));
c.push(...bullets([
  "A cleaned, documented dataset with a cleaning log and a data dictionary.",
  "An EDA report with at least eight figures and a multicollinearity assessment.",
  "A tuned, validated CO₂ model with test RMSE ≤ 12 g/km and R² ≥ 0.90, plus a high-emitter classifier with F1 ≥ 0.90.",
  "Prediction intervals, a segment-level error map and a forward-looking validation.",
  "A ranked list of CO₂-reduction levers with quantified fleet impact, and a list of limitations and next steps.",
]));
c.push(h2("9.1 Ethics and limitations"));
c.push(p("Vehicle specifications carry no personal data. The main risk is misuse of results: laboratory ratings are not real-world emissions, and associations found in observational data are not proof of causation. Both caveats will be repeated wherever recommendations are made."));

c.push(h1("References and data portals"));
c.push(...bullets([
  "Natural Resources Canada, Fuel Consumption Ratings – open.canada.ca (search “Fuel Consumption Ratings”).",
  "US Department of Energy / EPA, fueleconomy.gov – “Download Fuel Economy Data”.",
  "European Environment Agency – “CO₂ emissions from new passenger cars” monitoring data.",
  "UCI Machine Learning Repository – “Auto MPG” data set.",
  "scikit-learn documentation – model selection, preprocessing and inspection modules.",
]));
c.push(h1("Appendix A. Column mapping for real data"));
c.push(table(["Project column", "NRCan ratings", "EPA vehicles.csv"], [
  ["make", "Make", "make"], ["model_year", "Model year", "year"], ["vehicle_class", "Vehicle class", "VClass"],
  ["engine_size_l", "Engine size (L)", "displ"], ["cylinders", "Cylinders", "cylinders"], ["transmission", "Transmission", "trany"],
  ["fuel_type", "Fuel type", "fuelType"], ["co2_gkm", "CO₂ emissions (g/km)", "co2TailpipeGpm ÷ 1.609"],
  ["weight_kg, horsepower", "not available – join another source", "not available"],
], [30, 35, 35]));

build("../outputs/Week1_Research_Plan_Automotive_CO2.docx", { headerText: "Automotive CO₂ Prediction Project · Week 1 – Research Plan and Strategy", children: c });

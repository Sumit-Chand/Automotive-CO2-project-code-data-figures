const L = require("./lib.js");
const { p, h1, h2, h3, bullets, numbered, code, table, tcap, figure, callout, spacer, titleBlock, build, f0, f1, f2, f3, load } = L;
const V = load("evaluation_results.json"), M = load("modeling_results.json"), E = load("eda_results.json"), X = load("extra_checks.json");
const o = V.overall, I = V.intervals, TM = V.temporal, F = V.fleet, S = Object.fromEntries(V.scenarios.map((s) => [s.name, s]));
const bt = M.best_test_metrics, mean = M.split.test_mean;
const sc = (k) => S[Object.keys(S).find((n) => n.startsWith(k))];
const wt100 = sc("Cut curb weight by 100"), hyb = sc("ICE -> full"), mild = sc("ICE -> mild"), awd = sc("AWD"), cvt = sc("Switch"), dn = sc("Downsize"), combo = sc("Combined");
const fr = F.rows;

const c = [];
c.push(...titleBlock("Evaluation, Insights and Strategic Recommendations",
  "What the CO₂ model tells us, where it is weak, and what manufacturers should do with it", "Week 4 · Automotive Data Analysis"));

c.push(callout("Executive summary", [
  `**The model is accurate enough for design-stage screening.** On unseen model lines it predicts CO₂ with RMSE ${f1(o.rmse)} g/km (${f1(o.rmse / mean * 100)}% of the mean), R² ${f3(o.r2)}; ${f0(100 - o.share_err_gt15)}% of vehicles are within 15 g/km. A forward-looking test (train ≤ 2021, predict 2022–24) gives a similar RMSE of ${f1(TM.rmse)} g/km.`,
  `**Electrification is the largest single lever:** converting a conventional vehicle to a full hybrid lowers predicted CO₂ by ${f0(Math.abs(hyb.mean_pct))}% (${f0(Math.abs(hyb.mean_delta))} g/km); a mild hybrid by ${f0(Math.abs(mild.mean_pct))}%.`,
  `**Mass matters in every class:** each 100 kg removed saves about ${f1(Math.abs(wt100.mean_delta))} g/km (${f1(Math.abs(wt100.mean_pct))}%).`,
  `**Product mix is as powerful as engineering:** moving 10 points of sales from heavy SUVs/pickups to small SUVs saves ${f1(Math.abs(fr[4].delta))} g/km, more than moving 10 points to full hybrids (${f1(Math.abs(fr[1].delta))} g/km).`,
  `**A combined programme** (lighter, mild hybrids, smaller engines, plus 10 points of full hybrids and a mix shift) cuts fleet-average CO₂ by about **${f0(Math.abs(fr[6].pct))}%**, equal to about ${f0(fr[6].years_equiv)} model years of the historical 2 g/km-per-year progress.`,
  "**Important caveat:** the data are simulated. The *method* and the ranking of levers are demonstrated; the exact percentages must be confirmed on real registration or ratings data before any decision is taken.",
]));
c.push(spacer(60));

// 1 ----------------------------------------------------------------------
c.push(h1("1. Recap: how Weeks 1–3 lead here"));
c.push(table(["Week", "Contribution", "Key output"], [
  ["1", "Problem, hypotheses H1–H5, methodology, timeline", "Predict CO₂ (g/km) from design specifications; targets R² ≥ 0.90, RMSE ≤ 12, F1 ≥ 0.90"],
  ["2", "Cleaning log, imputation benchmark, EDA, multicollinearity study", `${E.clean_rows.toLocaleString()} clean rows; H1–H4 supported; leakage columns excluded`],
  ["3", "Group-aware validation, 8 models, tuning, classification", `Selected blend: test RMSE ${f2(bt.RMSE)}, R² ${f3(bt.R2)}; classifier F1 ${f3(M.classification_table[1].f1)}`],
  ["4", "Error analysis, uncertainty, temporal test, scenarios, strategy", "This report"],
], [8, 42, 50]));

// 2 ----------------------------------------------------------------------
c.push(h1("2. Critical evaluation of the model"));
c.push(h2("2.1 Overall accuracy against the Week 1 targets"));
c.push(table(["Criterion", "Target", "Achieved", "Met?"], [
  ["Test RMSE", "≤ 12 g/km", `${f2(bt.RMSE)} (95% CI ${f2(M.best_test_ci.rmse[0])}–${f2(M.best_test_ci.rmse[1])})`, "Yes"],
  ["Test R²", "≥ 0.90", `${f3(bt.R2)} (CI ${f3(M.best_test_ci.r2[0])}–${f3(M.best_test_ci.r2[1])})`, "Yes"],
  ["High-emitter F1", "≥ 0.90", f3(M.classification_table[1].f1), "Yes"],
  ["90% interval coverage", "88–92%", `${f1(I.conformal_rel.coverage)}% (relative conformal)`, "Yes"],
  ["Mean error (bias)", "≈ 0", `${f2(o.bias)} g/km (actual − predicted)`, "Yes"],
], [28, 18, 42, 12], { aligns: [L.L, L.L, L.L, L.C] }));
c.push(p(`In practical terms the typical error is ${f1(o.mae)} g/km (${f1(bt.MAPE)}%). ${f1(o.share_err_gt15)}% of vehicles are missed by more than 15 g/km and ${f1(o.share_err_gt20)}% by more than 20 g/km. Relative to the mean-only baseline (RMSE ${f1(M.model_table[0].RMSE)}), the model removes ${f0((1 - bt.RMSE ** 2 / M.model_table[0].RMSE ** 2) * 100)}% of the squared error.`, { before: 100 }));

c.push(h2("2.2 Where is the model weak? Segment analysis"));
c.push(...figure("fig20_error_by_segment", "Test RMSE (bars) and mean bias (red dots) by vehicle class and by powertrain.", 610));
const sg = V.seg_class;
c.push(tcap("Test error by vehicle class (bias = actual − predicted; positive = model under-predicts)."));
c.push(table(["Class", "n", "RMSE", "MAE", "MAPE %", "Bias"], Object.entries(sg).sort((a, b) => b[1].rmse - a[1].rmse).map(([k, v]) => [k, String(v.n), f2(v.rmse), f2(v.mae), f1(v.mape), (v.bias > 0 ? "+" : "") + f2(v.bias)]),
  [34, 10, 14, 14, 14, 14], { aligns: [L.L, L.R, L.R, L.R, L.R, L.R] }));
c.push(...bullets([
  `**Pickup trucks are the weakest segment:** RMSE ${f1(sg["Pickup Truck"].rmse)} and a systematic under-prediction of ${f1(sg["Pickup Truck"].bias)} g/km. With only ${sg["Pickup Truck"].n} test pickups the estimate is noisy, but it is also the segment with the fewest training examples and the highest emissions, and the segment where a log-linear model is extrapolating furthest.`,
  `**Small cars are the most accurate** (Sub-compact ${f1(sg["Sub-compact"].rmse)}, Compact ${f1(sg.Compact.rmse)} g/km), partly because their CO₂ is low: in *relative* terms all classes lie between ${f1(Math.min(...Object.values(sg).map((v) => v.mape)))}% and ${f1(Math.max(...Object.values(sg).map((v) => v.mape)))}% MAPE, so the model is consistently ~4% accurate rather than consistently ±9 g/km.`,
  `**Hybrids have low absolute but higher relative error** (MAPE ${f1(V.seg_powertrain["Full hybrid"].mape)}% vs ${f1(V.seg_powertrain.ICE.mape)}% for ICE), and the model slightly over-predicts them (bias ${f1(V.seg_powertrain["Full hybrid"].bias)} g/km), possibly because full hybrids are a small share of the training data.`,
  `**Brands differ:** RMSE ranges from ${f1(V.seg_make_rmse_range[0])} (${V.seg_make_rmse_range[2]}) to ${f1(V.seg_make_rmse_range[1])} (${V.seg_make_rmse_range[3]}). Premium brands carry model-line effects that specifications do not explain.`,
  `**Model year** has almost no effect on error (RMSE ${f1(V.seg_year["2015-17"].rmse)}, ${f1(V.seg_year["2018-20"].rmse)}, ${f1(V.seg_year["2021-24"].rmse)}), a good sign for stability.`,
]));
c.push(h3("Effect of imputed inputs"));
const ia = X.weight_imputed;
c.push(p(`Weight was imputed for ${ia.n_with} of ${ia.n_with + ia.n_without} test rows. These rows have an RMSE of **${f1(ia.rmse_with)} g/km versus ${f1(ia.rmse_without)} g/km** for rows with a measured weight, a ${f0((ia.rmse_with / ia.rmse_without - 1) * 100)}% increase. Three of the eight largest errors in the test set had imputed weight. Capturing curb weight reliably is therefore the cheapest accuracy gain available.`));
c.push(h3("Largest individual errors"));
c.push(table(["Class", "Make", "Year", "Engine L", "Weight kg", "Powertrain", "Actual", "Pred.", "Error"],
  V.worst_cases.map((w) => [w.vehicle_class, w.make, String(w.model_year), f1(w.engine_size_l), w.weight_kg === null ? "missing" : f0(w.weight_kg), w.powertrain, f0(w.co2_gkm), f0(w.pred), (w.res > 0 ? "+" : "") + f0(w.res)]),
  [20, 11, 8, 10, 12, 12, 9, 9, 9], { size: 16, aligns: [L.L, L.L, L.C, L.R, L.R, L.L, L.R, L.R, L.R] }));
c.push(p(`All eight are conventional (ICE) vehicles with errors of 26–32 g/km (roughly 10–20% of the actual value); there is no systematic pattern in class or year, which is what unmodelled model-line effects (the simulation includes a ±3% line-specific efficiency term) would look like. Such cases cannot be eliminated using specifications alone.`, { before: 100 }));

c.push(h2("2.3 Quantifying uncertainty: prediction intervals"));
c.push(p("A point prediction is rarely enough for a compliance decision. Three 90% interval methods were compared on the test set:"));
c.push(table(["Method", "Empirical coverage", "Mean width (g/km)", "Comment"], [
  ["Quantile gradient boosting (5th–95th percentile)", `${f1(I.quantile_gb.coverage)}%`, f1(I.quantile_gb.mean_width), "**Under-covers**: intervals too narrow"],
  ["Absolute conformal (± residual quantile)", `${f1(I.conformal_abs.coverage)}%`, f1(I.conformal_abs.mean_width), `± ${f1(I.conformal_abs.q)} g/km for everyone; uneven by class`],
  ["Relative conformal (± % of prediction)", `${f1(I.conformal_rel.coverage)}%`, f1(I.conformal_rel.mean_width), `± ${f1(I.conformal_rel.q_pct)}% of the predicted value`],
], [34, 18, 18, 30], { aligns: [L.L, L.R, L.R, L.L], highlightRow: 2 }));
c.push(...figure("fig21_prediction_intervals", "Left: relative-conformal 90% intervals for 70 random test vehicles. Right: empirical coverage by class for each method (target 90%).", 610));
c.push(p(`The conformal methods use out-of-fold residuals from the training data only (no test information). The quantile model covers just ${f0(I.quantile_gb.coverage)}% of actual values, so its intervals should not be trusted. The absolute interval is wrong in a predictable way: it gives 100% coverage for sub-compacts (the interval is too wide for low-CO₂ cars) but only ${f0(I.coverage_by_class_conf_abs["Pickup Truck"])}% for pickups. The **relative interval fits the multiplicative error structure** and keeps coverage between ${f0(Math.min(...Object.values(I.coverage_by_class_conf_rel)))}% and ${f0(Math.max(...Object.values(I.coverage_by_class_conf_rel)))}% across classes. **Operational rule of thumb: actual CO₂ lies within ±8.4% of the prediction in nine cases out of ten.** The weakest segment is sub-compacts (82%), where small absolute values magnify relative noise.`, { before: 100 }));

c.push(h2("2.4 Forward-looking (temporal) validation"));
c.push(p("A random group split asks “can we predict *new model lines*?” Manufacturers also ask “can a model built on history predict *next year’s* vehicles?” The model was re-fitted on model years ≤ 2021 and tested on 2022–2024."));
c.push(...figure("fig22_temporal_validation", "Temporal test: predicted vs actual (left), bias by forecast year (centre), and RMSE for different validation designs and model components (right).", 610));
c.push(table(["Model", "RMSE (g/km)", "Mean bias"], [
  ["Selected blend", f2(TM.rmse), f2(TM.bias)], ["Ridge on log(CO₂) component", f2(TM.ridge_log_rmse), f2(TM.ridge_log_bias)], ["Gradient-boosting component", f2(TM.gb_rmse), f2(TM.gb_bias)],
], [50, 25, 25], { aligns: [L.L, L.R, L.R], highlightRow: 0 }));
c.push(p(`Accuracy holds up (R² ${f3(TM.r2)}, RMSE ${f2(TM.rmse)} g/km, ${TM.n_train.toLocaleString()} training / ${TM.n_test.toLocaleString()} test rows). The model **over-predicts later years** by ${f1(Math.abs(TM.bias))} g/km on average, reaching about ${f1(Math.abs(TM.by_year["2023"].bias))} g/km in 2023–24: real technology progress was a little faster than the trend learned from earlier years. The gradient-boosting component drifts most (bias ${f1(TM.gb_bias)}) because trees cannot extrapolate a trend, while the linear component is almost unbiased (${f2(TM.ridge_log_bias)}). This supports (a) keeping a linear component in the blend and (b) **retraining every year and monitoring bias** as a drift alarm.`));

c.push(h2("2.5 Strengths and weaknesses"));
c.push(table(["Strengths", "Weaknesses"], [
  ["Accurate (R² 0.96) and well calibrated (bias −0.7 g/km)", "Trained on simulated data; real-world performance unknown"],
  ["Stable across years, classes and powertrains (MAPE 3.5–4.7%)", "Weakest for pickups, small segments and premium brands"],
  ["Validated on unseen model lines and on future years", "Imputed weight raises error by about 40%"],
  ["Transparent core (log-linear) with reliable ±8.4% intervals", "Drifts slowly as technology improves; needs yearly retraining"],
  ["Same model serves regression and screening (F1 0.94)", "Associations, not causal effects; excludes aerodynamics, gearing, battery size, EVs"],
], [50, 50]));

// 3 ----------------------------------------------------------------------
c.push(h1("3. Key insights"));
c.push(p("Each insight combines evidence from several weeks and is stated with its link to an industry challenge."));
c.push(table(["#", "Insight", "Evidence", "Industry relevance"], [
  ["1", "**Mass is the strongest design driver.**", `Largest permutation importance (+${f0(M.perm_importance[0].rmse_increase)} g/km RMSE); r = 0.87 with CO₂; −100 kg ≈ −${f1(Math.abs(wt100.mean_delta))} g/km (−${f1(Math.abs(wt100.mean_pct))}%)`, "Vehicle weight creep (batteries, safety, SUV bodies) works against regulation; lightweight materials carry cost"],
  ["2", "**Electrification gives the largest step change.**", `Full hybrid −${f0(Math.abs(hyb.mean_pct))}% (model) and −${f0(Math.abs(E.hybrid_vs_ice_pct_avg))}% (EDA, class-matched); mild hybrid −${f0(Math.abs(mild.mean_pct))}%`, "Hybrids as a bridge towards battery-electric vehicles; mild hybrids are a cheap partial step"],
  ["3", "**Progress is real but slow.**", `−${f1(Math.abs(E.trend.pct_change))}% over 2015–2024 (${f2(E.trend.slope_g_per_year)} g/km/yr) while hybrid share rose from ${f0(E.trend.hybrid_share_2015)}% to ${f0(E.trend.hybrid_share_2024)}%`, "Tightening fleet targets need faster change than the historical trend"],
  ["4", "**What is sold matters as much as what is engineered.**", `10 points of sales from heavy SUV/pickup to small SUV: −${f1(Math.abs(fr[4].delta))} g/km, versus −${f1(Math.abs(fr[1].delta))} g/km for 10 points of full hybrids`, "SUV-ization of the market; pricing and marketing become emission tools"],
  ["5", "**All-wheel drive has a measurable penalty.**", `+${f1(E.awd_vs_fwd_pct_avg)}% vs FWD within class (pickups +${f0(E.awd_vs_fwd_pct_by_class["Pickup Truck"])}%); AWD→FWD −${f1(Math.abs(awd.mean_pct))}% where feasible`, "AWD fitted as a fashion or default feature; disconnectable systems reduce the penalty"],
  ["6", "**Class labels hide large differences.**", "Minivans and sports cars (≈ 230 g/km) emit like standard SUVs; power-to-weight is only weakly related (r = 0.39) to CO₂", "Regulation by size or class should be complemented by direct specification-level modelling"],
  ["7", "**Compliance screening is feasible and the risk is at the margin.**", `F1 ${f2(M.classification_table[1].f1)}; ${f0(M.clf_errors.share_within_10)}% of errors lie within 10 g/km of the 200 g/km line`, "A grey-zone rule (e.g. ±8%) focuses expensive testing on borderline vehicles"],
], [4, 24, 38, 34], { size: 17 }));

// 4 ----------------------------------------------------------------------
c.push(h1("4. What-if scenario analysis"));
c.push(p("The model can be used as a simulator: change one specification of each eligible held-out vehicle, re-predict, and average the change. The comparison group is the unchanged prediction, so model bias largely cancels."));
c.push(...figure("fig23_scenarios", "Average change in predicted CO₂ per lever. Error bars are 95% bootstrap intervals over the vehicles in the test portfolio.", 560));
c.push(table(["Lever", "Eligible vehicles", "Change (g/km)", "Change (%)"], V.scenarios.slice().sort((a, b) => a.mean_pct - b.mean_pct).map((s) => [s.name.replace("->", "→"), String(s.n), f1(s.mean_delta), `${f1(s.mean_pct)} [${f1(s.ci_lo)}, ${f1(s.ci_hi)}]`]), [46, 16, 16, 22], { aligns: [L.L, L.R, L.R, L.R] }));
c.push(callout("How to interpret these numbers", [
  "**Associations, not experiments.** The model learned how CO₂ differs between vehicles; changing one attribute of a real vehicle may alter others (a lighter car is usually smaller, a downsized engine needs a turbo to keep power). Treat results as screening estimates.",
  "**The intervals are narrow because they describe only sampling variation across test vehicles**, not model error or the simulation assumptions. The ±8% prediction interval of Section 2.3 is the more honest uncertainty for a single vehicle.",
  "The ‘downsize by 0.5 L at constant horsepower’ lever is optimistic: keeping power requires turbocharging, whose small penalty is in the model but whose reliability and cost are not.",
]));

// 5 ----------------------------------------------------------------------
c.push(h1("5. Fleet-level simulation"));
c.push(p(`Regulation applies to the sales-weighted *fleet average*. The 1,159 test vehicles are treated as a product portfolio with a baseline predicted average of ${f1(F.baseline)} g/km, of which ${f0(F.ice_share * 100)}% are conventional vehicles. A shift of 10 percentage points of sales to full hybrids therefore changes the average by 0.10 × (${f1(F.hyb_delta_per_ice_vehicle)} g/km) = ${f1(fr[1].delta)} g/km. The mix lever moves 10 points of sales from standard SUVs and pickups (average ${f0(F.heavy_mean)} g/km) to small SUVs (${f0(F.small_suv_mean)} g/km).`));
c.push(...figure("fig24_fleet_scenarios", "Predicted fleet-average CO₂ under alternative strategies (horizontal axis truncated).", 560));
c.push(table(["Scenario", "Fleet CO₂ (g/km)", "Change", "≈ years of past progress"], fr.map((r) => [r.scenario, f1(r.fleet_co2), r.pct === 0 ? "–" : `${f1(r.pct)}%`, r.pct === 0 ? "–" : f1(r.years_equiv)]), [50, 18, 14, 18], { aligns: [L.L, L.R, L.R, L.R] }));
c.push(p(`The last column divides each saving by the historical improvement of ${f2(Math.abs(F.hist_slope))} g/km per year found in Week 2. For example, the full package is worth about ${f0(fr[6].years_equiv)} years of past progress. Components are added approximately (interaction effects between levers are ignored), so the combined figures are indicative.`, { before: 100 }));

// 6 ----------------------------------------------------------------------
c.push(h1("6. Strategic recommendations"));
c.push(table(["#", "Recommendation", "Evidence", "Expected impact", "Key performance indicator"], [
  ["R1", "**Prioritise hybridisation of high-volume ICE lines**, starting with mild hybrids where cost is the barrier.", "Insight 2; scenarios", `Mild: −${f0(Math.abs(mild.mean_pct))}% per vehicle; full: −${f0(Math.abs(hyb.mean_pct))}%; +10 pts full hybrids = −${f1(Math.abs(fr[1].delta))} g/km fleet`, "Share of sales electrified; g/km per electrified model"],
  ["R2", "**Run a lightweighting programme focused on SUVs, pickups and full-size cars.**", "Insight 1", `−5% mass = −${f1(Math.abs(fr[3].delta))} g/km fleet (−${f1(Math.abs(fr[3].pct))}%)`, "kg removed per model line; cost per kg saved"],
  ["R3", "**Manage the portfolio mix:** price and promote smaller SUVs against heavy SUVs/pickups.", "Insight 4", `10-point shift = −${f1(Math.abs(fr[4].delta))} g/km fleet`, "Sales share of heavy SUV/pickup; fleet-average g/km"],
  ["R4", "**Make AWD optional or disconnectable** where it is not essential.", "Insight 5", `−${f1(Math.abs(awd.mean_pct))}% on each converted vehicle`, "AWD take rate; g/km of AWD vs FWD variants"],
  ["R5", "**Deploy the model as a design-stage screening tool** with ±8% intervals and a grey-zone rule near regulatory limits.", "Sections 2.2–2.3; Insight 7", "Earlier detection of non-compliant variants; fewer late redesigns", "Share of variants predicted within ±8%; false-negative rate near limit"],
  ["R6", "**Improve data capture, especially curb weight.**", "Imputed-weight RMSE 12.7 vs 8.9", "≈ 30% lower error on affected rows", "% of records with measured weight"],
  ["R7", "**Retrain annually and monitor bias.**", "Temporal validation", "Prevents slow drift (bias −1.5 g/km after three years)", "Rolling mean bias; alarm at ±2 g/km"],
], [5, 28, 17, 26, 24], { size: 16 }));
c.push(p("Sequencing: R5–R7 are low-cost and can start immediately; R3 and R4 are commercial decisions achievable within one model cycle; R1 and R2 require capital and supplier programmes. Because levers overlap, they should be evaluated together with the model rather than summed by hand.", { before: 120 }));

// 7 ----------------------------------------------------------------------
c.push(h1("7. Limitations and critical discussion"));
c.push(...numbered([
  "**Simulated data (most important).** Relationships were built into the generator, so recovering them demonstrates the pipeline, not real-world facts. The strength of the log-linear model and the low noise are partly artefacts. Every number in Sections 3–6 should be recomputed on real ratings or registration data.",
  "**Association, not causation.** Scenario results come from a predictive model. Changing weight or powertrain on a real vehicle may require other changes; causal claims need engineering data or experiments.",
  "**Laboratory CO₂, not real-world.** Ratings differ from on-road emissions, and tailpipe CO₂ ignores electricity generation, manufacturing and fuel-supply emissions. Battery-electric and plug-in hybrid vehicles are outside the scope.",
  "**Missing physical drivers.** Aerodynamic drag, rolling resistance, gearing, battery size and calibration are not in the feature set; they are the likely source of the unexplained model-line differences (RMSE ≈ 9 g/km).",
  "**Small segments.** Pickups (39), diesel (31) and full hybrids (82) have few test rows; segment estimates have wide uncertainty.",
  "**Single split.** The headline test metrics come from one group split; the cross-validation (RMSE 9.68 ± 0.64) and temporal test (9.35) corroborate them but do not replace repeated resampling.",
  "**Interval calibration is uneven** for small segments (82% coverage for sub-compacts) and conformal guarantees assume stable conditions.",
  "**Model choice.** The linear dominance observed here may reverse on real data, where interactions are stronger.",
]));

// 8 ----------------------------------------------------------------------
c.push(h1("8. Improvements and further studies"));
c.push(table(["Next step", "Purpose", "Method / tool"], [
  ["Validate on real public data", "Test whether conclusions survive contact with reality", "NRCan/EPA ratings joined with EEA mass and power; same scripts"],
  ["Add physical features", "Reduce unexplained model-line error", "Drag area, rolling resistance, gear ratios, battery capacity"],
  ["Mixed-effects or Bayesian model with a model-line random effect", "Principled treatment of repeated model years and honest intervals", "statsmodels MixedLM, PyMC"],
  ["Monotonic gradient boosting and SHAP", "Keep flexibility but enforce sensible directions and explain individual predictions", "HistGradientBoosting monotonic_cst; shap"],
  ["Cost-effectiveness analysis", "Rank levers by € per g/km saved, not by g/km alone", "Add cost data per lever; optimisation"],
  ["Real-world and life-cycle emissions; EV extension", "Close the gap between lab, road and total emissions", "Real-driving-emissions data; LCA factors"],
  ["Deployment monitoring", "Detect drift early", "Rolling bias and coverage dashboard; scheduled retraining"],
], [32, 36, 32], { size: 17 }));

c.push(h1("9. Conclusion"));
c.push(p(`Across four weeks the project moved from a problem definition to a validated model and a ranked set of actions. A transparent, well-validated model predicts vehicle CO₂ to about ${f0(bt.MAPE)}% accuracy and its uncertainty can be stated as ±8.4%. Its simulations indicate that the three biggest opportunities are electrification, vehicle mass and the mix of vehicles sold, and that together they could plausibly deliver a double-digit percentage cut in fleet-average emissions. The most valuable next step is not a more complex algorithm but validation on real data, better capture of vehicle weight, and annual retraining with drift monitoring.`));

c.push(h1("Appendix. Reproducibility"));
c.push(table(["File", "Purpose"], [
  ["scripts/00_generate_data.py", "Creates the simulated raw dataset (seed 42)"],
  ["scripts/01_clean_and_eda.py", "Cleaning, imputation benchmark, EDA, multicollinearity (Week 2)"],
  ["scripts/02_modeling.py", "Pipelines, tuning, evaluation, classification (Week 3)"],
  ["scripts/03_evaluation_insights.py", "Segment errors, intervals, temporal test, scenarios, fleet (Week 4)"],
  ["scripts/04_extra_checks.py, 05_extra_analyses.py, 06_planning_figures.py", "Supporting numbers and Week 1 figures"],
  ["data/, figures/, results/, models/", "Datasets, all charts, JSON/CSV results, saved model"],
], [42, 58]));
c.push(p("Run order: 00 → 01 → 02 → 03 → 04 → 05. Python 3 with pandas, numpy, scipy, matplotlib, seaborn, scikit-learn and joblib.", { before: 100 }));

build("../outputs/Week4_Evaluation_Insights_and_Recommendations_Automotive_CO2.docx", { headerText: "Automotive CO₂ Prediction Project · Week 4 – Evaluation, Insights and Strategic Recommendations", children: c });

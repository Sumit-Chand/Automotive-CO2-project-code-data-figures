const L = require("./lib.js");
const { p, h1, h2, h3, bullets, numbered, code, table, tcap, figure, callout, spacer, titleBlock, build, f0, f1, f2, f3, load } = L;
const E = load("eda_results.json"), X = load("extra_analyses.json");
const T = E.trend, ks = E.kruskal_class;
const pct = (a, b) => ((a / b) * 100).toFixed(1);

const c = [];
c.push(...titleBlock("Data Acquisition, Cleaning and Exploratory Analysis",
  "Vehicle CO₂ emissions dataset: audit, cleaning log, EDA and multicollinearity study", "Week 2 · Automotive Data Analysis"));

c.push(callout("Key findings", [
  `Raw file: **${E.raw_shape[0].toLocaleString()} rows × ${E.raw_shape[1]} columns**. After cleaning: **${E.clean_rows.toLocaleString()} rows** covering ${E.n_families} model lines, model years 2015–2024.`,
  `Curb weight (r = ${f2(E.pearson_with_co2.weight_kg.pearson)}) and engine size (r = ${f2(E.pearson_with_co2.engine_size_l.pearson)}) are the strongest numeric correlates of CO₂; vehicle class alone explains about ${f0(ks.eta2 * 100)}% of rank variation (Kruskal–Wallis).`,
  `Within the same class, full hybrids emit **${f1(Math.abs(E.hybrid_vs_ice_pct_avg))}% less** than ICE vehicles and AWD emits **${f1(E.awd_vs_fwd_pct_avg)}% more** than FWD. Mean CO₂ fell **${f1(Math.abs(T.pct_change))}%** from 2015 to 2024.`,
  `The four fuel-rating columns are the target in disguise (r up to ${f3(E.corr_pairs["comb_l100km-co2"])}) and are excluded from modelling. Among legitimate predictors, engine size has VIF ${f1(E.vif_design.engine_size_l)}.`,
]));
c.push(spacer(60));

// 1 ----------------------------------------------------------------------
c.push(h1("1. Dataset: source, structure and key variables"));
c.push(h2("1.1 Source and why it is simulated"));
c.push(p("The brief allows an obtained **or simulated** public-style dataset. The work environment had no internet access, so real files (Natural Resources Canada fuel-consumption ratings, EPA fueleconomy.gov, EEA registrations) could not be downloaded. A generator (`00_generate_data.py`) produces a dataset whose columns and value ranges mimic those sources. **It is synthetic**: every row is artificial, brands are anonymised (Make_A … Make_L), and relationships between variables are built in by the generator. Findings therefore demonstrate the method; they must be re-checked on real data before being quoted as facts about the car market."));
c.push(p("The generator works at model-line level and then creates one row per model year:"));
c.push(...bullets([
  "720 model lines, each with a make, vehicle class (9 classes), base engine size, base weight, drivetrain and a small line-specific efficiency effect.",
  "Per year: engine redesigns, rising turbo adoption (5% → 50%), falling manual transmissions, growing mild and full hybrid share, power and weight variation.",
  "Consumption (L/100 km) is built from weight, engine size, power, drivetrain, transmission, technology progress and powertrain multipliers (full hybrid ×0.70, mild hybrid ×0.93, diesel ×0.82) with 3.5% random noise. CO₂ = consumption × 23.2 (petrol) or 26.5 (diesel) g per L/100 km.",
  "Realistic faults are then injected: inconsistent labels, impossible values, pounds entered as kilograms, sentinel values, missing values and duplicate rows.",
]));
c.push(h2("1.2 Structure and data dictionary"));
c.push(tcap("Data dictionary (raw file)."));
c.push(table(["Column", "Type", "Description", "Role"], [
  ["vehicle_id", "ID", "Unique row identifier", "Ignored"],
  ["model_family", "ID", "Model line (all years)", "Grouping for validation"],
  ["make", "Category", "Anonymised manufacturer (12)", "Feature"],
  ["model_year", "Integer", "2015–2024", "Feature"],
  ["vehicle_class", "Category", "9 classes: Sub-compact … Pickup Truck", "Feature"],
  ["engine_size_l", "Float", "Displacement in litres", "Feature"],
  ["cylinders", "Integer", "3, 4, 6, 8, 12", "Feature"],
  ["horsepower", "Integer", "Engine power (hp)", "Feature"],
  ["weight_kg", "Integer", "Curb weight (kg)", "Feature"],
  ["turbo", "Binary", "1 = turbocharged", "Feature"],
  ["transmission", "Category", "Automatic, Manual, CVT, DCT", "Feature"],
  ["drivetrain", "Category", "FWD, RWD, AWD", "Feature"],
  ["powertrain", "Category", "ICE, Mild hybrid, Full hybrid", "Feature"],
  ["fuel_type", "Category", "Regular, Premium, Diesel", "Feature"],
  ["city_l100km, hwy_l100km, comb_l100km, comb_mpg", "Float", "Fuel-consumption ratings", "**Excluded (leakage)**"],
  ["co2_gkm", "Integer", "Tailpipe CO₂ (g/km)", "**Target**"],
], [26, 10, 42, 22], { size: 17 }));
const d = E.describe;
c.push(tcap("Summary statistics of key numeric variables after cleaning."));
c.push(table(["Variable", "Mean", "Std", "Min", "Max"], ["engine_size_l", "horsepower", "weight_kg", "comb_l100km", "co2_gkm", "power_to_weight"].map((k) => [k, f1(d[k].mean), f1(d[k].std), f1(d[k].min), f1(d[k].max)]),
  [34, 16, 16, 17, 17], { aligns: [L.L, L.R, L.R, L.R, L.R] }));
c.push(p(`Vehicles per class: ` + Object.entries(E.class_counts).map(([k, v]) => `${k} ${v}`).join(", ") + `. About ${f1(E.high_emitter_share * 100)}% of vehicles emit at least 200 g/km.`, { before: 100 }));

// 2 ----------------------------------------------------------------------
c.push(h1("2. Data-quality audit"));
c.push(p("Before changing anything, the raw file was audited for missing values, duplicates, label variants and implausible numbers."));
c.push(...figure("fig01_missingness", "Missing values per column in the raw file and after impossible values were converted to missing.", 520));
c.push(p(`Weight is missing in ${E.raw_missing.weight_kg} raw rows, horsepower in ${E.raw_missing.horsepower}, drivetrain in ${E.raw_missing.drivetrain}. Missingness is **not random** for weight: ${X.weight_missing_by_band_pct["2015-17"]}% of 2015–17 rows lack weight versus ${X.weight_missing_by_band_pct["2018-20"]}% (2018–20) and ${X.weight_missing_by_band_pct["2021-24"]}% (2021–24), a pattern consistent with older records being less completely documented (missing at random given model year). Horsepower missingness is flat across periods (${X.hp_missing_by_band_pct["2015-17"]}–${X.hp_missing_by_band_pct["2021-24"]}%). The weight pattern matters because simply dropping rows would under-represent older vehicles.`));
c.push(tcap("Issues found in the raw file."));
c.push(table(["Issue", "Evidence"], [
  ["Duplicate rows", "70 exact duplicates"],
  ["Label variants", "vehicle_class had 19 spellings for 9 classes (e.g. “suv - small”, “SUV-Small ”); transmission 9 spellings for 4; fuel_type 7 for 3"],
  ["Pounds entered as kg", "24 weights between 4,753 and 5,606 (physically impossible for these vehicles)"],
  ["Impossible engine / cylinders", "Engine size 0, 0.2, 60 and 99 L; cylinders = 0"],
  ["Impossible power", "Horsepower 0, −1, −50 and 2,000"],
  ["Invalid target", "CO₂ of −1, 0 and sentinel 999 (9 rows)"],
  ["Year typo", "4 rows with model_year 2105"],
  ["Missing values", "Engine, power, weight, transmission, drivetrain, fuel type and combined consumption"],
], [24, 76]));

// 3 ----------------------------------------------------------------------
c.push(h1("3. Cleaning process"));
c.push(h2("3.1 Step-by-step log"));
c.push(tcap("Cleaning log with the rule applied and the number of rows (or label spellings) affected."));
c.push(table(["Step", "Rule", "Rows / spellings"], E.cleaning_log.filter((r) => r.rows > 0).map((r) => [r.step, r.rule, String(r.rows)]), [32, 56, 12], { aligns: [L.L, L.L, L.R], size: 17 }));
c.push(p("Notes on decisions that are not obvious:", { before: 100 }));
c.push(...bullets([
  "**Pounds → kilograms.** No real vehicle in these classes exceeds about 3,300 kg, so any weight above that is divided by 2.2046. The converted values (about 2,150–2,550 kg) fall in the normal range for heavy SUVs, pickups and full-size cars, which confirms the diagnosis.",
  "**Invalid → missing, not deleted.** Impossible engine, power and cylinder values are set to missing and treated with the imputation methods below, preserving the rest of each row.",
  "**Invalid target → deleted.** A wrong label cannot be imputed without circularity, so these 9 rows are dropped.",
  "**Combined consumption rebuilt from CO₂.** Because CO₂ = consumption × a fuel constant, 81 missing consumption values are recomputed exactly. This column is excluded from models anyway (leakage), but the rebuild keeps the EDA complete.",
]));
c.push(h3("Representative code"));
c.push(...code(`df = df.drop_duplicates()
df["vehicle_class"] = norm_label(df["vehicle_class"]).map(class_map)      # 19 spellings -> 9
m = df["weight_kg"] > 3300                                                  # pounds entered as kg
df.loc[m, "weight_kg"] = (df.loc[m, "weight_kg"] / 2.2046).round()
for col, (lo, hi) in {"engine_size_l": (0.6, 8.5), "horsepower": (30, 1500)}.items():
    df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan                  # impossible -> missing
df = df[(df.co2_gkm > 0) & (df.co2_gkm < 900)]                              # invalid target -> drop`));
c.push(...figure("fig02_outliers_before_after", "Box plots of four key variables before cleaning (top) and after cleaning (bottom). Note the 99 L engine, 2,000 hp and 999 g/km sentinel values in the raw data.", 600));

c.push(h2("3.2 Handling missing values: imputation benchmark"));
c.push(p(`Choosing an imputation method by habit (“use the median”) can add error. Instead, ${E.imputation_complete_rows.toLocaleString()} complete rows were used: 10% of known values were hidden, imputed by each method, and the mean absolute error was measured (average of five random masks).`));
c.push(...figure("fig08_imputation_benchmark", "Imputation error when 10% of known weights and horsepower values are hidden. *Family median uses other years of the same model line, which is not available for a brand-new design.", 520));
const ib = E.imputation_benchmark_MAE;
c.push(table(["Method", "Weight MAE (kg)", "Horsepower MAE (hp)"], Object.keys(ib.weight_kg).map((k) => [k, f1(ib.weight_kg[k]), f1(ib.horsepower[k])]), [46, 27, 27], { aligns: [L.L, L.R, L.R], highlightRow: 3 }));
c.push(p(`Global median is poor (weight error ${f0(ib.weight_kg["Global median"])} kg) because classes differ greatly in mass. Using class information cuts this to ${f0(ib.weight_kg["Class median"])} kg, and the **iterative regression imputer** is best among methods usable on new vehicles (${f0(ib.weight_kg["Iterative (BayesianRidge)"])} kg for weight, ${f1(ib.horsepower["Iterative (BayesianRidge)"])} hp for power, where it exploits the strong engine–power link). Weight remains the hard variable: a ${f0(ib.weight_kg["Iterative (BayesianRidge)"])} kg error is about 4% of mean weight, which is why Week 4 checks how much accuracy is lost on rows with imputed weight.`, { before: 100 }));
c.push(p("Two versions of the cleaned file are saved. `clean_imputed.csv` (all gaps filled; categorical gaps filled with the class mode, cylinders from the engine-size band) is used for EDA. `clean_unimputed.csv` keeps the gaps so that imputation can happen **inside** each cross-validation fold in Week 3, which prevents test-set information leaking into imputation."));

c.push(h2("3.3 Outlier detection and decision"));
const io = E.iqr_outliers;
c.push(tcap("Outliers flagged by the 1.5×IQR rule on the cleaned data."));
c.push(table(["Variable", "Flagged", "% of rows", "Upper fence", "Most common class among flagged"],
  ["engine_size_l", "horsepower", "weight_kg", "co2_gkm", "power_to_weight"].map((k) => [k, String(io[k].n_flagged), f1(io[k].pct), f1(io[k].upper_fence), io[k].n_flagged ? `${io[k].top_class} (${f0(io[k].top_class_share * 100)}%)` : "–"]),
  [20, 12, 14, 16, 38], { aligns: [L.L, L.R, L.R, L.R, L.L] }));
c.push(p(`After cleaning, no impossible values remain, but the IQR rule still flags ${io.horsepower.n_flagged} horsepower and ${io.engine_size_l.n_flagged} engine-size values. They are concentrated in Sports/Performance cars and Pickup Trucks. A multivariate check agrees: an Isolation Forest (1% contamination) flagged ${E.isoforest.n_flagged} vehicles, all of them pickups (${E.isoforest.by_class["Pickup Truck"]}) or sports cars (${E.isoforest.by_class["Sports/Performance"]}). Judged **within** class, only ${E.within_class_co2_outliers} vehicles (${f1(E.within_class_co2_outliers / E.clean_rows * 100)}%) have unusual CO₂.`, { before: 100 }));
c.push(callout("Decision", "These extreme values are **real, explainable vehicles** (large V8 engines, high-performance variants), not errors. They are kept, because deleting them would remove exactly the high-emitting vehicles that matter for fleet compliance. Models that are sensitive to extremes are protected by regularisation and by using a log-transformed target (Week 3)."));

c.push(h2("3.4 Transformations and engineered features"));
const sk = E.skew_kurt;
c.push(...bullets([
  `**Skewness check.** CO₂ is mildly right-skewed (${f2(sk.co2_gkm[0])}); horsepower is strongly right-skewed (${f2(sk.horsepower[0])}, excess kurtosis ${f1(sk.horsepower[1])}); engine size is ${f2(sk.engine_size_l[0])}; weight ${f2(sk.weight_kg[0])}. Because hybrids and diesel act multiplicatively on CO₂, a log transform of the target is tested in Week 3.`,
  "**Engineered features:** `power_to_weight` (hp per tonne), `hp_per_litre` and the binary `high_emitter` (CO₂ ≥ 200 g/km) for the classification extension.",
  "**Encoding and scaling** (one-hot for categories, standardisation for linear models) are placed inside the modelling pipeline rather than applied here, to avoid leakage.",
]));

// 4 ----------------------------------------------------------------------
c.push(h1("4. Exploratory data analysis"));
c.push(h2("4.1 Distributions"));
c.push(...figure("fig03_distributions", "Distributions of CO₂, combined consumption, engine size, horsepower, weight and power-to-weight after cleaning.", 600));
c.push(p(`CO₂ ranges from ${f0(d.co2_gkm.min)} to ${f0(d.co2_gkm.max)} g/km with a mean of ${f1(d.co2_gkm.mean)} and median close to 175, showing several modes that correspond to vehicle classes (compact cars near 145 g/km, standard SUVs near 240). Horsepower has a long right tail created by a small number of performance and truck engines. Weight is roughly symmetric between 1,000 and 2,600 kg.`));
c.push(h2("4.2 Emissions by vehicle class"));
c.push(...figure("fig04_co2_by_class", "CO₂ by vehicle class, sorted by median. The dashed line marks the 200 g/km high-emitter threshold.", 520));
const cc = E.co2_by_class;
c.push(table(["Class", "n", "Mean CO₂", "Median", "Std"], Object.entries(cc).map(([k, v]) => [k, String(v.count), f1(v.mean), f1(v.median), f1(v.std)]), [34, 14, 18, 17, 17], { aligns: [L.L, L.R, L.R, L.R, L.R] }));
c.push(p(`Sub-compact cars average ${f0(cc["Sub-compact"].mean)} g/km and pickups ${f0(cc["Pickup Truck"].mean)} g/km, more than double. Differences between classes are overwhelmingly significant (Kruskal–Wallis H = ${f0(ks.H)}, p < 0.001); the effect size η² ≈ ${f2(ks.eta2)} means class membership accounts for roughly ${f0(ks.eta2 * 100)}% of the rank variation in CO₂. Importantly, Minivans and Sports/Performance cars (≈ 230 g/km) emit about as much as standard SUVs, despite being very different vehicles; the class label alone is therefore not enough.`, { before: 100 }));
c.push(h2("4.3 Powertrain, fuel and drivetrain"));
c.push(...figure("fig05_co2_by_powertrain_fuel_drive", "CO₂ by powertrain, fuel type and drivetrain.", 600));
const pt = E.co2_by_powertrain, dt = E.co2_by_drivetrain;
c.push(p(`Raw group means: ICE ${f0(pt.ICE.mean)}, mild hybrid ${f0(pt["Mild hybrid"].mean)}, full hybrid ${f0(pt["Full hybrid"].mean)} g/km; FWD ${f0(dt.FWD.mean)}, AWD ${f0(dt.AWD.mean)}, RWD ${f0(dt.RWD.mean)} g/km. Raw means mix in vehicle size (RWD is common in pickups and sports cars), so comparisons were repeated **within each class** and averaged:`));
c.push(table(["Class", "Full hybrid vs ICE", "AWD vs FWD"], Object.keys(E.hybrid_vs_ice_pct_by_class).map((k) => [k, `${f1(E.hybrid_vs_ice_pct_by_class[k])}%`, E.awd_vs_fwd_pct_by_class[k] !== undefined ? `+${f1(E.awd_vs_fwd_pct_by_class[k])}%` : "–"]).concat([["**Average**", `**${f1(E.hybrid_vs_ice_pct_avg)}%**`, `**+${f1(E.awd_vs_fwd_pct_avg)}%**`]]), [40, 30, 30], { aligns: [L.L, L.R, L.R] }));
c.push(p(`The class-controlled hybrid benefit (${f1(E.hybrid_vs_ice_pct_avg)}%) is very stable across classes (−29% to −34%, and −23% for sports cars), supporting hypothesis H2. AWD adds ${f1(E.awd_vs_fwd_pct_avg)}% on average and as much as ${f1(E.awd_vs_fwd_pct_by_class["Pickup Truck"])}% for pickups (H4). Premium-fuel and diesel vehicles emit more than regular petrol vehicles on average (${f0(E.co2_by_fuel.Premium.mean)} and ${f0(E.co2_by_fuel.Diesel.mean)} vs ${f0(E.co2_by_fuel.Regular.mean)} g/km), but this reflects larger, more powerful vehicles using those fuels, not the fuel itself. CVT vehicles emit least among transmissions (${f0(E.co2_by_transmission.CVT.mean)} g/km), largely because CVTs are fitted to smaller cars.`, { before: 100 }));
c.push(h2("4.4 Relationships between numeric variables"));
c.push(...figure("fig06_scatter_relationships", "CO₂ against weight, engine size, horsepower and power-to-weight, coloured by powertrain. The black line is a linear fit.", 560));
const pr = E.pearson_with_co2;
c.push(p(`Weight (r = ${f2(pr.weight_kg.pearson)}), engine size (${f2(pr.engine_size_l.pearson)}) and horsepower (${f2(pr.horsepower.pearson)}) are strongly related to CO₂ (H1 supported). Two patterns deserve attention. First, the **green full-hybrid points sit well below the cloud** at every weight and engine size, so powertrain adds information that mass and displacement cannot. Second, power-to-weight is only weakly related (r = ${f2(pr.power_to_weight.pearson)}): in this data it is **how heavy and how large** a vehicle is, rather than how fast, that drives emissions. At high horsepower (above about 350 hp) points flatten or fall below the line, a sign of non-linearity that tree models may capture.`));
c.push(h2("4.5 Trends over time"));
c.push(...figure("fig07_trends", "Mean CO₂ by model year (left), powertrain mix (centre) and technology adoption (right).", 610));
c.push(p(`Mean CO₂ fell from ${f1(T.co2_2015)} g/km in 2015 to ${f1(T.co2_2024)} g/km in 2024 (**${f1(T.pct_change)}%**). A linear fit on yearly means gives a slope of ${f2(T.slope_g_per_year)} g/km per year (p < 0.001), supporting H3. Over the same period the hybrid share rose from ${f1(T.hybrid_share_2015)}% to ${f1(T.hybrid_share_2024)}%, turbocharging from ${f0(T.turbo_2015)}% to ${f0(T.turbo_2024)}%, and manual transmissions fell from ${f0(T.manual_2015)}% to ${f1(T.manual_2024)}%. Average weight barely changed (${f0(T.weight_2015)} → ${f0(T.weight_2024)} kg) and power rose ${f1((T.hp_2024 / T.hp_2015 - 1) * 100)}%, so the improvement came from technology and electrification rather than lighter or weaker vehicles.`));

// 5 ----------------------------------------------------------------------
c.push(h1("5. Multicollinearity analysis"));
c.push(p("Multicollinearity means that predictors carry overlapping information. It does not reduce a model’s predictive accuracy much, but it makes individual coefficients unstable and hard to interpret. Four diagnostics were used."));
c.push(h2("5.1 Correlation structure"));
c.push(...figure("fig09_correlation_heatmap", "Pearson correlation matrix. The four rating columns (city, highway, combined, MPG) are almost perfectly related to CO₂.", 480));
const cp = E.corr_pairs;
c.push(table(["Pair", "Correlation", "Interpretation"], [
  ["engine size – cylinders", f2(cp["engine-cylinders"]), "Cylinder count is a coarse version of displacement"],
  ["engine size – horsepower", f2(cp["engine-horsepower"]), "Larger engines make more power"],
  ["cylinders – horsepower", f2(cp["cylinders-horsepower"]), "Same chain"],
  ["engine size – weight", f2(cp["engine-weight"]), "Bigger engines are in bigger vehicles"],
  ["horsepower – weight", f2(cp["horsepower-weight"]), "Moderate"],
  ["combined L/100 km – CO₂", f3(cp["comb_l100km-co2"]), "**Same quantity in different units (leakage)**"],
  ["city L/100 km – CO₂", f3(cp["city-co2"]), "Leakage"],
  ["MPG – CO₂", f3(cp["comb_mpg-co2"]), "Leakage (inverse relationship)"],
], [34, 16, 50], { aligns: [L.L, L.R, L.L] }));
c.push(h2("5.2 Variance inflation factors and condition number"));
c.push(p("For each predictor j, VIF_j = 1 / (1 − R²_j). **Worked example from this data:** the engine-size VIF is " + f1(E.vif_design.engine_size_l) + ", which means that regressing engine size on the other five predictors gives R² = 1 − 1/" + f1(E.vif_design.engine_size_l) + " = " + f2(1 - 1 / E.vif_design.engine_size_l) + "; about " + f0((1 - 1 / E.vif_design.engine_size_l) * 100) + "% of its variance is explained by the others, and its coefficient variance is inflated " + f0(E.vif_design.engine_size_l) + "-fold (standard error about ×" + f1(Math.sqrt(E.vif_design.engine_size_l)) + ")."));
c.push(...figure("fig10_vif", "Left: VIF for all numeric columns on a log scale (red = rating columns). Right: VIF for design features with and without cylinders.", 600));
const va = E.vif_all_with_leakage, vd = E.vif_design, vr = E.vif_reduced_no_cylinders;
c.push(table(["Variable", "VIF incl. rating columns", "VIF design features", "VIF without cylinders"], Object.keys(va).map((k) => [k, f1(va[k]), vd[k] !== undefined ? f1(vd[k]) : "–", vr[k] !== undefined ? f1(vr[k]) : "–"]), [28, 24, 24, 24], { aligns: [L.L, L.R, L.R, L.R] }));
c.push(p(`Including the rating columns, combined consumption reaches VIF ${f0(va.comb_l100km)} and city/highway about ${f0(va.city_l100km)}. Among design features, engine size (${f1(vd.engine_size_l)}), horsepower (${f1(vd.horsepower)}) and cylinders (${f1(vd.cylinders)}) exceed the common threshold of 10 or approach it. The condition number of the standardised matrix is ${f1(E.condition_number.with_leakage)} with rating columns, ${f1(E.condition_number.design)} for design features and ${f1(E.condition_number.reduced)} without cylinders: a moderate level (values above about 30 are usually considered severe). Principal-component analysis shows that the first component carries ${f0(E.pca_explained_ratio_design[0] * 100)}% of the variance of the six design features and the first three ${f0((E.pca_explained_ratio_design[0] + E.pca_explained_ratio_design[1] + E.pca_explained_ratio_design[2]) * 100)}%, i.e. the predictors are effectively about three-dimensional.`, { before: 100 }));
c.push(h2("5.3 Coefficient stability by bootstrap"));
c.push(p("An ordinary-least-squares regression of standardised CO₂ on the design features was fitted to 400 bootstrap resamples, with and without cylinders. If collinearity matters, coefficients will vary a lot between resamples."));
c.push(...figure("fig11_bootstrap_coefficients", "Bootstrap distribution of standardised coefficients with and without the cylinders variable.", 560));
const bc = E.bootstrap_coef;
c.push(table(["Coefficient", "SD with cylinders", "SD without cylinders", "Change"], ["engine_size_l", "horsepower", "weight_kg"].map((k) => [k, f3(bc[k].with_cyl_sd), f3(bc[k].no_cyl_sd), `${f0((bc[k].no_cyl_sd / bc[k].with_cyl_sd - 1) * 100)}%`]), [34, 22, 24, 20], { aligns: [L.L, L.R, L.R, L.R] }));
c.push(p(`Removing cylinders makes the engine-size coefficient about ${f0((1 - bc.engine_size_l.no_cyl_sd / bc.engine_size_l.with_cyl_sd) * 100)}% more stable and shifts its mean from ${f3(bc.engine_size_l.with_cyl_mean)} to ${f3(bc.engine_size_l.no_cyl_mean)} (the displacement effect was partly attributed to cylinders). Horsepower and weight coefficients are already precise (SD ≤ 0.017) and unaffected.`, { before: 100 }));
c.push(h2("5.4 Decisions"));
c.push(table(["Finding", "Decision", "Reason"], [
  ["Rating columns are the target in other units", "**Drop** city, highway, combined, MPG from all models", "They would not exist at design stage; keeping them would produce a meaningless R² near 1"],
  ["Cylinders ≈ coarse engine size (r 0.93)", "Keep in the feature set but expect near-zero importance; test in Week 3", "Little unique information; slightly destabilises the engine coefficient"],
  ["Engine size and horsepower r = 0.91; VIF 12–21", "**Keep both**", "Goal is prediction, not coefficient interpretation; they carry different engineering information (displacement vs. tuning/turbo); Ridge and tree models tolerate overlap"],
  ["Condition number moderate (≈ 10)", "Use Ridge as a safeguard; compare with OLS", "Checks whether shrinkage changes predictions"],
  ["Interpretation needed", "Explain models with permutation importance and partial dependence, not raw coefficients", "These are robust to correlated predictors in a predictive sense"],
], [30, 32, 38], { size: 17 }));

// 6 ----------------------------------------------------------------------
c.push(h1("6. Summary of initial findings"));
c.push(table(["Hypothesis (Week 1)", "Evidence", "Verdict"], [
  ["H1: weight and engine size are the strongest predictors", `r = ${f2(pr.weight_kg.pearson)} and ${f2(pr.engine_size_l.pearson)}`, "Supported"],
  ["H2: full hybrids emit 20–35% less in the same class", `${f1(E.hybrid_vs_ice_pct_avg)}% average; −23% to −34% by class`, "Supported"],
  ["H3: CO₂ falls with model year", `${f2(T.slope_g_per_year)} g/km per year, p < 0.001`, "Supported"],
  ["H4: AWD emits more than FWD in the same class", `+${f1(E.awd_vs_fwd_pct_avg)}% on average`, "Supported"],
  ["H5: high emitters can be flagged (F1 ≥ 0.90)", "Tested in Week 3", "Pending"],
], [42, 40, 18]));
c.push(...numbered([
  "Vehicle class, mass and powertrain together separate low from high emitters; no single variable is enough.",
  "The fleet is improving at about 2 g/km per year, and the improvement is technology-driven.",
  "Data quality is itself a finding: impossible values affected about 1% of rows and missing values about 8%; weight, the most informative variable, is also the hardest to impute.",
  "Collinearity exists but is manageable; leakage is the more serious modelling hazard.",
]));
c.push(h1("7. Conclusion and next steps"));
c.push(p(`A raw file of ${E.raw_shape[0].toLocaleString()} rows was converted into ${E.clean_rows.toLocaleString()} analysis-ready rows through a fully logged process, and the exploration produced consistent, quantified answers to four of the five hypotheses. The main risks carried into modelling are (1) the leakage columns, (2) overlapping engine-related predictors, (3) imputation noise in weight, and (4) the near-duplicate structure of consecutive model years within a model line.`));
c.push(p("**Week 3** will build a leakage-free pipeline with group-aware validation (all years of a model line kept together), compare linear, regularised, log-target, random-forest and gradient-boosting models, tune hyper-parameters with documented code, and add a high-emitter classifier. **Beyond this project**, the same cleaning and EDA code should be re-run on a real public file, and the sensitivity of conclusions to the synthetic structure checked."));
c.push(h1("Appendix. Code listing: VIF and bootstrap"));
c.push(...code(`def vif(frame, cols):
    out = {}
    Z = (frame[cols] - frame[cols].mean()) / frame[cols].std()
    for c in cols:
        y = Z[c].values
        Xo = np.c_[np.ones(len(Z)), Z.drop(columns=c).values]
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        r2 = 1 - ((y - Xo @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / max(1 - r2, 1e-6)
    return out

def boot_coefs(cols, B=400, seed=1):
    rng = np.random.default_rng(seed)
    Z = (d[cols] - d[cols].mean()) / d[cols].std()
    y = ((d.co2_gkm - d.co2_gkm.mean()) / d.co2_gkm.std()).values
    Xm = np.c_[np.ones(len(Z)), Z.values]
    res = np.zeros((B, len(cols)))
    for b in range(B):
        ix = rng.integers(0, len(Z), len(Z))
        res[b] = np.linalg.lstsq(Xm[ix], y[ix], rcond=None)[0][1:]
    return pd.DataFrame(res, columns=cols)`));

build("../outputs/Week2_Data_Cleaning_and_EDA_Automotive_CO2.docx", { headerText: "Automotive CO₂ Prediction Project · Week 2 – Data Cleaning and Exploratory Analysis", children: c });

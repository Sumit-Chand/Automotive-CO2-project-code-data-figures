"""
00_generate_data.py
-------------------
Generates a SIMULATED vehicle fuel-consumption / CO2 dataset (2015-2024 model years).

Why simulate?  The assignment allows "obtain OR simulate" a public automotive dataset.
The structure and value ranges mimic public fuel-consumption ratings
(Natural Resources Canada "Fuel Consumption Ratings", US EPA fueleconomy.gov,
EU EEA CO2 monitoring), but all rows are artificial and brand names are anonymised
(Make_A ... Make_L) so that nothing here is mistaken for a statement about real brands.

The generator deliberately injects realistic data-quality problems (missing values,
unit errors, impossible values, label inconsistencies, duplicates) so that the
cleaning stage in Week 2 has genuine work to do.

To run the rest of the pipeline on REAL data, replace data/raw_vehicles.csv with a
file that has the same column names (see README.md for a column-mapping table).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# 1. Family-level (model-line) attributes
# ----------------------------------------------------------------------------
classes = {
    # name: (probability, weight_mean_kg, weight_sd, engine_mean_L, engine_sd)
    "Sub-compact":        (0.08, 1150,  80, 1.5, 0.30),
    "Compact":            (0.20, 1300,  90, 1.8, 0.40),
    "Mid-size":           (0.17, 1500, 100, 2.4, 0.60),
    "Full-size":          (0.08, 1800, 130, 3.4, 0.90),
    "SUV - Small":        (0.18, 1550, 110, 2.1, 0.50),
    "SUV - Standard":     (0.13, 2000, 160, 3.4, 0.90),
    "Pickup Truck":       (0.07, 2250, 220, 4.8, 1.20),
    "Minivan":            (0.04, 2050, 100, 3.5, 0.30),
    "Sports/Performance": (0.05, 1600, 180, 4.2, 1.30),
}
class_names = list(classes)
class_p = np.array([classes[c][0] for c in class_names])
class_p = class_p / class_p.sum()

makes = [f"Make_{c}" for c in "ABCDEFGHIJKL"]
make_p = np.array([.14, .12, .11, .10, .09, .08, .08, .07, .06, .06, .05, .04])
make_eff = dict(zip(makes, [0.96, 0.97, 1.00, 1.00, 1.01, 1.02, 1.03, 0.98, 1.05, 1.04, 1.06, 1.02]))
make_lux = dict(zip(makes, [0, 0, 0, 0, 0, 0.5, 0.5, 0, 1, 1, 1, 0.5]))  # premium-ness 0..1

N_FAMILIES = 720
YEARS = np.arange(2015, 2025)

families = []
for i in range(N_FAMILIES):
    cls = rng.choice(class_names, p=class_p)
    _, wm, wsd, em, esd = classes[cls]
    make = rng.choice(makes, p=make_p)
    eng = float(np.clip(rng.normal(em, esd), 1.0, 6.8))
    fam = dict(
        model_family=f"F{i:04d}",
        make=make,
        vehicle_class=cls,
        base_engine=eng,
        base_weight=wm + 55 * (eng - em) + 60 * make_lux[make] + rng.normal(0, wsd * 0.6),
        fam_effect=rng.normal(0, 0.03),  # model-line specific efficiency (random effect)
    )
    # drivetrain tendency
    if cls in ("SUV - Small", "SUV - Standard", "Minivan"):
        p = [0.45, 0.05, 0.50] if cls != "Minivan" else [0.85, 0.02, 0.13]
    elif cls == "Pickup Truck":
        p = [0.02, 0.48, 0.50]
    elif cls == "Sports/Performance":
        p = [0.05, 0.60, 0.35]
    elif cls == "Full-size":
        p = [0.50, 0.25, 0.25]
    else:
        p = [0.85, 0.05, 0.10]
    fam["drivetrain"] = rng.choice(["FWD", "RWD", "AWD"], p=p)
    families.append(fam)
fam_df = pd.DataFrame(families)

# ----------------------------------------------------------------------------
# 2. Yearly rows per family
# ----------------------------------------------------------------------------
rows = []
for f in families:
    n_years = int(np.clip(rng.normal(8.8, 1.8), 4, 10))
    start = rng.integers(2015, 2025 - n_years + 1)
    yrs = np.arange(start, start + n_years)
    eng = f["base_engine"]
    for y in yrs:
        # engine redesign every ~5 years (downsizing trend)
        if rng.random() < 0.12:
            eng = float(np.clip(eng + rng.choice([-0.4, -0.3, 0.2, 0.3, -0.5]), 1.0, 6.8))
        eng_r = round(eng, 1)
        cls = f["vehicle_class"]

        # turbo adoption grows over time and is more likely for small engines
        p_turbo = np.clip(0.06 + 0.045 * (y - 2015), 0, 0.6) * (1.0 if eng_r < 3.5 else 0.6)
        if cls == "Sports/Performance":
            p_turbo = min(0.85, p_turbo + 0.30)
        turbo = int(rng.random() < p_turbo)

        # power
        pow_factor = 1.30 if cls == "Sports/Performance" else (0.97 if cls == "Minivan" else 1.0)
        hp = (45 * eng_r + 40) * (1 + 0.25 * turbo) * pow_factor * (1 + 0.08 * make_lux[f["make"]])
        hp += rng.normal(0, 9)
        hp = float(np.clip(hp, 60, 760))

        weight = f["base_weight"] + rng.normal(0, 25) + 4 * (y - 2015)  # slow weight creep
        weight = float(np.clip(weight, 850, 3200))

        # cylinders from engine size
        if eng_r < 1.4:
            cyl = 3
        elif eng_r < 2.9:
            cyl = 4
        elif eng_r < 4.4:
            cyl = 6
        elif eng_r < 6.4:
            cyl = 8
        else:
            cyl = 12

        # transmission
        p_manual = max(0.04, 0.18 - 0.012 * (y - 2015))
        if cls == "Sports/Performance":
            p_manual = 0.22
        p_cvt = (0.10 + 0.02 * (y - 2015)) if cls in ("Compact", "Sub-compact", "SUV - Small", "Mid-size") else 0.02
        p_dct = 0.06
        p_auto = max(0.0, 1 - p_manual - p_cvt - p_dct)
        probs = np.array([p_auto, p_manual, p_cvt, p_dct])
        trans = rng.choice(["Automatic", "Manual", "CVT", "DCT"], p=probs / probs.sum())

        # powertrain electrification grows with year
        scale = 0.4 if cls in ("Pickup Truck", "Sports/Performance") else 1.0
        p_hev = scale * (0.015 + 0.012 * (y - 2015))
        p_mhev = scale * (0.02 + 0.016 * (y - 2015))
        u = rng.random()
        powertrain = "Full hybrid" if u < p_hev else ("Mild hybrid" if u < p_hev + p_mhev else "ICE")

        # fuel type
        p_diesel = 0.06 if (cls in ("SUV - Standard", "Pickup Truck", "Full-size", "Mid-size") and powertrain == "ICE") else 0.0
        p_prem = np.clip(0.08 + 0.5 * (hp > 250) + 0.30 * make_lux[f["make"]], 0, 0.95)
        u = rng.random()
        fuel = "Diesel" if u < p_diesel else ("Premium" if rng.random() < p_prem else "Regular")

        # ------------------ consumption model (L/100 km) ------------------
        comb = 1.5 + 3.2 * (weight / 1000) + 0.55 * eng_r + 0.0040 * (hp - 150)
        comb += 0.9 * max(0.0, weight - 1900) / 1000                         # non-linear heavy-vehicle penalty
        comb += {"FWD": 0.0, "RWD": 0.10, "AWD": 0.25 + 0.20 * weight / 1000}[f["drivetrain"]]
        comb += {"Automatic": 0.0, "Manual": -0.25, "CVT": -0.35, "DCT": -0.15}[trans]
        comb += 0.10 * turbo
        comb -= 0.055 * (y - 2015)                                           # technology progress
        comb *= {"ICE": 1.0, "Mild hybrid": 0.93, "Full hybrid": 0.70}[powertrain]
        comb *= 0.82 if fuel == "Diesel" else 1.0
        comb *= make_eff[f["make"]] * (1 + f["fam_effect"])
        comb *= np.exp(rng.normal(0, 0.035))
        comb = float(max(3.0, comb))

        co2_factor = 26.5 if fuel == "Diesel" else 23.2                      # g CO2 per (L/100km)
        co2 = comb * co2_factor * (1 + rng.normal(0, 0.004))

        hyb = powertrain == "Full hybrid"
        city = comb * (rng.uniform(0.88, 1.00) if hyb else rng.uniform(1.15, 1.30))
        hwy = comb * (rng.uniform(1.00, 1.10) if hyb else rng.uniform(0.80, 0.88))

        rows.append(dict(
            model_family=f["model_family"], make=f["make"], model_year=int(y),
            vehicle_class=cls, engine_size_l=eng_r, cylinders=cyl, horsepower=round(hp),
            weight_kg=round(weight), turbo=turbo, transmission=trans, drivetrain=f["drivetrain"],
            powertrain=powertrain, fuel_type=fuel,
            city_l100km=round(city, 1), hwy_l100km=round(hwy, 1), comb_l100km=round(comb, 1),
            comb_mpg=round(235.215 / comb, 1), co2_gkm=round(co2),
        ))

df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
df.insert(0, "vehicle_id", [f"V{i:05d}" for i in range(len(df))])
clean_truth = df.copy()
clean_truth.to_csv("data/_ground_truth_clean.csv", index=False)  # kept only for validating imputation

# ----------------------------------------------------------------------------
# 3. Inject realistic data-quality problems
# ----------------------------------------------------------------------------
d = df.copy()
n = len(d)

# 3a. inconsistent category labels
def messy(series, mapping, frac):
    s = series.copy()
    idx = rng.choice(s.index[s.isin(mapping)], size=int(frac * s.isin(mapping).sum()), replace=False)
    s.loc[idx] = [rng.choice(mapping[v]) for v in s.loc[idx]]
    return s

d["vehicle_class"] = messy(d["vehicle_class"], {
    "SUV - Small": ["suv - small", "SUV-Small ", "SUV: Small"],
    "SUV - Standard": ["suv - standard", "SUV: Standard", "SUV-Standard "],
    "Compact": ["compact", " Compact"],
    "Pickup Truck": ["pickup", "PICKUP TRUCK"],
}, 0.12)
d["transmission"] = messy(d["transmission"], {"Automatic": ["AUTO", "auto", "Automatic "], "Manual": ["manual", "MT"]}, 0.10)
d["fuel_type"] = messy(d["fuel_type"], {"Regular": ["regular", "REGULAR "], "Premium": ["premium", "PREMIUM"]}, 0.08)

# 3b. impossible / erroneous numeric values
def pick(k, mask=None):
    pool = d.index if mask is None else d.index[mask]
    return rng.choice(pool, size=k, replace=False)

d.loc[pick(10), "engine_size_l"] = rng.choice([0.0, 0.0, 99.0, 60.0, 0.2], size=10)
d.loc[pick(8), "cylinders"] = 0
heavy = d["weight_kg"] > 2150
idx = pick(25, heavy)                                     # unit error: pounds entered instead of kg
d.loc[idx, "weight_kg"] = (d.loc[idx, "weight_kg"] * 2.2046).round()
d.loc[pick(6), "horsepower"] = rng.choice([0, -1, 0, -50, 0, -1], size=6)
d.loc[pick(3), "horsepower"] = 2000
d.loc[pick(5), "co2_gkm"] = rng.choice([-1, 0, -1, 0, -1], size=5)
d.loc[pick(4), "co2_gkm"] = 999                           # sentinel value
d.loc[pick(4), "model_year"] = 2105                       # typo

# 3c. missing values (weight is MAR: older model years are missing more often)
p_miss_w = np.where(d["model_year"].between(2015, 2017), 0.09, 0.025)
d.loc[rng.random(n) < p_miss_w, "weight_kg"] = np.nan
d.loc[rng.random(n) < 0.03, "horsepower"] = np.nan
d.loc[rng.random(n) < 0.025, "drivetrain"] = np.nan
d.loc[rng.random(n) < 0.02, "transmission"] = np.nan
d.loc[rng.random(n) < 0.01, "engine_size_l"] = np.nan
d.loc[rng.random(n) < 0.015, "comb_l100km"] = np.nan
d.loc[rng.random(n) < 0.008, "fuel_type"] = np.nan

# 3d. exact duplicates
dups = d.sample(70, random_state=7)
d = pd.concat([d, dups], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

d.to_csv("data/raw_vehicles.csv", index=False)
print("raw shape:", d.shape)
print(d.head())
print("\nMissing values:\n", d.isna().sum()[d.isna().sum() > 0])

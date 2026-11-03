# Automotive CO2 prediction project (Weeks 1-4)

Run order (from this folder, Python 3 with pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, joblib):

    python scripts/00_generate_data.py      # simulated raw data -> data/raw_vehicles.csv
    python scripts/01_clean_and_eda.py      # Week 2: cleaning, EDA, multicollinearity
    python scripts/02_modeling.py           # Week 3: models, tuning, classification (several minutes on 1 CPU)
    python scripts/03_evaluation_insights.py# Week 4: errors, intervals, temporal test, scenarios
    python scripts/04_extra_checks.py; python scripts/05_extra_analyses.py; python scripts/06_planning_figures.py

NOTE: the dataset is SIMULATED (brands anonymised). To use real data (NRCan / EPA / EEA), replace
data/raw_vehicles.csv with a file using the same column names (see Appendix A of the Week 1 report)
and re-run; weight_kg and horsepower must be joined from another source.

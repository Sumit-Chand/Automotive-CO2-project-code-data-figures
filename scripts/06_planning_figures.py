"""Workflow diagram and project Gantt chart used in the Week-1 plan."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY, BLUE, ORG, GRN, GREY = "#1F4E79", "#2E86AB", "#F18F01", "#6A994E", "#8D99AE"

# ---------------- workflow ----------------
steps = [
    ("1. Acquire", "Public datasets\n(NRCan, EPA, EEA)\nor simulation", "Wk 1-2", NAVY),
    ("2. Clean", "Labels, duplicates,\nunits, invalid values,\nimputation", "Wk 2", BLUE),
    ("3. Explore", "Distributions, trends,\ncorrelation, VIF,\noutliers", "Wk 2", BLUE),
    ("4. Model", "Pipeline, group CV,\nRidge / RF / GBM,\ntuning", "Wk 3", ORG),
    ("5. Evaluate", "Segment errors,\nintervals, temporal\nvalidation", "Wk 4", GRN),
    ("6. Recommend", "What-if scenarios,\nfleet simulation,\nstrategy", "Wk 4", GRN),
]
fig, ax = plt.subplots(figsize=(12, 2.9))
ax.set_xlim(0, 12.2); ax.set_ylim(0, 3); ax.axis("off")
w, h, gap = 1.8, 1.9, 0.28
for i, (t, body, wk, col) in enumerate(steps):
    x = 0.1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.45), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=2))
    ax.add_patch(FancyBboxPatch((x, 0.45 + h - 0.5), w, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec=col))
    ax.text(x + w / 2, 0.45 + h - 0.25, t, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, 0.45 + (h - 0.5) / 2, body, ha="center", va="center", fontsize=8.6, color="#222")
    ax.text(x + w / 2, 0.2, wk, ha="center", va="center", fontsize=8.5, color=col, fontweight="bold")
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.4), (x + w + gap - 0.02, 1.4), arrowstyle="-|>", mutation_scale=14, color="#555", lw=1.4))
fig.savefig("figures/fig00_workflow.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------- gantt ----------------
tasks = [
    ("Problem identification & scoping", 1, 3, NAVY), ("Data-source survey & feasibility pilot", 2, 5, NAVY),
    ("Methodology, tools & plan write-up", 4, 7, NAVY),
    ("Data acquisition / simulation", 8, 9, BLUE), ("Cleaning & imputation benchmark", 9, 11, BLUE),
    ("EDA & multicollinearity study", 11, 13, BLUE), ("EDA report", 13, 14, BLUE),
    ("Split design & preprocessing pipeline", 15, 16, ORG), ("Baseline & linear models", 16, 17, ORG),
    ("Tuning RF / boosting, blend", 17, 19, ORG), ("Classification extension & diagnostics", 19, 20, ORG), ("Modelling report", 20, 21, ORG),
    ("Error analysis & intervals", 22, 23, GRN), ("Temporal validation & scenarios", 23, 25, GRN),
    ("Fleet simulation & recommendations", 25, 26, GRN), ("Final report & QA", 26, 28, GRN),
]
fig, ax = plt.subplots(figsize=(11, 5.2))
for i, (n, s, e, c) in enumerate(tasks):
    ax.barh(i, e - s + 1, left=s - 0.5, color=c, height=0.62)
ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in tasks], fontsize=8.5); ax.invert_yaxis()
for wk in range(1, 5):
    ax.axvline((wk - 1) * 7 + 0.5, color="#999", lw=0.8, ls="--")
    ax.text((wk - 1) * 7 + 4, -0.9, f"Week {wk}", ha="center", fontsize=10, fontweight="bold")
ax.axvline(28.5, color="#999", lw=0.8, ls="--")
for x, lab in [(7.5, "M1"), (14.5, "M2"), (21.5, "M3"), (28.5, "M4")]:
    ax.scatter([x], [len(tasks) - 0.2], marker="D", color="#C73E1D", zorder=5)
    ax.text(x, len(tasks) + 0.55, lab, ha="center", fontsize=9, color="#C73E1D", fontweight="bold")
ax.set_xlim(0.5, 29); ax.set_ylim(len(tasks) + 0.9, -1.3); ax.set_xlabel("Project day"); ax.grid(axis="x", alpha=0.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig("figures/fig_gantt.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("ok")

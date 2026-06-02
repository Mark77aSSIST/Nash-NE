# 6. Import from the saved results file and visualization

import json, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "axes.spines.top": False,
                     "axes.spines.right": False,
                     "figure.dpi": 130})


# Set the path to the drive folder which want to save
DRIVE_SAVE_DIR = "/content/drive/MyDrive/Nash_LLM_Experiment"
# RESULT_DIR = os.path.join(DRIVE_SAVE_DIR, "results")


# Load Results
def load_all_results(result_dir: str) -> dict:
    
    loaded = {}
    for ds in DATASET_CONFIGS.keys():
        path = os.path.join(result_dir, f"result_{ds}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded[ds] = json.load(f)
            print(f"  [{ds}] Loaded "
                  f"Nash {loaded[ds]['nash_accuracy']*100:.1f}%  "
                  f"n={loaded[ds]['n_samples']}")
        else:
            print(f"  [{ds}] No File - Exclude from visualization")
    return loaded

print(" Bringing up saved experimental results...")
ALL_RESULTS = load_all_results(DRIVE_SAVE_DIR)

if not ALL_RESULTS:
    raise FileNotFoundError(
        f"The result file does not exist.\n"
        f"Path: {DRIVE_SAVE_DIR}"
    )

valid_results = {ds: r for ds, r in ALL_RESULTS.items() if r}
ds_names      = list(valid_results.keys())
N_DS          = len(ds_names)
print(f"\n총 {N_DS}ea Dataset Results Load Complete\n")

# Visualization
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Nash-NE LLM Ensemble — Experiment Result",
             fontsize=15, fontweight="bold", y=1.01)

x = np.arange(N_DS)
w = 0.35

nash_accs, best_base_accs = [], []
for ds in ds_names:
    res = valid_results[ds]
    nash_accs.append(res["nash_accuracy"] * 100)
    others = {k: v for k,v in res["baselines"].items() if k != "Nash-NE"}
    best_base_accs.append(max(others.values()) * 100)

# Plot 1: Nash-NE vs Best-Baseline
ax1 = fig.add_subplot(3, 2, 1)
ax1.bar(x - w/2, best_base_accs, w, label="Best Baseline",
        color="#B4B2A9", edgecolor="none")
ax1.bar(x + w/2, nash_accs, w, label="Nash-NE",
        color="#1D9E75", edgecolor="none")
for i, (nb, na) in enumerate(zip(best_base_accs, nash_accs)):
    diff  = na - nb
    color = "#1D9E75" if diff >= 0 else "#E24B4A"
    ax1.annotate(f"{diff:+.1f}", xy=(x[i]+w/2, na+1.5),
                 ha="center", fontsize=8.5, color=color, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=9)
ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Nash-NE vs Best Baseline", fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_ylim(0, max(max(nash_accs), max(best_base_accs)) + 18)

# Plot 2: Accuracy Heatmap
ax2 = fig.add_subplot(3, 2, 2)
method_keys = ["Nash-NE", "Majority-Vote", "MoA (information sharing)"]
def get_sc_key(bl):
    return next((k for k in bl if "Self-Consistency" in k), None)
sc_key = get_sc_key(valid_results[ds_names[0]]["baselines"])
all_methods = method_keys + ([sc_key] if sc_key else [])
heatmap = []
for mk in all_methods:
    row = []
    for ds in ds_names:
        bl  = valid_results[ds]["baselines"]
        val = next((v for k,v in bl.items() if mk in k), 0.0)
        row.append(val * 100)
    heatmap.append(row)
im = ax2.imshow(np.array(heatmap), cmap="YlGn", aspect="auto", vmin=0, vmax=100)
ax2.set_xticks(range(N_DS)); ax2.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=8)
ax2.set_yticks(range(len(all_methods)))
labels = ["Nash-NE", "Majority-Vote", "MoA", "Self-Consis."][:len(all_methods)]
ax2.set_yticklabels(labels, fontsize=9)
ax2.set_title("Accuracy Heatmap (%)", fontweight="bold")
plt.colorbar(im, ax=ax2, label="%")
for i in range(len(all_methods)):
    for j in range(N_DS):
        v = heatmap[i][j]
        ax2.text(j, i, f"{v:.0f}", ha="center", va="center",
                 fontsize=8, color="white" if v > 65 else "black")

# Plot 3: NE fulfillment rate & average number of convergence iterations
ax3 = fig.add_subplot(3, 2, 3)
ne_rates  = [valid_results[ds]["ne_rate"]  * 100 for ds in ds_names]
avg_iters = [valid_results[ds]["avg_iter"]       for ds in ds_names]
ax3b = ax3.twinx()
ax3.bar(x, ne_rates, 0.5, label="NE Fulfillment Rate (%)", color="#534AB7", alpha=0.75)
ax3b.plot(x, avg_iters, "o-", color="#D85A30", linewidth=2,
          markersize=7, label="Mean Convergence Iteration")
ax3.set_xticks(x); ax3.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=9)
ax3.set_ylabel("Nash Condition Fulfillment Rate (%)", color="#534AB7")
ax3b.set_ylabel("Number of Mean Convergence Iteration", color="#D85A30")
ax3.set_title("Nash Balance Quality Indicators", fontweight="bold")
ax3.set_ylim(0, 125)
h1,l1 = ax3.get_legend_handles_labels(); h2,l2 = ax3b.get_legend_handles_labels()
ax3.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")

# Plot 4: Regret distribution boxplot
ax4 = fig.add_subplot(3, 2, 4)
regret_data = [
    [r["max_regret"] for r in valid_results[ds]["stage2_detail"]]
    for ds in ds_names
]
bp = ax4.boxplot(regret_data, labels=ds_names, patch_artist=True,
                 medianprops={"color": "#E24B4A", "linewidth": 2})
for patch in bp["boxes"]:
    patch.set_facecolor("#AFA9EC"); patch.set_alpha(0.7)
ax4.axhline(1e-3, color="#639922", linestyle="--", linewidth=1,
            label="NE threshold (0.001)")
ax4.set_yscale("log")
ax4.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=9)
ax4.set_ylabel("Maximum Regret (log)", fontsize=9)
ax4.set_title("Nash Condition Verification — Regret Distribution", fontweight="bold")
ax4.legend(fontsize=8)

# Plot 5: Elevation Width Horizontal Bar
ax5 = fig.add_subplot(3, 2, 5)
improvements = [na - nb for na, nb in zip(nash_accs, best_base_accs)]
colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in improvements]
bars = ax5.barh(ds_names, improvements, color=colors, edgecolor="none", height=0.55)
ax5.axvline(0, color="#444", linewidth=0.8)
for bar, val in zip(bars, improvements):
    ax5.text(val + (0.2 if val >= 0 else -0.2),
             bar.get_y() + bar.get_height()/2,
             f"{val:+.1f}%pt", va="center",
             ha="left" if val >= 0 else "right",
             fontsize=9, fontweight="bold")
ax5.set_xlabel("Nash-NE Enhancements (%pt vs Best Baseline)", fontsize=9)
ax5.set_title("Nash-NE performance improvement / decline", fontweight="bold")

# Plot 6: Hypothesis Test Summary Table
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis("off")
h_rows = []
for ds in ds_names:
    res = valid_results[ds]
    bl  = res["baselines"]
    mv  = bl.get("Majority-Vote", 0)
    moa = bl.get("MoA (information sharing)", 0)
    # ASCII text (DejaVu Sans)
    h1  = "O" if res["nash_accuracy"] > mv          else "X"
    h2  = "O" if res["nash_accuracy"] >= moa - 0.05 else "X"
    h3  = "O" if res["ne_rate"] >= 0.80             else "X"
    h_rows.append([ds, h1, h2, h3, f"{res['avg_regret']:.4f}"])

df_h = pd.DataFrame(
    h_rows,
    columns=["Dataset", "H1\n(Nash>MV)", "H2\n(Nash>=MoA)", "H3\n(NE>80%)", "Avg\nRegret"]
)
tbl = ax6.table(cellText=df_h.values, colLabels=df_h.columns,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#ddd")
    if row == 0:
        cell.set_facecolor("#534AB7")
        cell.set_text_props(color="white", fontweight="bold")
    elif col in (1, 2, 3) and row > 0:   
        val = cell.get_text().get_text()
        if val == "O":
            cell.set_facecolor("#EAF3DE")
            cell.set_text_props(color="#3B6D11", fontweight="bold")
        elif val == "X":
            cell.set_facecolor("#FCEBEB")
            cell.set_text_props(color="#A32D2D", fontweight="bold")

ax6.set_title("Hypothesis Verification Summary  (O = Support / X = Reject)",
              fontweight="bold", pad=16)

plt.tight_layout()

# Save
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig_name  = f"nash_plot_{EXPERIMENT_SCALE}_{timestamp}.png"
fig_path  = os.path.join(DRIVE_SAVE_DIR, fig_name)
plt.savefig(fig_path, bbox_inches="tight", dpi=150)
plt.show()
print(f" Graph Save Complete: {fig_path}")

# Final Summary Table
print("\n" + "="*70)
print("  Final Result Summary Table")
print("="*70)
rows = []
for ds in ds_names:
    res = valid_results[ds]
    bl  = res["baselines"]
    mv  = bl.get("Majority-Vote", 0)
    moa = bl.get("MoA (information sharing)", 0)
    sc  = next((v for k,v in bl.items() if "Self-Consistency" in k), 0)
    sb  = next((v for k,v in bl.items() if "Single-Best" in k), 0)
    nash = res["nash_accuracy"]
    rows.append({
        "Dataset"    : ds,
        "Task"       : res["task_type"],
        "n"          : res["n_samples"],
        "Single(%)"  : f"{sb*100:.1f}",
        "SC(%)"      : f"{sc*100:.1f}",
        "Vote(%)"    : f"{mv*100:.1f}",
        "MoA(%)"     : f"{moa*100:.1f}",
        "Nash-NE(%)": f"{nash*100:.1f}",
        "Δ"          : f"{(nash-max(sb,sc,mv,moa))*100:+.1f}",
        "NE-rate"    : f"{res['ne_rate']*100:.0f}%",
        "Regret"     : f"{res['avg_regret']:.5f}",
    })
df_final = pd.DataFrame(rows)
print(df_final.to_string(index=False))
print("="*70)
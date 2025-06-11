import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from pe4ci.library import start

# ------------------ CONFIG ------------------
PLATFORMS = ["openai", "llama3.3"]
CONCEPTS = ["gratitude", "ncb", "mm"]
CONCEPT_LABELS = {
    "gratitude": "Gratitude",
    "ncb": "Negative Core Beliefs",
    "mm": "Meaning Making",
}
BINS = 12

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True)

for row_idx, platform in enumerate(PLATFORMS):
    for col_idx, concept in enumerate(CONCEPTS):
        ax = axes[row_idx, col_idx]

        fewshot_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_few_results_train.xlsx",
        )
        zero_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_zero_results_dev.xlsx",
        )

        try:
            df_few = pd.read_excel(fewshot_path, sheet_name="results")
            df_zero = pd.read_excel(zero_path, sheet_name="results")
        except Exception as e:
            print(f"Error loading data for {platform} {concept}: {e}")
            continue

        # Get baseline F1s
        top_baseline = df_zero["F1"].max()
        bottom_baseline = df_zero["F1"].min()

        # Few-shot F1s
        top_f1 = df_few[df_few["category"] == "top"]["F1"]
        bottom_f1 = df_few[df_few["category"] == "bottom"]["F1"]
        all_f1 = pd.concat([top_f1, bottom_f1])
        bins = np.histogram_bin_edges(all_f1, bins=BINS)

        # Plot bottom first (white hatch), then top (gray)
        ax.hist(
            bottom_f1,
            bins=bins,
            alpha=0.5,
            label="Bottom Prompt (Few-Shot)" if row_idx == 0 and col_idx == 0 else None,
            edgecolor="black",
            color="white",
            linewidth=1.0,
            hatch="//",
        )
        ax.hist(
            top_f1,
            bins=bins,
            alpha=0.5,
            label="Top Prompt (Few-Shot)" if row_idx == 0 and col_idx == 0 else None,
            edgecolor="black",
            color="gray",
            linewidth=1.0,
        )

        # Add dashed lines for zero-shot baselines
        ax.axvline(
            bottom_baseline,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=(
                "Bottom Prompt (Zero-Shot Baseline)"
                if row_idx == 0 and col_idx == 0
                else None
            ),
        )
        ax.axvline(
            top_baseline,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=(
                "Top Prompt (Zero-Shot Baseline)"
                if row_idx == 0 and col_idx == 0
                else None
            ),
        )

        # Titles and labels
        concept_title = CONCEPT_LABELS[concept]
        ax.set_title(f"{concept_title} ({platform})", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel("Frequency", fontsize=11)
        ax.set_xlabel("F1 Score", fontsize=11)
        ax.set_xlim(0.3, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

# ------------------ STYLING ------------------
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, frameon=False, loc="lower center", ncol=2)
fig.suptitle(
    "Few-Shot F1 Score Distribution by Prompt Type (All Concepts & Platforms)",
    fontsize=16,
)
plt.rcParams["font.family"] = "times new roman"
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

output_path = os.path.join(start.RESULTS_DIR, "fewshot_f1_histograms_all_labeled.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot: {output_path}")

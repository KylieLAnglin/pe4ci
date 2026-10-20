# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from crisp.library import start

# %%
# ------------------ SETUP ------------------
PLATFORMS = ["openai", "llama3.3"]
CONCEPTS = ["gratitude", "mm", "ncb"]
CONCEPT_LABELS = {
    "gratitude": "Gratitude",
    "ncb": "Negative Core Beliefs",
    "mm": "Meaning Making",
}
PLATFORM_LABELS = {
    "openai": "gpt-5.4",
    "llama3.3": "llama3.3",
}
BINS = 12

OUTPUT_PATH = os.path.join(start.RESULTS_DIR, "fewshot_f1_histograms_all_labeled.png")
TABLE_OUTPUT_PATH = os.path.join(start.RESULTS_DIR, "fewshot_pct_above_baseline.xlsx")

# %%
# ------------------ PLOT ------------------
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True)

for row_idx, platform in enumerate(PLATFORMS):
    for col_idx, concept in enumerate(CONCEPTS):
        ax = axes[row_idx, col_idx]

        fewshot_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_few_results_train.xlsx",
        )
        zero_train_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_zero_results_train.xlsx",
        )
        zero_dev_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_zero_results_dev.xlsx",
        )

        try:
            df_few = pd.read_excel(fewshot_path, sheet_name="results")
            df_zero_train = pd.read_excel(zero_train_path, sheet_name="results")
            df_zero_dev = pd.read_excel(zero_dev_path, sheet_name="results")
        except Exception as e:
            print(f"Error loading data for {platform} {concept}: {e}")
            continue

        top_baseline_train = df_zero_train["F1"].max()
        bottom_baseline_train = df_zero_train["F1"].min()
        top_baseline_dev = df_zero_dev["F1"].max()
        bottom_baseline_dev = df_zero_dev["F1"].min()

        top_f1 = df_few[df_few["category"] == "top"]["F1"]
        bottom_f1 = df_few[df_few["category"] == "bottom"]["F1"]
        all_f1 = pd.concat([top_f1, bottom_f1])
        bins = np.histogram_bin_edges(all_f1, bins=BINS)

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
        ax.axvline(
            bottom_baseline_train,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="Bottom Prompt (Zero-Shot, Train)" if row_idx == 0 and col_idx == 0 else None,
        )
        ax.axvline(
            top_baseline_train,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Top Prompt (Zero-Shot, Train)" if row_idx == 0 and col_idx == 0 else None,
        )
        ax.axvline(
            bottom_baseline_dev,
            color="gray",
            linestyle="-",
            linewidth=1.2,
            label="Bottom Prompt (Zero-Shot, Dev)" if row_idx == 0 and col_idx == 0 else None,
        )
        ax.axvline(
            top_baseline_dev,
            color="black",
            linestyle="-",
            linewidth=1.2,
            label="Top Prompt (Zero-Shot, Dev)" if row_idx == 0 and col_idx == 0 else None,
        )

        concept_title = CONCEPT_LABELS[concept]
        ax.set_title(f"{concept_title} ({PLATFORM_LABELS[platform]})", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel("Frequency", fontsize=11)
        ax.set_xlabel("F1 Score", fontsize=11)
        ax.set_xlim(0.3, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, frameon=False, loc="lower center", ncol=3)
# fig.suptitle(
#     "Few-Shot F1 Score Distribution by Prompt Type (All Concepts & Platforms)",
#     fontsize=16,
# )
plt.rcParams["font.family"] = "times new roman"
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_PATH}")

# %%
# ------------------ SUMMARY TABLE ------------------
summary_rows = []

for platform in PLATFORMS:
    for concept in CONCEPTS:
        fewshot_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_few_results_train.xlsx",
        )
        zero_train_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_zero_results_train.xlsx",
        )

        try:
            df_few = pd.read_excel(fewshot_path, sheet_name="results")
            df_zero_train = pd.read_excel(zero_train_path, sheet_name="results")
        except Exception as e:
            print(f"Error loading data for {platform} {concept}: {e}")
            continue

        top_baseline_train = df_zero_train["F1"].max()
        bottom_baseline_train = df_zero_train["F1"].min()

        for category, baseline_f1 in [("top", top_baseline_train), ("bottom", bottom_baseline_train)]:
            category_f1 = df_few[df_few["category"] == category]["F1"]
            pct_above_baseline = (category_f1 > baseline_f1).mean() * 100
            summary_rows.append(
                {
                    "model": platform,
                    "construct": CONCEPT_LABELS[concept],
                    "category": category,
                    "baseline_f1": round(baseline_f1, 4),
                    "pct_above_baseline": round(pct_above_baseline, 1),
                }
            )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(TABLE_OUTPUT_PATH, index=False)
print(f"Saved: {TABLE_OUTPUT_PATH}")
print(summary_df.to_string(index=False))

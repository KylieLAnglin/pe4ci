# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from sklearn.metrics import f1_score

from crisp.library import start, metric_standard_errors

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
N_BOOTSTRAPS = 1000
RANDOM_STATE = 12

OUTPUT_PATH = os.path.join(start.RESULTS_DIR, "baseline_f1_histograms_all.png")
TABLE_OUTPUT_PATH = os.path.join(start.RESULTS_DIR, "baseline_f1_summary_stats.xlsx")

# %%
# ------------------ PLOT ------------------
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True)

for row_idx, platform in enumerate(PLATFORMS):
    for col_idx, concept in enumerate(CONCEPTS):
        ax = axes[row_idx, col_idx]

        train_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_zero_results_train.xlsx",
        )
        dev_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_zero_results_dev.xlsx",
        )
        alternative_dev_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_alternative_results_dev.xlsx",
        )

        df_train = pd.read_excel(train_path, sheet_name="results")
        df_dev = pd.read_excel(dev_path, sheet_name="results")
        df_alternative_dev = pd.read_excel(alternative_dev_path, sheet_name="results")

        df_gold = pd.read_excel(
            os.path.join(start.DATA_DIR, f"clean/{concept}_coding_final.xlsx"),
            usecols=["unique_text_id", "human_code"],
        )

        baseline_responses_path = os.path.join(
            start.DATA_DIR,
            f"responses_dev/{platform}_{concept}_baseline_zero_responses_dev.xlsx",
        )
        df_baseline_responses = pd.read_excel(baseline_responses_path).merge(
            df_gold, on="unique_text_id", how="inner"
        )

        alternative_responses_path = os.path.join(
            start.DATA_DIR,
            f"responses_dev/{platform}_{concept}_alternative_responses_dev.xlsx",
        )
        df_alt_responses = pd.read_excel(alternative_responses_path)

        all_f1 = df_train["F1"]

        top_row = df_dev.loc[df_dev["F1"].idxmax()]
        bottom_row = df_dev.loc[df_dev["F1"].idxmin()]
        top_f1 = top_row["F1"]
        bottom_f1 = bottom_row["F1"]

        median_dev_f1 = df_alternative_dev.loc[
            df_alternative_dev["alternative_type"] == "median", "F1"
        ].values[0]

        bins = np.histogram_bin_edges(all_f1, bins=BINS)

        ax.hist(
            all_f1,
            bins=bins,
            alpha=0.6,
            label="Baseline Prompts (Zero-Shot) in Training Dataset" if row_idx == 0 and col_idx == 0 else None,
            edgecolor="black",
            color="#aaaaaa",
            linewidth=1.0,
        )
        ax.axvline(
            bottom_f1, color="#888888", linestyle="--", linewidth=1.2,
            label="Bottom Prompt (Dev)" if row_idx == 0 and col_idx == 0 else None,
        )
        ax.axvline(
            top_f1, color="#111111", linestyle="--", linewidth=1.2,
            label="Top Prompt (Dev)" if row_idx == 0 and col_idx == 0 else None,
        )
        ax.axvline(
            median_dev_f1, color="#444444", linestyle="--", linewidth=1.2,
            label="Median Prompt (Dev)" if row_idx == 0 and col_idx == 0 else None,
        )

        concept_title = CONCEPT_LABELS[concept]
        platform_label = PLATFORM_LABELS[platform]
        ax.set_title(f"{concept_title} ({platform_label})", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel("Frequency", fontsize=11)
        ax.set_xlabel("F1 Score", fontsize=11)
        ax.set_xlim(0.3, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, frameon=False, loc="lower center", ncol=3)
# fig.suptitle(
#     "Histogram of Zero-Shot Baseline F1 Scores from Training Dataset, with Development Set Performace of Bottom, Median, and Top Prompts Indicated in Vertical Lines and Shaded Confidence Intervals",
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
        train_path = os.path.join(
            start.MAIN_DIR,
            f"results/{platform}_{concept}_baseline_zero_results_train.xlsx",
        )
        all_f1 = pd.read_excel(train_path, sheet_name="results")["F1"]
        summary_rows.append(
            {
                "model": platform,
                "construct": CONCEPT_LABELS[concept],
                "min": round(all_f1.min(), 4),
                "median": round(all_f1.median(), 4),
                "max": round(all_f1.max(), 4),
                "range": round(all_f1.max() - all_f1.min(), 4),
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(TABLE_OUTPUT_PATH, index=False)
print(f"Saved: {TABLE_OUTPUT_PATH}")
print(summary_df.to_string(index=False))

# %%
# ------------------ TRAIN VS DEV SIGNIFICANCE TEST ------------------
comparison_rows = []

for platform in PLATFORMS:
    for concept in CONCEPTS:
        df_gold = pd.read_excel(
            os.path.join(start.DATA_DIR, f"clean/{concept}_coding_final.xlsx"),
            usecols=["unique_text_id", "human_code"],
        )

        df_train_results = pd.read_excel(
            os.path.join(start.MAIN_DIR, f"results/{platform}_{concept}_baseline_zero_results_train.xlsx"),
            sheet_name="results",
        )
        df_dev_results = pd.read_excel(
            os.path.join(start.MAIN_DIR, f"results/{platform}_{concept}_baseline_zero_results_dev.xlsx"),
            sheet_name="results",
        )
        df_alternative_dev = pd.read_excel(
            os.path.join(start.MAIN_DIR, f"results/{platform}_{concept}_alternative_results_dev.xlsx"),
            sheet_name="results",
        )

        df_train_responses = pd.read_excel(
            os.path.join(start.DATA_DIR, f"responses_train/{platform}_{concept}_baseline_zero_responses_train.xlsx"),
        ).merge(df_gold, on="unique_text_id", how="inner")

        df_dev_responses = pd.read_excel(
            os.path.join(start.DATA_DIR, f"responses_dev/{platform}_{concept}_baseline_zero_responses_dev.xlsx"),
        ).merge(df_gold, on="unique_text_id", how="inner")

        df_alt_responses = pd.read_excel(
            os.path.join(start.DATA_DIR, f"responses_dev/{platform}_{concept}_alternative_responses_dev.xlsx"),
        )

        top_prompt_id = df_dev_results.loc[df_dev_results["F1"].idxmax(), "prompt_id"]
        bottom_prompt_id = df_dev_results.loc[df_dev_results["F1"].idxmin(), "prompt_id"]
        median_prompt_id = df_train_results.loc[
            (df_train_results["F1"] - df_train_results["F1"].median()).abs().idxmin(), "prompt_id"
        ]

        prompts = [
            ("top", top_prompt_id),
            ("bottom", bottom_prompt_id),
            ("median", median_prompt_id),
        ]

        for prompt_type, prompt_id in prompts:
            train_responses = df_train_responses[df_train_responses["prompt_id"] == prompt_id]
            train_f1 = df_train_results.loc[df_train_results["prompt_id"] == prompt_id, "F1"].values[0]
            train_participant_ids = train_responses["unique_text_id"].str.rsplit("_", n=1).str[0]
            train_se = metric_standard_errors.bootstrap_f1(train_responses["human_code"], train_responses["classification"], train_participant_ids, N_BOOTSTRAPS, RANDOM_STATE)

            if prompt_type == "median":
                dev_responses = df_alt_responses[df_alt_responses["alternative_type"] == "median"]
                dev_f1 = df_alternative_dev.loc[df_alternative_dev["alternative_type"] == "median", "F1"].values[0]
            else:
                dev_responses = df_dev_responses[df_dev_responses["prompt_id"] == prompt_id]
                dev_f1 = df_dev_results.loc[df_dev_results["prompt_id"] == prompt_id, "F1"].values[0]

            dev_participant_ids = dev_responses["unique_text_id"].str.rsplit("_", n=1).str[0]
            dev_se = metric_standard_errors.bootstrap_f1(dev_responses["human_code"], dev_responses["classification"], dev_participant_ids, N_BOOTSTRAPS, RANDOM_STATE)

            se_diff = np.sqrt(train_se**2 + dev_se**2)
            z = (dev_f1 - train_f1) / se_diff if se_diff > 0 else np.nan
            p_value = 2 * norm.sf(abs(z)) if not np.isnan(z) else np.nan

            comparison_rows.append({
                "model": PLATFORM_LABELS[platform],
                "construct": CONCEPT_LABELS[concept],
                "prompt": prompt_type,
                "train_f1": round(train_f1, 4),
                "dev_f1": round(dev_f1, 4),
                "diff (dev - train)": round(dev_f1 - train_f1, 4),
                "z": round(z, 3),
                "p_value": round(p_value, 4),
                "sig (p<0.05)": p_value < 0.05,
            })

comparison_df = pd.DataFrame(comparison_rows)
print("\nTrain vs. Dev F1 Significance Test (top, median, bottom prompts):")
print(comparison_df.to_string(index=False))
sig_cases = comparison_df[comparison_df["sig (p<0.05)"]]
if sig_cases.empty:
    print("\nNo statistically significant train-dev differences at p < 0.05.")
else:
    print("\nStatistically significant differences (p < 0.05):")
    print(sig_cases[["model", "construct", "prompt", "train_f1", "dev_f1", "diff (dev - train)", "z", "p_value"]].to_string(index=False))

# %%
# ------------------ TOP VS MEDIAN PROMPT SIGNIFICANCE TEST ------------------
top_vs_median_rows = []

for platform in PLATFORMS:
    for concept in CONCEPTS:
        df_gold = pd.read_excel(
            os.path.join(start.DATA_DIR, f"clean/{concept}_coding_final.xlsx"),
            usecols=["unique_text_id", "human_code"],
        )

        df_dev_results = pd.read_excel(
            os.path.join(start.MAIN_DIR, f"results/{platform}_{concept}_baseline_zero_results_dev.xlsx"),
            sheet_name="results",
        )
        df_alternative_dev = pd.read_excel(
            os.path.join(start.MAIN_DIR, f"results/{platform}_{concept}_alternative_results_dev.xlsx"),
            sheet_name="results",
        )

        df_dev_responses = pd.read_excel(
            os.path.join(start.DATA_DIR, f"responses_dev/{platform}_{concept}_baseline_zero_responses_dev.xlsx"),
        ).merge(df_gold, on="unique_text_id", how="inner")

        df_alt_responses = pd.read_excel(
            os.path.join(start.DATA_DIR, f"responses_dev/{platform}_{concept}_alternative_responses_dev.xlsx"),
        )

        top_row = df_dev_results.loc[df_dev_results["F1"].idxmax()]
        top_f1 = top_row["F1"]
        top_responses = df_dev_responses[df_dev_responses["prompt_id"] == top_row["prompt_id"]]
        top_participant_ids = top_responses["unique_text_id"].str.rsplit("_", n=1).str[0]
        top_se = metric_standard_errors.bootstrap_f1(top_responses["human_code"], top_responses["classification"], top_participant_ids, N_BOOTSTRAPS, RANDOM_STATE)

        median_f1 = df_alternative_dev.loc[df_alternative_dev["alternative_type"] == "median", "F1"].values[0]
        median_responses = df_alt_responses[df_alt_responses["alternative_type"] == "median"]
        median_participant_ids = median_responses["unique_text_id"].str.rsplit("_", n=1).str[0]
        median_se = metric_standard_errors.bootstrap_f1(median_responses["human_code"], median_responses["classification"], median_participant_ids, N_BOOTSTRAPS, RANDOM_STATE)

        se_diff = np.sqrt(top_se**2 + median_se**2)
        z = (top_f1 - median_f1) / se_diff if se_diff > 0 else np.nan
        p_value_one_sided = norm.sf(z) if not np.isnan(z) else np.nan

        top_vs_median_rows.append({
            "model": PLATFORM_LABELS[platform],
            "construct": CONCEPT_LABELS[concept],
            "top_f1": round(top_f1, 4),
            "median_f1": round(median_f1, 4),
            "diff (top - median)": round(top_f1 - median_f1, 4),
            "z": round(z, 3),
            "p_value (one-sided)": round(p_value_one_sided, 4),
            "sig (p<0.05)": p_value_one_sided < 0.05,
        })

top_vs_median_df = pd.DataFrame(top_vs_median_rows)
print("\nTop vs. Median Prompt Dev F1 Significance Test (one-sided, top > median):")
print(top_vs_median_df.to_string(index=False))
sig_top_vs_median = top_vs_median_df[top_vs_median_df["sig (p<0.05)"]]
if sig_top_vs_median.empty:
    print("\nTop prompt is not statistically significantly higher than the median prompt (p >= 0.05) in any case.")
else:
    print("\nCases where top prompt is significantly higher than median (p < 0.05):")
    print(sig_top_vs_median[["model", "construct", "top_f1", "median_f1", "diff (top - median)", "z", "p_value (one-sided)"]].to_string(index=False))

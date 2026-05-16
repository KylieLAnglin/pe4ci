# %%
import pandas as pd
import matplotlib.pyplot as plt

from crisp.library import start

# %%
# ------------------ SETUP ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
PLATFORMS = ["openai", "llama3.3"]

OUTPUT_PATH = start.RESULTS_DIR + "forest_combined.png"
ESTIMATES_PATH = start.RESULTS_DIR + "forest_estimates.xlsx"


def get_top_category_row(df, technique):
    technique_df = df[df["technique"] == technique].copy()
    technique_df = technique_df[technique_df["category"] == "top"]
    technique_df = technique_df.sort_values("F1", ascending=False)
    return technique_df.iloc[0]


def get_bottom_category_row(df, technique):
    technique_df = df[df["technique"] == technique].copy()
    technique_df = technique_df[technique_df["category"] == "bottom"]
    technique_df = technique_df.sort_values("F1", ascending=False)
    return technique_df.iloc[0]


# %%
# ------------------ PANEL 1: FIX BAD PROMPT ------------------
ROW_LABELS_1 = {
    "top_zero_minus_bottom_zero": "Combinatorial Empirical Prompting (Zero-shot)",
    "ape_bottom_minus_bottom_zero": "Automatic Prompt Engineering (Zero-shot)",
    "few_bottom_minus_bottom_zero": "Empirical Few-shot Prompting",
    "persona_bottom_minus_bottom_zero": "Persona (Zero-shot)",
    "cot_bottom_minus_bottom_zero": "Chain-of-Thought (Zero-shot)",
}

comparison_order_1 = [
    "top_zero_minus_bottom_zero",
    "ape_bottom_minus_bottom_zero",
    "few_bottom_minus_bottom_zero",
    "persona_bottom_minus_bottom_zero",
    "cot_bottom_minus_bottom_zero",
]

plot_rows_1 = []

for concept in CONCEPTS:
    for platform in PLATFORMS:
        results_df = pd.read_excel(
            start.RESULTS_DIR + f"{platform}_{concept}_dev_results.xlsx",
            sheet_name="results",
        )

        baseline_zero = results_df[results_df["technique"] == "baseline_zero"].sort_values("F1", ascending=False)
        top_zero_f1 = baseline_zero.iloc[0]["F1"]
        bottom_zero_f1 = baseline_zero.iloc[-1]["F1"]
        cot_bottom_f1 = results_df[results_df["technique"] == "cot_zero"].sort_values("F1", ascending=False).iloc[-1]["F1"]

        plot_rows_1.append({"panel": 1, "comparison": "top_zero_minus_bottom_zero", "label": ROW_LABELS_1["top_zero_minus_bottom_zero"], "concept": concept, "platform": platform, "difference": top_zero_f1 - bottom_zero_f1})
        plot_rows_1.append({"panel": 1, "comparison": "ape_bottom_minus_bottom_zero", "label": ROW_LABELS_1["ape_bottom_minus_bottom_zero"], "concept": concept, "platform": platform, "difference": get_bottom_category_row(results_df, "ape_zero")["F1"] - bottom_zero_f1})
        plot_rows_1.append({"panel": 1, "comparison": "few_bottom_minus_bottom_zero", "label": ROW_LABELS_1["few_bottom_minus_bottom_zero"], "concept": concept, "platform": platform, "difference": get_bottom_category_row(results_df, "baseline_few")["F1"] - bottom_zero_f1})
        plot_rows_1.append({"panel": 1, "comparison": "persona_bottom_minus_bottom_zero", "label": ROW_LABELS_1["persona_bottom_minus_bottom_zero"], "concept": concept, "platform": platform, "difference": get_bottom_category_row(results_df, "persona_zero")["F1"] - bottom_zero_f1})
        plot_rows_1.append({"panel": 1, "comparison": "cot_bottom_minus_bottom_zero", "label": ROW_LABELS_1["cot_bottom_minus_bottom_zero"], "concept": concept, "platform": platform, "difference": cot_bottom_f1 - bottom_zero_f1})

plot_df1 = pd.DataFrame(plot_rows_1)
mean_df1 = plot_df1.groupby(["panel", "comparison", "label"], as_index=False)["difference"].mean().rename(columns={"difference": "mean_difference"})

# %%
# ------------------ PANEL 2: IMPROVE GOOD PROMPT ------------------
ROW_LABELS_2 = {
    "few_top_minus_top_zero": "Empirical Few-shot Prompting",
    "ape_top_minus_top_zero": "Automatic Prompt Engineering (Zero-shot)",
    "persona_top_minus_top_zero": "Persona (Zero-shot)",
    "cot_top_minus_top_zero": "Chain-of-Thought (Zero-shot)",
}

comparison_order_2 = [
    "few_top_minus_top_zero",
    "ape_top_minus_top_zero",
    "persona_top_minus_top_zero",
    "cot_top_minus_top_zero",
]

plot_rows_2 = []

for concept in CONCEPTS:
    for platform in PLATFORMS:
        results_df = pd.read_excel(
            start.RESULTS_DIR + f"{platform}_{concept}_dev_results.xlsx",
            sheet_name="results",
        )

        top_zero_f1 = results_df[results_df["technique"] == "baseline_zero"].sort_values("F1", ascending=False).iloc[0]["F1"]
        cot_top_f1 = results_df[results_df["technique"] == "cot_zero"].sort_values("F1", ascending=False).iloc[0]["F1"]

        plot_rows_2.append({"panel": 2, "comparison": "few_top_minus_top_zero", "label": ROW_LABELS_2["few_top_minus_top_zero"], "concept": concept, "platform": platform, "difference": get_top_category_row(results_df, "baseline_few")["F1"] - top_zero_f1})
        plot_rows_2.append({"panel": 2, "comparison": "ape_top_minus_top_zero", "label": ROW_LABELS_2["ape_top_minus_top_zero"], "concept": concept, "platform": platform, "difference": get_top_category_row(results_df, "ape_zero")["F1"] - top_zero_f1})
        plot_rows_2.append({"panel": 2, "comparison": "persona_top_minus_top_zero", "label": ROW_LABELS_2["persona_top_minus_top_zero"], "concept": concept, "platform": platform, "difference": get_top_category_row(results_df, "persona_zero")["F1"] - top_zero_f1})
        plot_rows_2.append({"panel": 2, "comparison": "cot_top_minus_top_zero", "label": ROW_LABELS_2["cot_top_minus_top_zero"], "concept": concept, "platform": platform, "difference": cot_top_f1 - top_zero_f1})

plot_df2 = pd.DataFrame(plot_rows_2)
mean_df2 = plot_df2.groupby(["panel", "comparison", "label"], as_index=False)["difference"].mean().rename(columns={"difference": "mean_difference"})

# %%
# ------------------ PANEL 3: IMPROVE BEST PROMPT ------------------
ROW_LABELS_3 = {
    "ape_top_few_minus_top_few": "Automatic Prompt Engineering (Few-shot)",
    "persona_top_few_minus_top_few": "Persona (Few-shot)",
    "cot_top_few_minus_top_few": "Chain-of-Thought (Few-shot)",
    "explanation_top_few_minus_top_few": "Explanation (Few-shot)",
}

comparison_order_3 = [
    "ape_top_few_minus_top_few",
    "persona_top_few_minus_top_few",
    "cot_top_few_minus_top_few",
    "explanation_top_few_minus_top_few",
]

plot_rows_3 = []

for concept in CONCEPTS:
    for platform in PLATFORMS:
        results_df = pd.read_excel(
            start.RESULTS_DIR + f"{platform}_{concept}_dev_results.xlsx",
            sheet_name="results",
        )

        baseline_top_few_f1 = get_top_category_row(results_df, "baseline_few")["F1"]

        plot_rows_3.append({"panel": 3, "comparison": "ape_top_few_minus_top_few", "label": ROW_LABELS_3["ape_top_few_minus_top_few"], "concept": concept, "platform": platform, "difference": get_top_category_row(results_df, "ape_few")["F1"] - baseline_top_few_f1})
        plot_rows_3.append({"panel": 3, "comparison": "persona_top_few_minus_top_few", "label": ROW_LABELS_3["persona_top_few_minus_top_few"], "concept": concept, "platform": platform, "difference": get_top_category_row(results_df, "persona_few")["F1"] - baseline_top_few_f1})
        plot_rows_3.append({"panel": 3, "comparison": "cot_top_few_minus_top_few", "label": ROW_LABELS_3["cot_top_few_minus_top_few"], "concept": concept, "platform": platform, "difference": get_top_category_row(results_df, "cot_few")["F1"] - baseline_top_few_f1})
        plot_rows_3.append({"panel": 3, "comparison": "explanation_top_few_minus_top_few", "label": ROW_LABELS_3["explanation_top_few_minus_top_few"], "concept": concept, "platform": platform, "difference": get_top_category_row(results_df, "explanation_few")["F1"] - baseline_top_few_f1})

plot_df3 = pd.DataFrame(plot_rows_3)
mean_df3 = plot_df3.groupby(["panel", "comparison", "label"], as_index=False)["difference"].mean().rename(columns={"difference": "mean_difference"})

# %%
# ------------------ SHARED X AXIS BOUNDS ------------------
all_differences = pd.concat([plot_df1["difference"], plot_df2["difference"], plot_df3["difference"]])
x_pad = (all_differences.max() - all_differences.min()) * 0.1
xlim = (all_differences.min() - x_pad, all_differences.max() + x_pad)

# %%
# ------------------ PLOT ------------------
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1,
    figsize=(8, 11),
    gridspec_kw={"height_ratios": [5, 4, 4]},
)

panels = [
    (ax1, plot_df1, mean_df1, comparison_order_1, ROW_LABELS_1, "Impact of Technique on Worst-Performing Zero-Shot Prompt,\nas Estimated in Development Set"),
    (ax2, plot_df2, mean_df2, comparison_order_2, ROW_LABELS_2, "Impact of Technique on Best-Performing Zero-Shot Prompt,\nas Estimated in Development Set"),
    (ax3, plot_df3, mean_df3, comparison_order_3, ROW_LABELS_3, "Impact of Technique on Best-Performing Few-Shot Prompt,\nas Estimated in Development Set"),
]

for ax, plot_df, mean_df, comparison_order, row_labels, title in panels:
    y_positions = {comparison: position for comparison, position in zip(comparison_order, range(len(comparison_order), 0, -1))}

    for comparison in comparison_order:
        comparison_df = plot_df[plot_df["comparison"] == comparison]
        y = y_positions[comparison]

        ax.scatter(
            comparison_df["difference"],
            [y] * len(comparison_df),
            s=30,
            facecolors="white",
            edgecolors="black",
            linewidths=1,
            zorder=2,
        )

        mean_difference = mean_df[mean_df["comparison"] == comparison]["mean_difference"].iloc[0]

        ax.scatter(
            mean_difference,
            y,
            s=110,
            color="black",
            zorder=3,
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(xlim)
    ax.set_yticks([y_positions[item] for item in comparison_order])
    ax.set_yticklabels([row_labels[item] for item in comparison_order])
    ax.set_title(title)

ax3.set_xlabel("Difference in F1")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_PATH}")

# %%
# ------------------ EXPORT ESTIMATES TO EXCEL ------------------
all_estimates = pd.concat([plot_df1, plot_df2, plot_df3], ignore_index=True)
all_means = pd.concat([mean_df1, mean_df2, mean_df3], ignore_index=True)

wide_estimates = (
    all_estimates[["panel", "label", "concept", "platform", "difference"]]
    .pivot_table(index=["panel", "label"], columns=["concept", "platform"], values="difference")
)
wide_estimates.columns = [f"{concept}_{platform}" for concept, platform in wide_estimates.columns]
wide_estimates = wide_estimates.reset_index()

with pd.ExcelWriter(ESTIMATES_PATH, engine="openpyxl") as writer:
    wide_estimates.to_excel(writer, sheet_name="estimates", index=False)
    all_means[["panel", "label", "mean_difference"]].to_excel(writer, sheet_name="means", index=False)

print(f"Saved: {ESTIMATES_PATH}")

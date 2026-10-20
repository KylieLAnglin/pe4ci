# %%
import math
import pandas as pd

from crisp.library import start

# %%
# ------------------ SETUP ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
PLATFORMS = ["openai", "llama3.3"]

OUTPUT_PATH = start.RESULTS_DIR + "top_bottom_zero_shot_significance.xlsx"

# %%
# ------------------ COMPUTE SIGNIFICANCE ------------------
all_rows = []

for concept in CONCEPTS:
    for platform in PLATFORMS:
        results_path = start.RESULTS_DIR + f"{platform}_{concept}_dev_results.xlsx"

        results_df = pd.read_excel(results_path, sheet_name="results")
        baseline_df = results_df[results_df["technique"] == "baseline_zero"].copy()
        baseline_df = baseline_df.sort_values("F1", ascending=False)

        top_row = baseline_df.iloc[0]
        bottom_row = baseline_df.iloc[-1]

        top_f1 = top_row["F1"]
        bottom_f1 = bottom_row["F1"]
        top_se = top_row["F1 SE"]
        bottom_se = bottom_row["F1 SE"]

        difference = top_f1 - bottom_f1
        pooled_se = math.sqrt(top_se**2 + bottom_se**2)
        z_score = difference / pooled_se if pooled_se > 0 else 0

        if z_score > 2.58:
            significance = "***"
        elif z_score > 1.96:
            significance = "**"
        elif z_score > 1.64:
            significance = "*"
        else:
            significance = ""

        all_rows.append(
            {
                "concept": concept,
                "platform": platform,
                "top_prompt_id": top_row["prompt_id"],
                "bottom_prompt_id": bottom_row["prompt_id"],
                "top_f1": round(top_f1, 3),
                "top_f1_se": round(top_se, 3),
                "bottom_f1": round(bottom_f1, 3),
                "bottom_f1_se": round(bottom_se, 3),
                "difference_top_minus_bottom": round(difference, 3),
                "pooled_se": round(pooled_se, 3),
                "z_score": round(z_score, 3),
                "significance": significance,
                "top_significantly_higher": z_score > 1.64,
            }
        )

# %%
# ------------------ SAVE ------------------
significance_df = pd.DataFrame(all_rows)
significance_df.to_excel(OUTPUT_PATH, index=False, sheet_name="results")
print(f"Saved: {OUTPUT_PATH}")

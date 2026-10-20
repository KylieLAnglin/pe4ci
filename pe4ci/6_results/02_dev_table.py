# %%
import math
import pandas as pd

from pe4ci.library import start

# %%
# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM

DEV_RESULTS_PATH = start.RESULTS_DIR + f"{PLATFORM}_{CONCEPT}_dev_results.xlsx"
EXPORT_TABLE_PATH = start.RESULTS_DIR + f"{PLATFORM}_{CONCEPT}_dev_table.xlsx"

TECHNIQUE_LABELS = {
    "baseline_zero": ("Baseline", "Zero-Shot"),
    "baseline_few": ("Baseline", "Few-Shot"),
    "ape_zero": ("APE", "Zero-Shot"),
    "ape_few": ("APE", "Few-Shot"),
    "persona_zero": ("Persona", "Zero-Shot"),
    "persona_few": ("Persona", "Few-Shot"),
    "cot_zero": ("Chain-of-Thought", "Zero-Shot"),
    "cot_few": ("Chain-of-Thought", "Few-Shot"),
    "explanation_few": ("Explanations", "Few-Shot"),
}

# %%
# ------------------ LOAD DEV RESULTS ------------------
results_df = pd.read_excel(DEV_RESULTS_PATH, sheet_name="results")

baseline_row = (
    results_df[results_df["technique"] == "baseline_zero"]
    .sort_values("F1", ascending=False)
    .iloc[0]
)

# %%
# ------------------ BUILD TABLE ------------------
# compare the best prompt from each technique to the top zero-shot baseline prompt
table_rows = []

for technique_name, (technique_label, shot_label) in TECHNIQUE_LABELS.items():
    technique_df = results_df[results_df["technique"] == technique_name]

    if technique_df.empty:
        continue

    best_row = technique_df.sort_values("F1", ascending=False).iloc[0]

    pooled_se = math.sqrt(best_row["F1 SE"] ** 2 + baseline_row["F1 SE"] ** 2)
    z_score = abs(best_row["F1"] - baseline_row["F1"]) / pooled_se if pooled_se > 0 else 0
    p_value = math.erfc(z_score / math.sqrt(2))

    table_rows.append(
        {
            "Technique": technique_label,
            "Shots": shot_label,
            "F1": round(best_row["F1"], 2),
            "F1 SE": round(best_row["F1 SE"], 2),
            "Difference from Baseline": round(best_row["F1"] - baseline_row["F1"], 2),
            "p-value": round(p_value, 3),
            "prompt_id": best_row["prompt_id"],
            "prompt": best_row["prompt"],
        }
    )

table_df = pd.DataFrame(table_rows)
print(table_df[["Technique", "Shots", "F1", "F1 SE", "Difference from Baseline", "p-value"]])

# %%
# ------------------ EXPORT ------------------
table_df.to_excel(EXPORT_TABLE_PATH, index=False)
print(f"Saved: {EXPORT_TABLE_PATH}")

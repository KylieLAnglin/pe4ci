# %%
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pe4ci.library import start, metric_standard_errors

# %%
# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM

N_BOOTSTRAPS = 1000
RANDOM_STATE = 12

RESPONSES_DIR = start.DATA_DIR + "responses_dev/"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"
EXPORT_RESULTS_PATH = start.RESULTS_DIR + f"{PLATFORM}_{CONCEPT}_dev_results.xlsx"

TECHNIQUE_GROUP_COLUMNS = {
    "baseline_zero": [
        "prompt_id",
        "context_part_num",
        "task_part_num",
        "defn_part_num",
        "num_guidance_items",
        "guidance_part_nums",
    ],
    "baseline_few": ["prompt_id", "num_examples"],
    "ape_zero": ["prompt_id", "generation", "variant_id"],
    "ape_few": ["prompt_id", "num_examples", "sample_id"],
    "persona_zero": ["prompt_id", "persona"],
    "persona_few": ["prompt_id"],
    "cot_zero": ["prompt_id"],
    "cot_few": ["prompt_id", "combination"],
    "explanation_few": ["prompt_id", "combination"],
}

# %%
# ------------------ LOAD GOLD CODES ------------------
df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

# %%
# ------------------ ESTIMATE RESULTS FOR EACH TECHNIQUE ------------------
all_metric_rows = []

for technique_name, group_col in TECHNIQUE_GROUP_COLUMNS.items():
    response_path = RESPONSES_DIR + f"{PLATFORM}_{CONCEPT}_{technique_name}_responses_dev.xlsx"

    # techniques after the baseline are optional, so skip any that were not run
    if not os.path.exists(response_path):
        print(f"Skipping {technique_name}: {response_path} not found")
        continue

    response_df = pd.read_excel(response_path)
    scored_df = response_df.merge(df_gold, on="unique_text_id", how="inner")
    scored_df = scored_df.dropna(subset=["human_code", "classification"]).copy()

    technique_rows = []

    for group_values, prompt_group in scored_df.groupby(group_col, dropna=False):
        human_codes = prompt_group["human_code"]
        classifications = prompt_group["classification"]
        participant_ids = prompt_group["unique_text_id"].str.rsplit("_", n=1).str[0]

        result_row = {"technique": technique_name}

        for column_name, column_value in zip(group_col, list(group_values)):
            result_row[column_name] = column_value

        result_row["Accuracy"] = accuracy_score(human_codes, classifications)
        result_row["Precision"] = precision_score(human_codes, classifications, zero_division=0)
        result_row["Recall"] = recall_score(human_codes, classifications, zero_division=0)
        result_row["F1"] = f1_score(human_codes, classifications, zero_division=0)
        result_row["Accuracy SE"] = metric_standard_errors.bootstrap_accuracy(human_codes, classifications, participant_ids, N_BOOTSTRAPS, RANDOM_STATE)
        result_row["Precision SE"] = metric_standard_errors.bootstrap_precision(human_codes, classifications, participant_ids, N_BOOTSTRAPS, RANDOM_STATE)
        result_row["Recall SE"] = metric_standard_errors.bootstrap_recall(human_codes, classifications, participant_ids, N_BOOTSTRAPS, RANDOM_STATE)
        result_row["F1 SE"] = metric_standard_errors.bootstrap_f1(human_codes, classifications, participant_ids, N_BOOTSTRAPS, RANDOM_STATE)
        result_row["prompt"] = prompt_group["prompt"].iloc[0]

        technique_rows.append(result_row)

    all_metric_rows.extend(technique_rows)
    print(f"Processed {technique_name}: {len(technique_rows)} rows")

# %%
# ------------------ EXPORT ------------------
results_df = pd.DataFrame(all_metric_rows)
results_df.to_excel(EXPORT_RESULTS_PATH, index=False, sheet_name="results")
print(f"Saved: {EXPORT_RESULTS_PATH}")

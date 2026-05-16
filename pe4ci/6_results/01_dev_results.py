# %%
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from crisp.library import start, metric_standard_errors

# %%
# ------------------ SETUP ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
PLATFORMS = ["openai", "llama3.3"]

N_BOOTSTRAPS = 1000
RANDOM_STATE = 12

RESPONSES_DIR = start.DATA_DIR + "responses_dev/"
RESULTS_DIR = start.RESULTS_DIR

TECHNIQUE_CONFIGS = {
    "baseline_zero": {
        "group_col": [
            "prompt_id",
            "context_part_num",
            "task_part_num",
            "defn_part_num",
            "num_guidance_items",
            "guidance_part_nums",
        ],
    },
    "baseline_few": {
        "group_col": ["prompt_id", "category", "num_examples"],
    },
    "ape_zero": {
        "group_col": ["prompt_id", "category", "generation", "variant_id"],
    },
    "ape_few": {
        "group_col": ["prompt_id", "category", "num_examples", "sample_id"],
    },
    "persona_zero": {
        "group_col": ["prompt_id", "category", "persona"],
    },
    "persona_few": {
        "group_col": ["prompt_id", "category"],
    },
    "cot_zero": {
        "group_col": "prompt_id",
    },
    "cot_few": {
        "group_col": ["prompt_id", "category", "combination"],
    },
    "explanation_few": {
        "group_col": ["prompt_id", "category", "combination"],
    },
}

# %%
# ------------------ COMPUTE AND EXPORT RESULTS ------------------
for concept in CONCEPTS:
    gold_path = start.DATA_DIR + f"clean/{concept}_coding_final.xlsx"

    df_gold = pd.read_excel(gold_path)
    df_gold = df_gold[["unique_text_id", "human_code"]].copy()

    for platform in PLATFORMS:
        print(f"Processing {platform} - {concept}...")

        all_metric_rows = []

        for technique_name, technique_config in TECHNIQUE_CONFIGS.items():
            response_file = f"{platform}_{concept}_{technique_name}_responses_dev.xlsx"
            response_path = RESPONSES_DIR + response_file

            if not os.path.exists(response_path):
                print(f"Skipping {technique_name}: {response_path} not found")
                continue

            response_df = pd.read_excel(response_path)
            scored_df = response_df.merge(df_gold, on="unique_text_id", how="inner")

            group_col = technique_config["group_col"]
            if isinstance(group_col, str):
                group_col = [group_col]

            scored_df = scored_df.dropna(subset=["human_code", "classification"]).copy()
            prompt_groups = scored_df.groupby(group_col, dropna=False)

            technique_rows = []

            for group_values, prompt_group in prompt_groups:
                human_codes = prompt_group["human_code"]
                classifications = prompt_group["classification"]
                participant_ids = prompt_group["unique_text_id"].str.rsplit("_", n=1).str[0]

                if isinstance(group_values, tuple):
                    group_values_list = list(group_values)
                else:
                    group_values_list = [group_values]

                accuracy = accuracy_score(human_codes, classifications)
                precision = precision_score(human_codes, classifications, zero_division=0)
                recall = recall_score(human_codes, classifications, zero_division=0)
                f1 = f1_score(human_codes, classifications, zero_division=0)

                _, acc_se = metric_standard_errors.bootstrap_accuracy(human_codes, classifications, participant_ids, N_BOOTSTRAPS, RANDOM_STATE)
                _, prec_se = metric_standard_errors.bootstrap_precision(human_codes, classifications, participant_ids, N_BOOTSTRAPS, RANDOM_STATE)
                _, rec_se = metric_standard_errors.bootstrap_recall(human_codes, classifications, participant_ids, N_BOOTSTRAPS, RANDOM_STATE)
                _, f1_se = metric_standard_errors.bootstrap_f1(human_codes, classifications, participant_ids, N_BOOTSTRAPS, RANDOM_STATE)

                result_row = {"technique": technique_name}

                for column_name, column_value in zip(group_col, group_values_list):
                    result_row[column_name] = column_value

                result_row["Accuracy"] = round(accuracy, 3)
                result_row["Precision"] = round(precision, 3)
                result_row["Recall"] = round(recall, 3)
                result_row["F1"] = round(f1, 3)
                result_row["Accuracy SE"] = round(acc_se, 3)
                result_row["Precision SE"] = round(prec_se, 3)
                result_row["Recall SE"] = round(rec_se, 3)
                result_row["F1 SE"] = round(f1_se, 3)
                result_row["prompt"] = prompt_group["prompt"].iloc[0]

                technique_rows.append(result_row)

            all_metric_rows.extend(technique_rows)
            print(f"Processed {technique_name}: {len(technique_rows)} rows")

        output_path = RESULTS_DIR + f"{platform}_{concept}_dev_results.xlsx"
        results_df = pd.DataFrame(all_metric_rows)
        results_df.to_excel(output_path, index=False, sheet_name="results")
        print(f"Saved: {output_path}")

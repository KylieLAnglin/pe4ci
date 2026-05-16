# %%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from crisp.library import start, metric_standard_errors

# %%
# ------------------ SETUP ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
PLATFORM = "openai"

N_BOOTSTRAPS = 1000
RANDOM_STATE = 12
MISCLASSIFIED_SAMPLE_SIZE = 15
AGREEMENT_SAMPLE_SIZE = 5

RESPONSES_DIR = start.DATA_DIR + "responses_test/"
RESULTS_DIR = start.RESULTS_DIR

# %%
# ------------------ COMPUTE AND EXPORT TEST METRICS ------------------
misclassified_sheets = {}

for concept in CONCEPTS:
    print(f"Processing {concept}...")

    response_df = pd.read_excel(RESPONSES_DIR + f"{concept}_best_responses_test.xlsx")

    group_col = ["prompt_id", "platform", "technique", "category"]
    scored_df = response_df.dropna(subset=["human_code", "classification"]).copy()
    prompt_groups = scored_df.groupby(group_col, dropna=False)

    metric_rows = []

    for group_values, prompt_group in prompt_groups:
        human_codes = prompt_group["human_code"]
        classifications = prompt_group["classification"]
        participant_ids = prompt_group["unique_text_id"].str.rsplit("_", n=1).str[0]

        accuracy = accuracy_score(human_codes, classifications)
        precision = precision_score(human_codes, classifications, zero_division=0)
        recall = recall_score(human_codes, classifications, zero_division=0)
        f1 = f1_score(human_codes, classifications, zero_division=0)
        specificity = recall_score(human_codes, classifications, pos_label=0, zero_division=0)
        human_codes_arr = np.array(human_codes)
        classifications_arr = np.array(classifications)
        nn_rate = np.sum((human_codes_arr == 0) & (classifications_arr == 0)) / len(human_codes_arr)

        acc_ci_lower, acc_ci_upper = metric_standard_errors.bootstrap_metric_ci(human_codes, classifications, participant_ids, accuracy_score, N_BOOTSTRAPS, RANDOM_STATE)
        prec_ci_lower, prec_ci_upper = metric_standard_errors.bootstrap_metric_ci(human_codes, classifications, participant_ids, lambda y, yp: precision_score(y, yp, zero_division=0), N_BOOTSTRAPS, RANDOM_STATE)
        rec_ci_lower, rec_ci_upper = metric_standard_errors.bootstrap_metric_ci(human_codes, classifications, participant_ids, lambda y, yp: recall_score(y, yp, zero_division=0), N_BOOTSTRAPS, RANDOM_STATE)
        f1_ci_lower, f1_ci_upper = metric_standard_errors.bootstrap_metric_ci(human_codes, classifications, participant_ids, lambda y, yp: f1_score(y, yp, zero_division=0), N_BOOTSTRAPS, RANDOM_STATE)
        spec_ci_lower, spec_ci_upper = metric_standard_errors.bootstrap_metric_ci(human_codes, classifications, participant_ids, lambda y, yp: recall_score(y, yp, pos_label=0, zero_division=0), N_BOOTSTRAPS, RANDOM_STATE)
        nn_ci_lower, nn_ci_upper = metric_standard_errors.bootstrap_metric_ci(human_codes, classifications, participant_ids, lambda y, yp: np.sum((np.array(y) == 0) & (np.array(yp) == 0)) / len(y), N_BOOTSTRAPS, RANDOM_STATE)

        result_row = {"technique": "final"}
        for column_name, column_value in zip(group_col, list(group_values)):
            result_row[column_name] = column_value

        result_row["Accuracy"] = round(accuracy, 3)
        result_row["Precision"] = round(precision, 3)
        result_row["Recall"] = round(recall, 3)
        result_row["F1"] = round(f1, 3)
        result_row["Specificity"] = round(specificity, 3)
        result_row["NN Rate"] = round(nn_rate, 3)
        result_row["Accuracy CI Lower"] = round(acc_ci_lower, 3)
        result_row["Accuracy CI Upper"] = round(acc_ci_upper, 3)
        result_row["Precision CI Lower"] = round(prec_ci_lower, 3)
        result_row["Precision CI Upper"] = round(prec_ci_upper, 3)
        result_row["Recall CI Lower"] = round(rec_ci_lower, 3)
        result_row["Recall CI Upper"] = round(rec_ci_upper, 3)
        result_row["F1 CI Lower"] = round(f1_ci_lower, 3)
        result_row["F1 CI Upper"] = round(f1_ci_upper, 3)
        result_row["Specificity CI Lower"] = round(spec_ci_lower, 3)
        result_row["Specificity CI Upper"] = round(spec_ci_upper, 3)
        result_row["NN Rate CI Lower"] = round(nn_ci_lower, 3)
        result_row["NN Rate CI Upper"] = round(nn_ci_upper, 3)
        result_row["prompt"] = prompt_group["prompt"].iloc[0]

        metric_rows.append(result_row)

    output_path = RESULTS_DIR + f"{PLATFORM}_{concept}_test_results.xlsx"
    pd.DataFrame(metric_rows).to_excel(output_path, index=False, sheet_name="results")
    print(f"Saved: {output_path}")

    text_df = pd.read_excel(
        start.DATA_DIR + f"clean/{concept}_coding_final.xlsx",
        usecols=["unique_text_id", "text"],
    )
    misclassified = response_df[response_df["classification"] != response_df["human_code"]].merge(
        text_df, on="unique_text_id", how="left"
    )
    misclassified_sheets[concept] = (
        misclassified[["unique_text_id", "text", "human_code", "classification"]]
        .drop_duplicates(subset="unique_text_id")
        .sample(n=MISCLASSIFIED_SAMPLE_SIZE, random_state=RANDOM_STATE)
        .reset_index(drop=True)
        .sort_values(by="human_code", ascending=False)
    )

# %%
# ------------------ SAVE MISCLASSIFIED EXAMPLES ------------------
misclassified_output_path = RESULTS_DIR + "misclassified_examples.xlsx"
with pd.ExcelWriter(misclassified_output_path) as writer:
    for concept, df_sheet in misclassified_sheets.items():
        df_sheet.to_excel(writer, sheet_name=concept, index=False)
print(f"Saved: {misclassified_output_path}")

# %%
# ------------------ SAVE AGREEMENT EXAMPLES ------------------
agreement_sheets = {}

for concept in CONCEPTS:
    response_df = pd.read_excel(RESPONSES_DIR + f"{concept}_best_responses_test.xlsx")

    text_df = pd.read_excel(
        start.DATA_DIR + f"clean/{concept}_coding_final.xlsx",
        usecols=["unique_text_id", "text"],
    )
    agreements = (
        response_df[response_df["classification"] == response_df["human_code"]]
        .merge(text_df, on="unique_text_id", how="left")
        [["unique_text_id", "text", "human_code", "classification"]]
        .drop_duplicates(subset="unique_text_id")
    )

    positive_sample = agreements[agreements["human_code"] == 1].sample(
        n=min(AGREEMENT_SAMPLE_SIZE, (agreements["human_code"] == 1).sum()), random_state=RANDOM_STATE
    )
    negative_sample = agreements[agreements["human_code"] == 0].sample(
        n=min(AGREEMENT_SAMPLE_SIZE, (agreements["human_code"] == 0).sum()), random_state=RANDOM_STATE
    )
    sampled_ids = set(positive_sample["unique_text_id"]) | set(negative_sample["unique_text_id"])
    remaining_pool = agreements[~agreements["unique_text_id"].isin(sampled_ids)]
    fill_sample = remaining_pool.sample(
        n=min(AGREEMENT_SAMPLE_SIZE, len(remaining_pool)), random_state=RANDOM_STATE
    )
    agreement_sheets[concept] = (
        pd.concat([positive_sample, negative_sample, fill_sample])
        .reset_index(drop=True)
        .sort_values(by="human_code", ascending=False)
    )

agreements_output_path = RESULTS_DIR + "agreement_examples.xlsx"
with pd.ExcelWriter(agreements_output_path) as writer:
    for concept, df_sheet in agreement_sheets.items():
        df_sheet.to_excel(writer, sheet_name=concept, index=False)
print(f"Saved: {agreements_output_path}")

# %%
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pe4ci.library import start, metric_standard_errors

# %%
# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM

N_BOOTSTRAPS = 1000
RANDOM_STATE = 12

RESPONSE_PATH = start.DATA_DIR + f"responses_test/{PLATFORM}_{CONCEPT}_best_responses_test.xlsx"
TEXT_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"
EXPORT_RESULTS_PATH = start.RESULTS_DIR + f"{PLATFORM}_{CONCEPT}_test_results.xlsx"
EXPORT_MISCLASSIFIED_PATH = start.RESULTS_DIR + f"{PLATFORM}_{CONCEPT}_misclassified_test.xlsx"

METRICS = {
    "Accuracy": lambda y, yp: accuracy_score(y, yp),
    "Specificity": lambda y, yp: recall_score(y, yp, pos_label=0, zero_division=0),
    "Precision": lambda y, yp: precision_score(y, yp, zero_division=0),
    "Recall": lambda y, yp: recall_score(y, yp, zero_division=0),
    "F1": lambda y, yp: f1_score(y, yp, zero_division=0),
}

# %%
# ------------------ LOAD TEST RESPONSES ------------------
response_df = pd.read_excel(RESPONSE_PATH)
scored_df = response_df.dropna(subset=["human_code", "classification"]).copy()

human_codes = scored_df["human_code"]
classifications = scored_df["classification"]
participant_ids = scored_df["unique_text_id"].str.rsplit("_", n=1).str[0]

# %%
# ------------------ ESTIMATE METRICS WITH 95% CONFIDENCE INTERVALS ------------------
result_rows = []

for metric_name, metric_fn in METRICS.items():
    estimate = metric_fn(human_codes, classifications)
    ci_lower, ci_upper = metric_standard_errors.bootstrap_metric_ci(
        human_codes, classifications, participant_ids, metric_fn, N_BOOTSTRAPS, RANDOM_STATE
    )

    result_rows.append(
        {
            "Metric": metric_name,
            "Estimate": round(estimate, 2),
            "CI Lower": round(ci_lower, 2),
            "CI Upper": round(ci_upper, 2),
        }
    )

results_df = pd.DataFrame(result_rows)
print(f"Technique: {scored_df['technique'].iloc[0]}")
print(f"Prompt ID: {scored_df['prompt_id'].iloc[0]}")
print(results_df)

# %%
# ------------------ MISCLASSIFIED TEXTS ------------------
# useful for reviewing disagreements and refining the codebook
text_df = pd.read_excel(TEXT_PATH, usecols=["unique_text_id", "text"])

misclassified_df = (
    scored_df[scored_df["classification"] != scored_df["human_code"]]
    .merge(text_df, on="unique_text_id", how="left")
    [["unique_text_id", "text", "human_code", "classification"]]
    .sort_values(by="human_code", ascending=False)
)

# %%
# ------------------ EXPORT ------------------
results_df.to_excel(EXPORT_RESULTS_PATH, index=False, sheet_name="results")
print(f"Saved: {EXPORT_RESULTS_PATH}")

misclassified_df.to_excel(EXPORT_MISCLASSIFIED_PATH, index=False)
print(f"Saved: {EXPORT_MISCLASSIFIED_PATH}")

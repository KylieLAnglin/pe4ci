# 02_baseline_dev.py
import pandas as pd
from tqdm import tqdm
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from pe4ci.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT

PLATFORM = start.PLATFORM
MODEL = start.MODEL

SAMPLE = start.SAMPLE
SEED = start.SEED

print(f"Running baseline dev on {CONCEPT} with {PLATFORM} {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)
RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_dev/{PLATFORM}_{CONCEPT}_baseline_zero_responses_dev.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD PROMPTS ------------------
train_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
top_id = train_results.loc[train_results["F1"].idxmax(), "prompt_id"]
bottom_id = train_results.loc[train_results["F1"].idxmin(), "prompt_id"]

prompt_df = train_results[train_results["prompt_id"].isin([top_id, bottom_id])]
prompt_df["prompt"] = prompt_df["prompt"].str.replace("Text:", "", regex=False)

# ------------------ GENERATE RESPONSES ------------------

for row in tqdm(
    prompt_df.itertuples(), total=len(prompt_df), desc="Evaluating Prompts"
):
    print(row)
    prompt_text = row.prompt
    prompt_id = row.prompt_id

response_rows = []
for row in tqdm(
    prompt_df.itertuples(), total=len(prompt_df), desc="Evaluating Prompts"
):
    prompt_text = row.prompt
    prompt_id = row.prompt_id
    responses = classify.evaluate_prompt(
        prompt_text=prompt_text,
        prompt_id=prompt_id,
        df=df,
        platform=PLATFORM,
        temperature=0.0001,
    )
    response_rows.extend(responses)

# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(RESPONSE_PATH, index=False)
long_df = pd.read_excel(RESPONSE_PATH)

# ------------------ EXPORT METRICS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)

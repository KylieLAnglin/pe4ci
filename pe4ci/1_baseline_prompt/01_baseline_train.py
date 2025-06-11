# 01_baseline_train.py
import os
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from pe4ci.library import start, classify

# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED

print(f"Running baseline train on {CONCEPT} with {PLATFORM} {MODEL} in train set")


# ------------------ PATHS ------------------
PROMPT_PATH = start.DATA_DIR + f"prompts/{CONCEPT}_baseline_variants.xlsx"
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"

RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_baseline_zero_responses_train.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
if SAMPLE:
    df = df.sample(5, random_state=SEED)


# ------------------ LOAD PROMPTS ------------------
prompt_df = pd.read_excel(PROMPT_PATH, sheet_name="baseline")
prompt_df["prompt_id"] = prompt_df.index

# ------------------ COLLECT RESPONSES ------------------
response_rows = []
for row in tqdm(
    prompt_df.itertuples(), total=len(prompt_df), desc="Evaluating Prompts", position=0
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
# ------------------ EXPORT RESULTS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    sheet_name="results",
    include_se=False,
)

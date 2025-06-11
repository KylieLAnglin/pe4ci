# 1_baseline_prompt/04_fewshot_dev.py
# %%
import json
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

print(f"Running COT few-shot evaluation on {CONCEPT} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
BASELINE_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_cot_few_results_train.xlsx"
)

RESPONSE_PATH = (
    start.DATA_DIR + f"responses_dev/{PLATFORM}_{CONCEPT}_cot_few_responses_dev.xlsx"
)
RESULTS_PATH = start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_cot_few_results_dev.xlsx"

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
df = df[df.text.notna() & df.human_code.notna()]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD FEWSHOT EXAMPLES ------------------
with open(EXAMPLES_PATH, "r") as f:
    fewshot_samples = json.load(f)

# ------------------ LOAD TOP & BOTTOM BASELINE PROMPTS ------------------
baseline_df = pd.read_excel(BASELINE_RESULTS_PATH, sheet_name="results")
top_prompt = baseline_df.loc[baseline_df["F1"].idxmax(), "prompt"]
bottom_prompt = baseline_df.loc[baseline_df["F1"].idxmin(), "prompt"]

# ------------------ LOAD TRAINING FEWSHOT RESULTS ------------------

prompt_df = pd.read_excel(BASELINE_RESULTS_PATH, sheet_name="results")

prompt_df = prompt_df[
    prompt_df.groupby("category")["F1"].transform(max) == prompt_df["F1"]
]
prompt_df = prompt_df.reset_index(drop=True)

# ------------------ GENERATE RESPONSES ------------------
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
# %%

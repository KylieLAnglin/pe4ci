# 1_baseline_prompt/01_explanation_fewshot_dev.py
# %%

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

print(f"Running explanation few-shot evaluation on {CONCEPT} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
TRAIN_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_explanation_few_results_train.xlsx"
)

DEV_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_dev/{PLATFORM}_{CONCEPT}_explanation_few_responses_dev.xlsx"
)
DEV_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_explanation_few_results_dev.xlsx"
)

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
df = df[df.text.notna() & df.human_code.notna()]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD BEST PROMPTS FROM TRAIN ------------------
train_df = pd.read_excel(TRAIN_RESULTS_PATH, sheet_name="results")

best_prompts = (
    train_df.sort_values(["category", "F1"], ascending=[True, False])
    .groupby("category", as_index=False)
    .first()
    .reset_index(drop=True)
)

# ------------------ EVALUATE ON DEV SET ------------------
response_rows = []
for row in tqdm(
    best_prompts.itertuples(), total=len(best_prompts), desc="Evaluating Prompts"
):
    prompt_text = row.prompt
    prompt_id = row.prompt_id

    eval_rows = classify.evaluate_prompt(
        prompt_text=prompt_text,
        prompt_id=prompt_id,
        df=df,
        platform=PLATFORM,
        temperature=0.0001,
    )

    for r in eval_rows:
        r["prompt_id"] = prompt_id
        r["category"] = row.category
        r["prompt"] = prompt_text

    response_rows.extend(eval_rows)

# ------------------ EXPORT ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(DEV_RESPONSE_PATH, index=False)
long_df = pd.read_excel(DEV_RESPONSE_PATH)

classify.export_results_to_excel(
    df=long_df,
    output_path=DEV_RESULTS_PATH,
    group_col=[
        "prompt_id",
        "category",
    ],
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)

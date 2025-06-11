# 1_baseline_prompt/03_fewshot_train.py
import json
import os
import random
import sys
import numpy as np
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

random.seed(SEED)
rng = np.random.default_rng(SEED)

print(f"Running few-shot setup and evaluation on {CONCEPT} with {MODEL} in train set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)

FEWSHOT_EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
EXPORT_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_baseline_few_responses_train.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_train.xlsx"
)

# ------------------ LOAD TRAINING DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
df = df[df.text.notna() & df.human_code.notna()]

df_fewshot_pool = df[df.train_use == "example"].copy()
df_eval = df[df.train_use == "eval"].copy()

if SAMPLE:
    df_eval = df_eval.sample(5, random_state=SEED)

# ------------------ LOAD TOP AND BOTTOM PROMPTS ------------------
results_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
top_prompt = results_df.loc[results_df["F1"].idxmax(), "prompt"]
bottom_prompt = results_df.loc[results_df["F1"].idxmin(), "prompt"]
top_prompt = top_prompt.replace("Text:", "")
bottom_prompt = bottom_prompt.replace("Text:", "")

# ------------------ GENERATE OR LOAD FEWSHOT SAMPLES ------------------

if os.path.exists(FEWSHOT_EXAMPLES_PATH):
    print(
        f"Few-shot samples already exist at {FEWSHOT_EXAMPLES_PATH}, loading from file..."
    )
    with open(FEWSHOT_EXAMPLES_PATH, "r") as f:
        sample_examples = json.load(f)

else:
    print("Generating new few-shot samples...")
    sample_examples = []
    for sample_num in range(1, 51):
        num_examples = rng.integers(1, 11)
        sampled = df_fewshot_pool.sample(num_examples, random_state=sample_num)
        examples = [
            {"text": text, "label": int(label)}
            for text, label in zip(sampled["text"], sampled["human_code"])
        ]
        sample_examples.append(
            {
                "sample_id": int(sample_num),
                "num_examples": int(num_examples),
                "examples": examples,
            }
        )

    os.makedirs(os.path.dirname(FEWSHOT_EXAMPLES_PATH), exist_ok=True)
    with open(FEWSHOT_EXAMPLES_PATH, "w") as f:
        json.dump(sample_examples, f, indent=2)
    print(f"Saved {len(sample_examples)} few-shot samples to {FEWSHOT_EXAMPLES_PATH}")

# ------------------ EVALUATE SAMPLES ------------------
response_rows = classify.evaluate_fewshot_prompt_combinations(
    samples=sample_examples,
    df_eval=df_eval,
    prompt_dict={"top": top_prompt, "bottom": bottom_prompt},
    platform=PLATFORM,
    temperature=0.0001,
    prefix="fewshot",
)
# ------------------ EXPORT RESPONSES AND METRICS ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)

classify.export_results_to_excel(
    df=long_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id", "category", "num_examples"],
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)

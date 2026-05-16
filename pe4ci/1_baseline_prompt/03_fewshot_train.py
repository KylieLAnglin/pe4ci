# %%
import json
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from crisp.library import start, classify

# %%
# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = False
SEED = start.SEED
TEMPERATURE = 0.0001

RUN_CLASSIFICATIONS = True
SAVE_EVERY = 1
RUN_TAG = "_sample" if SAMPLE else ""
NUM_FEWSHOT_SAMPLES = 50
FEWSHOT_EXAMPLES_MIN = 4
FEWSHOT_EXAMPLES_MAX = 10

random.seed(SEED)
rng = np.random.default_rng(SEED)

print(f"Running few-shot setup and evaluation on {CONCEPT} with {MODEL} in train set")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

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
TEMP_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_baseline_few_responses_train_PROGRESS{RUN_TAG}.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_train.xlsx"
)

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "train"].copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

df = df.merge(df_gold, on="unique_text_id", how="inner")

df_fewshot_pool = df[df["train_use"] == "example"].copy()
df_eval = df[df["train_use"] == "eval"].copy()

if SAMPLE:
    df_eval = df_eval.sample(5, random_state=SEED).copy()

# %%
# ------------------ LOAD TOP AND BOTTOM PROMPTS ------------------
results_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
top_prompt = results_df.loc[results_df["F1"].idxmax(), "prompt"]
bottom_prompt = results_df.loc[results_df["F1"].idxmin(), "prompt"]

print(IMPORT_RESULTS_PATH)
print("Top prompt:", top_prompt)
print("Bottom prompt:", bottom_prompt)

# %%
# ------------------ GENERATE OR LOAD FEWSHOT SAMPLES ------------------
if os.path.exists(FEWSHOT_EXAMPLES_PATH):
    print(f"Few-shot samples already exist at {FEWSHOT_EXAMPLES_PATH}, loading from file...")
    with open(FEWSHOT_EXAMPLES_PATH, "r") as f:
        sample_examples = json.load(f)

else:
    print("Generating new few-shot samples...")
    sample_examples = []
    seen_sets = set()

    while len(sample_examples) < NUM_FEWSHOT_SAMPLES:
        sample_id = len(sample_examples) + 1
        num_examples = int(rng.integers(FEWSHOT_EXAMPLES_MIN, FEWSHOT_EXAMPLES_MAX + 1))

        sampled_indices = rng.choice(
            df_fewshot_pool.index,
            size=num_examples,
            replace=False,
        )
        sampled = df_fewshot_pool.loc[sampled_indices].copy()

        example_ids = tuple(sorted(sampled["unique_text_id"].tolist()))
        if example_ids in seen_sets:
            continue

        seen_sets.add(example_ids)

        examples = [
            {"text": text, "label": int(label)}
            for text, label in zip(sampled["text"], sampled["human_code"])
        ]

        sample_examples.append(
            {
                "sample_id": sample_id,
                "num_examples": num_examples,
                "examples": examples,
            }
        )

    os.makedirs(os.path.dirname(FEWSHOT_EXAMPLES_PATH), exist_ok=True)
    with open(FEWSHOT_EXAMPLES_PATH, "w") as f:
        json.dump(sample_examples, f, indent=2)

    print(f"Saved {len(sample_examples)} few-shot samples to {FEWSHOT_EXAMPLES_PATH}")

# %%
# ------------------ EVALUATE SAMPLES ------------------
if RUN_CLASSIFICATIONS:
    if SAMPLE:
        sample_examples_to_run = sample_examples[:2]
    else:
        sample_examples_to_run = sample_examples

    response_rows = []

    try:
        existing_df = pd.read_excel(TEMP_RESPONSE_PATH)
        completed_prompt_ids = set(existing_df["prompt_id"].unique())
        response_rows = existing_df.to_dict("records")
        print(f"Found existing progress for {len(completed_prompt_ids)} prompts")
    except FileNotFoundError:
        completed_prompt_ids = set()
        print("No existing progress file found")

    prompt_dict = {
        "top": top_prompt,
        "bottom": bottom_prompt,
    }

    counter = 0

    for sample in tqdm(sample_examples_to_run, desc="Evaluating Few-shot Samples"):

        sample_id = sample["sample_id"]
        num_examples = sample["num_examples"]

        example_block = "\n\n".join(
            [
                f'Text: "{ex["text"]}"\nAnswer: ```{"yes" if ex["label"] == 1 else "no"}```'
                for ex in sample["examples"]
            ]
        )

        for category, base_prompt in prompt_dict.items():
            if "{TEXT}" not in base_prompt:
                raise ValueError("base_prompt must contain {TEXT}")
            if "Here is the text:" not in base_prompt:
                raise ValueError('base_prompt must contain the phrase "Here is the text:"')

            prompt_id = f"{category}_fewshot_{sample_id}_n{num_examples}"

            if prompt_id in completed_prompt_ids:
                continue

            fewshot_prompt = base_prompt.replace(
                "Here is the text:",
                f"Here are some examples:\n{example_block}\n\nHere is the text:",
            )

            rows = classify.get_classifications_from_prompt(
                prompt_text=fewshot_prompt,
                prompt_id=prompt_id,
                df=df_eval[["unique_text_id", "text"]].copy(),
                platform=PLATFORM,
                temperature=TEMPERATURE,
            )

            for response_row in rows:
                response_row["category"] = category
                response_row["num_examples"] = num_examples
                response_row["sample_id"] = sample_id

            response_rows.extend(rows)
            completed_prompt_ids.add(prompt_id)

            counter += 1

            if counter % SAVE_EVERY == 0:
                pd.DataFrame(response_rows).to_excel(TEMP_RESPONSE_PATH, index=False)
                print(f"Saved progress after {counter} new prompts")

    long_df = pd.DataFrame(response_rows)
    long_df.to_excel(TEMP_RESPONSE_PATH, index=False)
    print(f"Saved: {TEMP_RESPONSE_PATH}")
    long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
    print(f"Saved: {EXPORT_RESPONSE_PATH}")

# %%
# ------------------ EXPORT RESPONSES AND METRICS ------------------
long_df = pd.read_excel(EXPORT_RESPONSE_PATH)
scored_df = long_df.merge(df_gold, on="unique_text_id", how="inner")

classify.export_results_to_excel(
    df=scored_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id", "category", "num_examples", "sample_id"],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {EXPORT_RESULTS_PATH}")

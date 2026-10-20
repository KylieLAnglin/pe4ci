# %%
import random
from itertools import product

import pandas as pd
from tqdm import tqdm

from pe4ci.library import start, classify

# %%
# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = False
SEED = start.SEED
TEMPERATURE = 0.0001

SAVE_EVERY = 1
RUN_TAG = "_sample" if SAMPLE else ""
NUM_SAMPLE_VARIANTS = 3

COT_TASK_SUFFIX = (
    " First, state what you notice in the text as it relates to the definition. "
    "Then, report your final answer as either ```yes``` or ```no```."
)

random.seed(SEED)

print(f"Running CoT Fewshot Training for {CONCEPT} on {PLATFORM} with {MODEL}")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

IMPORT_BASELINE_PROMPTS = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)
FILLED_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_template_w_reasoning.xlsx"

EXPORT_RESPONSE_PATH = (
    start.DATA_DIR + f"responses_train/{PLATFORM}_{CONCEPT}_cot_few_responses_train.xlsx"
)
TEMP_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_cot_few_responses_train_PROGRESS{RUN_TAG}.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_cot_few_results_train.xlsx"
)

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "train"].copy()

df_gold = pd.read_excel(GOLD_PATH)[["unique_text_id", "participant_id", "human_code"]]

df_eval = df[df["train_use"] == "eval"].copy()

if SAMPLE:
    df_eval = df_eval.sample(5, random_state=SEED).copy()

# %%
# ------------------ LOAD BASE PROMPTS ------------------
prompt_df = pd.read_excel(IMPORT_BASELINE_PROMPTS, sheet_name="results")

top_prompt = prompt_df.loc[prompt_df["F1"].idxmax(), "prompt"] + COT_TASK_SUFFIX

# %%
# ------------------ LOAD FILLED FILE ------------------
top_df = pd.read_excel(FILLED_PATH)

# %%
# ------------------ BUILD VARIANTS ------------------
top_variants_all = []
for combo in product([1, 2], repeat=len(top_df)):
    parts = []
    for i, row in enumerate(top_df.itertuples()):
        cot = row.exemplar_cot1 if combo[i] == 1 else row.exemplar_cot2
        label = "yes" if row.label == 1 else "no"
        parts.append(
            f'Text: "{row.text}"\n'
            f"Reasoning: {cot}\n"
            f"Answer: ```{label}```"
        )
    example_block = "\n\n".join(parts)
    if "Here is the text:" not in top_prompt:
        raise ValueError('Prompt must contain "Here is the text:"')
    full_prompt = top_prompt.replace(
        "Here is the text:",
        f"Here are some examples:\n{example_block}\n\nHere is the text:",
    )
    top_variants_all.append(
        {
            "prompt_id": f"topcot_{''.join(map(str, combo))}",
            "prompt": full_prompt,
            "combination": "".join(map(str, combo)),
        }
    )

# %%
# ------------------ SELECT VARIANTS ------------------
top_all_ones = "1" * len(top_df)
top_all_twos = "2" * len(top_df)

top_fixed_variants = []
top_remaining_variants = []
for variant in top_variants_all:
    if variant["combination"] in [top_all_ones, top_all_twos]:
        top_fixed_variants.append(variant)
    else:
        top_remaining_variants.append(variant)

top_variants = top_fixed_variants + random.sample(top_remaining_variants, NUM_SAMPLE_VARIANTS)

# %%
# ------------------ RESUME FROM PROGRESS FILE ------------------
try:
    existing_df = pd.read_excel(TEMP_RESPONSE_PATH)
    completed_prompt_ids = set(existing_df["prompt_id"].unique())
    response_rows = existing_df.to_dict("records")
    print(f"Found existing progress for {len(completed_prompt_ids)} prompts")
except FileNotFoundError:
    completed_prompt_ids = set()
    response_rows = []
    print("No existing progress file found")

# %%
# ------------------ RUN MODEL ------------------
counter = 0

for variant in tqdm(top_variants, desc="CoT Variants"):

    if variant["prompt_id"] in completed_prompt_ids:
        continue

    rows = classify.get_classifications_from_prompt(
        prompt_text=variant["prompt"],
        prompt_id=variant["prompt_id"],
        df=df_eval[["unique_text_id", "text"]].copy(),
        platform=PLATFORM,
        temperature=TEMPERATURE,
    )

    for response_row in rows:
        response_row["combination"] = variant["combination"]

    response_rows.extend(rows)
    completed_prompt_ids.add(variant["prompt_id"])

    counter += 1

    if counter % SAVE_EVERY == 0:
        pd.DataFrame(response_rows).to_excel(TEMP_RESPONSE_PATH, index=False)
        print(f"Saved progress after {counter} new prompts")

# %%
# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(TEMP_RESPONSE_PATH, index=False)
print(f"Saved: {TEMP_RESPONSE_PATH}")
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
print(f"Saved: {EXPORT_RESPONSE_PATH}")

# %%
# ------------------ SCORE AND EXPORT RESULTS ------------------
long_df = pd.read_excel(EXPORT_RESPONSE_PATH)
scored_df = long_df.merge(df_gold, on="unique_text_id", how="inner")

classify.export_results_to_excel(
    df=scored_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id", "combination"],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {EXPORT_RESULTS_PATH}")

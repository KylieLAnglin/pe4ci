# %%
import os
import random
from itertools import product

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

SAVE_EVERY = 1
NUM_SAMPLE_VARIANTS = 3
RUN_TAG = "_sample" if SAMPLE else ""

random.seed(SEED)

print(
    f"Running Explanation Fewshot Training for {CONCEPT} on {PLATFORM} with model {MODEL} and sample = {SAMPLE}."
)

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"
EXAMPLES_PATH = start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
IMPORT_RESULTS_PATH = start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_train.xlsx"
IMPORT_BASELINE_PROMPTS = start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
TOP_FILLED_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_top_template_w_reasoning.xlsx"
BOTTOM_FILLED_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_bottom_template_w_reasoning.xlsx"
EXPORT_RESPONSE_PATH = start.DATA_DIR + f"responses_train/{PLATFORM}_{CONCEPT}_explanation_few_responses_train.xlsx"
TEMP_RESPONSE_PATH = start.DATA_DIR + f"responses_train/{PLATFORM}_{CONCEPT}_explanation_few_responses_train_PROGRESS{RUN_TAG}.xlsx"
EXPORT_RESULTS_PATH = start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_explanation_few_results_train.xlsx"

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "train"].copy()

df_gold = pd.read_excel(GOLD_PATH)[["unique_text_id", "human_code"]]

df_eval = df[df["train_use"] == "eval"].copy()

if SAMPLE:
    df_eval = df_eval.sample(5, random_state=SEED).copy()

# %%
# ------------------ IDENTIFY BEST FEWSHOT SETS ------------------
baseline_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")

top_prompt_example_id = int(
    baseline_df.loc[baseline_df["F1"].idxmax(), "prompt_id"].split("_")[2]
)
bottom_prompt_example_id = int(
    baseline_df.loc[baseline_df["F1"].idxmin(), "prompt_id"].split("_")[2]
)

# %%
# ------------------ LOAD BASE PROMPTS ------------------
prompt_df = pd.read_excel(IMPORT_BASELINE_PROMPTS, sheet_name="results")

top_prompt = prompt_df.loc[prompt_df["F1"].idxmax(), "prompt"]
bottom_prompt = prompt_df.loc[prompt_df["F1"].idxmin(), "prompt"]

# %%
# ------------------ LOAD FEWSHOT EXAMPLES ------------------
example_combinations = pd.read_json(EXAMPLES_PATH)

top_examples = example_combinations.loc[
    example_combinations["sample_id"] == top_prompt_example_id, "examples"
].iloc[0]

bottom_examples = example_combinations.loc[
    example_combinations["sample_id"] == bottom_prompt_example_id, "examples"
].iloc[0]

# %%
# ------------------ LOAD FILLED FILES ------------------
if not os.path.exists(TOP_FILLED_PATH):
    raise FileNotFoundError(f"Missing: {TOP_FILLED_PATH}")

if not os.path.exists(BOTTOM_FILLED_PATH):
    raise FileNotFoundError(f"Missing: {BOTTOM_FILLED_PATH}")

top_examples_expl_df = pd.read_excel(TOP_FILLED_PATH)
bottom_examples_expl_df = pd.read_excel(BOTTOM_FILLED_PATH)

# %%
# ------------------ BUILD PROMPT VARIANTS ------------------
top_expl_variants_all = []
for combo in product([1, 2], repeat=len(top_examples_expl_df)):
    parts = []

    for i, row in enumerate(top_examples_expl_df.itertuples()):
        explanation = row.exemplar_cot1 if combo[i] == 1 else row.exemplar_cot2
        answer = "yes" if row.label == 1 else "no"

        parts.append(
            f'Text: "{row.text}"\n'
            f"Answer: ```{answer}```\n"
            f"Explanation: {explanation}"
        )

    example_block = "\n\n".join(parts)

    if "Here is the text:" not in top_prompt:
        raise ValueError('Prompt must contain "Here is the text:"')

    full_prompt = top_prompt.replace(
        "Here is the text:",
        f"Here are some examples:\n{example_block}\n\nHere is the text:",
    )

    top_expl_variants_all.append(
        {
            "prompt_id": f"top_explain_{''.join(str(c) for c in combo)}",
            "prompt": full_prompt,
            "combination": "".join(str(c) for c in combo),
        }
    )

bottom_expl_variants_all = []
for combo in product([1, 2], repeat=len(bottom_examples_expl_df)):
    parts = []

    for i, row in enumerate(bottom_examples_expl_df.itertuples()):
        explanation = row.exemplar_cot1 if combo[i] == 1 else row.exemplar_cot2
        answer = "yes" if row.label == 1 else "no"

        parts.append(
            f'Text: "{row.text}"\n'
            f"Answer: ```{answer}```\n"
            f"Explanation: {explanation}"
        )

    example_block = "\n\n".join(parts)

    if "Here is the text:" not in bottom_prompt:
        raise ValueError('Prompt must contain "Here is the text:"')

    full_prompt = bottom_prompt.replace(
        "Here is the text:",
        f"Here are some examples:\n{example_block}\n\nHere is the text:",
    )

    bottom_expl_variants_all.append(
        {
            "prompt_id": f"bot_explain_{''.join(str(c) for c in combo)}",
            "prompt": full_prompt,
            "combination": "".join(str(c) for c in combo),
        }
    )

# %%
# ------------------ SELECT FIXED + RANDOM COMBINATIONS ------------------
all_ones_top = next(v for v in top_expl_variants_all if set(v["combination"]) == {"1"})
all_twos_top = next(v for v in top_expl_variants_all if set(v["combination"]) == {"2"})
rest_top = [v for v in top_expl_variants_all if v not in [all_ones_top, all_twos_top]]
top_expl_variants = [all_ones_top, all_twos_top] + random.sample(
    rest_top, min(NUM_SAMPLE_VARIANTS, len(rest_top))
)

all_ones_bottom = next(
    v for v in bottom_expl_variants_all if set(v["combination"]) == {"1"}
)
all_twos_bottom = next(
    v for v in bottom_expl_variants_all if set(v["combination"]) == {"2"}
)
rest_bottom = [
    v for v in bottom_expl_variants_all if v not in [all_ones_bottom, all_twos_bottom]
]
bottom_expl_variants = [all_ones_bottom, all_twos_bottom] + random.sample(
    rest_bottom, min(NUM_SAMPLE_VARIANTS, len(rest_bottom))
)

if SAMPLE:
    top_expl_variants = top_expl_variants[:5]
    bottom_expl_variants = bottom_expl_variants[:5]

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

for variant in tqdm(top_expl_variants, desc="Top explanation prompts"):

    if variant["prompt_id"] in completed_prompt_ids:
        continue

    rows = classify.get_classifications_from_prompt(
        prompt_text=variant["prompt"],
        prompt_id=variant["prompt_id"],
        df=df_eval[["unique_text_id", "text"]].copy(),
        platform=PLATFORM,
        temperature=TEMPERATURE,
    )

    for row in rows:
        row["combination"] = variant["combination"]
        row["category"] = "top"

    response_rows.extend(rows)
    completed_prompt_ids.add(variant["prompt_id"])

    counter += 1

    if counter % SAVE_EVERY == 0:
        pd.DataFrame(response_rows).to_excel(TEMP_RESPONSE_PATH, index=False)
        print(f"Saved progress after {counter} new prompts")

for variant in tqdm(bottom_expl_variants, desc="Bottom explanation prompts"):

    if variant["prompt_id"] in completed_prompt_ids:
        continue

    rows = classify.get_classifications_from_prompt(
        prompt_text=variant["prompt"],
        prompt_id=variant["prompt_id"],
        df=df_eval[["unique_text_id", "text"]].copy(),
        platform=PLATFORM,
        temperature=TEMPERATURE,
    )

    for row in rows:
        row["combination"] = variant["combination"]
        row["category"] = "bottom"

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
    group_col=["prompt_id", "combination", "category"],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {EXPORT_RESULTS_PATH}")

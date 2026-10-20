# %%
import random

import pandas as pd

from pe4ci.library import start

# %%
# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
SEED = start.SEED

random.seed(SEED)

print(f"Creating CoT Fewshot Template for {CONCEPT} on {PLATFORM}")

EXAMPLES_PATH = start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_train.xlsx"
)

TEMPLATE_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_template.xlsx"
FILLED_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_template_w_reasoning.xlsx"

# %%
# ------------------ IDENTIFY BEST FEWSHOT SET ------------------
baseline_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")

top_row = baseline_df.loc[baseline_df["F1"].idxmax()]
top_example_id = int(top_row["prompt_id"].split("_")[1])

# %%
# ------------------ LOAD FEWSHOT EXAMPLES ------------------
example_combinations = pd.read_json(EXAMPLES_PATH)

top_examples = example_combinations.loc[
    example_combinations["sample_id"] == top_example_id, "examples"
].iloc[0]

# %%
# ------------------ EXPORT TEMPLATE EXCEL ------------------
template_df = pd.DataFrame(top_examples)[["text", "label"]].copy()
template_df["exemplar_cot1"] = ""
template_df["exemplar_cot2"] = ""

template_df.to_excel(TEMPLATE_PATH, index=False)
print(f"Saved: {TEMPLATE_PATH}")

print("\nMANUAL STEP REQUIRED:")
print("1. Fill in columns exemplar_cot1 and exemplar_cot2 with two reasoning variants per example")
print("2. Save as:")
print(FILLED_PATH)

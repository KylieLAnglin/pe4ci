# %%
import random

import pandas as pd

from crisp.library import start

# %%
# ------------------ SETUP ------------------
CONCEPT = "mm"
PLATFORM = "llama3.3"
SEED = start.SEED

random.seed(SEED)

print(f"Creating CoT Fewshot Templates for {CONCEPT} on {PLATFORM}")

# This script always runs locally, even for llama.
# Use openai- path names.
PATH_PLATFORM = "openai" if PLATFORM.startswith("llama") else PLATFORM

EXAMPLES_PATH = start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_train.xlsx"
)

TOP_TEMPLATE_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_top_template.xlsx"
BOTTOM_TEMPLATE_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_bottom_template.xlsx"

TOP_FILLED_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_top_template_w_reasoning.xlsx"
BOTTOM_FILLED_PATH = start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_bottom_template_w_reasoning.xlsx"

# %%
# ------------------ IDENTIFY BEST FEWSHOT SET ------------------
baseline_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")

top_row = baseline_df.loc[baseline_df["F1"].idxmax()]
bottom_row = baseline_df.loc[baseline_df["F1"].idxmin()]

top_example_id = int(top_row["prompt_id"].split("_")[2])
bottom_example_id = int(bottom_row["prompt_id"].split("_")[2])

# %%
# ------------------ LOAD FEWSHOT EXAMPLES ------------------
example_combinations = pd.read_json(EXAMPLES_PATH)

top_examples = example_combinations.loc[
    example_combinations["sample_id"] == top_example_id, "examples"
].iloc[0]
bottom_examples = example_combinations.loc[
    example_combinations["sample_id"] == bottom_example_id, "examples"
].iloc[0]

# %%
# ------------------ EXPORT TEMPLATE EXCELS ------------------
top_template_df = pd.DataFrame(top_examples)[["text", "label"]].copy()
top_template_df["exemplar_cot1"] = ""
top_template_df["exemplar_cot2"] = ""

bottom_template_df = pd.DataFrame(bottom_examples)[["text", "label"]].copy()
bottom_template_df["exemplar_cot1"] = ""
bottom_template_df["exemplar_cot2"] = ""

top_template_df.to_excel(TOP_TEMPLATE_PATH, index=False)
print(f"Saved: {TOP_TEMPLATE_PATH}")
bottom_template_df.to_excel(BOTTOM_TEMPLATE_PATH, index=False)
print(f"Saved: {BOTTOM_TEMPLATE_PATH}")

print("\nMANUAL STEP REQUIRED:")
print("1. Add columns: exemplar_cot1, exemplar_cot2")
print("2. Save as:")
print(TOP_FILLED_PATH)
print(BOTTOM_FILLED_PATH)

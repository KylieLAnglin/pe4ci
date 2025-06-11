# 4_zero_shot_cot/01_cot_dev.py
# %%
import os
import re
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
TEMPERATURE = 0.00001

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)
RESPONSE_PATH = (
    start.DATA_DIR + f"responses_dev/{PLATFORM}_{CONCEPT}_cot_zero_responses_dev.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_cot_zero_results_dev.xlsx"
)

# ------------------ CONSTANTS ------------------
COT_SUFFIX = (
    " First, explain your reasoning step by step. "
    "Then, state your final answer — either Yes or No — using the format: Final Answer: [Yes or No]"
)


# ------------------ PARSE FUNCTION ------------------
def parse_final_answer(response_text: str) -> int:
    match = re.search(r"final answer\s*:\s*(yes|no)", response_text, re.IGNORECASE)
    if match:
        return 1 if match.group(1).lower() == "yes" else 0
    return 0  # fallback


# ------------------ LOAD TOP + BOTTOM BASELINE PROMPTS ------------------
baseline_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
top_prompt = baseline_results.loc[baseline_results["F1"].idxmax(), "prompt"].replace(
    "Text:", ""
)
bottom_prompt = baseline_results.loc[baseline_results["F1"].idxmin(), "prompt"].replace(
    "Text:", ""
)

prompt_df = pd.DataFrame(
    [
        {"prompt_id": "top_cot", "prompt": top_prompt + COT_SUFFIX},
        {"prompt_id": "bottom_cot", "prompt": bottom_prompt + COT_SUFFIX},
    ]
)

# ------------------ LOAD TEXT DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
df = df[df.text.notna() & df.human_code.notna()]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ EVALUATE ------------------
all_rows = []
for row in tqdm(
    prompt_df.itertuples(), total=len(prompt_df), desc="Evaluating CoT Prompts"
):
    rows = classify.evaluate_prompt(
        prompt_text=row.prompt,
        prompt_id=row.prompt_id,
        df=df,
        platform=PLATFORM,
        temperature=TEMPERATURE,
        parser_fn=parse_final_answer,
    )
    all_rows.extend(rows)

# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(all_rows)
long_df.to_excel(RESPONSE_PATH, index=False)

# ------------------ EXPORT METRICS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    include_se=True,
)
# %%

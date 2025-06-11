# 3_persona/02_persona_dev.py
# %%
import os
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

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_persona_zero_results_train.xlsx"
)
EXPORT_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_dev/{PLATFORM}_{CONCEPT}_persona_zero_responses_dev.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_persona_zero_results_dev.xlsx"
)

# ------------------ LOAD DEV DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
df = df[df.text.notna() & df.human_code.notna()]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ SELECT BEST PROMPTS ------------------
results_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")

top_row = (
    results_df[results_df["category"] == "top"]
    .sort_values("F1", ascending=False)
    .iloc[0]
)
bottom_row = (
    results_df[results_df["category"] == "bottom"]
    .sort_values("F1", ascending=False)
    .iloc[0]
)

prompt_df = pd.DataFrame([top_row, bottom_row])
prompt_df["prompt_id"] = ["top_best_persona", "bottom_best_persona"]
prompt_df["prompt"] = prompt_df["prompt"].str.replace("Text:", "", regex=False)

# ------------------ RUN CLASSIFICATION ------------------
all_rows = []
for row in tqdm(
    prompt_df.itertuples(), total=len(prompt_df), desc="Evaluating Prompts"
):
    rows = classify.evaluate_prompt(
        prompt_text=row.prompt,
        prompt_id=row.prompt_id,
        df=df,
        platform=PLATFORM,
        temperature=0.0001,
    )
    all_rows.extend(rows)

# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(all_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
long_df = pd.read_excel(EXPORT_RESPONSE_PATH)

# ------------------ EXPORT METRICS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id"],
    prompt_col="prompt",
    include_se=True,
)
# %%

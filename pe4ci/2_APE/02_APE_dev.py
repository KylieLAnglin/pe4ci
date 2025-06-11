# 2_APE/02_APE_dev.py
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

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
PROMPT_PATH_TOP = start.RESULTS_DIR + f"{PLATFORM}_{CONCEPT}_ape_top_results_train.xlsx"
PROMPT_PATH_BOTTOM = (
    start.RESULTS_DIR + f"{PLATFORM}_{CONCEPT}_ape_bottom_results_train.xlsx"
)


EXPORT_RESPONSE_PATH = (
    start.DATA_DIR + f"responses_dev/{PLATFORM}_{CONCEPT}_ape_zero_responses_dev.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_ape_zero_results_dev.xlsx"
)


# ------------------ LOAD BEST PROMPTS ------------------
def load_best_prompts(path_top, path_bottom):
    df_top = pd.read_excel(path_top)
    df_top["category"] = "top"

    df_bottom = pd.read_excel(path_bottom)
    df_bottom["category"] = "bottom"

    df = pd.concat([df_top, df_bottom], ignore_index=True)
    df["prompt_id"] = (
        df.generation.astype(int).astype(str)
        + "_"
        + df.variant_id.astype(int).astype(str)
        + "_"
        + df.category
    )
    df["prompt"] = df["prompt"].str.replace("Text:", "", regex=False)
    df = df.set_index("prompt_id")

    top_id = df[df.category == "top"]["f1_score"].idxmax()
    bottom_id = df[df.category == "bottom"]["f1_score"].idxmax()

    return df.loc[[top_id, bottom_id]]


prompt_df = load_best_prompts(PROMPT_PATH_TOP, PROMPT_PATH_BOTTOM)

# ------------------ LOAD TEXT DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ EVALUATE PROMPTS ------------------
all_rows = []
for prompt_id, prompt_text in prompt_df["prompt"].items():
    rows = classify.evaluate_prompt(prompt_text, prompt_id, df, PLATFORM, 0.0001)
    all_rows.extend(rows)

# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(all_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
long_df = pd.read_excel(EXPORT_RESPONSE_PATH)

# ------------------ EXPORT METRICS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)
# %%

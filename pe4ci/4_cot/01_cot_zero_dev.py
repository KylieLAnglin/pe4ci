# %%
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
TEMPERATURE = 0.00001

COT_SUFFIX = (
    " First, state what you notice in the text as it relates to the definition. "
    "Then, report your final answer as either ```yes``` or ```no```."
)

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in dev set")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)
RESPONSE_PATH = (
    start.DATA_DIR + f"responses_dev/{PLATFORM}_{CONCEPT}_cot_zero_responses_dev.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_cot_zero_results_dev.xlsx"
)

# %%
# ------------------ LOAD PROMPTS ------------------
baseline_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")

top_prompt = baseline_results.loc[baseline_results["F1"].idxmax(), "prompt"]
bottom_prompt = baseline_results.loc[baseline_results["F1"].idxmin(), "prompt"]

prompt_df = pd.DataFrame(
    [
        {"prompt_id": "top_cot", "prompt": top_prompt + COT_SUFFIX},
        {"prompt_id": "bottom_cot", "prompt": bottom_prompt + COT_SUFFIX},
    ]
)

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "dev"].copy()

if SAMPLE:
    df = df.sample(5, random_state=SEED).copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

# %%
# ------------------ EVALUATE ------------------
all_rows = []

for row in tqdm(
    prompt_df.itertuples(index=False),
    total=len(prompt_df),
    desc="Evaluating CoT Prompts",
):
    rows = classify.get_classifications_from_prompt(
        prompt_text=row.prompt,
        prompt_id=row.prompt_id,
        df=df[["unique_text_id", "text"]].copy(),
        platform=PLATFORM,
        temperature=TEMPERATURE,
    )

    all_rows.extend(rows)

# %%
# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(all_rows)
long_df.to_excel(RESPONSE_PATH, index=False)
print(f"Saved: {RESPONSE_PATH}")

# %%
# ------------------ EXPORT METRICS ------------------
long_df = pd.read_excel(RESPONSE_PATH)
scored_df = long_df.merge(df_gold, on="unique_text_id", how="inner")

classify.export_results_to_excel(
    df=scored_df,
    output_path=RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {RESULTS_PATH}")

# %%
import pandas as pd
from tqdm import tqdm

from crisp.library import start, classify

# %%
# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED
TEMPERATURE = 0.0001

print(f"Running baseline dev on {CONCEPT} with {PLATFORM} {MODEL} in dev set")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)
RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_dev/{PLATFORM}_{CONCEPT}_baseline_zero_responses_dev.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
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
# ------------------ LOAD PROMPTS ------------------
train_results = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")

top_id = train_results.loc[train_results["F1"].idxmax(), "prompt_id"]
bottom_id = train_results.loc[train_results["F1"].idxmin(), "prompt_id"]

prompt_df = train_results[train_results["prompt_id"].isin([top_id, bottom_id])].copy()

# %%
# ------------------ GENERATE RESPONSES ------------------
response_rows = []

for row in tqdm(
    prompt_df.itertuples(index=False),
    total=len(prompt_df),
    desc="Evaluating Prompts",
):
    responses = classify.get_classifications_from_prompt(
        prompt_text=row.prompt,
        prompt_id=row.prompt_id,
        df=df[["unique_text_id", "text"]].copy(),
        platform=PLATFORM,
        temperature=TEMPERATURE,
    )

    for response_row in responses:
        response_row["context_part_num"] = row.context_part_num
        response_row["task_part_num"] = row.task_part_num
        response_row["defn_part_num"] = row.defn_part_num
        response_row["num_guidance_items"] = row.num_guidance_items
        response_row["guidance_part_nums"] = row.guidance_part_nums

    response_rows.extend(responses)

# %%
# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(RESPONSE_PATH, index=False)
print(f"Saved: {RESPONSE_PATH}")

# %%
# ------------------ EXPORT METRICS ------------------
long_df = pd.read_excel(RESPONSE_PATH)
scored_df = long_df.merge(df_gold, on="unique_text_id", how="inner")

classify.export_results_to_excel(
    df=scored_df,
    output_path=RESULTS_PATH,
    group_col=[
        "prompt_id",
        "context_part_num",
        "task_part_num",
        "defn_part_num",
        "num_guidance_items",
        "guidance_part_nums",
    ],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {RESULTS_PATH}")

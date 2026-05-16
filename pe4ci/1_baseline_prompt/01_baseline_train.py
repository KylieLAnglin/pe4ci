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
TEMPERATURE = 0.0001
SAVE_EVERY = 1

RUN_TAG = "_sample" if SAMPLE else ""
print(f"Running baseline train on {CONCEPT} with {PLATFORM} {MODEL} in train set")

PROMPT_PATH = start.DATA_DIR + f"prompts/{CONCEPT}_baseline_variants.xlsx"
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_baseline_zero_responses_train.xlsx"
)
TEMP_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_baseline_zero_responses_train_PROGRESS{RUN_TAG}.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)
print(DATA_PATH)

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "train"].copy()

if SAMPLE:
    df = df.sample(5, random_state=SEED).copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

# %%
# ------------------ LOAD PROMPTS ------------------
prompt_df = pd.read_excel(PROMPT_PATH, sheet_name="baseline")
prompt_df = prompt_df.rename(columns={"baseline_prompt_id": "prompt_id"})

if SAMPLE:
    prompt_df = prompt_df.head(n=2).copy()

# %%
# ------------------ RESUME FROM PROGRESS FILE ------------------
response_rows = []

try:
    existing_df = pd.read_excel(TEMP_RESPONSE_PATH)
    completed_prompt_ids = set(existing_df["prompt_id"].unique())
    response_rows = existing_df.to_dict("records")
    print(f"Found existing progress for {len(completed_prompt_ids)} prompts")
except FileNotFoundError:
    completed_prompt_ids = set()
    print("No existing progress file found")

prompts_to_run = prompt_df[~prompt_df["prompt_id"].isin(completed_prompt_ids)].copy()

print(f"Prompts already completed: {len(completed_prompt_ids)}")
print(f"Prompts remaining: {len(prompts_to_run)}")

# %%
# ------------------ COLLECT RESPONSES ------------------
for i, row in enumerate(
    tqdm(
        prompts_to_run.itertuples(index=False),
        total=len(prompts_to_run),
        desc="Evaluating Prompts",
        position=0,
    ),
    start=1,
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

    if i % SAVE_EVERY == 0:
        pd.DataFrame(response_rows).to_excel(TEMP_RESPONSE_PATH, index=False)
        print(f"Saved progress after {i} new prompts")

# %%
# ------------------ FINAL SAVE ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(TEMP_RESPONSE_PATH, index=False)
print(f"Saved: {TEMP_RESPONSE_PATH}")
long_df.to_excel(RESPONSE_PATH, index=False)
print(f"Saved: {RESPONSE_PATH}")

# %%
# ------------------ SCORE AND EXPORT RESULTS ------------------
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
    include_se=False,
)
print(f"Saved: {RESULTS_PATH}")

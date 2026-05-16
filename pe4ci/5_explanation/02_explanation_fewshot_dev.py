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

print(f"Running explanation few-shot evaluation on {CONCEPT} with {MODEL} in dev set")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"
TRAIN_RESULTS_PATH = start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_explanation_few_results_train.xlsx"
DEV_RESPONSE_PATH = start.DATA_DIR + f"responses_dev/{PLATFORM}_{CONCEPT}_explanation_few_responses_dev.xlsx"
DEV_RESULTS_PATH = start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_explanation_few_results_dev.xlsx"

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "dev"].copy()

if SAMPLE:
    df = df.sample(5, random_state=SEED).copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

# %%
# ------------------ LOAD BEST PROMPTS FROM TRAIN ------------------
train_df = pd.read_excel(TRAIN_RESULTS_PATH, sheet_name="results")

best_prompts = (
    train_df.sort_values(["category", "F1"], ascending=[True, False])
    .drop_duplicates(subset=["category"], keep="first")
    .reset_index(drop=True)
)

# %%
# ------------------ EVALUATE ON DEV SET ------------------
response_rows = []

for row in tqdm(
    best_prompts.itertuples(index=False),
    total=len(best_prompts),
    desc="Evaluating Prompts",
):
    eval_rows = classify.get_classifications_from_prompt(
        prompt_text=row.prompt,
        prompt_id=row.prompt_id,
        df=df[["unique_text_id", "text"]].copy(),
        platform=PLATFORM,
        temperature=TEMPERATURE,
    )

    for r in eval_rows:
        r["category"] = row.category
        r["combination"] = row.combination

    response_rows.extend(eval_rows)

# %%
# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(DEV_RESPONSE_PATH, index=False)
print(f"Saved: {DEV_RESPONSE_PATH}")

# %%
# ------------------ SCORE AND EXPORT RESULTS ------------------
long_df = pd.read_excel(DEV_RESPONSE_PATH)
scored_df = long_df.merge(df_gold, on="unique_text_id", how="inner")

classify.export_results_to_excel(
    df=scored_df,
    output_path=DEV_RESULTS_PATH,
    group_col=["prompt_id", "category", "combination"],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {DEV_RESULTS_PATH}")

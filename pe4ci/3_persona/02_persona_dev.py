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

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in dev set")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

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

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "dev"].copy()

if SAMPLE:
    df = df.sample(5, random_state=SEED).copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

# %%
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

# %%
# ------------------ RUN CLASSIFICATION ------------------
all_rows = []

for row in tqdm(
    prompt_df.itertuples(index=False),
    total=len(prompt_df),
    desc="Evaluating Prompts",
):
    rows = classify.get_classifications_from_prompt(
        prompt_text=row.prompt,
        prompt_id=row.prompt_id,
        df=df[["unique_text_id", "text"]].copy(),
        platform=PLATFORM,
        temperature=TEMPERATURE,
    )

    for r in rows:
        r["category"] = row.category
        r["persona"] = row.persona

    all_rows.extend(rows)

# %%
# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(all_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
print(f"Saved: {EXPORT_RESPONSE_PATH}")

# %%
# ------------------ EXPORT METRICS ------------------
long_df = pd.read_excel(EXPORT_RESPONSE_PATH)
scored_df = long_df.merge(df_gold, on="unique_text_id", how="inner")

classify.export_results_to_excel(
    df=scored_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id", "category", "persona"],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {EXPORT_RESULTS_PATH}")

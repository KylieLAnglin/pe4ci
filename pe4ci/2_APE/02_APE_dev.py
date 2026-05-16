# %%
import pandas as pd

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

# %%
# ------------------ LOAD BEST PROMPTS ------------------
df_top = pd.read_excel(PROMPT_PATH_TOP)
df_top["category"] = "top"

df_bottom = pd.read_excel(PROMPT_PATH_BOTTOM)
df_bottom["category"] = "bottom"

prompt_df = pd.concat([df_top, df_bottom], ignore_index=True)
prompt_df["prompt_id"] = (
    prompt_df["generation"].astype(int).astype(str)
    + "_"
    + prompt_df["variant_id"].astype(int).astype(str)
    + "_"
    + prompt_df["category"]
)

top_prompt_df = (
    prompt_df[prompt_df["category"] == "top"]
    .sort_values("f1_score", ascending=False)
    .head(1)
)
bottom_prompt_df = (
    prompt_df[prompt_df["category"] == "bottom"]
    .sort_values("f1_score", ascending=False)
    .head(1)
)

prompt_df = pd.concat([top_prompt_df, bottom_prompt_df], ignore_index=True)

# %%
# ------------------ LOAD TEXT DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "dev"].copy()

if SAMPLE:
    df = df.sample(5, random_state=SEED).copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

# %%
# ------------------ EVALUATE PROMPTS ------------------
response_rows = []

for row in prompt_df.itertuples(index=False):
    rows = classify.get_classifications_from_prompt(
        prompt_text=row.prompt,
        prompt_id=row.prompt_id,
        df=df[["unique_text_id", "text"]].copy(),
        platform=PLATFORM,
        temperature=TEMPERATURE,
    )

    for response_row in rows:
        response_row["category"] = row.category
        response_row["generation"] = row.generation
        response_row["variant_id"] = row.variant_id

    response_rows.extend(rows)

# %%
# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
print(f"Saved: {EXPORT_RESPONSE_PATH}")

# %%
# ------------------ EXPORT METRICS ------------------
long_df = pd.read_excel(EXPORT_RESPONSE_PATH)
scored_df = long_df.merge(df_gold, on="unique_text_id", how="inner")

classify.export_results_to_excel(
    df=scored_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id", "category", "generation", "variant_id"],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {EXPORT_RESULTS_PATH}")

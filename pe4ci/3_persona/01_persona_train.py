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

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in train set")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

PROMPT_PATH = start.DATA_DIR + f"prompts/{CONCEPT}_baseline_variants.xlsx"

TRAIN_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)
EXPORT_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_persona_zero_responses_train.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_persona_zero_results_train.xlsx"
)

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "train"].copy()

if SAMPLE:
    df = df.sample(5, random_state=SEED).copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

# %%
# ------------------ LOAD BASELINE PROMPTS ------------------
train_results = pd.read_excel(TRAIN_RESULTS_PATH, sheet_name="results")

top_id = train_results.loc[train_results["F1"].idxmax(), "prompt_id"]
bottom_id = train_results.loc[train_results["F1"].idxmin(), "prompt_id"]

prompt_df = pd.read_excel(PROMPT_PATH, sheet_name="baseline")
prompt_df = prompt_df.rename(columns={"baseline_prompt_id": "prompt_id"})

top_prompt = prompt_df.loc[prompt_df.prompt_id == top_id, "prompt"].values[0]
bottom_prompt = prompt_df.loc[prompt_df.prompt_id == bottom_id, "prompt"].values[0]

# %%
# ------------------ LOAD PERSONAS ------------------
personas_df = pd.read_excel(PROMPT_PATH, sheet_name="persona")
PERSONAS = personas_df["persona"].tolist()

# %%
# ------------------ CREATE PERSONA COMBOS ------------------
combos = []
combo_id = 1

for persona in PERSONAS:
    for category, base_prompt in [("top", top_prompt), ("bottom", bottom_prompt)]:
        modified_prompt = persona.strip() + "\n\n" + base_prompt

        combos.append(
            {
                "prompt_id": combo_id,
                "prompt": modified_prompt,
                "category": category,
                "persona": persona,
            }
        )
        combo_id += 1

combo_df = pd.DataFrame(combos)

# %%
# ------------------ GENERATE RESPONSES ------------------
response_rows = []

for row in tqdm(
    combo_df.itertuples(index=False),
    total=len(combo_df),
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
    group_col=["prompt_id", "category", "persona"],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {EXPORT_RESULTS_PATH}")

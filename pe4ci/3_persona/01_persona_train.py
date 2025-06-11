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

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in train set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
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


# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD BASELINE PROMPTS ------------------
train_results = pd.read_excel(TRAIN_RESULTS_PATH, sheet_name="results")
top_id = train_results.loc[train_results["F1"].idxmax(), "prompt_id"]
bottom_id = train_results.loc[train_results["F1"].idxmin(), "prompt_id"]

prompt_df = pd.read_excel(PROMPT_PATH, sheet_name="baseline")
prompt_df["prompt_id"] = prompt_df.index
prompt_df["prompt"] = prompt_df["prompt"].str.replace("Text: ", "", regex=False)

top_prompt = prompt_df.loc[prompt_df.prompt_id == top_id, "prompt"].values[0]
bottom_prompt = prompt_df.loc[prompt_df.prompt_id == bottom_id, "prompt"].values[0]

personas_df = pd.read_excel(PROMPT_PATH, sheet_name="persona")
PERSONAS = personas_df["persona"].tolist()
# ------------------ CREATE PERSONA COMBOS ------------------
combos = []
combo_id = 1
for persona in PERSONAS:
    for category, base_prompt in [("top", top_prompt), ("bottom", bottom_prompt)]:
        modified_prompt = persona + base_prompt
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

# ------------------ GENERATE RESPONSES ------------------
response_rows = []
for row in tqdm(combo_df.itertuples(), total=len(combo_df), desc="Evaluating Prompts"):
    rows = classify.evaluate_prompt(
        prompt_text=row.prompt,
        prompt_id=row.prompt_id,
        df=df,
        platform=PLATFORM,
        temperature=0.0001,
    )

    for r in rows:
        r["category"] = row.category
        r["persona"] = row.persona

    response_rows.extend(rows)

# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
long_df = pd.read_excel(EXPORT_RESPONSE_PATH)

# ------------------ EXPORT METRICS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id", "category"],
    prompt_col="prompt",
    include_se=True,
)
# %%

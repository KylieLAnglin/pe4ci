# %%
import json
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

print(f"Running persona few-shot evaluation on {CONCEPT} with {MODEL} in dev set")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
PERSONA_ZERO_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_persona_zero_results_dev.xlsx"
)
FEWSHOT_BASELINE_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_dev.xlsx"
)
RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_dev/{PLATFORM}_{CONCEPT}_persona_few_responses_dev.xlsx"
)
RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_persona_few_results_dev.xlsx"
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
# ------------------ LOAD FEWSHOT EXAMPLES ------------------
with open(EXAMPLES_PATH, "r") as f:
    fewshot_samples = json.load(f)

fewshot_samples_df = pd.DataFrame(fewshot_samples)

fewshot_results = pd.read_excel(FEWSHOT_BASELINE_RESULTS_PATH, sheet_name="results")

top_example_id = int(
    fewshot_results.loc[fewshot_results["F1"].idxmax(), "prompt_id"].split("_")[2]
)
bottom_example_id = int(
    fewshot_results.loc[fewshot_results["F1"].idxmin(), "prompt_id"].split("_")[2]
)

top_examples = fewshot_samples_df.loc[
    fewshot_samples_df["sample_id"] == top_example_id, "examples"
].iloc[0]
bottom_examples = fewshot_samples_df.loc[
    fewshot_samples_df["sample_id"] == bottom_example_id, "examples"
].iloc[0]

# %%
# ------------------ LOAD PERSONA PROMPTS ------------------
baseline_df = pd.read_excel(PERSONA_ZERO_RESULTS_PATH, sheet_name="results")

top_prompt = baseline_df.loc[baseline_df["F1"].idxmax(), "prompt"]
bottom_prompt = baseline_df.loc[baseline_df["F1"].idxmin(), "prompt"]

# %%
# ------------------ BUILD FEWSHOT PROMPTS ------------------
top_block = "\n\n".join(
    [
        f'Text: "{ex["text"]}"\nAnswer: ```{"yes" if ex["label"] == 1 else "no"}```'
        for ex in top_examples
    ]
)
bottom_block = "\n\n".join(
    [
        f'Text: "{ex["text"]}"\nAnswer: ```{"yes" if ex["label"] == 1 else "no"}```'
        for ex in bottom_examples
    ]
)

if "Here is the text:" not in top_prompt:
    raise ValueError('Prompt must contain "Here is the text:"')
top_full_prompt = top_prompt.replace(
    "Here is the text:",
    f"Here are some examples:\n{top_block}\n\nHere is the text:",
)

if "Here is the text:" not in bottom_prompt:
    raise ValueError('Prompt must contain "Here is the text:"')
bottom_full_prompt = bottom_prompt.replace(
    "Here is the text:",
    f"Here are some examples:\n{bottom_block}\n\nHere is the text:",
)

prompt_df = pd.DataFrame(
    {
        "prompt_id": [
            f"{PLATFORM}_{CONCEPT}_persona_top_fewshot_{top_example_id}",
            f"{PLATFORM}_{CONCEPT}_persona_bottom_fewshot_{bottom_example_id}",
        ],
        "prompt": [top_full_prompt, bottom_full_prompt],
        "category": ["top", "bottom"],
    }
)

# %%
# ------------------ GENERATE RESPONSES ------------------
response_rows = []

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

    response_rows.extend(rows)

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
    group_col=["prompt_id", "category"],
    prompt_col="prompt",
    y_true_col="human_code",
    y_pred_col="classification",
    sheet_name="results",
    include_se=True,
)
print(f"Saved: {RESULTS_PATH}")

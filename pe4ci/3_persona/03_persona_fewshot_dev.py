# %%
import os
import json
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

print(f"Running few-shot evaluation on {CONCEPT} with {MODEL} in dev set")

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"

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

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "dev"]
df = df[df.text.notna() & df.human_code.notna()]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

# ------------------ LOAD FEWSHOT EXAMPLES ------------------
with open(EXAMPLES_PATH, "r") as f:
    fewshot_samples = json.load(f)

fewshot_samples_df = pd.DataFrame(fewshot_samples)

fewshot_results = pd.read_excel(FEWSHOT_BASELINE_RESULTS_PATH, sheet_name="results")

top_example_id = fewshot_results.loc[fewshot_results["F1"].idxmax(), "prompt_id"]
top_example_id = int(top_example_id.split("_")[2])

bottom_example_id = fewshot_results.loc[fewshot_results["F1"].idxmin(), "prompt_id"]
bottom_example_id = int(bottom_example_id.split("_")[2])

top_examples = fewshot_samples_df[
    fewshot_samples_df["sample_id"] == top_example_id
].iloc[0]["examples"]
bottom_examples = fewshot_samples_df[
    fewshot_samples_df["sample_id"] == bottom_example_id
].iloc[0]["examples"]

# ------------------ LOAD TOP & BOTTOM BASELINE PROMPTS ------------------
baseline_df = pd.read_excel(PERSONA_ZERO_RESULTS_PATH, sheet_name="results")
top_prompt = baseline_df.loc[baseline_df["F1"].idxmax(), "prompt"]
bottom_prompt = baseline_df.loc[baseline_df["F1"].idxmin(), "prompt"]

top_prompt = top_prompt.replace("Text:", "").strip()
bottom_prompt = bottom_prompt.replace("Text:", "").strip()

# ------------------ APPEND FEWSHOT EXAMPLES TO PROMPTS ------------------
label_format_fn = lambda label: "Yes" if label == 1 else "No"
top_example_block = "\n".join(
    [
        f'Text: "{ex["text"]}"\nAnswer: {label_format_fn(ex["label"])}'
        for ex in top_examples
    ]
)
bottom_example_block = "\n".join(
    [
        f'Text: "{ex["text"]}"\nAnswer: {label_format_fn(ex["label"])}'
        for ex in bottom_examples
    ]
)

top_full_prompt = f"{top_prompt}\nHere are some examples:\n{top_example_block}\n\n"
bottom_full_prompt = (
    f"{bottom_prompt}\nHere are some examples:\n{bottom_example_block}\n\n"
)

prompt_df = pd.DataFrame(
    {
        "prompt_id": [
            f"{PLATFORM}_{CONCEPT}_persona_top_fewshot_{top_example_id}",
            f"{PLATFORM}_{CONCEPT}_persona_bottom_fewshot_{bottom_example_id}",
        ],
        "prompt": [top_full_prompt, bottom_full_prompt],
    }
)


# ------------------ GENERATE RESPONSES ------------------
response_rows = []
for row in tqdm(
    prompt_df.itertuples(), total=len(prompt_df), desc="Evaluating Prompts"
):
    prompt_text = row.prompt
    prompt_id = row.prompt_id
    responses = classify.evaluate_prompt(
        prompt_text=prompt_text,
        prompt_id=prompt_id,
        df=df,
        platform=PLATFORM,
        temperature=0.0001,
    )
    response_rows.extend(responses)

# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(response_rows)
long_df.to_excel(RESPONSE_PATH, index=False)
long_df = pd.read_excel(RESPONSE_PATH)

# ------------------ EXPORT METRICS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)
# %%

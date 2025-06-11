# 1_baseline_prompt/01_explanation_fewshot_train.py
# %%

from itertools import product
import pandas as pd
from tqdm import tqdm
import random
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

# ------------------ PATHS ------------------
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
EXAMPLES_PATH = (
    start.DATA_DIR + f"fewshot_examples/{CONCEPT}_fewshot_train_samples.json"
)
IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_few_results_train.xlsx"
)
IMPORT_BASELINE_PROMPTS = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_train.xlsx"
)

EXCEL_TOP_EXPL_PATH = (
    start.DATA_DIR + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_top_examples.xlsx"
)
EXCEL_BOTTOM_EXPL_PATH = (
    start.DATA_DIR
    + f"fewshot_examples/{PLATFORM}_{CONCEPT}_cot_few_bottom_examples.xlsx"
)

EXPORT_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_explanation_few_responses_train.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_explanation_few_results_train.xlsx"
)

print(
    f"Running Explanation Fewshot Training for {CONCEPT} on {PLATFORM} with model {MODEL} and sample = {SAMPLE}."
)

# ------------------ LOAD TRAINING DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[(df.split_group == "train") & (df.text.notna()) & (df.human_code.notna())]
df = df[df.train_use == "eval"]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

print("test classification")
classify.evaluate_prompt(
    prompt_text="Classify the following text as positive or negative. Text:",
    prompt_id="test",
    df=df.head(1),
    platform=start.PLATFORM,
    temperature=0.0001,
)
print("test classification done")
# ------------------ IDENTIFY BEST PROMPT IDs ------------------
baseline_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
top_prompt_id = int(
    baseline_df.loc[baseline_df["F1"].idxmax(), "prompt_id"].split("_")[2]
)
bottom_prompt_id = int(
    baseline_df.loc[baseline_df["F1"].idxmin(), "prompt_id"].split("_")[2]
)

prompt_df = pd.read_excel(IMPORT_BASELINE_PROMPTS, sheet_name="results")
top_prompt = (
    prompt_df.loc[prompt_df["F1"].idxmax(), "prompt"].replace("Text:", "").strip()
)
bottom_prompt = (
    prompt_df.loc[prompt_df["F1"].idxmin(), "prompt"].replace("Text:", "").strip()
)

# ------------------ LOAD FEWSHOT EXAMPLES ------------------
example_combinations = pd.read_json(EXAMPLES_PATH)
example_combinations = example_combinations[
    example_combinations["sample_id"].isin([top_prompt_id, bottom_prompt_id])
]

top_examples = example_combinations.loc[
    example_combinations["sample_id"] == top_prompt_id, "examples"
].values[0]
bottom_examples = example_combinations.loc[
    example_combinations["sample_id"] == bottom_prompt_id, "examples"
].values[0]

top_examples_expl_df = pd.read_excel(EXCEL_TOP_EXPL_PATH)
bottom_examples_expl_df = pd.read_excel(EXCEL_BOTTOM_EXPL_PATH)


# ------------------ GENERATE EXPLANATION PROMPT VARIANTS ------------------
def generate_explanation_prompt_variants(example_df, base_prompt, id_prefix="explain"):
    n = len(example_df)
    variants = []
    for combo in product([1, 2], repeat=n):
        parts = []
        for i, row in enumerate(example_df.itertuples()):
            explanation = row.exemplar_cot1 if combo[i] == 1 else row.exemplar_cot2
            answer = "Yes" if row.label == 1 else "No"
            block = f'Text: "{row.text}"\nAnswer: {answer}\n Explanation: {explanation}'
            parts.append(block)

        example_block = "\n\n".join(parts)
        full_prompt = f"{base_prompt}\nHere are some examples:\n{example_block}\n\n"
        prompt_id = f"{id_prefix}_{''.join(str(c) for c in combo)}"

        variants.append(
            {
                "prompt_id": prompt_id,
                "prompt_block": full_prompt,
                "combination": combo,
            }
        )

    return variants


# ------------------ SELECT FIXED + RANDOM COMBINATIONS ------------------
def select_with_anchors(variants, seed, total=5):
    random.seed(seed)
    all_ones = next(v for v in variants if set(v["combination"]) == {1})
    all_twos = next(v for v in variants if set(v["combination"]) == {2})
    rest = [v for v in variants if v not in [all_ones, all_twos]]
    sampled = random.sample(rest, min(total - 2, len(rest)))
    return [all_ones, all_twos] + sampled


top_expl_variants_all = generate_explanation_prompt_variants(
    top_examples_expl_df, base_prompt=top_prompt, id_prefix="top_explain"
)
bottom_expl_variants_all = generate_explanation_prompt_variants(
    bottom_examples_expl_df, base_prompt=bottom_prompt, id_prefix="bot_explain"
)

top_expl_variants = select_with_anchors(top_expl_variants_all, SEED)
bottom_expl_variants = select_with_anchors(bottom_expl_variants_all, SEED)

if SAMPLE:
    top_expl_variants = top_expl_variants[:5]
    bottom_expl_variants = bottom_expl_variants[:5]


# ------------------ EVALUATE ------------------
def evaluate_expl_variants(
    prompt_variants, df_eval, platform, temperature=0.0001, verbose=True
):
    response_rows = []
    iterator = (
        tqdm(prompt_variants, desc="Evaluating Explanation Prompts")
        if verbose
        else prompt_variants
    )

    for variant in iterator:
        prompt_id = variant["prompt_id"]
        prompt_block = variant["prompt_block"]

        eval_rows = classify.evaluate_prompt(
            prompt_text=prompt_block,
            prompt_id=prompt_id,
            df=df_eval,
            platform=platform,
            temperature=temperature,
        )

        for row in eval_rows:
            row["prompt_id"] = prompt_id
            row["combination"] = "".join(str(c) for c in variant.get("combination", []))
            row["prompt"] = prompt_block

        response_rows.extend(eval_rows)

    return response_rows


top_response_rows = evaluate_expl_variants(top_expl_variants, df, PLATFORM)
bottom_response_rows = evaluate_expl_variants(bottom_expl_variants, df, PLATFORM)

# Add category
for row in top_response_rows:
    row["category"] = "top"
for row in bottom_response_rows:
    row["category"] = "bottom"

# ------------------ EXPORT ------------------
all_rows = top_response_rows + bottom_response_rows
long_df = pd.DataFrame(all_rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
long_df = pd.read_excel(EXPORT_RESPONSE_PATH)

classify.export_results_to_excel(
    df=long_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col=["prompt_id", "combination", "category"],
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
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
RUN_TAG = "_sample" if SAMPLE else ""

NUM_VARIANTS = 5
NUM_GENERATIONS = 1 if SAMPLE else 5

META_INSTRUCTIONS1 = (
    "Generate a variation of the following prompt while keeping the output format. "
    "You can add important information or remove unnecessary information. "
    "Instruction:\n"
)
META_INSTRUCTIONS2 = "\nOutput only the new instruction."

PROMPT_ADDENDUM = """Here is the text:
{TEXT}
Remember, report your final classification as either ```yes``` or ```no``` surrounded by triple backticks."""

ALTERNATIVE_ADDENDUM = """Here is the text:
{TEXT}
Respond only ```yes``` or ```no``` surrounded by triple backticks."""

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in train set")

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"

IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)

TRACKING_PATHS = {
    "top": {
        "csv": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_top_results_train.xlsx",
        "progress": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_top_results_train_PROGRESS{RUN_TAG}.xlsx",
        "fig": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_evolution_top_train.png",
    },
    "bottom": {
        "csv": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_bottom_results_train.xlsx",
        "progress": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_bottom_results_train_PROGRESS{RUN_TAG}.xlsx",
        "fig": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_evolution_bottom_train.png",
    },
}

EXPORT_RESPONSE_PATH = (
    start.DATA_DIR + f"responses_train/{PLATFORM}_{CONCEPT}_ape_zero_responses_train.xlsx"
)
TEMP_RESPONSE_PATH = (
    start.DATA_DIR
    + f"responses_train/{PLATFORM}_{CONCEPT}_ape_zero_responses_train_PROGRESS{RUN_TAG}.xlsx"
)
EXPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_ape_zero_results_train.xlsx"
)

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "train"].copy()

if SAMPLE:
    df = df.sample(5, random_state=SEED).copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "human_code"]].copy()

prompt_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")

# %%
# ------------------ STRIP PROMPT ADDENDUM ------------------
original_prompts = prompt_df["prompt"].copy()
prompt_df["prompt"] = prompt_df["prompt"].str.replace(
    PROMPT_ADDENDUM, "", regex=False
)

if (original_prompts == prompt_df["prompt"]).all():
    raise ValueError(
        "PROMPT_ADDENDUM removal failed - no replacements were made. "
        "Check PROMPT_ADDENDUM format."
    )

# %%
# ------------------ APE LOOP ------------------
prompts = [
    ("top", prompt_df["F1"].idxmax()),
    ("bottom", prompt_df["F1"].idxmin()),
]

if start.PLATFORM.startswith("llama"):
    addendum = ALTERNATIVE_ADDENDUM
else:
    addendum = PROMPT_ADDENDUM

# ------------------ RESUME RESPONSES ------------------
try:
    existing_response_df = pd.read_excel(TEMP_RESPONSE_PATH)
    all_response_rows = existing_response_df.to_dict("records")
    print(f"Found existing response progress: {len(existing_response_df)} rows")
except FileNotFoundError:
    all_response_rows = []
    print("No existing response progress file found")

# ------------------ COLLECT ALL RESPONSES ------------------
for cat, index in prompts:
    print(f"\n======== Evaluating {cat.upper()} Prompt Seed ========")

    output_tracking_file = TRACKING_PATHS[cat]["csv"]
    tracking_progress_file = TRACKING_PATHS[cat]["progress"]

    try:
        tracking_df = pd.read_excel(tracking_progress_file)
        tracking_records = tracking_df.to_dict("records")
        print(f"Found existing {cat} tracking progress: {len(tracking_df)} rows")
    except FileNotFoundError:
        tracking_df = pd.DataFrame()
        tracking_records = []
        print(f"No existing {cat} tracking progress file found")

    current_prompt = prompt_df.loc[index, "prompt"]

    for generation in range(NUM_GENERATIONS):
        generation_num = generation + 1
        print(f"\n=== Generation {generation_num} ===")

        if not tracking_df.empty:
            generation_tracking = tracking_df[
                tracking_df["generation"] == generation_num
            ].copy()
        else:
            generation_tracking = pd.DataFrame()

        if len(generation_tracking) == NUM_VARIANTS:
            best_row = generation_tracking.loc[
                generation_tracking["f1_score"].idxmax()
            ]
            current_prompt = best_row["prompt"].replace("\n" + addendum, "")
            print(f"Generation {generation_num} already completed; skipping")
            continue

        variants = classify.generate_prompt_variants(
            model_provider=PLATFORM,
            base_prompt=current_prompt,
            metaprompt1=META_INSTRUCTIONS1,
            metaprompt2=META_INSTRUCTIONS2,
            num_variants=NUM_VARIANTS,
        )

        variant_scores = []

        for i, variant in enumerate(variants):
            print(f"Evaluating variant {i + 1}...")
            prompt_id = f"{cat}_gen{generation_num}_var{i + 1}"

            rows = classify.get_classifications_from_prompt(
                prompt_text=variant + "\n" + addendum,
                prompt_id=prompt_id,
                df=df[["unique_text_id", "text"]].copy(),
                platform=PLATFORM,
                temperature=TEMPERATURE,
            )

            for response_row in rows:
                response_row["category"] = cat
                response_row["generation"] = generation_num
                response_row["variant_id"] = i + 1

            all_response_rows.extend(rows)

            rows_df = pd.DataFrame(rows)
            scored_df = rows_df.merge(df_gold, on="unique_text_id", how="inner")

            mask = scored_df["classification"].notna() & scored_df["human_code"].notna()
            y_true = scored_df.loc[mask, "human_code"]
            y_pred = scored_df.loc[mask, "classification"]

            if len(y_true) == 0:
                f1 = float("nan")
            else:
                f1 = f1_score(y_true, y_pred, zero_division=0)

            print(f"Variant F1: {f1:.4f}")
            variant_scores.append((variant, f1))

            tracking_records.append(
                {
                    "generation": generation_num,
                    "variant_id": i + 1,
                    "prompt_id": prompt_id,
                    "prompt": variant + "\n" + addendum,
                    "f1_score": f1,
                }
            )

        best_variant, best_f1 = max(variant_scores, key=lambda x: x[1])
        print(f"Best variant selected with F1 {best_f1:.4f}")
        current_prompt = best_variant

        tracking_df = pd.DataFrame(tracking_records)

        pd.DataFrame(all_response_rows).to_excel(TEMP_RESPONSE_PATH, index=False)
        tracking_df.to_excel(tracking_progress_file, index=False)
        print(f"Saved progress after {cat} generation {generation_num}")

    tracking_df = pd.DataFrame(tracking_records)
    tracking_df.to_excel(output_tracking_file, index=False)

# %%
# ------------------ SAVE ALL RESPONSES ------------------
long_df = pd.DataFrame(all_response_rows)
long_df.to_excel(TEMP_RESPONSE_PATH, index=False)
print(f"Saved: {TEMP_RESPONSE_PATH}")
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
print(f"Saved: {EXPORT_RESPONSE_PATH}")

# %%
# ------------------ SCORE AND EXPORT RESULTS ------------------
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

# %%
# ------------------ PLOT RESULTS ------------------
for cat, _ in prompts:
    tracking_df = pd.read_excel(TRACKING_PATHS[cat]["csv"])
    tracking_df["generation"] = tracking_df["generation"].astype(int)
    tracking_df["f1_score"] = tracking_df["f1_score"].astype(float)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        tracking_df["generation"],
        tracking_df["f1_score"],
        color="black",
        alpha=0.7,
    )
    plt.plot(
        tracking_df.groupby("generation")["f1_score"].max(),
        marker="o",
        linestyle="-",
        color="black",
    )
    plt.xlabel("Generation")
    plt.ylabel("F1 Score")
    plt.title(f"Prompt F1 Scores Across Generations - {cat.title()} Seed")
    plt.grid(True, linestyle="--", color="gray", alpha=0.7)
    plt.tight_layout()
    plt.savefig(TRACKING_PATHS[cat]["fig"])
    plt.show()
    print(f"Saved: {TRACKING_PATHS[cat]['fig']}")

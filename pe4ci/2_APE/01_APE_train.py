# A3_ape_training.py
# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from pe4ci.library import start, classify

# ------------------ CONSTANTS ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
MODEL = start.MODEL
SAMPLE = start.SAMPLE
SEED = start.SEED
TEMPERATURE = 0.0001

print(f"Running {CONCEPT} on {PLATFORM} with {MODEL} in train set")

NUM_VARIANTS = 5
NUM_GENERATIONS = 5

META_INSTRUCTIONS1 = "Generate a variation of the following prompt while keeping the output format. You can add important information or remove unnecessary information. Instruction:\n"
META_INSTRUCTIONS2 = "\nOutput only the new instruction."

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"
IMPORT_RESULTS_PATH = (
    start.MAIN_DIR + f"results/{PLATFORM}_{CONCEPT}_baseline_zero_results_dev.xlsx"
)

TRACKING_PATHS = {
    "top": {
        "csv": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_top_results_train.xlsx",
        "fig": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_evolution_top_train.png",
    },
    "bottom": {
        "csv": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_bottom_results_train.xlsx",
        "fig": f"{start.RESULTS_DIR}{PLATFORM}_{CONCEPT}_ape_evolution_bottom_train.png",
    },
}

# ------------------ LOAD DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "train"]
df = df[df.text.notna() & df.human_code.notna()]
if SAMPLE:
    df = df.sample(5, random_state=SEED)

prompt_df = pd.read_excel(IMPORT_RESULTS_PATH, sheet_name="results")
prompt_df = prompt_df.rename(columns={"Prompt": "prompt"})

# ------------------ MAIN SCRIPT ------------------
prompts = [
    ("top", prompt_df["F1"].idxmax()),
    ("bottom", prompt_df["F1"].idxmin()),
]

for cat, index in prompts:
    print(f"\n======== Evaluating {cat.upper()} Prompt Seed ========")
    OUTPUT_TRACKING_FILE = TRACKING_PATHS[cat]["csv"]
    FIGURE_FILE = TRACKING_PATHS[cat]["fig"]

    current_prompt = prompt_df.loc[index, "prompt"]
    tracking_records = []

    for generation in range(NUM_GENERATIONS):
        print(f"\n=== Generation {generation+1} ===")
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
            prompt_id = f"{cat}_gen{generation+1}_var{i+1}"
            rows = classify.evaluate_prompt(
                prompt_text=variant,
                prompt_id=prompt_id,
                df=df,
                platform=PLATFORM,
                temperature=TEMPERATURE,
            )
            f1 = f1_score(
                [r["human_code"] for r in rows],
                [r["classification"] for r in rows],
            )
            print(f"Variant F1: {f1:.4f}")
            variant_scores.append((variant, f1))

            tracking_records.append(
                {
                    "generation": generation + 1,
                    "variant_id": i + 1,
                    "prompt": variant,
                    "f1_score": f1,
                }
            )

        best_variant, best_f1 = max(variant_scores, key=lambda x: x[1])
        print(f"Best variant selected with F1 {best_f1:.4f}")
        current_prompt = best_variant

    tracking_df = pd.DataFrame(tracking_records)
    tracking_df.to_excel(OUTPUT_TRACKING_FILE, index=False)

# ------------------ PLOT RESULTS ------------------
for cat, _ in prompts:
    tracking_df = pd.read_excel(TRACKING_PATHS[cat]["csv"])
    tracking_df["generation"] = tracking_df["generation"].astype(int)
    tracking_df["f1_score"] = tracking_df["f1_score"].astype(float)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        tracking_df["generation"], tracking_df["f1_score"], color="black", alpha=0.7
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
# %%

# A0_baseline_prep.py
import pandas as pd
import numpy as np
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from pe4ci.library import start

# ------------------ SETUP ------------------
np.random.seed(start.SEED)

CONCEPT = start.CONCEPT
PROMPT_PATH = start.DATA_DIR + f"prompts/{CONCEPT}_baseline_variants.xlsx"
DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"


# ------------------ LOAD VARIANTS ------------------
part1_df = pd.read_excel(PROMPT_PATH, sheet_name="part1", index_col="part_num")
part2_df = pd.read_excel(PROMPT_PATH, sheet_name="part2", index_col="part_num")
opt_df = pd.read_excel(PROMPT_PATH, sheet_name="opt", index_col="opt_num")

# ------------------ GENERATE 50 RANDOM COMBOS ------------------
sample_prompts = []
for sample_num in range(1, 51):
    part1 = part1_df.sample(1, random_state=sample_num).iloc[0, 0]
    part2 = part2_df.sample(1, random_state=sample_num).iloc[0, 0]

    num_opts = np.random.randint(0, 4)
    opt_text = ""
    if num_opts > 0:
        opts = opt_df.sample(num_opts, random_state=sample_num).iloc[:, 0]
        opt_text = " ".join(opts.tolist()) + " "

    full_prompt = f"{part1} {opt_text}{part2}".strip()
    sample_prompts.append(full_prompt)

# ------------------ EXPORT TO EXCEL ------------------
prompt_df = pd.DataFrame({"baseline_prompt_id": range(1, 51), "prompt": sample_prompts})

with pd.ExcelWriter(
    PROMPT_PATH, mode="a", engine="openpyxl", if_sheet_exists="replace"
) as writer:
    prompt_df.to_excel(writer, sheet_name="baseline", index=False)

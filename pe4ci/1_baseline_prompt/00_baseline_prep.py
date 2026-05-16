# %%
import pandas as pd
import numpy as np

from crisp.library import start

# %%
# ------------------ SETUP ------------------
rng = np.random.default_rng(start.SEED)

CONCEPT = start.CONCEPT
PROMPT_PATH = start.DATA_DIR + f"prompts/{CONCEPT}_baseline_variants.xlsx"

N_PROMPTS = 50
GUIDANCE_MEAN_TARGET = 5

# %%
# ------------------ LOAD VARIANTS ------------------
part1_context_df = pd.read_excel(PROMPT_PATH, sheet_name="part1_context")
part2_task_df = pd.read_excel(PROMPT_PATH, sheet_name="part2_task")
part3_defn_df = pd.read_excel(PROMPT_PATH, sheet_name="part3_defn")
part4_guidance_df = pd.read_excel(PROMPT_PATH, sheet_name="part4_guidance")
final_pretext_df = pd.read_excel(PROMPT_PATH, sheet_name="final_pretext")
final_posttext_df = pd.read_excel(PROMPT_PATH, sheet_name="final_posttext")

FINAL_PRETEXT = final_pretext_df.loc[0, "prompt_part"]
FINAL_POSTTEXT = final_posttext_df.loc[0, "prompt_part"]

all_context_part_nums = part1_context_df["part_num"].tolist()
all_task_part_nums = part2_task_df["part_num"].tolist()
all_defn_part_nums = part3_defn_df["part_num"].tolist()
all_guidance_part_nums = part4_guidance_df["part_num"].tolist()

n_guidance_total = len(all_guidance_part_nums)
# probability that each guidance item is included in a given prompt
guidance_inclusion_probability = GUIDANCE_MEAN_TARGET / n_guidance_total

# %%
# ------------------ GENERATE UNIQUE COMBINATIONS ------------------
sample_prompts = []
seen_combinations = set()

# Always include the most basic version first: all 0s, no guidance
basic_context_part_num = 0
basic_task_part_num = 0
basic_defn_part_num = 0
basic_guidance_part_nums = tuple()

basic_context_text = part1_context_df.loc[
    part1_context_df["part_num"] == basic_context_part_num, "prompt_part"
].iloc[0]
basic_task_text = part2_task_df.loc[
    part2_task_df["part_num"] == basic_task_part_num, "prompt_part"
].iloc[0]
basic_defn_text = part3_defn_df.loc[
    part3_defn_df["part_num"] == basic_defn_part_num, "prompt_part"
].iloc[0]

basic_prompt = "\n".join(
    [basic_context_text, basic_task_text, basic_defn_text, FINAL_PRETEXT, "{TEXT}", FINAL_POSTTEXT]
)

sample_prompts.append(
    {
        "baseline_prompt_id": 1,
        "context_part_num": basic_context_part_num,
        "task_part_num": basic_task_part_num,
        "defn_part_num": basic_defn_part_num,
        "num_guidance_items": 0,
        "guidance_part_nums": "",
        "prompt": basic_prompt,
    }
)

seen_combinations.add(
    (
        basic_context_part_num,
        basic_task_part_num,
        basic_defn_part_num,
        basic_guidance_part_nums,
    )
)

while len(sample_prompts) < N_PROMPTS:
    context_part_num = rng.choice(all_context_part_nums)
    task_part_num = rng.choice(all_task_part_nums)
    defn_part_num = rng.choice(all_defn_part_nums)

    num_guidance = rng.binomial(n=n_guidance_total, p=guidance_inclusion_probability)

    if num_guidance == 0:
        guidance_part_nums = tuple()
    else:
        guidance_part_nums = tuple(
            sorted(
                rng.choice(all_guidance_part_nums, size=num_guidance, replace=False).tolist()
            )
        )

    combo_key = (context_part_num, task_part_num, defn_part_num, guidance_part_nums)

    if combo_key in seen_combinations:
        continue

    seen_combinations.add(combo_key)

    context_text = part1_context_df.loc[
        part1_context_df["part_num"] == context_part_num, "prompt_part"
    ].iloc[0]
    task_text = part2_task_df.loc[
        part2_task_df["part_num"] == task_part_num, "prompt_part"
    ].iloc[0]
    defn_text = part3_defn_df.loc[
        part3_defn_df["part_num"] == defn_part_num, "prompt_part"
    ].iloc[0]

    prompt_parts = [context_text, task_text, defn_text]
    if len(guidance_part_nums) > 0:
        guidance_texts = part4_guidance_df.loc[
            part4_guidance_df["part_num"].isin(guidance_part_nums), "prompt_part"
        ].tolist()
        prompt_parts.extend(guidance_texts)
    prompt_parts.extend([FINAL_PRETEXT, "{TEXT}", FINAL_POSTTEXT])
    full_prompt = "\n".join(prompt_parts)

    sample_prompts.append(
        {
            "baseline_prompt_id": len(sample_prompts) + 1,
            "context_part_num": context_part_num,
            "task_part_num": task_part_num,
            "defn_part_num": defn_part_num,
            "num_guidance_items": len(guidance_part_nums),
            "guidance_part_nums": ",".join(map(str, guidance_part_nums)),
            "prompt": full_prompt,
        }
    )

prompt_df = pd.DataFrame(sample_prompts)

# %%
# ------------------ EXPORT TO EXCEL ------------------
with pd.ExcelWriter(
    PROMPT_PATH,
    mode="a",
    engine="openpyxl",
    if_sheet_exists="replace",
) as writer:
    prompt_df.to_excel(writer, sheet_name="baseline", index=False)
print(f"Saved: {PROMPT_PATH}")

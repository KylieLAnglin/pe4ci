# %%
import pandas as pd
from pe4ci.library import start, classify

# %%
# ------------------ SETUP ------------------
CONCEPT = start.CONCEPT
PLATFORM = start.PLATFORM
TEMPERATURE = 0.0001

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx"
GOLD_PATH = start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx"
DEV_RESULTS_PATH = start.RESULTS_DIR + f"{PLATFORM}_{CONCEPT}_dev_results.xlsx"
EXPORT_RESPONSE_PATH = start.DATA_DIR + f"responses_test/{PLATFORM}_{CONCEPT}_best_responses_test.xlsx"

# %%
# ------------------ GET BEST DEV PROMPT ------------------
results_df = pd.read_excel(DEV_RESULTS_PATH, sheet_name="results")
best = results_df.sort_values("F1", ascending=False).iloc[0]

print(f"Best technique: {best['technique']}")
print(f"Best prompt ID: {best['prompt_id']}")
print(f"Best dev F1: {best['F1']}")

# %%
# ------------------ LOAD TEST DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df["split_group"] == "test"].copy()

df_gold = pd.read_excel(GOLD_PATH)
df_gold = df_gold[["unique_text_id", "participant_id", "human_code"]].copy()

# %%
# ------------------ EVALUATE BEST PROMPT ON TEST SET ------------------
rows = classify.get_classifications_from_prompt(
    prompt_text=best["prompt"],
    prompt_id=best["prompt_id"],
    df=df[["unique_text_id", "text"]].copy(),
    platform=PLATFORM,
    temperature=TEMPERATURE,
)

# %%
# ------------------ SAVE RESPONSES ------------------
response_df = pd.DataFrame(rows)
response_df = response_df.merge(df_gold, on="unique_text_id", how="left")
response_df["technique"] = best["technique"]

response_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
print(f"Saved: {EXPORT_RESPONSE_PATH}")

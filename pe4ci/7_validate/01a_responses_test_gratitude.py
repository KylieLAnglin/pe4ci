# %%
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from pe4ci.library import start, classify


# %%
# ------------------ CONSTANTS ------------------
CONCEPT = "gratitude"

DATA_PATH = start.DATA_DIR + f"clean/{CONCEPT}.xlsx"

EXPORT_RESPONSE_PATH = start.DATA_DIR + f"responses_dev/{CONCEPT}_best_test.xlsx"
EXPORT_RESULTS_PATH = start.MAIN_DIR + f"results/{CONCEPT}_best_results_test.xlsx"

# %%
# ------------------ GET PROMPT ------------------
results_df = pd.read_excel(start.RESULTS_DIR + "long_results_dev.xlsx")
results_df = results_df[results_df["concept"] == CONCEPT]
results_df = results_df.sort_values("f1", ascending=False)

best = results_df.iloc[0]
best_platform = best["platform"]
best_technique = best["technique"]
best_few = best["few"]
if best_few == True:
    best_few = "few"
else:
    best_few = "zero"
best_category = best["category"]

print(f"Best platform: {best_platform}")
print(f"Best technique: {best_technique}")
print(f"Best few-shot: {best_few}")
print(f"Best category: {best_category}")

# import prompt
file = f"{best_platform}_{CONCEPT}_{best_technique}_{best_few}_results_dev.xlsx"
prompt_df = pd.read_excel(start.MAIN_DIR + f"results/{file}", sheet_name="results")
prompt_df = prompt_df.sort_values("F1", ascending=False)
prompt = prompt_df.iloc[0]["prompt"].replace("Text: Text:", "")
prompt_id = prompt_df.iloc[0]["prompt_id"]
# %%

# ------------------ LOAD TEXT DATA ------------------
df = pd.read_excel(DATA_PATH)
df = df[df.split_group == "test"]

# %%
# ------------------ EVALUATE PROMPTS ------------------
all_rows = []
rows = classify.evaluate_prompt(prompt, prompt_id, df, best_platform, 0.0001)

# ------------------ SAVE RESPONSES ------------------
long_df = pd.DataFrame(rows)
long_df.to_excel(EXPORT_RESPONSE_PATH, index=False)

# ------------------ EXPORT METRICS ------------------
classify.export_results_to_excel(
    df=long_df,
    output_path=EXPORT_RESULTS_PATH,
    group_col="prompt_id",
    prompt_col="prompt",
    sheet_name="results",
    include_se=True,
)

# %%
import pandas as pd
from crisp.library import start, classify

# %%
# ------------------ SETUP ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
PLATFORMS = ["openai", "llama3.3"]

TEMPERATURE = 0.0001

RESPONSES_DIR = start.DATA_DIR + "responses_test/"
RESULTS_DIR = start.RESULTS_DIR

# %%
# ------------------ RUN EACH CONCEPT ------------------
for concept in CONCEPTS:
    DATA_PATH = start.DATA_DIR + f"clean/{concept}_coding_final.xlsx"
    EXPORT_RESPONSE_PATH = RESPONSES_DIR + f"{concept}_best_responses_test.xlsx"
    EXPORT_RESULTS_PATH = RESULTS_DIR + f"{concept}_best_results_test.xlsx"

    # ------------------ GET BEST DEV PROMPT ------------------
    all_dev_results = []

    for platform in PLATFORMS:
        platform_df = pd.read_excel(RESULTS_DIR + f"{platform}_{concept}_dev_results.xlsx", sheet_name="results")
        platform_df["platform"] = platform
        all_dev_results.append(platform_df)

    results_df = pd.concat(all_dev_results, ignore_index=True)
    results_df = results_df.sort_values("F1", ascending=False)

    best = results_df.iloc[0]

    best_platform = best["platform"]
    best_technique = best["technique"]
    best_prompt = best["prompt"].replace("Text: Text:", "")
    best_prompt_id = best["prompt_id"]
    best_category = best["category"]

    print(f"\n--- {concept} ---")
    print(f"Best platform: {best_platform}")
    print(f"Best technique: {best_technique}")
    print(f"Best category: {best_category}")
    print(f"Best prompt ID: {best_prompt_id}")
    print(f"Best dev F1: {best['F1']}")

    # ------------------ LOAD TEST DATA ------------------
    df = pd.read_excel(DATA_PATH)
    df = df[df["split_group"] == "test"].copy()

    # ------------------ EVALUATE BEST PROMPT ON TEST SET ------------------
    rows = classify.get_classifications_from_prompt(
        prompt_text=best_prompt,
        prompt_id=best_prompt_id,
        df=df,
        platform=best_platform,
        temperature=TEMPERATURE,
    )

    # ------------------ SAVE RESPONSES ------------------
    response_df = pd.DataFrame(rows)

    response_df = response_df.merge(
        df[["unique_text_id", "human_code"]],
        on="unique_text_id",
        how="left",
    )

    response_df["concept"] = concept
    response_df["platform"] = best_platform
    response_df["technique"] = best_technique
    response_df["category"] = best_category

    response_df.to_excel(EXPORT_RESPONSE_PATH, index=False)
    print(f"Saved responses: {EXPORT_RESPONSE_PATH}")

    # ------------------ EXPORT TEST METRICS ------------------
    classify.export_results_to_excel(
        df=response_df,
        output_path=EXPORT_RESULTS_PATH,
        group_col="prompt_id",
        prompt_col="prompt",
        y_true_col="human_code",
        y_pred_col="classification",
        sheet_name="results",
        include_se=False,
    )
    print(f"Saved results: {EXPORT_RESULTS_PATH}")

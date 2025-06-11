import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from pe4ci.library import start

# ------------------ CONFIG ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
PLATFORMS = ["openai", "llama3.3"]
TECHNIQUES = ["baseline", "ape", "persona", "cot", "explanation"]
STRATEGIES = ["zero", "few"]
RESULTS_DIR = start.RESULTS_DIR
RESULTS_SUFFIX = "_results_dev.xlsx"


# ------------------ HELPER FUNCTION ------------------
def assign_top_bottom_category(df):
    """
    Assign 'top' to the row with higher F1 and 'bottom' to the other.
    Assumes input df has two rows (or one).
    """
    df = df.copy()
    if len(df) < 2:
        df["category"] = "top" if df["F1"].iloc[0] >= 0.5 else "bottom"
    else:
        top_idx = df["F1"].idxmax()
        bottom_idx = df["F1"].idxmin()
        df.loc[top_idx, "category"] = "top"
        df.loc[bottom_idx, "category"] = "bottom"
    return df


# ------------------ COLLECT RESULTS ------------------
records = []

for platform in PLATFORMS:
    for concept in CONCEPTS:
        for technique in TECHNIQUES:
            for strategy in STRATEGIES:
                filename = (
                    f"{platform}_{concept}_{technique}_{strategy}{RESULTS_SUFFIX}"
                )
                filepath = os.path.join(RESULTS_DIR, filename)
                if not os.path.exists(filepath):
                    continue
                try:
                    df = pd.read_excel(filepath, sheet_name="results")
                    df = df[["F1", "F1 SE"]].copy()
                    df = assign_top_bottom_category(df)

                    for _, row in df.iterrows():
                        records.append(
                            {
                                "platform": platform,
                                "concept": concept,
                                "technique": technique,
                                "few": strategy == "few",
                                "category": row["category"],
                                "f1": row["F1"],
                                "f1_se": row["F1 SE"],
                            }
                        )
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

# ------------------ CREATE LONG DATAFRAME ------------------
long_df = pd.DataFrame(records)
# drop all missing values for any column
long_df.dropna(inplace=True)
long_df.to_excel(start.RESULTS_DIR + "long_results_dev.xlsx", index=False)
# (Optional) Preview
print(long_df.head())

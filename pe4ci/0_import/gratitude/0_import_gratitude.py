# %%
from pe4ci.library import start
import pandas as pd
import numpy as np

# ------------------ SETUP ------------------
CONCEPT = "gratitude"
SEED = start.SEED
NUM_SAMPLES_PER_CLASS = 400
NUM_EXAMPLES_PER_CLASS = 30
SPLIT_FRACTION = 1 / 3
RANDOM_ORDER_RANGE = 100001
np.random.seed(SEED)

# %%
# ------------------ LOAD DATA ------------------
df = pd.read_csv(start.DATA_DIR + "raw/goemotions_1.csv")
df = df.rename(columns={"gratitude": "human_code"})

positive_df = df[df["human_code"] == 1].sample(n=NUM_SAMPLES_PER_CLASS, random_state=SEED)
negative_df = df[df["human_code"] == 0].sample(n=NUM_SAMPLES_PER_CLASS, random_state=SEED)
df = pd.concat([positive_df, negative_df], ignore_index=True)

# %%
# ------------------ DEFINE COLUMNS ------------------
df["construct"] = CONCEPT
df["participant_id"] = df["id"].astype(str)
df["study"] = "emotion_classify"
df["question"] = "emotion"
df["unique_text_id"] = (
    df["participant_id"].astype(str) + "_" + df["study"] + "_" + df["question"]
)

df = df[
    [
        "participant_id",
        "study",
        "question",
        "text",
        "unique_text_id",
        "construct",
        "human_code",
    ]
].copy()

print(f"Total N: {len(df)}")
print(f"Positive N: {df['human_code'].sum()}")
print(f"Negative N: {(df['human_code'] == 0).sum()}")

# %%
# ------------------ SPLIT ------------------
np.random.seed(1)
df["random_number"] = np.random.randint(1, RANDOM_ORDER_RANGE, df.shape[0])
df = df.sort_values(by=["random_number"]).reset_index(drop=True)
df["order"] = np.arange(1, df.shape[0] + 1)

# split 1/3, 1/3, 1/3
n = df["order"].max()
train_size = int(n * SPLIT_FRACTION)
dev_size = int(n * SPLIT_FRACTION)
test_size = n - train_size - dev_size 

df["train"] = df["order"] <= train_size
df["dev"] = (df["order"] > train_size) & (df["order"] <= train_size + dev_size)
df["test"] = df["order"] > train_size + dev_size

df["split_group"] = np.where(df["train"], "train", "")
df["split_group"] = np.where(df["dev"], "dev", df["split_group"])
df["split_group"] = np.where(df["test"], "test", df["split_group"])

# %%
# ------------------ ASSIGN EXAMPLE TRAIN USE ------------------
df["train_use"] = pd.Series(dtype="object")

train_rows = df[df["split_group"] == "train"].copy()

positive_train = train_rows[train_rows["human_code"] == 1]
negative_train = train_rows[train_rows["human_code"] == 0]

positive_examples = positive_train.sample(n=NUM_EXAMPLES_PER_CLASS, random_state=SEED)
negative_examples = negative_train.sample(n=NUM_EXAMPLES_PER_CLASS, random_state=SEED)

example_sample = pd.concat([positive_examples, negative_examples])

df.loc[example_sample.index, "train_use"] = "example"

remaining_train_mask = (df["split_group"] == "train") & (df["train_use"].isna())
df.loc[remaining_train_mask, "train_use"] = "eval"

# %%
# ------------------ EXPORT ------------------
coding_df = df[
    [
        "participant_id",
        "split_group",
        "train_use",
        "study",
        "question",
        "unique_text_id",
        "text",
        "human_code",
    ]
].copy()

coding_df.to_excel(start.DATA_DIR + f"clean/{CONCEPT}_coding_final.xlsx", index=False)

# text-only file for prompting
text_df = coding_df.drop(columns=["human_code"]).copy()
text_df.to_excel(start.DATA_DIR + f"clean/{CONCEPT}_final.xlsx", index=False)

print(f"Saved: {start.DATA_DIR + f'clean/{CONCEPT}_coding_final.xlsx'}")
print(f"Saved: {start.DATA_DIR + f'clean/{CONCEPT}_final.xlsx'}")

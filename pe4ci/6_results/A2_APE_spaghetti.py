import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from pe4ci.library import start

# ------------------ SETUP ------------------
CONCEPTS = ["gratitude", "ncb", "mm"]
PLATFORMS = ["openai", "llama3.3"]
CATEGORIES = ["top", "bottom"]

CONCEPT_LABELS = {
    "gratitude": "Gratitude",
    "ncb": "Negative Core Beliefs",
    "mm": "Meaning Making",
}
# Grayscale line styles by concept
LINESTYLES = {
    "gratitude": "solid",
    "ncb": "dashed",
    "mm": "dotted",
}

# Gray colors by platform
COLORS = {
    "openai": "black",
    "llama3.3": "dimgray",
}

df_all = []

for platform in PLATFORMS:
    for concept in CONCEPTS:
        for category in CATEGORIES:
            file_path = os.path.join(
                start.RESULTS_DIR,
                f"{platform}_{concept}_ape_{category}_results_train.xlsx",
            )
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue

            df = pd.read_excel(file_path)
            df["platform"] = platform
            df["concept"] = concept
            df["category"] = category
            df_all.append(df)

df_full = pd.concat(df_all, ignore_index=True)

# ------------------ CREATE FIGURE ------------------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True, sharey=True)

for i, category in enumerate(["bottom", "top"]):  # bottom on top
    ax = axes[i]
    df_cat = df_full[df_full["category"] == category]

    for (platform, concept), group in df_cat.groupby(["platform", "concept"]):
        label = f"{CONCEPT_LABELS[concept]} ({platform})"
        linestyle = LINESTYLES[concept]
        color = COLORS[platform]

        group = group.groupby("generation")["f1_score"].max().reset_index()
        ax.plot(
            group["generation"],
            group["f1_score"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            alpha=0.9,
        )

        # Highlight max F1
        max_idx = group["f1_score"].idxmax()
        ax.scatter(
            group.loc[max_idx, "generation"],
            group.loc[max_idx, "f1_score"],
            color="black",
            edgecolor="white",
            zorder=5,
            s=70,
        )

    ax.set_title(f"{category.capitalize()} Prompts", fontsize=13)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    if i == 1:
        ax.set_xlabel("Generation", fontsize=11)

# ------------------ STYLING ------------------
axes[0].legend(title="Construct / Platform", fontsize=9, loc="lower right")
# fig.suptitle("Max F1 per Generation by Prompt Type", fontsize=15)
plt.xticks(sorted(df_full["generation"].unique()))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.rcParams["font.family"] = "times new roman"

# ------------------ SAVE ------------------
output_path = os.path.join(start.RESULTS_DIR, "ape_spaghetti_top_bottom_grayscale.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved plot to: {output_path}")

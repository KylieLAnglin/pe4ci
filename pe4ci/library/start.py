# ------------------ UPDATE HERE ------------------
# PLATFORM = "openai", "llama3.3"
# CONCEPT =  "gratitude", "mm", "ncb",
# SAMPLE = True, False
# Code can only run out of the box with openai and gratitude.
# Llama requires some modification for a high performance computing environment
# NCB and MM require data that are not publically available.

PLATFORM = "openai"
CONCEPT = "gratitude"
SAMPLE = False


CODE_DIR = "/Users/kla21002/pe4ci/code/"
MAIN_DIR = "/Users/kla21002/pe4ci/"
DATA_DIR = MAIN_DIR + "data/"
RESULTS_DIR = MAIN_DIR + "results/"

# ------------------ MODEL ------------------
if PLATFORM == "openai":
    MODEL = "gpt-4.1-2025-04-14"
elif PLATFORM == "llama3.3":  # requires HPC environment
    MODEL = "llama3.3:latest"


# ------------------ OTHERS ------------------
SEED = 123

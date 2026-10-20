# ------------------ UPDATE HERE ------------------
# PLATFORM = "openai", "llama3.3"
# CONCEPT = "gratitude", "mm", "ncb"
# SAMPLE = True, False
PLATFORM = "openai"
CONCEPT = "gratitude"
SAMPLE = False

# Code runs out of the box with openai and gratitude.
# llama3.3 requires an Ollama server (e.g., in a high performance computing environment).
# NCB and MM require data that are not publicly available.


# ------------------ PATHS ------------------
MAIN_DIR = "/path/to/project/"
DATA_DIR = MAIN_DIR + "data/"
RESULTS_DIR = MAIN_DIR + "results/"


# ------------------ MODEL ------------------
if PLATFORM == "openai":
    MODEL = "gpt-5.4-2026-03-05"
elif PLATFORM == "llama3.3":
    MODEL = "llama3.3:latest"

# ------------------ OTHERS ------------------
SEED = 123

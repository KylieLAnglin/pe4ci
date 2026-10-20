# PE4CI: Prompt Engineering for Construct Identification

This repository has two branches:

* **`main`**: minimal code for following the prompt engineering procedure recommended in the manuscript. Start here if you want to adapt the procedure to your own construct.
* **`replication`** (this branch): the full code used to produce the results, tables, and figures in the manuscript, including the bottom baseline prompt comparisons.

If you use or adapt this code, please cite:

Anglin, K. L., Milan, S., Hernandez, B., & Ventura, C. (2025). Improving Alignment Between Human and Machine Codes: An Empirical Assessment of Prompt Engineering for Construct Identification in Psychology. arXiv preprint arXiv:2512.03818. https://arxiv.org/abs/2512.03818

## Replication

The manuscript reports results for three constructs (gratitude, meaning making, and negative core beliefs) and two models (OpenAI gpt-5.4 and llama3.3).

* **Gratitude** uses public data from GoEmotions:
  Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. arXiv preprint arXiv:2005.00547. https://github.com/google-research/google-research/tree/master/goemotions
* **Negative core beliefs (`ncb`) and meaning making (`mm`)** require data that are not publicly available. The prompt variants for these constructs are included in `pe4ci/` as examples and for transparency.

Note LLM responses are not deterministic, so re-running the code will likely produce results close to, but not identical to, those reported.

llama3.3 requires an Ollama server, and likely a high-performance computing environment. See https://kb.uconn.edu/space/SH/27688501251/Ollama and https://github.com/hernandezb3/llama-on-uconn-hpc for one setup, though exact instructions will depend on the environment.

## Setup

1. Install the package and dependencies:

        conda create -n pe4ci_env python=3.10
        conda activate pe4ci_env
        pip install -r requirements.txt
        pip install -e .

2. Create `pe4ci/library/secrets.py` (ignored by git) containing only:

        OPENAI_API_KEY = "your-openai-key-here"

3. Edit `pe4ci/library/start.py`:
    * `MAIN_DIR`: the folder where data and results will be stored
    * `PLATFORM`: `"openai"` or `"llama3.3"`
    * `CONCEPT`: `"gratitude"`, `"mm"`, or `"ncb"`

4. Create the following folders inside `MAIN_DIR`:

        data/raw/
        data/clean/
        data/prompts/
        data/fewshot_examples/
        data/responses_train/
        data/responses_dev/
        data/responses_test/
        results/

5. Copy `pe4ci/{CONCEPT}_baseline_variants.xlsx` into `data/prompts/`.

6. For gratitude, download `goemotions_1.csv` from the GoEmotions repository into `data/raw/`.

## Running the Code

Run folders in order, and files within each folder in order. Steps 0 through 5 run for a single `PLATFORM` and `CONCEPT` (set in `start.py`). Repeat them for each platform and construct you want to replicate.

| Folder | Purpose |
|---|---|
| `0_import/gratitude/` | Sample GoEmotions texts and split into train, dev, and test |
| `1_baseline_prompt/` | Generate 50 combinatorial baseline prompts, evaluate them in train, then select few-shot example sets for the top and bottom prompts |
| `2_APE/` | Automatic prompt engineering, seeded with the top and bottom baseline prompts |
| `3_persona/` | Append personas to the baseline prompts |
| `4_cot/` | Zero-shot and few-shot chain-of-thought |
| `5_explanation/` | Few-shot examples with explanations |
| `6_results/` | Combine dev results across techniques and produce dev tables and the forest plot |
| `7_validate/` | Evaluate the best dev prompt in the test set and produce final tables |

Notes:

* `4_cot/0_cot_fewshot_prep.py` sets `CONCEPT` and `PLATFORM` directly at the top of the script rather than reading them from `start.py`. It exports templates for the few-shot examples; reasoning for each example must then be written by hand (columns `exemplar_cot1` and `exemplar_cot2`) and saved under the `_w_reasoning` file names printed by the script. `5_explanation/` reuses these files.
* The `figure_*.py` scripts, `6_results/`, and `7_validate/` loop over every construct and platform in their `CONCEPTS` and `PLATFORMS` lists. If you only replicate a subset (e.g., gratitude with OpenAI), edit these lists at the top of each script.

## Adapting to a New Construct

See the `main` branch.

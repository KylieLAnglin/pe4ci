# PE4CI: Prompt Engineering for Construct Identification

This repository has two branches:

* **`main`** (this branch): recommended code for implementing combinatorial empirical prompting, and additional techniques.
* **`replication`**: the full code used to produce the results, tables, and figures in the manuscript.

If you use or adapt this code, please cite:

Anglin, K. L., Milan, S., Hernandez, B., & Ventura, C. (2025). Empirical Prompt Engineering for Construct Identification with Large Language Models
. arXiv preprint arXiv:2512.03818. https://doi.org/10.48550/arXiv.2512.03818

## Organizaiton

1_baseline_prompt folder implements combinatorial empirical prompting, zero and few-shot. 

2-5 provide additional techniques.

6_results compares performance in a development dataset; and 7_validate assesses performance in a testing dataset. 



## Setup

1. Install the package and dependencies (Python 3.10+):

        conda create -n pe4ci_env python=3.10
        conda activate pe4ci_env
        pip install -r requirements.txt
        pip install -e .

2. Create `pe4ci/library/secrets.py` (ignored by git) containing only:

        OPENAI_API_KEY = "your-openai-key-here"

3. Edit `pe4ci/library/start.py`:
    * `MAIN_DIR`: the folder where data and results will be stored
    * `PLATFORM`: `"openai"` or `"llama3.3"` (llama3.3 requires an Ollama server, e.g., https://github.com/hernandezb3/llama-on-uconn-hpc)
    * `CONCEPT`: a short name for your construct (used in file names)
    * `SAMPLE`: `True` runs each script on a few texts to test the pipeline cheaply
    * `RESUME`: `True` continues an interrupted training run from its `_PROGRESS` file

4. Create the following folders inside `MAIN_DIR`:

        data/clean/
        data/prompts/
        data/fewshot_examples/
        data/responses_train/
        data/responses_dev/
        data/responses_test/
        results/

## Preparing Your Inputs

### Data

Save two files in `data/clean/`:

`{CONCEPT}_final.xlsx`, one row per text. See 0_import_gratitude for example. 

| Column | Description |
|---|---|
| `participant_id` | ID of the person who wrote the text |
| `unique_text_id` | ID unique to the text |
| `text` | the text to classify |
| `split_group` | `train`, `dev`, or `test` |
| `train_use` | for training texts, `example` (pool of candidate few-shot examples) or `eval` (used to evaluate prompts); blank otherwise |

`{CONCEPT}_coding_final.xlsx`, the human codes:

| Column | Description |
|---|---|
| `participant_id` | ID of the person who wrote the text |
| `unique_text_id` | ID unique to the text |
| `human_code` | 1 if the construct is present, 0 otherwise |

`0_import_gratitude.py` shows how these files were created from the public GoEmotions dataset and can serve as a template. To run it as-is, download `goemotions_1.csv` from https://github.com/google-research/google-research/tree/master/goemotions into `data/raw/`.

### Codebook Components

Copy one of the example workbooks (`pe4ci/gratitude_baseline_variants.xlsx`, `pe4ci/mm_baseline_variants.xlsx`, or `pe4ci/ncb_baseline_variants.xlsx`) to `data/prompts/{CONCEPT}_baseline_variants.xlsx` and replace the content with your own codebook. The workbook has one sheet per prompt component:

| Sheet | Contents |
|---|---|
| `part1_context` | variants of a description of the context (e.g., who wrote the text and why) |
| `part2_task` | variants of a description of the task |
| `part3_defn` | variants of the construct definition |
| `part4_guidance` | inclusion/exclusion criteria drawn from your codebook |
| `final_pretext`, `final_posttext` | formatting instructions placed before and after the text |
| `persona` | candidate personas (only needed for `3_persona/`) |

Keep `final_pretext` and `final_posttext` as they appear in the examples, or update code to match your preferred formatting. 

`1_baseline_prompt/00_baseline_prep.py` generates 50 prompts by randomly combining these components and saves them to a `baseline` sheet in the same workbook.

## Running the Code

Run folders and files in order. Folders 2-5 are optional. `6_results/` only includes techniques that were run.


## Outputs

* `data/responses_*/`: every LLM response, by technique and split
* `results/*_results_train.xlsx`, `results/*_results_dev.xlsx`: metrics for each prompt evaluated by each technique
* `results/{PLATFORM}_{CONCEPT}_dev_table.xlsx`: comparison of techniques in the development set
* `results/{PLATFORM}_{CONCEPT}_test_results.xlsx`: final metrics with 95% confidence intervals
* `results/{PLATFORM}_{CONCEPT}_misclassified_test.xlsx`: test texts where the LLM and human codes disagree, useful for reviewing the codebook

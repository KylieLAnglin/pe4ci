# PE4CI Setup Instructions

This code can be adapted to employ empricial prompt engineering for a wide range of constructs of interest.

If you use or adapt this code, please cite: 
Anglin, K. L., Milan, S., Hernandez, B., & Ventura, C. (2025). Improving Alignment Between Human and Machine Codes: An Empirical Assessment of Prompt Engineering for Construct Identification in Psychology. arXiv preprint arXiv:2512.03818.
https://arxiv.org/abs/2512.03818

The code can also be used to (non-deterministically) replicate the "gratitude" results in the above paper.
That dataset is from: https://github.com/google-research/google-research/tree/master/goemotions
Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. arXiv Preprint arXiv:2005.00547. https://doi.org/https://doi.org/10.48550/arXiv.2005.00547
* Negative Core Beliefs (NCB) and Meaning Making (MM) tasks require data that are not publicly available.

Notes: 
* LLaMA requires additional setup in a high-performance computing environment. See:
https://kb.uconn.edu/space/SH/27688501251/Ollama and
https://github.com/hernandezb3/llama-on-uconn-hpc for more information though exact instructions will depend on the environment. 



PREREQUISITES:
---

1. Create a secrets.py File for API Access

In the `pe4ci/library/` folder, create a file named `secrets.py` containing just:

    OPENAI_API_KEY = "your-openai-key-here"

- This file is included in `.gitignore` and must not be committed.
- Do not include any other credentials in this file.

---

2. Update start.py

Edit `pe4ci/library/start.py` to define the correct file paths and settings:

- Input file paths 
- Output directories
- Selected platform (e.g., "openai", "llama3")
- Target construct (e.g., "gratitude", "ncb", "mm")

---

3. Install Dependencies

Recommended Python version: 3.10+

Using pip:
    pip install -r requirements.txt

Using conda:
    conda create -n pe4ci_env python=3.10
    conda activate pe4ci_env
    pip install -r requirements.txt

---

4. Running the Code

Code must be run in order of folders (e.g., files in 1_baseline_prompt must be run before 2_APE file) and files within folders (e.g., 00_baseline_prep.py before 01_baseline_train.py)

FOR ADAPTATION:

The code assume a {CONSTRUCT}_final.xlsx and {CONSTRUCT}_coding_final.xlsx. 
{CONSTRUCT}_final.xlsx should contain the following columns: 
participant_id
split_group (train, dev, test)
train_use (eval or example)
study (can be a default number)
question (can be a default number)
unique_text_id (id unique to the text)
text

{CONSTRUCT}_coding_final.xlsx. should contain atleast:
unique_text_id
human_code

A {CONSTRUCT}_baseline_variants.xlsx file can be created by replacing the content of the uploaded examples with guidance relevant to the construct of interest. 

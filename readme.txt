# PE4CI Setup Instructions

This code can be used to (non-deterministically) replicate the "gratitude" results in Prompt Engineering for Construct Identification. 

Notes: 
* LLaMA requires additional setup in a high-performance computing environment. See:
https://kb.uconn.edu/space/SH/27688501251/Ollama and
https://github.com/hernandezb3/llama-on-uconn-hpc for more information though exact instructions will depend on the environment. 

* Negative Core Beliefs (NCB) and Meaning Making (MM) tasks require data that are not publicly available.

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

- Input file paths (e.g., train/dev CSVs)
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

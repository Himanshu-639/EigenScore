# INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection (ICLR-2024)

<div align="center"><img src="https://github.com/alibaba/eigenscore/blob/main/data/datasets/fig.png" width="750" /></div>

* This repository contains the code for the paper **[INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/abs/2402.03744)** (ICLR 2024).
* Contact: `chench@zju.edu.cn` / `ercong.cc@alibaba-inc.com`.

---

## 🛠️ Setup & Installation

### 1. Create Virtual Environment

#### Using `venv` (Recommended):
```bash
python -m venv .venv

# On Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# On Linux / macOS
source .venv/bin/activate
```

#### Or Using Conda:
```bash
conda env create -f environment.yml
conda activate eigenscore-py310
```

---

### 2. Install Dependencies

#### GPU (CUDA Support):
Install PyTorch with CUDA support (e.g. CUDA 12.6 or 12.4):

```bash
# Windows / Linux with CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Or CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Install Required Packages:
```bash
pip install transformers sentence-transformers datasets evaluate rouge_score accelerate pandarallel persist-to-disk matplotlib ipdb scikit-learn pandas tqdm
```

---

## 📂 Datasets

The repository supports multiple question-answering benchmark datasets:
* **`coqa`**: Conversational Question Answering (bundled under `data/datasets/coqa-dev-v1.0.json`).
* **`SQuAD`**: Stanford Question Answering Dataset (bundled under `data/datasets/dev-v2.0.json`).
* **`triviaqa`**: TriviaQA benchmark (loaded via Hugging Face `datasets`).
* **`nq_open`**: Natural Questions Open-domain QA (loaded via Hugging Face `datasets`).

---

## 🚀 Running the Code

The runner scripts (`run.ps1` and `run.sh`) run the full end-to-end pipeline:
1. **Model Generation & EigenScore Extraction** via `pipeline.generate`
2. **Evaluation & AUROC Computation** via `func.evalFunc`
3. **Automatic Log Saving & Plot Generation**

---

### Option A: On Windows (PowerShell)

#### Basic Run (Default: GPU, `facebook/opt-125m`, `coqa` dataset):
```powershell
.\run.ps1
```

#### Custom Parameter Run:
```powershell
.\run.ps1 -Model "facebook/opt-125m" -Dataset "coqa" -Fraction 0.01 -NumGen 5 -ProjectInd 1
```

#### Run on other datasets:
```powershell
# SQuAD dataset
.\run.ps1 -Model "facebook/opt-125m" -Dataset "SQuAD" -Fraction 0.01 -NumGen 5 -ProjectInd 2

# TriviaQA dataset
.\run.ps1 -Model "facebook/opt-125m" -Dataset "triviaqa" -Fraction 0.01 -NumGen 5 -ProjectInd 3

# Natural Questions dataset
.\run.ps1 -Model "facebook/opt-125m" -Dataset "nq_open" -Fraction 0.01 -NumGen 5 -ProjectInd 4
```

#### Available PowerShell Parameters:
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `-Model` | `facebook/opt-125m` | Hugging Face model identifier or local checkpoint folder path |
| `-Dataset` | `coqa` | Dataset to evaluate: `coqa`, `SQuAD`, `triviaqa`, `nq_open` |
| `-Fraction` | `0.01` | Fraction of dataset to evaluate (`0.01` = 1%, `1.0` = 100%) |
| `-NumGen` | `5` | Number of stochastic generations per prompt ($K$ samples for covariance matrix) |
| `-Device` | `cuda:0` | Compute device (`cuda:0` or `cpu`) |
| `-ProjectInd` | `0` | Run identifier for caching output |
| `-Temperature` | `0.5` | Sampling temperature |
| `-TopP` | `0.99` | Nucleus sampling Top-P |
| `-TopK` | `10` | Top-K sampling |
| `-Overwrite` | `False` | Switch flag to overwrite existing cached `.pkl` |

---

### Option B: On Linux / macOS / Git Bash (`run.sh`)

```bash
# Default run
bash run.sh

# Custom run: bash run.sh [MODEL] [DATASET] [FRACTION] [NUM_GEN] [DEVICE] [PROJECT_IND]
bash run.sh facebook/opt-125m coqa 0.01 5 cuda:0 1
```

---

### Option C: Direct Python Execution

#### Step 1: Run Generation & Extract Internal States
```bash
python -m pipeline.generate --model facebook/opt-125m --dataset coqa --fraction_of_data_to_use 0.01 --num_generations_per_prompt 5 --device cuda:0 --project_ind 1
```

#### Step 2: Run Evaluation Metrics & Plotting
```bash
python -m func.evalFunc ./data/output/facebook_opt-125m_coqa_1/0.pkl
```

---

## 📊 Outputs & Evaluation

* **Execution Logs**: Saved in `data/output/run_<model>_<dataset>_<timestamp>.log`.
* **Output Data (`.pkl`)**: Saved in `data/output/<model>_<dataset>_<project_ind>/0.pkl`.
* **ROC Curves**: Generated and saved to `Figure/AUROC_<dataset>.png`.

---

## 📖 Citation

If you use this code or paper in your work, please cite:

```bibtex
@article{chen2024inside,
  title={INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection},
  author={Chen, Chao and Liu, Kai and Chen, Ze and Gu, Yi and Wu, Yue and Tao, Mingyuan and Fu, Zhihang and Ye, Jieping},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024}
}
```


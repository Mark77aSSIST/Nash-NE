# From Echo Chambers to Equilibrium: Evaluating Non-Cooperative Isolation in LLM Ensembles under Benchmark Saturation

Files and guides to reproduce the experiments of the paper "From Echo Chambers to Equilibrium: Evaluating Non-Cooperative Isolation in LLM Ensembles under Benchmark Saturation".

## 📝 Authors
Mukeun Choi_1 and Taeyeon Oh_2*

* 1 : Seoul AI School, aSSIST University, Seoul, Republic of Korea
* 1 : SDG Management School, Geneva, Swiss
* 2 : Seoul AI School, aSSIST University, Seoul, Republic of Korea
* \* : Corresponding Author

## 📦 File Path

| Path | Description |
|------|------|
| `figures` | Figures of paper |
| `resultData` | Experiment Results |
| `scripts` | Experiment Code |

### 🔍 Experiment Code in scripts path

| File | Description |
|------|------|
| `01_Environment_Setting.py` | Experiments Environment Setup |
| `02_Dataset_Loader.py` | Dataset Loading |
| `03_LLM_Model_Rapper.py` | LLM Model Setup |
| `04_Algorithm.py` | Algorithm Define and Setup |
| `05_Experiment.py` | Experiment Execute |
| `06_Visualization.py` | Figure & Table Generation |

### 📚 Reference File

| File | Description |
|------|------|
| `README.md` | This File |

### 🧠 Using Model

| Model | Provider | API string |
|---|---|---|
| Gemini 2.5 Flash | Google DeepMind | `gemini-2.5-flash` |
| GPT-5 mini | OpenAI | `gpt-5-mini-2025-08-07` |
| Claude Haiku 4.5 | Anthropic | `claude-haiku-4-5-20251001` |

### 💾 Dataset

| Dataset | Task | Split | n (use) | Whole Size |
|---|---|---|---|---|
| GSM8K | math | test | 1,319 | 1,319 (100%) |
| MMLU-Pro | mcq | test | 1,000 | 12,032 (8.3%) |
| GPQA Diamond | mcq | train* | 198 | 198 (100%) |
| TruthfulQA | mcq | validation | 817 | 817 (100%) |
| ARC-Challenge | mcq | test | 1,172 | 1,172 (100%) |
| BIG-Bench Hard | mcq | test | 1,000 | 6,511 (15.4%) |
| MedQA-USMLE | mcq | test | 1,000 | 1,273 (78.6%) |

## 🔥 Key Results

| Dataset | Single-Best | Majority-Vote | Self-Consistency | MoA | **Nash-NE** | Δ |
|---|---|---|---|---|---|---|
| GSM8K | 95.60% | 96.29% | 83.47% | 79.98% | **90.45%** | −5.84%pt |
| MMLU-Pro | 72.50% | 70.70% | 65.40% | 66.80% | **74.60%** | +2.10%pt |
| GPQA Diamond | 45.45% | 35.35% | 27.78% | 39.90% | **52.02%** | +6.57%pt |
| TruthfulQA | 91.68% | 93.39% | 87.64% | 90.58% | **93.64%** | +0.24%pt |
| ARC-Challenge | 94.62% | 96.93% | 94.11% | 66.21% | **95.31%** | −1.62%pt |
| BIG-Bench Hard | 86.10% | 86.30% | 61.90% | 57.60% | **81.70%** | −4.60%pt |
| MedQA-USMLE | 63.20% | 64.40% | 40.00% | 42.00% | **64.10%** | −0.30%pt |

> Δ = Nash-NE − Best Baseline

## 🔤 Hypothesis Verification Summary

| Hypothesis | Adoption Rate | Result |
|---|---|---|
| H1: Nash-NE > Majority-Vote | 3/7 (42.9%) | Partial support in non-saturated benchmarks |
| H2: Nash-NE ≥ MoA | **7/7 (100%)** | **Adopting All Datasets** |
| H3: NE fulfillment rate ≥ 80% | 0/7 (0%) | quality_score Binary Limit |

## 🚀 Quick Start
1) Login to Google Colab environment (A100 GPU 40GB RAM)
2) Copy and paste Python files in the "scripts" folder into each cell in order of number
3) Connect Google Drive to your colab file
4) Setup a API key of 3 LLMs and Hugging-face into the Google Colab
5) Set up and verify the path in the code
6) Run each cell's code in order
7) You can check and download the results

---

## 🔬 Technical Details

### Requirement

```
python >= 3.10
openai >= 1.0.0
anthropic >= 0.40.0
google-genai >= 1.0.0
datasets >= 2.18.0
numpy >= 1.26.0
scipy >= 1.12.0
tqdm >= 4.66.0
```

### API Setup

Set to the Secrets tab or environment variable in your Google Colab environment.

```python
# Colab Secrets 사용 (Recommend)
from google.colab import userdata
OPENAI_API_KEY    = userdata.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
GEMINI_API_KEY    = userdata.get('GEMINI_API_KEY')
```

### Package Version

```
openai==1.82.0
anthropic==0.52.0
google-genai==1.16.0
datasets==3.6.0
numpy==1.26.4
scipy==1.13.0
```

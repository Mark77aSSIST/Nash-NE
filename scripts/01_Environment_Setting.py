# 0. Package installation
!pip install -q openai anthropic google-genai
!pip install -q datasets numpy scipy matplotlib seaborn tqdm
!pip install -q scikit-learn

# 1. Setup
import os, random
import numpy as np
from google.colab import userdata

# API Key
OPENAI_API_KEY    = userdata.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = userdata.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY    = userdata.get("GEMINI_API_KEY")

ACTIVE_MODELS = []
if OPENAI_API_KEY:    ACTIVE_MODELS.append("gpt")
if ANTHROPIC_API_KEY: ACTIVE_MODELS.append("claude")
if GEMINI_API_KEY:    ACTIVE_MODELS.append("gemini")

assert len(ACTIVE_MODELS) >= 2, " Minimum double of API Key is needed"
print(f" Activitation Model: {ACTIVE_MODELS}")

# Generation Setup
GEN_CFG = {
    "n_candidates": 3,   
    "temperature" : 0.7,
    "max_tokens"  : 512,
}

# Nash Equilibrium Setup
NASH_CFG = {
    "alpha"    : 0.7,
    "nash_temp": 0.3,
    "max_iter" : 500,
    "tol"      : 1e-7,
}

# Experiment Scale Selection
#
#  "pilot"  : for pipeline working check
#  "minimal": 
#  "standard": 
#  "full"   : Whole Dataset Using
#
EXPERIMENT_SCALE = "full"   

_SCALE_CFG = {
    #            gsm8k  mmlu  gpqa  truthful  medqa   bbh   arc_challenge
    "pilot"  : {  20,    20,   20,     20,       20,   20,             20},   
    "minimal": { 100,   100,   50,    100,      100,  100,            100},   
    "standard":{ 200,   200,  100,    200,      200,  200,            200},    
    "full"   : {1319,  1000,  198,    817,     1000, 1000,           1172},   # max
}

# Sample size by Experiment scale
_N = {
    "pilot"   : {"gsm8k": 20,   "mmlu_pro": 20,  "gpqa_diamond": 20,  "truthfulqa": 20, "medqa": 20,  "bbh": 20,  "arc_challenge": 20},
    "minimal" : {"gsm8k": 100,  "mmlu_pro": 100, "gpqa_diamond": 50,  "truthfulqa": 100, "medqa": 100, "bbh": 100, "arc_challenge": 100},
    "standard": {"gsm8k": 200,  "mmlu_pro": 200, "gpqa_diamond": 100, "truthfulqa": 200, "medqa": 200, "bbh": 200, "arc_challenge": 200},
    "full"    : {"gsm8k": 1319, "mmlu_pro": 1000, "gpqa_diamond": 198, "truthfulqa": 817, "medqa": 1000,"bbh": 1000, "arc_challenge": 1172},
}[EXPERIMENT_SCALE]

# Setup for 7 of Datasets
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_path"    : "openai/gsm8k",
        "hf_name"    : "main",
        "split"      : "test",
        "task_type"  : "math",
        "answer_type": "number",
        "n_samples"  : _N["gsm8k"],
        "n_choices"  : None,
    },
    "mmlu_pro": {
        "hf_path"    : "TIGER-Lab/MMLU-Pro",
        "hf_name"    : None,
        "split"      : "test",
        "task_type"  : "mcq",
        "answer_type": "letter",
        "n_samples"  : _N["mmlu_pro"],
        "n_choices"  : 10,
    },
    "gpqa_diamond": {
        "hf_path"    : "Idavidrein/gpqa",
        "hf_name"    : "gpqa_diamond",
        "split"      : "train",
        "task_type"  : "mcq",
        "answer_type": "letter",
        "n_samples"  : _N["gpqa_diamond"],   
        "n_choices"  : 4,
    },
    "truthfulqa": {
        "hf_path"    : "truthful_qa",
        "hf_name"    : "multiple_choice",
        "split"      : "validation",
        "task_type"  : "mcq",
        "answer_type": "letter",
        "n_samples"  : _N["truthfulqa"],
        "n_choices"  : None,
    },
    "medqa": {
        "hf_path"    : "bigbio/med_qa",
        "hf_name"    : "med_qa_en_bigbio_qa",
        "split"      : "test",
        "task_type"  : "mcq",
        "answer_type": "letter",
        "n_samples"  : _N["medqa"],
        "n_choices"  : 4,
    },
    "bbh": {
        "hf_path"    : "lukaemon/bbh",
        "hf_name"    : None,        
        "split"      : "test",
        "task_type"  : "mcq",
        "answer_type": "letter",
        "n_samples"  : _N["bbh"],
        "n_choices"  : None,        
    },
    "arc_challenge": {
        "hf_path"    : "allenai/ai2_arc",
        "hf_name"    : "ARC-Challenge",
        "split"      : "test",
        "task_type"  : "mcq",
        "answer_type": "letter",
        "n_samples"  : _N["arc_challenge"],
        "n_choices"  : 4,
    },
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
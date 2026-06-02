# 2. Dataset Loader for 7 of Datasets

import re, random, json
import numpy as np
import requests, random
from datasets import load_dataset
from huggingface_hub import login
from google.colab import userdata

# HuggingFace Token Login
HF_TOKEN = userdata.get("HF_TOKEN")        # Using a Colab Secrets
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("HuggingFace Login Complete")
else:
    print("No HF_TOKEN → Dataset Loading unavailable")


# Dataset Loader for each datasets

def _load_gsm8k(cfg: dict) -> list:
    raw = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    idx = random.sample(range(len(raw)), min(cfg["n_samples"], len(raw)))
    samples = []
    for i in idx:
        item = raw[i]
        m    = re.search(r"####\s*([\-\d,\.]+)", item["answer"])
        answer = m.group(1).replace(",", "").strip() if m else ""
        samples.append({
            "dataset": "gsm8k", "question": item["question"],
            "answer": answer, "choices": None,
            "reference": None, "test_code": None, "entry_point": None,
        })
    return samples


def _load_mmlu_pro(cfg: dict) -> list:
    LETTERS = "ABCDEFGHIJ"
    raw = load_dataset(cfg["hf_path"], split=cfg["split"])
    idx = random.sample(range(len(raw)), min(cfg["n_samples"], len(raw)))
    samples = []
    for i in idx:
        item   = raw[i]
        opts   = item["options"]
        answer = LETTERS[item["answer_index"]]
        choices = {LETTERS[j]: o for j, o in enumerate(opts)}
        choice_str = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        question = f"{item['question']}\n\n{choice_str}"
        samples.append({
            "dataset": "mmlu_pro", "question": question,
            "answer": answer, "choices": choices,
            "reference": None, "test_code": None, "entry_point": None,
        })
    return samples


def _load_gpqa_diamond(cfg: dict) -> list:
    LETTERS = "ABCD"
    raw = load_dataset(
        cfg["hf_path"], cfg["hf_name"],
        split=cfg["split"],
        token=True,          
    )
    idx = random.sample(range(len(raw)), min(cfg["n_samples"], len(raw)))
    samples = []
    for i in idx:
        item    = raw[i]
        correct = item["Correct Answer"]
        wrongs  = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        all_ch = [correct] + wrongs
        random.shuffle(all_ch)                    
        correct_idx = all_ch.index(correct)
        answer  = LETTERS[correct_idx]
        choices = {LETTERS[j]: ch for j, ch in enumerate(all_ch)}
        choice_str = "\n".join(f"{LETTERS[j]}. {ch}" for j, ch in enumerate(all_ch))
        question = f"{item['Question']}\n\n{choice_str}"
        samples.append({
            "dataset": "gpqa_diamond", "question": question,
            "answer": answer, "choices": choices,
            "reference": None, "test_code": None, "entry_point": None,
        })
    return samples


def _load_truthfulqa(cfg: dict) -> list:
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    raw = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    idx = random.sample(range(len(raw)), min(cfg["n_samples"], len(raw)))
    samples = []
    for i in idx:
        item   = raw[i]
        mc1    = item["mc1_targets"]
        ch_list = mc1["choices"]
        labels  = mc1["labels"]
        if 1 not in labels:
            continue
        correct_idx = labels.index(1)
        answer  = LETTERS[correct_idx]
        choices = {LETTERS[j]: ch for j, ch in enumerate(ch_list)}
        choice_str = "\n".join(f"{LETTERS[j]}. {ch}" for j, ch in enumerate(ch_list))
        question = f"{item['question']}\n\n{choice_str}"
        samples.append({
            "dataset": "truthfulqa", "question": question,
            "answer": answer, "choices": choices,
            "reference": None, "test_code": None, "entry_point": None,
        })
    return samples

def _load_humaneval(cfg):
    raw = load_dataset(cfg["hf_path"], split=cfg["split"])
    idx = random.sample(range(len(raw)), min(cfg["n_samples"], len(raw)))
    samples = []
    for i in idx:
        item = raw[i]
        samples.append({
            "dataset": "humaneval", "question": item["prompt"],
            "answer": item["canonical_solution"],
            "choices": None,
            "reference": item["canonical_solution"],
            "test_code": item["test"],
            "entry_point": item["entry_point"],
        })
    return samples

def _load_medqa(cfg: dict) -> list:
    raw = load_dataset(
        split="test",
    )
    idx = random.sample(range(len(raw)), min(cfg["n_samples"], len(raw)))
    samples = []

    for i in idx:
        item    = raw[i]
        options = item["options"]   # {"A": "...", "B": "...", ...}
        answer  = str(item["answer_idx"]).strip().upper()

        choice_str = "\n".join(
            f"({k}) {v}" for k, v in sorted(options.items())
        )
        question = f"{item['question']}\n\n{choice_str}"

        samples.append({
            "dataset" : "medqa",
            "question": question,
            "answer"  : answer,
            "choices" : options,
        })

    return samples


def _load_bbh(cfg: dict) -> list:
    MCQ_SUBSETS = [
        "causal_judgement", "date_understanding", "disambiguation_qa",
        "formal_fallacies", "geometric_shapes", "hyperbaton",
        "logical_deduction_five_objects", "logical_deduction_seven_objects",
        "logical_deduction_three_objects", "movie_recommendation",
        "penguins_in_a_table", "reasoning_about_colored_objects",
        "ruin_names", "salient_translation_error_detection", "snarks",
        "temporal_sequences", "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects", "web_of_lies",
    ]

    import re
    all_items = []

    for subset in MCQ_SUBSETS:
        try:
            raw = load_dataset(
                "lukaemon/bbh",
                subset,
                split="test",
            )
            for item in raw:
                text   = item["input"]
                target = item["target"].strip()

                choices_raw = re.findall(r'\(([A-F])\)\s+([^\n(]+)', text)
                if not choices_raw:
                    continue

                choices = {k.upper(): v.strip() for k, v in choices_raw}
                answer  = re.sub(r'[^A-F]', '', target.upper())
                if not answer:
                    continue

                all_items.append({
                    "dataset" : "bbh",
                    "question": text,
                    "answer"  : answer,
                    "choices" : choices,
                    "subset"  : subset,
                })
        except Exception as e:
            print(f"  BBH subset '{subset}' Load failed: {e}")
            continue

    if not all_items:
        raise ValueError("Could not load a BBH MCQ question")

    idx     = random.sample(range(len(all_items)),
                            min(cfg["n_samples"], len(all_items)))
    samples = [all_items[i] for i in idx]
    print(f"    BBH: {len(samples)}ea sampling of {len(all_items)}ea MCQ ")
    return samples


def _load_arc_challenge(cfg: dict) -> list:
    raw = load_dataset(
        cfg["hf_path"],
        cfg["hf_name"],
        split=cfg["split"],
    )
    idx = random.sample(range(len(raw)), min(cfg["n_samples"], len(raw)))

    NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    samples = []

    for i in idx:
        item    = raw[i]
        labels  = item["choices"]["label"]
        texts   = item["choices"]["text"]
        raw_key = str(item["answerKey"]).strip()

        choices = {}
        for lbl, txt in zip(labels, texts):
            key = NUM_TO_LETTER.get(lbl, lbl.upper())
            choices[key] = txt

        answer     = NUM_TO_LETTER.get(raw_key, raw_key.upper())
        choice_str = "\n".join(
            f"({k}) {v}" for k, v in sorted(choices.items())
        )
        question = f"{item['question']}\n\n{choice_str}"

        samples.append({
            "dataset" : "arc_challenge",
            "question": question,
            "answer"  : answer,
            "choices" : choices,
        })

    return samples

# Loader Map
_LOADERS = {
    "gsm8k"       : _load_gsm8k,
    "mmlu_pro"    : _load_mmlu_pro,
    "gpqa_diamond": _load_gpqa_diamond,
    "truthfulqa"  : _load_truthfulqa,
    "medqa"        : _load_medqa,
    "bbh"          : _load_bbh,
    "arc_challenge": _load_arc_challenge,
}


def load_dataset_samples(dataset_name: str) -> list:
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"  📥 {dataset_name:15s} ({cfg['hf_path']}) ... ", end="", flush=True)
    try:
        samples = _LOADERS[dataset_name](cfg)
        print(f" {len(samples)}ea")
        return samples
    except Exception as e:
        print(f" Error: {e}")
        return []


# Run Full Load
ALL_SAMPLES = {}
print("\n📥 Start loading the dataset:")
for ds_name in DATASET_CONFIGS:
    ALL_SAMPLES[ds_name] = load_dataset_samples(ds_name)

loaded   = {k: v for k, v in ALL_SAMPLES.items() if v}
failed   = {k: v for k, v in ALL_SAMPLES.items() if not v}
total    = sum(len(v) for v in loaded.values())

print(f"\n{'─'*45}")
print(f" Loading Success: {len(loaded)}ea Datasets, Total {total}ea Samples")
if failed:
    print(f"  Load Failed: {list(failed.keys())}")
    print("   → Failed datasets are automatically excluded from the experiment.")
print(f"{'─'*45}")
# 5. Experiment
# 5.1 Baseline

from collections import Counter

# B1. Single Best LLM
def baseline_single_best(dataset_name: str, stage2_results: list) -> dict:
    
    model_accs = {}
    for m in ACTIVE_MODELS:
        correct = sum(
            evaluate_correctness(
                dataset_name,
                extract_answer(dataset_name, r["candidates"][m][0]),
                r["sample"],
            )
            for r in stage2_results
        )
        model_accs[m] = correct / len(stage2_results)

    best  = max(model_accs, key=model_accs.get)
    return {
        "name"     : f"Single-Best ({best})",
        "accuracy" : model_accs[best],
        "per_model": model_accs,
    }


# B2. Majority Vote 
def baseline_majority_vote(dataset_name: str, stage2_results: list) -> dict:
    
    task    = DATASET_CONFIGS[dataset_name]["task_type"]
    correct = 0

    for r in stage2_results:
        all_candidates = [
            (extract_answer(dataset_name, resp), resp)
            for m in ACTIVE_MODELS
            for resp in r["candidates"][m]
        ]
        all_candidates = [(a, resp) for a, resp in all_candidates if a]

        if not all_candidates:
            continue

        if task in ("math", "mcq"):
            texts = [a for a, _ in all_candidates]
            voted = Counter(texts).most_common(1)[0][0]

        else:  
            passing = [
                code for code, resp in all_candidates
                if evaluate_correctness(dataset_name, code, r["sample"])
            ]
            voted = passing[0] if passing else all_candidates[0][0]

        if evaluate_correctness(dataset_name, voted, r["sample"]):
            correct += 1

    return {
        "name"    : "Majority-Vote",
        "accuracy": correct / len(stage2_results),
    }


# B3. Self-Consistency
def baseline_self_consistency(dataset_name: str, stage2_results: list) -> dict:

    task   = DATASET_CONFIGS[dataset_name]["task_type"]
    model  = ACTIVE_MODELS[0]
    correct = 0

    for r in stage2_results:
        candidates = [
            (extract_answer(dataset_name, resp), resp)
            for resp in r["candidates"][model]
        ]
        candidates = [(a, resp) for a, resp in candidates if a]

        if not candidates:
            continue

        if task in ("math", "mcq"):
            texts = [a for a, _ in candidates]
            voted = Counter(texts).most_common(1)[0][0]

        else:  # code
            passing = [
                code for code, resp in candidates
                if evaluate_correctness(dataset_name, code, r["sample"])
            ]
            voted = passing[0] if passing else candidates[0][0]

        if evaluate_correctness(dataset_name, voted, r["sample"]):
            correct += 1

    return {
        "name"    : f"Self-Consistency ({model})",
        "accuracy": correct / len(stage2_results),
    }


# B4. MoA
def baseline_moa(dataset_name: str, samples: list,
                 n_eval: int = 20) -> dict:

    task        = DATASET_CONFIGS[dataset_name]["task_type"]
    prompt      = SYSTEM_PROMPTS[dataset_name]
    n_moa       = min(n_eval, len(samples))
    correct     = 0
    N           = len(ACTIVE_MODELS)

    print(f"  📋 MoA baseline [{dataset_name}] "
          f"({n_moa}개, round-robin aggregator, 병렬 Round1)")

    for idx, sample in enumerate(
            tqdm(samples[:n_moa], desc=f"MoA·{dataset_name}", leave=False)):

        def _r1_call(model_name):
            return model_name, _call_with_semaphore(
                model_name, sample["question"],
                temperature  =GEN_CFG["temperature"],
                max_tokens   =GEN_CFG["max_tokens"],
                system_prompt=prompt,
            )

        round1 = {}
        with ThreadPoolExecutor(max_workers=N) as executor:
            for model_name, resp in executor.map(
                lambda m: _r1_call(m), ACTIVE_MODELS
            ):
                round1[model_name] = resp

        aggregator = ACTIVE_MODELS[idx % N]
        others = {m: r for m, r in round1.items() if m != aggregator}
        refs   = "\n\n".join(f"[Model {m}]:\n{r}" for m, r in others.items())

        if task in ("math", "mcq"):
            moa_q = (
                f"{sample['question']}\n\n"
                f"[Reference answers from other models]:\n{refs}\n\n"
                "Considering the references above, provide your best final answer:"
            )
        else:
            moa_q = (
                f"{sample['question']}\n\n"
                f"[Reference implementations from other models]:\n{refs}\n\n"
                "Review the references and provide a correct, improved implementation:"
            )

        r_final = _call_with_semaphore(
            aggregator, moa_q,
            temperature  =GEN_CFG["temperature"],
            max_tokens   =GEN_CFG["max_tokens"],
            system_prompt=prompt,
        )
        ans = extract_answer(dataset_name, r_final)
        if evaluate_correctness(dataset_name, ans, sample):
            correct += 1

    agg_dist = {
        m: sum(1 for i in range(n_moa) if ACTIVE_MODELS[i % N] == m)
        for m in ACTIVE_MODELS
    }
    return {
        "name"    : "MoA (information sharing)",
        "accuracy": correct / n_moa,
        "n_eval"  : n_moa,
        "agg_dist": agg_dist,
        "note"    : f"{n_moa}sample, round-robin, Parallel Round1",
    }


# 5.2 Common Function Definition

import json, time, os, datetime
import numpy as np
from google.colab import drive

TARGET_DATASETS = list(DATASET_CONFIGS.keys())
ALL_RESULTS = {}

DRIVE_SAVE_DIR = "/content/drive/MyDrive/Nash_LLM_Experiment"   # Setup a saving path of google drive 
RESULT_DIR = os.path.join(DRIVE_SAVE_DIR, "results")   

os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
print(f" Check the storage path: {DRIVE_SAVE_DIR}")

def _ds_path(dataset_name: str) -> str:
    return os.path.join(DRIVE_SAVE_DIR, f"result_{dataset_name}.json")

def save_dataset_result(dataset_name: str, result: dict):
    path = _ds_path(dataset_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2,
                  default=lambda o: str(o))
    print(f"  Drive Save: {path}")

def load_dataset_result(dataset_name: str) -> dict | None:
    path = _ds_path(dataset_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

MODEL_PRICES = {
    "gpt": {"in": 0.25, "out": 2.00},     # gpt-5-mini-2025-08-07
    "claude": {"in": 1.00, "out": 5.00},  # claude-haiku-4-5-20251001
    "gemini": {"in": 0.30, "out": 2.50}   # gemini-2.5-flash
}

MODEL_REGISTRY = {
    "gpt": "gpt-5-mini-2025-08-07",
    "claude": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
}
EXPERIMENT_META = {
    "models": {k: MODEL_REGISTRY[k] for k in ACTIVE_MODELS},
    "n_candidates": GEN_CFG["n_candidates"],
    "nash_alpha": NASH_CFG["alpha"],
    "nash_temp": NASH_CFG["nash_temp"],
    "nash_max_iter": NASH_CFG["max_iter"],
    "nash_tol": NASH_CFG["tol"],
    "random_seed": RANDOM_SEED,
    "experiment_scale": EXPERIMENT_SCALE,
    "datasets": TARGET_DATASETS,
}


def run_single_dataset(dataset_name: str) -> dict:
    
    print(f"\n{'='*60}")
    print(f"  Experiment: {dataset_name.upper()}")
    print(f"  Model: {' | '.join(MODEL_REGISTRY.get(m,m) for m in ACTIVE_MODELS)}")
    print(f"{'='*60}")
    t0 = time.time()

    samples = ALL_SAMPLES.get(dataset_name, [])
    if not samples:
        print(f"  No Sample - Passed")
        return {}

    s1 = run_stage1_dataset(dataset_name, samples)
    s2 = stage2_nash_dataset(dataset_name, s1)

    print(f"\n📐 [{dataset_name}] Baseline evaluation")
    b_single = baseline_single_best(dataset_name, s2)
    b_vote   = baseline_majority_vote(dataset_name, s2)
    b_sc     = baseline_self_consistency(dataset_name, s2)
    b_moa    = baseline_moa(dataset_name, samples, n_eval=len(samples))

    nash_acc = float(np.mean([r["is_correct"]          for r in s2]))
    ne_rate  = float(np.mean([r["ne_satisfied"]        for r in s2]))
    avg_reg  = float(np.mean([r["max_regret"]          for r in s2]))
    avg_iter = float(np.mean([r["nash_info"]["n_iter"] for r in s2]))

    result = {
        "dataset"         : dataset_name,
        "task_type"       : DATASET_CONFIGS[dataset_name]["task_type"],
        "n_samples"       : len(samples),
        "models_used"     : MODEL_REGISTRY,
        "experiment_scale": EXPERIMENT_SCALE,
        "nash_accuracy"   : nash_acc,
        "ne_rate"         : ne_rate,
        "avg_regret"      : avg_reg,
        "avg_iter"        : avg_iter,
        "elapsed_sec"     : round(time.time() - t0, 1),
        "baselines": {
            b_single["name"]: b_single["accuracy"],
            b_vote["name"]  : b_vote["accuracy"],
            b_sc["name"]    : b_sc["accuracy"],
            b_moa["name"]   : b_moa["accuracy"],
            "Nash-NE"       : nash_acc,
        },
        "stage2_detail": [
            {
                "idx"         : r["idx"],
                "is_correct"  : bool(r["is_correct"]),
                "ne_satisfied": bool(r["ne_satisfied"]),
                "max_regret"  : float(r["max_regret"]),
                "n_iter"      : int(r["nash_info"]["n_iter"]),
                "nash_answer" : str(r["nash_answer"])[:100],
                "ground_truth": str(r["sample"]["answer"])[:100],
                "nash_strategies": {
                    ACTIVE_MODELS[i]: r["strategies"][i].tolist()
                    for i in range(len(ACTIVE_MODELS))
                },
            }
            for r in s2
        ],
    }

    # Summary
    print(f"\n [{dataset_name}] Result (consumption: {result['elapsed_sec']}s)")
    print(f"  {'Methodology':<38} {'Accuracy':>8}")
    print(f"  {'-'*47}")
    for name, acc in sorted(result["baselines"].items(), key=lambda x: x[1]):
        mark = "  ◀ Nash-NE" if name == "Nash-NE" else ""
        print(f"  {name:<38} {acc*100:>7.1f}%{mark}")
    print(f"\n  NE Satisfaction rate: {ne_rate*100:.1f}%  |  "
          f"Average regret: {avg_reg:.5f}  |  Average convergence: {avg_iter:.0f}회")
    return result


def run_dataset_cell(dataset_name: str):
    # If already have a saved result, restore it and save it to the drive immediately after launch if it's a new experiment.
    print(f"\n{'='*60}")
    print(f"  📂  [{dataset_name}] Ready to execute")
    print(f"{'='*60}")

    # Saved result check
    cached = load_dataset_result(dataset_name)
    if cached:
        print(f"\n  Already Saved Results Found — Retest Skip")
        print(f"   Nash-NE: {cached['nash_accuracy']*100:.1f}%  |  "
              f"n={cached['n_samples']}  |  {cached['elapsed_sec']}s")
        print(f"\n   To retest, delete the file below and run it again:")
        print(f"   {_ds_path(dataset_name)}")
        return cached

    # New experiment execute
    result = run_single_dataset(dataset_name)
    if result:
        save_dataset_result(dataset_name, result)
    return result


print(" Common Function Load Complete")
print(f"   Save Path: {DRIVE_SAVE_DIR}")
print(f"   Experiment Scale: {EXPERIMENT_SCALE}")
print(f"   Target Dataset: {list(DATASET_CONFIGS.keys())}")


# 5.3 Experiment for GSM8K
result_gsm8k = run_dataset_cell("gsm8k")


# 5.4 Experiment for MMLU-Pro
result_mmlu_pro = run_dataset_cell("mmlu_pro")


# 5.5 Experiment for GPQA Diamond
result_gpqa = run_dataset_cell("gpqa_diamond")


# 5.6 Experiment for TruthfulQA
result_truthful = run_dataset_cell("truthfulqa")

 
# 5.7 Experiment for MedQA-USMLE
result_medqa = run_dataset_cell("medqa")


# 5.8 Experiment for BIG-Bench Hard
result_bbh = run_dataset_cell("bbh")


# 5.9 Experiment for ARC-Challenge
result_arc = run_dataset_cell("arc_challenge")



# 5.10 Consolidated Save Completed Results + Final Summary

ALL_RESULTS = {}
missing = []

for ds in DATASET_CONFIGS.keys():
    r = load_dataset_result(ds)
    if r:
        ALL_RESULTS[ds] = r
        print(f"  [{ds}] Load Complete — Nash {r['nash_accuracy']*100:.1f}%")
    else:
        missing.append(ds)
        print(f"  [{ds}] No Result — Cell 9-{list(DATASET_CONFIGS.keys()).index(ds)+1} Requires execution")

# Save consolidated files
if ALL_RESULTS:
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        DRIVE_SAVE_DIR,
        f"nash_all_results_{EXPERIMENT_SCALE}_{timestamp}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"experiment_meta": EXPERIMENT_META, "results": ALL_RESULTS},
            f, ensure_ascii=False, indent=2, default=lambda o: str(o)
        )
    print(f"\n  Save consolidated files: {summary_path}")

# Final Summary Table
if ALL_RESULTS:
    print(f"\n{'='*65}")
    print(f"  Final Summary ({len(ALL_RESULTS)}/7ea Complete)")
    print(f"{'='*65}")
    print(f"  {'Dataset':<18} {'Nash':>7} {'Best-BL':>8} "
          f"{'Δ':>7} {'NE fulfillment':>7} {'Regret':>10}")
    print(f"  {'-'*60}")
    for ds, res in ALL_RESULTS.items():
        bl      = res["baselines"]
        best_bl = max((v for k,v in bl.items() if k != "Nash-NE"), default=0)
        delta   = res["nash_accuracy"] - best_bl
        sign    = "+" if delta >= 0 else ""
        print(f"  {ds:<18} {res['nash_accuracy']*100:>6.1f}%"
              f" {best_bl*100:>7.1f}%"
              f" {sign}{delta*100:>5.1f}%pt"
              f" {res['ne_rate']*100:>6.0f}%"
              f" {res['avg_regret']:>10.5f}")

if missing:
    print(f"\n  Not completed: {missing}") 
# 4. Experiment Algorithm
# 4.1 Evaluator by Datasets

import re
import subprocess
import tempfile
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# System Prompt by Datasets
SYSTEM_PROMPTS = {
    "gsm8k": (
        "You are a precise math problem solver. "
        "Solve the problem step-by-step. "
        "End your response with '#### <number>' where <number> is "
        "the final numeric answer only. No commas in the number."
    ),
    "mmlu_pro": (
        "You are a knowledgeable expert answering multiple-choice questions. "
        "Think briefly, then respond with ONLY the single letter of your answer "
        "(A, B, C, D, E, F, G, H, I, or J). "
        "No explanation after the letter."
    ),
    "gpqa_diamond": (
        "You are a PhD-level expert in biology, physics, and chemistry. "
        "Analyze carefully and respond with ONLY the letter of your answer "
        "(A, B, C, or D). No explanation."
    ),
    "truthfulqa": (
        "You are a truthful assistant that avoids falsehoods. "
        "Choose the most factually accurate answer. "
        "Respond with ONLY the letter of your answer. No explanation."
    ),
    "medqa": (
        "You are a medical expert. Answer the following USMLE-style "
        "medical question. Choose the single best answer from the options "
        "provided. Reason step by step, then end with: "
        "'The answer is (X)' where X is the letter of your choice."
    ),
    "bbh": (
        "You are a logical reasoning expert. Answer the following question "
        "carefully. If it is multiple choice, choose the single best answer. "
        "Reason step by step, then end with: "
        "'The answer is (X)' where X is the letter or value."
    ),
    "arc_challenge": (
        "You are a science expert. Answer the following science question. "
        "Choose the single best answer from the options provided. "
        "Reason step by step, then end with: "
        "'The answer is (X)' where X is the letter of your choice."
    ),
}

# Answer Extraction
def extract_answer(dataset_name: str, response: str) -> str:
    
    if not response:
        return ""

    if dataset_name == "gsm8k":
        m = re.search(r"####\s*([\-\d,\.]+)", response)
        if m:
            return m.group(1).replace(",", "").strip()
        nums = re.findall(r"[\-]?\d+(?:\.\d+)?", response)
        return nums[-1] if nums else ""

    elif dataset_name in ("mmlu_pro", "gpqa_diamond", "truthfulqa"):
        
        m = re.search(
            r"(?:answer\s+is|answer:|correct(?:\s+answer)?\s*:)\s*([A-J])\b",
            response, re.I
        )
        if m:
            return m.group(1).upper()
        
        for line in [response.strip().split("\n")[0],
                     response.strip().split("\n")[-1]]:
            m = re.match(r"^\s*([A-J])[.\):\s]?$", line.strip(), re.I)
            if m:
                return m.group(1).upper()
        
        m = re.search(r"\b([A-J])\b", response)
        return m.group(1).upper() if m else ""

    elif dataset_name in ("medqa", "bbh", "arc_challenge"):
        
        m = re.search(
            r'[Tt]he answer is\s*[\(\[]?([A-Fa-f])[\)\]]?',
            response
        )
        if m:
            return m.group(1).upper()

        m = re.search(
            r'(?:Answer:|answer:)\s*[\(\[]?([A-Fa-f])[\)\]]?',
            response
        )
        if m:
            return m.group(1).upper()

        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for line in reversed(lines):
            m = re.match(r'^[\(\[]?([A-Fa-f])[\)\]]?[\s\.:\,]?$', line)
            if m:
                return m.group(1).upper()

        matches = re.findall(r'\b([A-Fa-f])\b', response)
        if matches:
            return matches[-1].upper()

        return ""

    return ""


# Quality Score (Nash Payoff Function)
# (math/mcq Task)
def quality_score(dataset_name: str, candidates_k: list,
                  sample: dict) -> np.ndarray:

    task = DATASET_CONFIGS[dataset_name]["task_type"]
    scores = np.zeros(len(candidates_k))

    answers = [extract_answer(dataset_name, r) for r in candidates_k]

    from collections import Counter
    answer_counts = Counter(a for a in answers if a)
    most_common   = answer_counts.most_common(1)[0][0] if answer_counts else None

    for k, (resp, ans) in enumerate(zip(candidates_k, answers)):
        is_correct = float(evaluate_correctness(dataset_name, ans, sample))

        # Consistency weighting: 1.0, if multiple answers match, 0.5 if outlier
        consistency = 1.0 if ans == most_common else 0.5

        # Continuous Quality Score: Correctness × Consistency
        scores[k] = is_correct * consistency + (1 - is_correct) * (1 - consistency) * 0.1

    return scores


# Consistency Score (Nash Consistency Function)
def consistency_score(dataset_name: str, resp_a: str, resp_b: str) -> float:
    
    ans_a = extract_answer(dataset_name, resp_a)
    ans_b = extract_answer(dataset_name, resp_b)

    if dataset_name == "gsm8k":
        if not ans_a or not ans_b:
            return 0.0
        try:
            return 1.0 if abs(float(ans_a) - float(ans_b)) < 1e-6 else 0.0
        except ValueError:
            return 1.0 if ans_a == ans_b else 0.0

    elif dataset_name in ("mmlu_pro", "gpqa_diamond", "truthfulqa"):
        return 1.0 if ans_a.upper() == ans_b.upper() else 0.0

    elif dataset_name in ("humaneval"):
        if not ans_a.strip() or not ans_b.strip():
            return 0.0
        return _tfidf_cosine(ans_a, ans_b)

    return 0.0


# The final determination of the correct answer
def evaluate_correctness(dataset_name: str,
                         predicted: str, sample: dict) -> bool:
    
    task = DATASET_CONFIGS[dataset_name]["task_type"]

    if task == "math":
        gt = sample["answer"]
        try:
            return abs(float(predicted) - float(gt)) < 1e-6
        except (ValueError, TypeError):
            return str(predicted).strip() == str(gt).strip()

    elif task == "mcq":
        return str(predicted).upper() == str(sample["answer"]).upper()

    return False


# Matrix Builder (Enter Nash Algorithm)
def build_quality_matrix(dataset_name: str,
                         candidates: dict, sample: dict) -> np.ndarray:
    
    N, K = len(ACTIVE_MODELS), GEN_CFG["n_candidates"]
    Q = np.zeros((N, K))

    for i, m in enumerate(ACTIVE_MODELS):
        
        candidates_k = [
            candidates[m][k] if k < len(candidates.get(m, [])) else ""
            for k in range(K)
        ]
        # Quality_score returns K-dimensional array → full row substitution
        Q[i, :] = quality_score(dataset_name, candidates_k, sample)

    return Q

def build_consistency_matrix(dataset_name: str, candidates: dict) -> np.ndarray:
    """C[i, k, j, l] = consistency(r_{ik}, r_{jl})  shape: (N, K, N, K)"""
    model_names = list(candidates.keys())
    N, K = len(model_names), GEN_CFG["n_candidates"]
    C = np.zeros((N, K, N, K))
    for i, m_i in enumerate(model_names):
        for k in range(K):
            r_ik = candidates[m_i][k] if k < len(candidates[m_i]) else ""
            for j, m_j in enumerate(model_names):
                for l in range(K):
                    if i == j:
                        C[i, k, j, l] = 1.0
                        continue
                    r_jl = candidates[m_j][l] if l < len(candidates[m_j]) else ""
                    C[i, k, j, l] = consistency_score(dataset_name, r_ik, r_jl)
    return C


# Internal Helper
def _rouge_l(pred: str, ref: str) -> float:
    
    p_tok = pred.lower().split()   # ROUGE-L F1 (LCS based)
    r_tok = ref.lower().split()
    m, n  = len(p_tok), len(r_tok)
    if m == 0 or n == 0:
        return 0.0
    # LCS via DP
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            curr[j+1] = prev[j] + 1 if p_tok[i] == r_tok[j] else max(curr[j], prev[j+1])
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    prec   = lcs / m
    rec    = lcs / n
    return round(2 * prec * rec / (prec + rec), 4)

def _tfidf_cosine(a: str, b: str) -> float:
    # TF-IDF cosine similarity
    try:
        mat = TfidfVectorizer(max_features=500).fit_transform([a, b])
        return float(np.clip(cosine_similarity(mat[0:1], mat[1:2])[0][0], 0, 1))
    except Exception:
        return 0.0


print(" DatasetEvaluator Ready (7 Datasets Support)")


# 4.2 Nash Equilibrium Algorithm

def find_nash_equilibrium(quality_matrix: np.ndarray,
                          consistency_matrix: np.ndarray,
                          alpha: float = None,
                          temperature: float = None,
                          max_iter: int = None,
                          tol: float = None,
                          verbose: bool = False) -> tuple:

    alpha       = alpha       if alpha       is not None else NASH_CFG["alpha"]
    temperature = temperature if temperature is not None else NASH_CFG["nash_temp"]
    max_iter    = max_iter    if max_iter    is not None else NASH_CFG["max_iter"]
    tol         = tol         if tol         is not None else NASH_CFG["tol"]

    N, K = quality_matrix.shape

    # Initialization: Equal distribution (uninformed)
    strategies = [np.ones(K) / K for _ in range(N)]
    history = []

    for t in range(max_iter):
        new_strategies = []

        for i in range(N):
            marginal = np.zeros(K)

            for k in range(K):
                q = alpha * quality_matrix[i, k]   # Quality Components (independent of other models)
                c = 0.0                            # Consistency component (expectation consistency according to σ_{-i})
                for j in range(N):
                    if i == j:
                        continue
                    
                    c += np.dot(strategies[j], consistency_matrix[i, k, j, :])   # E_{σ_j}[consistency(r_{ik}, R_j)] = Σ_l σ_j(l) * C[i,k,j,l]
                if N > 1:
                    c /= (N - 1)

                marginal[k] = q + (1 - alpha) * c

            # Softmax optimal response (numerical stability: subtract maximum)
            marginal -= marginal.max()
            sigma = np.exp(marginal / temperature)
            sigma /= sigma.sum()
            new_strategies.append(sigma)

        # Check convergence
        delta = max(np.max(np.abs(new_strategies[i] - strategies[i]))
                    for i in range(N))
        history.append(delta)
        strategies = new_strategies

        if verbose and (t % 50 == 0 or delta < tol):
            print(f"  iter {t+1:4d} | Δ = {delta:.2e}")

        if delta < tol:
            break

    # Nash condition verification (calculate the regret value)
    nash_check = _verify_nash(strategies, quality_matrix, consistency_matrix, alpha)

    info = {
        "n_iter"     : len(history),
        "converged"  : delta < tol,
        "final_delta": delta,
        "history"    : history,
        "nash_check" : nash_check,
    }
    return strategies, info


def _verify_nash(strategies, quality_matrix, consistency_matrix, alpha):

    N, K = quality_matrix.shape
    results = []

    for i in range(N):
        # Expected rewards of the current mix strategy
        current_eu = 0.0
        best_deviation = -np.inf

        for k in range(K):
            q = alpha * quality_matrix[i, k]
            c = 0.0
            for j in range(N):
                if i == j:
                    continue
                c += np.dot(strategies[j], consistency_matrix[i, k, j, :])
            if N > 1:
                c /= (N - 1)

            uk = q + (1 - alpha) * c
            current_eu   += strategies[i][k] * uk
            best_deviation = max(best_deviation, uk)

        regret = best_deviation - current_eu
        results.append({
            "model"        : i,
            "expected_util": round(current_eu,    4),
            "best_dev_util": round(best_deviation, 4),
            "regret"       : round(regret,         6),
            "NE_satisfied" : regret < 1e-3,
        })
    return results


def aggregate_by_nash(candidates_dict: dict, strategies: list) -> str:

    model_names = list(candidates_dict.keys())
    K = EXPERIMENT_CFG["n_candidates"]

    # Nash selection response for each model (argmax σ_i)
    selected = []
    weights  = []
    for i, m in enumerate(model_names):
        best_k = int(np.argmax(strategies[i]))
        if best_k < len(candidates_dict[m]):
            resp   = candidates_dict[m][best_k]
            weight = float(strategies[i][best_k])
            selected.append((resp, weight))

    if not selected:
        return ""

    # Weighted voting based on extracted answers
    vote_tally: dict[str, float] = {}
    for resp, w in selected:
        ans = extract_answer(resp)
        if ans:
            vote_tally[ans] = vote_tally.get(ans, 0.0) + w

    if vote_tally:
        return max(vote_tally, key=vote_tally.get)

    # Fallback: Return the highest weighted response
    return max(selected, key=lambda x: x[1])[0]


print(" Nash equilibrium algorithm ready")


# 4.3 Non-cooperative independence creation

from tqdm import tqdm

def _diversity_stats(dataset_name: str, candidates: dict, sample: dict) -> dict:
    
    task = DATASET_CONFIGS[dataset_name]["task_type"]
    stats = {}

    for m, resps in candidates.items():
        if task in ("math", "mcq"):
            extracted = [extract_answer(dataset_name, r) for r in resps]
            unique_count = len(set(a for a in extracted if a))
            correct_count = sum(
                1 for a in extracted
                if a and a.upper() == str(sample["answer"]).upper()
            )
        else:  
            signatures = [r.strip()[:120] for r in resps if r.strip()]
            unique_count  = len(set(signatures))
            correct_count = -1

        stats[m] = {
            "unique_count" : unique_count,
            "correct_count": correct_count,
        }
    return stats


# Limiting the number of simultaneous requests per model (adjusted according to API)
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_API_SEMAPHORES = {
    "gpt"   : threading.Semaphore(3),  # OpenAI: Has room
    "claude": threading.Semaphore(3),  # Anthropic: Has room
    "gemini": threading.Semaphore(2),  # Google: Conservatively restricted
}

def _call_with_semaphore(model_name: str, question: str,
                         temperature: float, max_tokens: int,
                         system_prompt: str) -> str:
    
    sem = _API_SEMAPHORES.get(model_name, threading.Semaphore(2))
    with sem:
        return call_model(model_name, question,
                          temperature=temperature,
                          max_tokens=max_tokens,
                          system_prompt=system_prompt)

def stage1_generate_sample(dataset_name: str, sample: dict) -> dict:
    # Implementation of a non-cooperative gaming environment - Each model is fully isolated
    prompt   = SYSTEM_PROMPTS[dataset_name]
    question = sample["question"]

    def _generate_for_model(model_name: str) -> tuple[str, list[str]]:
        
        resps = []
        for _ in range(GEN_CFG["n_candidates"]):
            resp = _call_with_semaphore(
                model_name, question,
                temperature  =GEN_CFG["temperature"],
                max_tokens   =GEN_CFG["max_tokens"],
                system_prompt=prompt,
            )
            resps.append(resp)
        return model_name, resps

    # Three-Model Concurrent Calls (Parallel Between Models)
    candidates = {}
    with ThreadPoolExecutor(max_workers=len(ACTIVE_MODELS)) as executor:
        futures = {
            executor.submit(_generate_for_model, m): m
            for m in ACTIVE_MODELS
        }
        for future in as_completed(futures):
            model_name, resps = future.result()
            candidates[model_name] = resps

    return candidates


MAX_CONCURRENT_SAMPLES = 3  

def run_stage1_dataset(dataset_name: str, samples: list) -> list:
    
    task = DATASET_CONFIGS[dataset_name]["task_type"]
    print(f"\n🔬 [{dataset_name}] Non-cooperative generation "
          f"({len(samples)}ea, task={task}, "
          f"Parallel={MAX_CONCURRENT_SAMPLES}sample×{len(ACTIVE_MODELS)}Model)")

    results = [None] * len(samples)  

    def _process_sample(idx_sample: tuple) -> tuple[int, dict]:
        idx, sample = idx_sample
        candidates = stage1_generate_sample(dataset_name, sample)
        diversity  = _diversity_stats(dataset_name, candidates, sample)
        return idx, {
            "dataset"   : dataset_name,
            "idx"       : idx,
            "sample"    : sample,
            "candidates": candidates,
            "diversity" : diversity,
        }

    # Parallel processing between samples
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SAMPLES) as executor:
        futures = {
            executor.submit(_process_sample, (i, s)): i
            for i, s in enumerate(samples)
        }
        with tqdm(total=len(samples), desc=f"Stage1·{dataset_name}") as pbar:
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    print(f"\n  Sample {idx} error: {e}")
                    results[idx] = None
                pbar.update(1)

    # Remove None (Error Sample)
    results = [r for r in results if r is not None]

    # Output Diversity Summary
    K = GEN_CFG["n_candidates"]
    avg_uniq = np.mean([
        np.mean([r["diversity"][m]["unique_count"] for m in ACTIVE_MODELS])
        for r in results
    ])
    print(f"  Average Unique Response Count: {avg_uniq:.2f}/{K}")
    return results


# 4.4 Create Nash Balance and Choose the Best Agreement

def aggregate_by_nash(dataset_name: str,
                      candidates: dict,
                      strategies: list) -> str:
    
    model_names = list(candidates.keys())
    task        = DATASET_CONFIGS[dataset_name]["task_type"]

    # Gather Nash best candidates for each model
    selected = []
    for i, m in enumerate(model_names):
        best_k  = int(np.argmax(strategies[i]))
        resp    = candidates[m][best_k] if best_k < len(candidates[m]) else ""
        weight  = float(strategies[i][best_k])
        selected.append((resp, weight))

    if not selected:
        return ""

    # math / mcq: weighted voting
    if task in ("math", "mcq"):
        
        vote: dict[str, float] = {}
        for resp, w in selected:
            ans = extract_answer(dataset_name, resp)
            if ans:
                vote[ans] = vote.get(ans, 0.0) + w

        if not vote:
            return ""

        max_weight = max(vote.values())

        tied_answers = sorted(
            [ans for ans, w in vote.items() if w == max_weight]
        )
        return tied_answers[0]

    # code: Nash Weighted Top Candidate Selection
    else:
        best_resp, _ = max(selected, key=lambda x: x[1])
        return extract_answer(dataset_name, best_resp)


def stage2_nash_dataset(dataset_name: str,
                        stage1_results: list) -> list:
    # Apply Nash equilibrium across datasets
    task    = DATASET_CONFIGS[dataset_name]["task_type"]
    results = []
    print(f"\n⚖️  [{dataset_name}] Nash equilibrium derivation (task={task})")

    for r1 in tqdm(stage1_results, desc=f"Nash·{dataset_name}"):
        sample     = r1["sample"]
        candidates = r1["candidates"]

        # Configuring Quality and Consistency Matrixes
        Q = build_quality_matrix(dataset_name, candidates, sample)
        C = build_consistency_matrix(dataset_name, candidates)

        # Navigating Nash Equilibrium
        strategies, info = find_nash_equilibrium(
            Q, C,
            alpha      = NASH_CFG["alpha"],
            temperature= NASH_CFG["nash_temp"],
            max_iter   = NASH_CFG["max_iter"],
            tol        = NASH_CFG["tol"],
        )

        # Derivation of final answers and determination of correct answers
        nash_ans   = aggregate_by_nash(dataset_name, candidates, strategies)
        is_correct = evaluate_correctness(dataset_name, nash_ans, sample)

        # Nash condition verification (regret calculation)
        ne_satisfied = all(r["NE_satisfied"] for r in info["nash_check"])
        max_regret   = max(r["regret"]       for r in info["nash_check"])

        results.append({
            **r1,
            "nash_answer" : nash_ans,
            "strategies"  : strategies,
            "nash_info"   : info,
            "ne_satisfied": ne_satisfied,
            "max_regret"  : max_regret,
            "is_correct"  : is_correct,
        })

    # Summary
    acc      = np.mean([r["is_correct"]   for r in results])
    ne_rate  = np.mean([r["ne_satisfied"] for r in results])
    avg_reg  = np.mean([r["max_regret"]   for r in results])
    avg_iter = np.mean([r["nash_info"]["n_iter"] for r in results])

    print(f"  └ Nash Accuracy: {acc*100:.1f}%  |  "
          f"NE Satisfaction rate: {ne_rate*100:.1f}%  |  "
          f"Average regret: {avg_reg:.5f}  |  "
          f"Average convergence: {avg_iter:.0f}times")
    return results
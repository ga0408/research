# Dream-RSI: Recursive Self-Improvement through Evolving Worlds — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Dream-RSI_Recursive_Self-Improvement_through_Evolving_Worlds_2026_Google.md) / 원본: [arXiv:2609.14858](https://arxiv.org/abs/2609.14858) · [GitHub](https://github.com/zhengkid/Dream-RSI) · [Project Page](https://dream-rsi.com)

---

## 1. Problem Setting & Motivation

### 1.1 The Meta-Exploration Dilemma in Long-Horizon Discovery
> "Recursive self-improvement (RSI) has emerged as an ambitious goal for autonomous AI systems. A common mechanism underlying RSI is an iterative discovery loop wherein agents generate candidate solutions, evaluate outcomes, incorporate feedback, and refine future iterations... As agent capabilities improve and self-improvement targets become challenging, discovery increasingly requires long-horizon exploration over vast search spaces, often spanning thousands of proposal–evaluation cycles."
>
> "Existing approaches have largely relied on manually designed exploration strategies that remain largely fixed throughout discovery. Fixed strategies cannot improve from accumulated discovery experience and may repeatedly allocate computation to ineffective search directions. Recent work therefore seeks to optimize exploration policies online during discovery, but doing so faces two fundamental bottlenecks. First, feedback is delayed and expensive at the meta level: unlike evaluating an individual candidate, assessing an exploration policy requires observing how it shapes the subsequent discovery process over many proposal–evaluation cycles. Second, the meta-policy space is vast: a newly proposed policy may perform poorly, so many alternatives may need to be tried. Together, these challenges make meta-level improvement particularly costly: each policy may require a long online rollout before receiving useful feedback, making it difficult to efficiently close the self-improvement loop at the exploration layer."

### 1.2 Discovery History as a Replay Simulator (World Model Analogy)
> "To address these bottlenecks, our key intuition is simple: a fast and inexpensive simulator of discovery would allow many exploration policies to be evaluated before costly online deployment. Surprisingly, completed discovery histories already provide such a simulator. While prior work treats past discovery history merely as static textual context or training data for weight fine-tuning, a completed discovery process inherently records a structured tree of past exploration decisions and their realized code-execution outcomes."
>
> "Drawing an analogy to model-based reinforcement learning and World Models, once organized into a discovery tree, this history can serve as a replay simulator. As illustrated in Figure 2, an alternative exploration strategy can navigate this pre-recorded tree to traverse different subsets of recorded branches, in different orders, with different parallel groupings and stopping decisions. Because all execution outcomes are already saved in the tree, evaluating a new strategy requires only reading past records without rerunning the underlying discovery agent or evaluator. This transforms meta-policy improvement from an expensive online trial-and-error process into a fast, simulation-based 'dreaming' procedure."

---

## 2. Dream-RSI Framework Formulation

### 2.1 Discovery Trees & Decision Interface
A discovery tree is rooted at `r`, representing the initial workspace state. Each non-root node `v` has a unique primary parent, identifying where the exploration attempt begins. Node `v` records the filesystem snapshot, generated code artifact, diagnostic logs, and scalar task score `s_v`.

```
Set of eligible nodes for exploration:
A(T) = {r} ∪ {v ∈ T : v is a leaf of T}

Feasible parallel batch action for worker budget W ≥ 1:
A(T; W) = {C ⊆ A(T) : |C| ≤ W}
```

### 2.2 Online Rollout (Iteration t)
Let `t = 1, 2, ...` index outer recursive discovery rounds, starting from initial exploration policy `π_1` and empty history `H_0 = ()`.
- At round `k ≤ K_1`, policy `π_t` selects a node batch `C_t^k ∈ A(T_t^k; W)`.
- Each worker executes generation and evaluation in parallel, creating child nodes attached to parent nodes in `C_t^k`, yielding tree `T_t^{k+1}`.
- Upon termination (empty batch or `k = K_1`), the completed discovery tree `T_t` is archived:
```
H_t = H_{t-1} ∪ {T_t}
```

### 2.3 Offline Evaluation & Dreaming
During the offline phase of iteration `t`, history `H_t` is held fixed while `M ≥ 1` candidate policy versions `π_t^0, ..., π_t^{M-1}` (with `π_t^0 = π_t`) are evaluated across each historical tree `T_i` (`i = 1, ..., t`).
- Replay starts at `T_i^{m, 0} = {r}`.
- At decision round `k`, candidate policy `π_t^m` selects batch `C_i^{m,k} ∈ A(T_i^{m,k}; W)`.
- Replay transitions deterministically without invoking the LLM coding agent or test suite:
```
T_i^{m, k+1} = T_i^{m, k} ∪ ⋃_{v ∈ C_i^{m,k}} Child(v; T_i, T_i^{m, k})
```
- For `v ≠ r`, `Child(v; T_i, T_i^{m, k})` reveals `v`'s recorded child on tree `T_i`.
- For `v = r`, replay returns the earliest-created unobserved child of `r` outside `T_i^{m,k}`, opening a new branch.

### 2.4 Replay Objective
Let `N_i^m = |T_i^{m, k_i^{m,*}}| - 1` be the number of revealed non-root nodes (effective generations), and `k_i^{m,*}` be the number of completed decision rounds.
For fixed coefficients `β_1, β_2 ≥ 0`, the replay score on world `T_i` is defined as:

```
V_i^m = max_{v ∈ T_i^{m, k_i^{m,*}}} s_v  -  β_1 · N_i^m  +  β_2 · ( N_i^m / max{1, k_i^{m,*}} )
        [   discovery quality   ]    [ execution cost ]    [    parallelism bonus    ]
```

- Term 1: Maximum solution quality achieved in the replay rollout.
- Term 2: Linear penalty on total discovery requests evaluated.
- Term 3: Reward for average generation attempts per round (encourages parallel batching over sequential crawling).

### 2.5 Policy Improvement and Non-Degradation Guarantee
The aggregate evaluation score across all `t` historical replay worlds is:
```
V^m = (1 / t) · ∑_{i=1}^t V_i^m
```
The LLM policy-development agent reviews execution traces, scores, and prior revision diffs to produce candidate `π_t^{m+1}`.
After `M` iterations, the next deployed policy is:
```
π_{t+1} = π_t^{m*}   where   m* = argmax_{m ∈ {0, ..., M-1}} V^m
```
Because the candidate pool contains `π_t^0 = π_t`, the selection strictly guarantees:
```
V^{m*} ≥ V^0
```
The newly selected policy cannot degrade average historical performance on `H_t` before online deployment.

---

## 3. Mathematical Benchmark Definitions (Appendix A)

### 3.1 Lasso Regularization Path
Given feature matrix `X ∈ ℝ^{n × p}`, target vector `y ∈ ℝ^n`, and sequence `λ_1 > ··· > λ_K`:
```
F_k(w) = (1 / (2n)) · ‖y - Xw‖_2² + λ_k · ‖w‖_1
w*_k = argmin_{w ∈ ℝ^p} F_k(w)
```
Correctness constraint:
```
F_k(w_tilde_k) ≤ F_k(w_k,sklearn) + 10^-6   for all k ∈ {1, ..., K}
```
Timing score across instance set `I`:
```
R_search = ( ∏_{i ∈ I} t_i )^(-1 / |I|)
```

### 3.2 Sum-Difference Problem
Find a finite set `A ⊂ ℤ` maximizing the normalized sum-to-difference ratio:
```
Γ(A) = log(|A + A| / |A|) / log(|A - A| / |A|)
where A + A = {a + a' : a, a' ∈ A},  A - A = {a - a' : a, a' ∈ A}
```

### 3.3 Circle Packing in a Unit Square
For `n ∈ {26, 32}`, determine centers `(x_i, y_i) ∈ [0, 1]²` and radii `r_i ≥ 0` maximizing:
```
∑_{i=1}^n r_i
subject to:
r_i ≤ x_i ≤ 1 - r_i,   r_i ≤ y_i ≤ 1 - r_i
(x_i - x_j)² + (y_i - y_j)² ≥ (r_i + r_j)²   for all 1 ≤ i < j ≤ n
```

### 3.4 Autocorrelation Inequalities
For integrable `f : ℝ → ℝ` supported on `[-1/4, 1/4]` with `∫_{-1/4}^{1/4} f(x) dx = 1`:
```
Autoconvolution: (f * f)(t) = ∫_ℝ f(t - x) f(x) dx,   t ∈ [-1/2, 1/2]

Problem 1 (Non-negative f):
Minimize Φ_1(f) = max_{t ∈ [-1/2, 1/2]} (f * f)(t)

Problem 2 (Non-negative f):
Maximize Φ_2(f) = ‖f * f‖_2² / ( ‖f * f‖_1 · ‖f * f‖_∞ )

Problem 3 (Signed f):
Minimize Φ_3(f) = max_{t ∈ [-1/2, 1/2]} |(f * f)(t)|
```

---

## 4. Key Experimental Results

### 4.1 Lasso Regularization Path (Algorithm Engineering)
Wall-clock runtime (ms) on six held-out datasets (Gisette, RCV1, DNA, Leukemia, Colon, Duke Breast):

| Method | LLM Backbone | Compute (Calls) | Non-bio (Gisette) | Non-bio (RCV1) | Bio (DNA) | Bio (Leukemia) | Bio (Colon) | Bio (Duke) | Average (ms) |
|---|---|---|---|---|---|---|---|---|---|
| sklearn | — | — | 11,275.2 | 252,881.7 | 93.8 | 227.2 | 229.8 | 374.0 | 44,180.3 |
| glmnet | — | — | 9,063.6 | 73,072.8 | 351.9 | 45.0 | 24.2 | 47.7 | 13,767.5 |
| SimpleTES | GPT-OSS-120B | 51,200 | 3,141.9 | 19,625.6 | 15.9 | 15.5 | 11.6 | 18.1 | 3,804.8 |
| SimpleTES † | GPT-OSS-120B | 51,200 | 8,651.0 | 41,143.1 | 37.6 | 28.2 | 19.5 | 31.1 | 8,318.4 |
| **Recursive Fixed** | Gemini-3.1-Pro | 550 | 1,861.8 | 19,550.1 | 41.5 | 26.1 | 14.5 | 28.4 | 3,587.1 |
| **Dream-RSI** | Gemini-3.1-Pro | **317** | 2,841.0 | 14,616.0 | 49.9 | 30.2 | 16.4 | 32.5 | **2,931.0** |
| **Recursive Fixed** | Gemini-3.7-Flash | 3,200 | 1,133.1 | 13,873.0 | 29.8 | 24.1 | 15.7 | 24.4 | 2,516.7 |
| **Dream-RSI** | Gemini-3.7-Flash | **1,879** | 1,091.9 | 12,923.4 | 31.4 | 21.0 | 12.2 | 23.6 | **2,350.6** |

### 4.2 Mathematical Optimization Benchmarks

| Method | LLM | Sum-Diff (↑) | Autocorrelation (↓) | Circle Packing (↑) |
|---|---|---|---|---|
| AlphaEvolve | Gemini-2.0 Pro + Flash | — | 1.455700 | 2.635862 |
| AlphaEvolveV2 | Gemini-2.0 Pro + Flash | 1.121936 | — | 2.635983 |
| OpenEvolve | — | — | 1.460000 | — |
| CodeEvolve | — | — | — | 2.635980 |
| ShinkaEvolve | Mixed | — | 1.457800 | 2.635982 |
| TTS-Discovery | Qwen3-8B | — | — | 2.635983 |
| ThetaEvolve | Distilled-Qwen3-8B | — | 1.493000 | 2.635983 |
| EvoX | Gemini-3.0-Pro | — | 1.458900 | 2.635900 |
| SimpleTES | GPT-OSS-120B (51.2k) | 1.143975 | 1.453675 | 2.635983 |
| **Recursive Fixed** | Gemini-3.1-Pro | 1.144047 | 1.456001 | 2.635983 |
| **Dream-RSI** | Gemini-3.1-Pro (<1k) | **1.145427** | 1.456375 | **2.635983** |

### 4.3 GPU Kernel Engineering (KernelBench)
- **VGG16**: 2.43× fewer generations for comparable execution speed.
- **LayerNorm**: 1.79× fewer generations for comparable execution speed.
- **ConvDiv**: 2.09× performance speedup (1/ms) under identical generation budgets.
- **ConvMax**: 1.44× performance speedup under identical generation budgets.

### 4.4 Inductive Bias Analysis (Figure 5 & 6)
- **Prompt Semantic Guidance Pitfall**: In ConvDiv, injecting abstracted textual advice ("Guidance") into prompts caused exploration to lock into narrow sub-spaces. Unguided Dream-RSI reached ~1.898 1/ms, whereas guided Dream-RSI stalled around ~1.48 1/ms. Fixed exploration suffered an even worse drop (1.28 → 0.63 1/ms).
- **Dynamic Behavior Evolution**: In ConvDiv rounds E0–E8, Dream-RSI started with 110 attempts, dropped to 50 attempts during steady refinement (E4), and dynamically increased attempts back to 92/80/91 when approaching performance plateaus, driving final speed from 0.427 to 1.898 1/ms.

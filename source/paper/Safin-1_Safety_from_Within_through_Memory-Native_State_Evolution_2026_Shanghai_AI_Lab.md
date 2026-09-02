# Safin-1: Safety from Within through Memory-Native State Evolution — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Safin-1_Safety_from_Within_through_Memory-Native_State_Evolution_2026_Shanghai_AI_Lab.md) / 원본: [arXiv:2609.00092](https://arxiv.org/abs/2609.00092) · [GitHub](https://github.com/AI45Lab/Safin-1)

---

## 1. Abstract & Introduction

### 1.1 Core Motivation: Safety from Within
> "Long-horizon complex tasks require foundation models to accumulate information, maintain internal states, and adapt over extended interactions. In these settings, safety should be an intrinsic property of the model itself, rather than a behavioral constraint that depends solely on external safeguards or post-hoc alignment procedures such as supervised fine-tuning. This motivates Safety from Within, in which safety-relevant capabilities are represented and invoked through the model's native computation rather than relying solely on external safeguards." (Section 1, Page 1)

### 1.2 Limitations of Conventional Recurrent Models
> "Self-attention preserves fine-grained access to prior tokens but incurs quadratic training cost and a key–value cache that grows with sequence length. Recurrent and linear-attention models instead compress the causal prefix into a fixed-size state, enabling efficient constant-memory decoding. Selective state-space models and delta-rule recurrences substantially improve how this state is written, revised, and forgotten. However, conventional recurrent designs expose only the latest state for direct reading. Once an earlier association is attenuated or overwritten, its previous representation may no longer be recoverable from the current state alone." (Section 1, Page 3)

---

## 2. Technical Methodology: MARCH Architecture

### 2.1 Context-Derived State Anchors
> "Let `T = [t_1, ..., t_L]` be a sequence of `L` text tokens. An anchoring policy specifies an ordered set of text boundaries `B = {b_m}_{m=1}^M`, where `0 = b_0 < b_1 < ... < b_M ≤ L`. We insert one occurrence of a shared learned anchor embedding after each boundary:
```
T_tilde = ∥_{m=1}^M ( [t_{b_{m-1}+1}, ..., t_{b_m}] ∥ [ξ_m] ) ∥ [t_{b_M+1}, ..., t_L]
```
> where `∥` denotes sequence concatenation, and `ξ_m` is the `m`-th occurrence of the shared anchor embedding `ξ`. Text and anchor positions have distinct computational roles. Text positions apply the base recurrent update, allowing the state to evolve continuously across boundaries. Immediately after processing `t_{b_m}`, the architecture checkpoints the resulting cumulative state:
```
A^(m,ℓ) = S_{b_m}^(ℓ) ∈ ℝ^{d_v × d_k},    m = 1, ..., M
```
> Because the recurrence is not reset at anchor boundaries, `A^(m,ℓ)` represents the cumulative prefix up to position `b_m`, rather than only the most recent segment. We therefore refer to it as a state anchor. The ordered bank `{A^(1,ℓ), ..., A^(M,ℓ)}` preserves the temporal trajectory of a single recurrent memory before later updates attenuate or overwrite earlier contents." (Section 2.1, Page 5-6)

### 2.2 Content-Conditioned Anchor Metadata
> "Let `u_m^(ℓ)` denote the normalized input representation of anchor position `ξ_m` at layer `ℓ`. The anchor position reads only its aligned state checkpoint:
```
q_m^(ℓ) = W_q^(ℓ) u_m^(ℓ),    o_m^(ℓ) = A^(m,ℓ) q_m^(ℓ)
```
> The same input representation is projected into a compact routing key:
```
κ_m^(ℓ) = W_k^(ℓ) u_m^(ℓ) ∈ ℝ^{d_r}
```
> The aligned readout is passed through the standard output projection and residual pathway, making the next-layer anchor representation `u_m^(ℓ+1)` dependent on its associated checkpoint `A^(m,ℓ)`. Consequently, although all anchor positions share the same input embedding, their representations—and therefore their routing keys—become state-dependent after the first layer. The router thus addresses anchors according to their retained contents rather than their temporal indices alone." (Section 2.1, Page 6)

### 2.3 Content-Routed State Retrieval & Null Option
> "For a text token at position `t`, the causally available state anchors are indexed by `V_t = {m ∈ {1, ..., M} | b_m < t}`. In the dynamic-memory setting, these anchors form the non-null candidate set `C_t = V_t`. The routing module projects the normalized hidden state `x_t` into a routing query and scores it against the key of each candidate:
```
ρ_t = W_R x_t,    a_{t,j} = ρ_t^T κ_j,    j ∈ C_t
```
> To allow the model to bypass state retrieval, we augment the candidate set with a null option `∅`, whose payload is fixed to zero, `A(∅) = 0`. Its query-dependent logit is `n_t = w_∅^T x_t + b_∅`. Let `C_tilde_t = C_t ∪ {∅}` denote the augmented candidate set:
```
s_{t,j} = a_{t,j}  (if j ∈ C_t),    s_{t,j} = n_t  (if j = ∅)
π_{t,j} = exp(s_{t,j}) / ∑_{r ∈ C_tilde_t} exp(s_{t,r})
```
> Given the routing probabilities, the candidate-state readouts are weighted and added to the current-state readout using the underlying recurrence's native state-read query `q_t`:
```
o_t = S_t q_t + ∑_{j ∈ C_tilde_t} π_{t,j} A(j) q_t
```
> This additive formulation preserves the original recurrent path and introduces state retrieval as an auxiliary branch without modifying the underlying recurrent update." (Section 2.2, Page 6)

### 2.4 Persistent Capability States (Safety State)
> "Beyond context-derived anchors, the routed state bank at each state-bearing layer can also host `J` learnable persistent capability states, `{P_j}_{j=1}^J`. Each persistent state has the same matrix-valued structure as a dynamic anchor, and its routing key `κ_{p_j}` is generated through the same state-aware metadata pathway defined in Equations (3)–(4). Unlike dynamic anchors, persistent states are learned parameters rather than checkpoints constructed from the current sequence. They are therefore available from the first text token without consuming positions in the input sequence.
```
C_t = V_t ∪ {p_1, ..., p_J},    A(p_j) = P_j
```
> Dynamic anchors and persistent states are scored by the same routing query and jointly normalized with the null option through a single softmax. Consequently, the router assigns each persistent state a token- and context-dependent contribution rather than adding it with a fixed weight at every position.
> In Safin-1, we instantiate this interface for safety. The layer-wise collection `{P_safe^(ℓ)}_ℓ` constitutes the Safety State. During specialization, we keep the language-model backbone frozen and optimize only this collection... Once learned, the Safety State can be attached to or removed from the routed state bank without changing the shared backbone parameters." (Section 2.3, Page 7)

### 2.5 Efficient Producer–Reader Implementation
> "The state-routing computation is organized into a recurrent producer and a fused state reader. The producer inherits the hardware-efficient chunkwise kernel of the underlying recurrent mixer (e.g. Gated DeltaNet). Following the I/O-aware principles of FlashAttention, the reader jointly tiles query tokens and state candidates, reuses each candidate tile across a block of queries, and fuses routing-score computation, normalization, and weighted state readout into a streaming reduction. This schedule avoids materializing either the dense token-to-candidate routing matrix or the substantially larger tensor of per-candidate matrix-valued readouts, reducing intermediate storage and HBM traffic. Safin-1 uses Top-4 routing by default." (Section 2.4, Page 7-8)

---

## 3. Experimental Results

### 3.1 0.8B Small-Scale Controlled Validation
- **Commonsense Reasoning (Table 1)**:
  - Gated DeltaNet: 40.11% macro-average
  - Log-Linear Gated DeltaNet: 39.99% macro-average
  - MARCH (Gated DeltaNet + MARCH): **41.48%** macro-average (+1.37 pp over GDN, beating 24-layer Transformer at 41.38%)
- **LongBench 12 Tasks (Table 2)**:
  - Gated DeltaNet: 11.90% macro-average
  - Log-Linear GDN: 12.51% macro-average
  - MARCH: **14.87%** macro-average (+25.0% relative gain over GDN)
- **In-Context Retrieval 6 Tasks (Table 3)**:
  - Gated DeltaNet: 19.20% macro-average
  - Log-Linear GDN: 20.52% macro-average
  - MARCH: **23.31%** macro-average (+13.6% relative gain over strongest GDN baseline)
- **Generality across Recurrent Backbones (Table 4)**:
  - GDN: 31.58% → w/ MARCH: **46.43%** (+47.0% relative gain on 6-Task NIAH Avg)
  - KDA: 31.64% → w/ MARCH: **41.38%** (+30.8% relative gain on 6-Task NIAH Avg)
  - GDN-2: 40.52% → w/ MARCH: **44.80%** (+10.6% relative gain on 6-Task NIAH Avg)

### 3.2 Large-Scale Scaling (4B & 35B-A3B Backbones, Table 8)
- **4B Scale (Qwen3.5 vs Safin-1)**:
  - MMLU-Pro: 58.27% → **67.55%** (+9.28 pp)
  - GPQA-Diamond: 57.58% → **58.59%** (+1.01 pp)
  - AIME 2025 (Avg@64): 55.57% → **63.59%** (+8.02 pp)
  - MBPP: 59.20% → **65.40%** (+6.20 pp)
  - Average Jailbreak ASR (↓): 7.25% → **6.65%** (-0.60 pp)
  - Over-Refusal XSTest ORR (↓): 8.80% → **8.60%** (-0.20 pp)
- **35B-A3B MoE Scale (Qwen3.5 vs Safin-1)**:
  - MMLU-Pro: 75.33% → **75.98%** (+0.65 pp)
  - GPQA-Diamond: 64.65% → **71.21%** (+6.56 pp)
  - AIME 2025 (Avg@64): 71.88% → **80.10%** (+8.22 pp)
  - MBPP: 77.60% → **78.60%** (+1.00 pp)
  - HumanEval: 89.02% → **90.24%** (+1.22 pp)
  - FORTRESS ASR (↓): 23.80% → **17.80%** (-6.00 pp)
  - Average Jailbreak ASR (↓): 6.89% → **5.89%** (-1.00 pp)

### 3.3 Safety State Specialization vs Matched LoRA (Table 9)
- **4B Scale**:
  - Average ASR (↓): Base 6.65% → Safety State **3.84%** (-42.3% relative reduction) vs LoRA 5.33%
  - XSTest Over-Refusal ORR (↓): Base 8.60% → Safety State **9.00%** (+0.40 pp) vs LoRA 13.60% (+5.00 pp)
  - Capability Retention Average (↑): Base 71.61% → Safety State **69.86%** vs LoRA 69.05%
- **35B-A3B Scale**:
  - Average ASR (↓): Base 5.89% → Safety State **2.81%** (-52.3% relative reduction) vs LoRA 3.04%
  - XSTest Over-Refusal ORR (↓): Base 17.60% → Safety State **19.60%** (+2.00 pp) vs LoRA 27.60% (+10.00 pp)
  - Capability Retention Average (↑): Base 82.07% → Safety State **78.64%** vs LoRA 79.36% (MATH 93.32% vs LoRA 93.12%, AIME 74.21% vs LoRA 72.61%)

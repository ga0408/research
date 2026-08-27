# Addressable Memory for Video World Models — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Addressable_Memory_for_Video_World_Models_2026_NVIDIA.md) / 원본: [arXiv:2608.07408](https://arxiv.org/abs/2608.07408)

## 1. Problem Formulation & Failures in Long-Horizon Video World Models (Sec. 2)

Autoregressive (AR) video world models generate video chunk-by-chunk while attending over a Key-Value (KV) cache of prior context. Visual persistence degrades rapidly when rollouts exceed the training context length due to two coupled failures:

1. **Addressability Failure (Out-of-Distribution RoPE Offsets)**:
   - Queries at timestamp `q` attending to past tokens at `t` produce relative offsets `δ_{q,k} = q - t`.
   - When rollouts extend beyond the training context `Δt_train`, positional embeddings (e.g., RoPE) are queried out-of-distribution (OOD), causing attention-based memory retrieval to fail even if past tokens are stored.

2. **Phase Cancellation Failure in Naive Cache Compression**:
   - Naively averaging RoPE-rotated keys across frame window `M`:
     ```
     K_bar^naive_f = (1 / M) * ∑_{m=1}^M R(θ_f * t_m) K^f_m
     ```
   - Rotating keys by different angles `θ_f * t_m` prior to averaging causes vectors at distant timestamps to point in opposite directions, cancelling high-frequency features and suppressing attention scores.

---

## 2. Virtual Position Assignment: WorldTrace Slot Indexing (Def. 1, Sec. 3.1)

To keep summary slots within the trained temporal RoPE range `[Δt_min, Δt_max]`, WorldTrace partitions the local attention window `L_attn` into a verbatim recent window `R` (`N_r` slots) and a compressed summary cache `S` (`N_s` slots), where `N_s + N_r = L_attn`.

### Definition 1 (WorldTrace Slot Indexing)
Given current query position `q`, local attention window `L_attn = N_s + N_r`, and summary cache size `N_s`, the virtual position of summary slot `s` is defined as:
```
t^v_s = q - (L_attn - 1 - s),    s = 0, ..., N_s - 1
```

- **In-distribution guarantee**: `t^v_s ∈ [t^v_min, t^v_max]` holds at any generation horizon `N`.
- **Slot-rank anchoring**: Offsets `(L_attn - 1 - s)` depend solely on slot rank `s`, preventing the **Block-relative collapse** where all distant slots map to the same capped minimum position.

---

## 3. Canonical Key Averaging: WorldTrace-Field Operator (Def. 2, Sec. 3.3)

WorldTrace-Field preserves temporal coherence for smooth continuation by averaging unrotated keys in canonical space and re-encoding at the slot's virtual position.

### Definition 2 (Canonical Key Averaging / WorldTrace-Field)
For each temporal head-dimension pair `f`, the compressed key at virtual position `t^v` is:
```
K^field_f(t^v) = R(θ_f * t^v) * (1 / M) * ∑_{m=1}^M R(-θ_f * t_m) K^f_{t_m}
```

- **Canonical domain unrotation**: Each key `K^f_{t_m}` is multiplied by `R(-θ_f * t_m)` to remove its original temporal RoPE rotation before averaging.
- **Re-encoding**: The unrotated canonical mean is re-rotated by `R(θ_f * t^v)` to match the slot rank's virtual position phase.
- **Mean Attention Preservation (Prop. 2)**: Re-encoding canonical averages preserves pre-softmax mean attention scores over source frames without phase cancellation.

---

## 4. Frozen Landmark Keys: WorldTrace-Landmark Operator (Sec. 3.4)

WorldTrace-Landmark optimizes for episodic recall by storing verbatim scene-entry frames at detected environmental transitions.

1. **Scene-Entry Detection**:
   - Computes cosine distance between consecutive canonical keys: `dist(K_can(t), K_can(t-1))`.
   - If distance exceeds threshold `τ`, frame `t` is marked as a scene-entry landmark.
2. **Frozen Canonical Key Assignment**:
   - Selected landmark frames store frozen canonical keys: `K^landmark_can = R(-θ * t_landmark) * K_{t_landmark}`.
   - At query time `q`, landmark slot `s` is dynamically re-rotated to slot `s`'s virtual position `t^v_s`:
     ```
     K^landmark_s(t^v_s) = R(θ_f * t^v_s) * K^landmark_can
     ```
3. **Slot Eviction Policy**:
   - Maintains up to `N_s` most recent landmarks; when capacity is exceeded, the oldest landmark is evicted.

---

## 5. LoopBench Benchmark & Experimental Results (Sec. 4)

### LoopBench Protocol
LoopBench evaluates visual persistence by generating closed-loop camera trajectories (`A → B → ... → A`) and comparing regenerated return-leg frames with initial reference frames `A` at geometrically aligned camera poses.

- **Geometries**: ABA (reversal), ABCA (triangle), ABCDA (square).
- **Metric**: Position-Aligned CLIP (PAC ↑), TempSSIM (↑), Local Scene Drift (↓).

### Key Results Summary
- **Temporal Coherence (MG2-1.3B, Horizon N = 48 AR chunks)**:
  - **Sliding Window Baseline**: TempSSIM = 0.472, Scene Drift = 0.0305
  - **Block-Relative Baseline**: TempSSIM = 0.530, Scene Drift = 0.0339
  - **WorldTrace-Field (Ours)**: TempSSIM = **0.545** (+15.5% relative vs sliding window at N = 16), Scene Drift = **0.0295**

- **Episodic Recall (LoopBench ABA Revisit)**:
  - **Sliding Window Baseline**: PAC = 0.638
  - **WorldTrace-Landmark (Ours)**: PAC = **0.833** (+19.5% improvement in episodic recall), faithfully reconstructing initial scene `A` after long detours without retraining.

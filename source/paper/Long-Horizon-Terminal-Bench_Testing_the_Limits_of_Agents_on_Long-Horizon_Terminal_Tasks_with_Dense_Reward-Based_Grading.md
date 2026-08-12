# Long-Horizon-Terminal-Bench — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Long-Horizon-Terminal-Bench_Testing_the_Limits_of_Agents_on_Long-Horizon_Terminal_Tasks_with_Dense_Reward-Based_Grading.md) / 원본: [arXiv:2607.08964](https://arxiv.org/abs/2607.08964)

## Subtask-based Grading (§2.2)

각 task는 deterministic grader가 컨테이너 내에서 rollout 종료 후 채점. 단일 pass/fail 대신 subtask 집합 {s₁, ..., s_K}에 대해 정규화 점수 rₖ ∈ [0, 1] 산출.

**Task reward 공식:**

```
R = Σₖ₌₁ᴷ (wₖ · rₖ) / Σₖ₌₁ᴷ wₖ
```

- wₖ: non-negative 가중치 (기본 모두 동일, 최종 goal에 더 높은 가중치 부여 가능)
- 성공 기준: R ≥ τ (relaxed threshold, 예: τ = 0.95)

**세 가지 subtask 유형:**

1. **Binary subtasks**: 환경 상태에 대한 Boolean 조건 (unit test 통과 여부, 서비스 응답 등). rₖ ∈ {0, 1}
2. **Continuous/thresholded subtasks**: 정량 목표 (metric 재현, speedup 달성 등). 허용 오차 내 1.0, 오차 증가에 따라 선형 감소 → 0
3. **Episode-aggregating subtasks**: 캠페인형 task (게임, 반복 audit)에서 에피소드 집계. 성공 플래그 비율 또는 simulator 평균 정규화 보상

## Task Construction (§2.3)

각 task의 구성 요소:
- reproducible public asset-generation scripts
- weak baseline implementation
- official gold solution (hidden verifier에서 1.0 달성 필요)
- multi-step solve.sh
- hidden verifier

**Public vs Hidden 체계:**
- Public checks: CLI 동작, 파일 포맷, 단순 예제만 검증 → **낮은 가중치, agent에게 visible**
- Hidden stress cases: 더 어려운 입력 동적 생성 (nested manifests, gzip+base64 wrappers, renamed fields, missing values, injected noise, rotated/cropped images, anomalous frames, alternative coordinate conventions) → **높은 가중치, agent에게 invisible**

난이도 교정: Deepseek-V4-Pro를 1.5시간 예산으로 반복 실행하여 해결 가능하지만 어렵도록 조정. 120 candidate tasks에서 46개 선별.

## Main Results (§3.1)

| Model | Pass@1 R≥0.9 | Pass@1 R≥0.95 | Pass@1 R≥1.0 | Mean R |
| --- | --- | --- | --- | --- |
| GPT-5.5 | 17.4% (8) | 15.2% (7) | 10.9% (5) | 0.37 |
| MiniMax M3 | 13.0% (6) | 6.5% (3) | 2.2% (1) | 0.30 |
| Kimi K2.7 Code | 13.0% (6) | 6.5% (3) | 2.2% (1) | 0.31 |
| DeepSeek V4 Pro | 8.7% (4) | 6.5% (3) | 0.0% (0) | 0.29 |
| Qwen3.7 Max | 6.5% (3) | 4.3% (2) | 0.0% (0) | 0.27 |
| Doubao Seed 2.1 Pro | 4.3% (2) | 4.3% (2) | 0.0% (0) | 0.27 |
| Gemini 3.1 Pro | 6.5% (3) | 4.3% (2) | 0.0% (0) | 0.31 |
| GLM 5.1 | 8.7% (4) | 4.3% (2) | 0.0% (0) | 0.31 |
| GPT-5.3 Codex | 6.5% (3) | 4.3% (2) | 0.0% (0) | 0.35 |
| GPT-5.4 | 6.5% (3) | 2.2% (1) | 0.0% (0) | 0.27 |
| GLM 5.2 | 4.3% (2) | 2.2% (1) | 0.0% (0) | 0.32 |
| Qwen3.6 Plus | 6.5% (3) | 2.2% (1) | 0.0% (0) | 0.31 |
| Hy3 | 4.3% (2) | 2.2% (1) | 0.0% (0) | 0.39 |
| Kimi K2.6 | 4.3% (2) | 0.0% (0) | 0.0% (0) | 0.25 |
| Grok 4.20 | 2.2% (1) | 0.0% (0) | 0.0% (0) | 0.08 |

## Cost Analysis (§3.3, Table 1)

| Model | Tokens/task (M) | Episodes/task | Time/task (min) | Cost/task ($) |
| --- | --- | --- | --- | --- |
| GPT-5.5 | 4.16 | 208 | 72.9 | 21.46 |
| MiniMax M3 | 20.20 | 314 | 90.0 | 6.13 |
| Kimi K2.7 Code | 8.54 | 183 | 85.4 | 8.31 |
| DeepSeek V4 Pro | 14.45 | 321 | 83.6 | 6.32 |
| Qwen3.7 Max | 6.13 | 218 | 83.5 | 7.78 |
| Doubao Seed 2.1 Pro | 5.80 | 183 | 91.7 | 5.16 |
| Gemini 3.1 Pro | 3.55 | 148 | 85.0 | 7.61 |
| GLM 5.1 | 5.84 | 120 | 92.6 | 5.13 |
| GPT-5.3 Codex | 4.57 | 299 | 80.7 | 8.20 |
| GLM 5.2 | 8.43 | 195 | 89.3 | 11.93 |
| Qwen3.6 Plus | 8.67 | 194 | 88.6 | 4.47 |
| Hy3 | 17.21 | 258 | 91.3 | 2.47 |
| GPT-5.4 | 10.90 | 302 | 79.3 | 27.57 |
| Kimi K2.6 | 10.27 | 188 | 92.5 | 9.94 |
| Grok 4.20 | 16.23 | 288 | 69.5 | 20.63 |
| **Average** | **9.66** | **228** | **85.1** | **10.21** |

## Failure Mode Analysis (§3.4–3.5)

**종료 원인 분포 (전체 unresolved runs 기준):**
- Timeout (90분 예산 만료): **79%** (518/660) — mean reward 0.10~0.35
- Early exit (agent 자발적 종료): **19%** — 일부는 false finish
- Harness error: **3%**

**False finish 정의:** R ≥ 0.75에서 agent가 자발적 종료 (hidden verifier 미충족). 14건 식별.
- 예: Kimi K2.7 Code가 duckdb-optimizer-closure에서 R=0.92에서 정지
- 예: 7개 모델이 apex-law433-matter에서 R=0.80~0.87에서 정지 (약 20분 잔여)

**핵심 통찰:**
- 주요 병목은 local execution correctness가 아닌 **long-horizon completion**
- Agent들이 많은 국소적 올바른 행동을 연결할 수 있지만, horizon 만료 전 최종 산출물 완성 불가
- Dense reward가 없으면 이러한 패턴 구분 불가 (binary grading은 모두 동일 failure로 취급)

## Dense Reward의 가치 (§3.2)

- R ≥ 1.0 기준: 15개 모델 중 10개가 0 task pass → binary grading 시 대다수 동점
- 전체 15×46 runs 중:
  - R < 0.05 (무의미 진전): 227 runs (32.9%)
  - 0.05 ≤ R < 0.95 (부분 진전): 433 runs (62.8%) → binary에서는 모두 failure
  - R ≥ 0.95 (pass): 30 runs (4.3%)
- Near-misses (0.75 ≤ R < 0.95): 73건 → pass(30건)의 2배 이상
- Pass rate와 mean reward는 Spearman ρ = 0.56 (moderate correlation, 다른 랭킹 가능)

## Task Distribution (§2.4, Figure 2)

| Category | Count | % |
| --- | --- | --- |
| Software & reverse engineering | 7 | 15% |
| Earth, climate & energy | 6 | 13% |
| Multimodal & imaging analysis | 6 | 13% |
| Scientific computing & simulation | 6 | 13% |
| Research reproduction & ML | 5 | 11% |
| Systems, performance & security | 5 | 11% |
| APEX professional workflows | 4 | 9% |
| Interactive games | 4 | 9% |
| Logic & constraint puzzles | 3 | 7% |
| **Total** | **46** | **100%** |

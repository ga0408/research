> [paper] https://arxiv.org/abs/2607.08964

# Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading

## Summary & Outline

**한 줄 요약:** 46개의 containerized long-horizon terminal task로 구성된 벤치마크로, subtask 기반 dense reward 채점을 통해 15개 frontier 모델의 장시간 자율 실행 능력을 평가한 결과, 최강 모델도 15.2% pass@1(τ=0.95)에 그쳐 long-horizon completion이 핵심 병목임을 시사.

**논문 구조 outline:**
1. Introduction — 기존 벤치마크의 한계(binary 평가, sparse reward)와 long-horizon 과제의 필요성
2. Long-Horizon-Terminal-Bench — task formulation, subtask-based grading, dataset construction, task composition
3. Experiments — main results, dense reward 가치, cost analysis, failure pattern 분석, false finish 분석
4. Related Work — terminal/SWE 벤치마크, long-horizon 자율성 측정, process reward, agent harness
5. Conclusion
6. Appendix — 46개 task 전체 목록

## Problem & Motivation

- **연구 배경:** LLM 기반 AI agent는 단기·단순 task에서 빠르게 발전하고 있으나, 실제 전문가 워크플로는 수백 step, 수십 분~수시간의 지속적 실행을 요구함. 논문 재현, ML 파이프라인 디버깅, 멀티모달 데이터 감사 등이 대표적.
- **풀고자 하는 문제:** long-horizon terminal task에서의 agent 능력 평가. 기존 벤치마크는 terminal task를 binary(pass/fail)로만 채점하여 부분 진전을 포착하지 못함.
- **기존 접근의 한계:**
  - SWE-Bench, Terminal-Bench 등은 수분~최대 1시간 내 완료되는 단기 task 위주 → real workflow의 난이도 과소평가
  - outcome-only(binary) grading은 매우 sparse한 reward signal 제공 → 대부분 단계를 완수하고 마지막에 실패한 agent와 처음부터 실패한 agent를 동일하게 "실패"로 취급
  - METR의 time-horizon 분석 등은 horizon 길이를 일급 변수로 다루지만 여전히 binary 성공 신호에 의존

## Contributions

1. **벤치마크 설계 (데이터셋):** 9개 카테고리, 46개 long-horizon terminal task로 구성된 LHTB(Long-Horizon-Terminal-Bench) 공개. 각 task는 containerized terminal 환경에서 수십 분~수시간 실행
2. **평가 방법론 (dense reward grading):** 각 task를 의미론적 subtask로 분해하고 deterministic grader가 부분 점수를 부여하는 subtask-based grading 도입. binary pass/fail이 놓치는 부분 진전을 포착
3. **대규모 실증 평가:** 15개 frontier 모델 평가 및 장시간 실행 통계(평균 9.9M tokens, 231 episodes, 85.3 min/task) 제공
4. **실패 패턴 분석:** timeout, early exit, false finish 등의 패턴을 dense reward 기반으로 정량 분석하여 long-horizon completion이 핵심 병목임을 시사

## Method

### 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                 Long-Horizon-Terminal-Bench                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Task Construction          Agent Execution         Grading │
│                                                              │
│  ┌─────────────┐           ┌──────────────┐    ┌─────────┐  │
│  │ Real-world   │           │ Harbor +     │    │Hidden   │  │
│  │ workflow     │──build──▶ │ Terminus-2   │──▶ │verifier │  │
│  │              │           │ agent harness│    │         │  │
│  │ Docker image │           │              │    │Subtask  │  │
│  │ + solve.sh   │           │ 100s of      │    │scoring  │  │
│  │ + hidden     │           │ episodes     │    │         │  │
│  │   verifier   │           │ over 90 min  │    │R = Σwr/Σw│ │
│  └─────────────┘           └──────────────┘    └─────────┘  │
│                                                              │
│  Public checks         Hidden stress tests                   │
│  (low weight, visible)  (high weight, invisible)             │
│  - CLI behavior         - nested manifests                   │
│  - file formats         - gzip+base64 wrappers                │
│  - simple examples      - renamed fields                     │
│                          - missing values / noise            │
│                          - rotated/cropped images            │
│                          - anomalous frames                  │
│                          - alt coordinate conventions        │
└─────────────────────────────────────────────────────────────┘
```

### Task Formulation

LHTB task는 Terminal-Bench formulation을 따름: (1) 자연어 instruction, (2) Docker image, (3) task config file, (4) oracle/simulator. 차이점은 **각 subtask가 수십~수백 번의 개별 조작과 수분~수시간의 작업을 요구**하도록 설계되었다는 것.

### Subtask-based Grading

각 task는 K개의 subtask {s₁, ..., s_K}로 분해되며, deterministic grader가 각 subtask에 대해 정규화 점수 rₖ ∈ [0, 1] 산출. 전체 task reward는 가중 평균:

```
R = Σ(wₖ · rₖ) / Σ wₖ
```

세 가지 subtask 유형:

| 유형 | 설명 | 예시 | 점수 |
| --- | --- | --- | --- |
| Binary | 환경 상태에 대한 Boolean 조건 | unit test 통과, 서비스 응답 | rₖ ∈ {0, 1} |
| Continuous/Thresholded | 정량 목표, 허용 오차 내 1.0→선형 감소→0 | metric 재현, speedup 달성 | rₖ ∈ [0, 1] |
| Episode-aggregating | 캠페인형 task의 에피소드 집계 | 게임 승리 비율, simulator 평균 보상 | rₖ ∈ [0, 1] |

성공 기준: R ≥ τ (relaxed: τ = 0.95, perfect: τ = 1.0). 모든 보고 메트릭에 task reward R 사용.

상세 발췌 → [excerpt](../source/paper/Long-Horizon-Terminal-Bench_Testing_the_Limits_of_Agents_on_Long-Horizon_Terminal_Tasks_with_Dense_Reward-Based_Grading.md)

### Dataset Construction

120개 candidate task에서 품질 필터링 후 46개 선별. 난이도 교정은 Deepseek-V4-Pro를 1.5시간 예산으로 반복 실행하여 "어렵지만 원칙적으로 해결 가능" 수준으로 조정. 각 task는:
- reproducible public asset-generation scripts
- weak baseline implementation
- official gold solution (hidden verifier 1.0 달성 필요)
- multi-step solve.sh
- deliberately broken terminal-only project (agent가 코드 검사·명령 실행·artifact 분석 후 pipeline 수리)

**Overfitting 방지:** public checks는 CLI 동작·파일 포맷·단순 예제만 검증하고 낮은 가중치. 대부분 보상은 hidden stress test(gzip+base64, nested manifests, renamed fields, noise, rotated images 등)에 할당됨.

### Task Composition

46개 task는 9개 카테고리로 분포 (Figure 2):

| Category | Tasks | % |
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

단일 카테고리가 지배하지 않으며, 게임·멀티모달 감사가 가장 많고 나머지는 과학·엔지니어링 워크플로에 걸쳐 분산됨.

## Experiments & Results

### Benchmark Datasets

- **LHTB 자체:** 46개 containerized terminal task, 9개 카테고리
- 각 task는 90분 timeout, Harbor framework + Terminus-2 agent harness로 실행
- 비교 벤치마크: Terminal-Bench 2 (대부분 20분 내 완료), SWE-Bench

### Setup

- **Models (15):** GPT-5.5, GPT-5.4, GPT-5.3 Codex, DeepSeek V4 Pro, Gemini 3.1 Pro, GLM 5.1, GLM 5.2, Kimi K2.6, Kimi K2.7 Code, MiniMax M3, Qwen3.7 Max, Qwen3.6 Plus, Doubao Seed 2.1 Pro, Hy3, Grok 4.20
- **Agent harness:** Harbor framework + Terminus-2 (GPT-5.3은 Codex harness 사용)
- **Metrics:** pass@1 (R ≥ 0.95), mean normalized reward, episodes/task, time/task, token/dollar cost
- **Budget:** 90분 per task

### Results

**리더보드 (R ≥ 0.95 기준):**

| Tier | Model | Pass@1 (R≥0.95) | Mean R |
| --- | --- | --- | --- |
| Top | GPT-5.5 | 15.2% (7/46) | 0.37 |
| 2nd | MiniMax M3, Kimi K2.7 Code, DeepSeek V4 Pro | 6.5% (3/46) each | 0.30, 0.31, 0.29 |
| Mid | Qwen3.7 Max, Doubao Seed 2.1 Pro, Gemini 3.1 Pro, GLM 5.1, GPT-5.3 Codex | 4.3% (2/46) each | 0.27~0.35 |
| Low | GPT-5.4, GLM 5.2, Qwen3.6 Plus, Hy3 | 2.2% (1/46) each | 0.27~0.39 |
| Zero | Kimi K2.6, Grok 4.20 | 0.0% (0/46) | 0.25, 0.08 |

- 최강 GPT-5.5도 15.2%에 불과 → 매우 큰 개선 여분(headroom)
- R ≥ 1.0에서는 대다수(15개 중 10개)가 0% pass → binary 평가 시 구분 불가

**실행 규모 (평균):** 9.66M tokens, 228 episodes, 85.1 min, \$10.21/task → 기존 terminal/code 벤치마크 대비 압도적 규모

상세 결과 및 cost 표 → [excerpt](../source/paper/Long-Horizon-Terminal-Bench_Testing_the_Limits_of_Agents_on_Long-Horizon_Terminal_Tasks_with_Dense_Reward-Based_Grading.md)

### Findings & Implications

**1. Dense reward가 모델 랭킹에 필수 (§3.2)**
- 전체 690 runs(15×46) 중: R < 0.05 = 32.9%, 0.05 ≤ R < 0.95 = 62.8%, R ≥ 0.95 = 4.3%
- 62.8%의 부분 진전 run이 binary 평가에서는 모두 "실패"로 처리됨
- Near-misses (0.75 ≤ R < 0.95): 73건 → pass(30건)의 2.4배. Kimi K2.6은 pass 0건이지만 near-miss 5건 → binary로는 무능으로 보이나 실제로는 임박 달성

**2. 주요 병목은 long-horizon completion (§3.4)**
- Unresolved runs의 79%가 timeout (90분 예산 만료 시 agent가 아직 작업 중)
- Timeout run들의 mean reward는 0.10~0.35 → 완료에 한참 못 미침
- Agent들은 많은 국소적 올바른 행동을 연결할 수 있지만, horizon 만료 전 최종 산출물 완성 불가
- → short-horizon 벤치마크는 "올바르게 행동할 수 있는가"를 측정하지만, LHTB는 "긴 horizon을 예산 관리하며 완수할 수 있는가"를 추가 측정

**3. False finish와 약한 자기 검증 (§3.5)**
- 자발적 종료 124건 중 14건이 R ≥ 0.75에서 정지 (false finish)
- 예: Kimi K2.7 Code가 duckdb-optimizer-closure에서 R=0.92 정지, 7개 모델이 apex-law433-matter에서 R=0.80~0.87 정지 (약 20분 잔여)
- 현재 agent들은 완료를 체계적으로 과대 추정하고 최종 검증에 투자不足
- Early exit 평균 reward로 모델 간 차이 구분 가능: Kimi K2.7 Code(0.51), MiniMax M3(0.42) vs Kimi K2.6(0.11)

**4. Cost–reward 분석 (§3.3)**
- 평균 \$10.2/task (\$2.5~\$28 범위)
- Pareto frontier: GPT-5.5(최고 성능, \$21/task) — MiniMax M3, DeepSeek V4 Pro(중간 성능, ~\$6) — Hy3(최저 비용, \$2.5)
- GPT-5.4는 GPT-5.5보다 비싸면서(\$28 vs \$21) 성능이 낮음 → 더 약한 모델이 더 많은 episode(302 vs 208)로 더 높은 비용 발생
- 더 높은 추론 비용 지출만으로는 long-horizon 성능 보장 불가

## Analysis

### Strengths & Significance

- **Dense reward 설계의 실증적 가치:** binary 평가가 무너지는 난이도에서 dense reward가 모델 간 의미 있는 차이를 포착함을 명확히 입증. 62.8%의 부분 진전 run이 binary에서는 동일한 "실패"로 처리되는 문제를 수치로 제시
- **다양한 도메인:** 9개 카테고리·21개 고수준 카테고리에 걸친 46개 task로 단일 도메인 편향 최소화. EDA(chip design), SLAM, 기후 과학, DICOM radiology 등 실제 전문가 워크플로 포함
- **Overfitting 방지 설계:** public(low weight) vs hidden(high weight) 체계로 hard-coding 또는 superficial patching이 높은 점수를 얻지 못하도록 설계
- **실패 패턴 정량화:** timeout(79%) vs early exit(19%) vs harness error(3%) 분해, false finish 개념 도입 → 단순한 성능 비교를 넘어 "왜 실패하는가"에 대한 통찰 제공
- **실행 규모의 현실성:** 평균 9.9M tokens, 85.3 min/task로 real-world long-horizon workload의 비용·시간 규모를 realistic하게 반영

### Limitations

- **표본 크기:** 46개 task는 9개 카테고리에 분산되어 카테고리당 3~7개 → 카테고리별 통계적 강도 제한
- **단일 harness 의존:** Terminus-2 harness를 기본 사용(GPT-5.3만 Codex harness). harness 간 영향을 분리하지 못함 [48]
- **난이도 교정의 주관성:** Deepseek-V4-Pro 기준으로 교정했으나, 이 모델 자체의 능력 변화에 따라 난이도가 변할 수 있음
- **Hidden verifier의 결정론적 한계:** 환경 상태 기반 deterministic grader는 강건하지만, 창의적 해법이나 대안적 접근을 penalty할 수 있음
- **시간 제한:** 90분 고정. 모델 추론 속도 개선에 따른 상대적 난이도 변화를 반영하지 못함

### Future Work / Improvements

- **자기 검증 능력 평가:** false finish 패턴을 systematic으로 측정하는 세부 메트릭(예: stopping calibration) 개발
- **Context/token 관리 전략 비교:** long-horizon에서 context 압축·compaction 전략이 성능에 미치는 영향 분석
- **Harness-모델 상호작용:** 다양한 harness(OpenHands, SWE-agent 등) 간 비교로 모델 능력과 harness 효과 분리
- **난이도 스펙트럼 확장:** 현재 대부분 "Hard"로 분류 → 쉬운 task를 더 포함하여 난이도-성능 곡선 도출
- **Process reward 활용:** dense reward를 평가뿐 아니라 agent 학습 시 intermediate signal로 활용하는 방안

## References

- Project page: https://zli12321.github.io/LHTB/
- arXiv: https://arxiv.org/abs/2607.08964
- Terminal-Bench [22]: arXiv:2601.11868
- APEX-Agents [40]: arXiv:2601.14242
- METR time-horizon [14]: arXiv:2503.14499
- RE-Bench [43]: arXiv:2411.15114
- Harbor framework [10]: https://doi.org/10.5281/zenodo.20953922

# Chain-of-Experience for Continual LLM Improvement — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Chain-of-Experience_for_Continual_LLM_Improvement_2026_Bytedance_Seed.md) / 원본: [arXiv:2608.18027](https://arxiv.org/abs/2608.18027)

---

## 1. Abstract & Problem Formulation

### 1.1 Abstract
> "Humans continuously learn from experience, whereas conventional large language model (LLM) evaluations ignore the models’ ability to improve through inference-time interaction. In this paper, we study how LLMs learn from iterative experience at test time, a setting we refer to as Chain-of-Experience (CoE), where models accumulate experiential traces through iterative interactions with self or environmental feedback to form a continual improvement loop beyond zero-shot inference. We instantiate CoE with diverse feedback mechanisms, including model self-feedback and environmental signals such as correctness or public coding test pass rates, and evaluate across math, coding, and knowledge domains using 8 LLMs, including GPT-5, Gemini-2.5 Pro, Claude-4.5 Sonnet. Our study shows that leveraging iterative experience consistently outperforms feedback-free baselines, achieving substantial gains with self feedback alone, alongside a 5.6% overall improvement and 19% lower API cost across tasks and models. We further show that combining complementary feedback channels (e.g., model and correctness signals) yields additional gains, and that CoE delivers higher accuracy per token than existing test-time strategies. We observe a positive correlation between LLM base ability and improvement capacity, and show that models remain robust under weak or spurious feedback, with different feedback contributing to distinct improvement aspects and most gains emerging early in the iterations."

---

## 2. Chain-of-Experience (CoE) Formalization & Feedback Spectrum

### 2.1 Sequential Formulation
기존 QA 패러다임:
636a \sim P(A \mid Q)636

환경 피드백 변수 $ 도입:
636f \sim P'(F \mid Q, a)636

Chain-of-Experience (CoE) 순차적 의사결정 프로세스:
636a_t \sim P(a_t \mid Q, e_0, e_1, \dots, e_{t-1})636
여기서  = (a_i, f_i)$는 $번째 반복(iteration)에서의 시도(action $)와 피드백($) 쌍으로 구성된 경험(experience) 튜플이다.

전체 피드백 적용 수식:
636a_t \sim P(a_t \mid Q, (a_0, f_0), (a_1, f_1), \dots, (a_{t-1}, f_{t-1}))636

### 2.2 4-Type Feedback Spectrum
1. **No feedback ( = \emptyset$)**:
   636a_t \sim P(a_t \mid Q, a_0, a_1, \dots, a_{t-1})636
   외부 신호 없이 모델 자체의 과거 시도 궤적에 대한 내적 반성(self-reflection)에만 의존.
2. **Execution feedback ( = \mathcal{E}(Q, a_i)$)**:
   인터프리터나 단위 테스트 환경에서 코드를 실행하여 얻는 런타임 오류 로그, 테스트 케이스 통과/실패 결과.
3. **Model feedback ( = \mathcal{M}_{fb}(Q, a_i)$)**:
   보조 LLM 또는 자기 자신($\mathcal{M}_{fb}$)이 생성하는 자연어 비평(critique), 점수, 또는 구조화된 평가.
4. **Correctness feedback ( = \mathbf{1}\{a_i 	ext{ is correct}\} \in \{0, 1\}$)**:
   도메인 검증기(verifier) 또는 오라클이 제공하는 이진 정답 여부 신호 (상한선 reference 역할).

---

## 3. Core Benchmark Results (Table 3 & Table 5)

### Table 3: Average Performance Comparison (%) Across LLMs on 6 Benchmarks
> "Average performance comparison (%) across different LLMs on different datasets. For baselines, ICL, ACE, DC stands for few-shot in-context learning, agentic context engineering, and dynamic cheatsheet, respectively."

| Method | AIME 2025 | LiveCodeBench (V6) | LiveBench (Code) | OmniMath | GPQA Diamond | EvaLearn | Average |
|---|---|---|---|---|---|---|---|
| **ICL** | 71.83% | 62.50% | 65.46% | 53.12% | 78.45% | 40.99% | 62.06% |
| **ACE** | 71.98% | 66.94% | 69.38% | 50.33% | 76.58% | 42.54% | 62.96% |
| **DC** | 73.33% | 63.59% | 68.58% | 48.64% | 79.56% | 42.68% | 62.73% |
| **w/o Feedback (NF)** | 77.78% | 72.57% | 60.16% | 65.17% | 80.02% | 44.91% | 66.77% |
| **Reasoning-high** | 69.05% | 70.63% | 55.46% | 61.81% | 76.21% | 39.58% | 62.12% |
| **Reasoning-low** | 60.48% | 61.03% | 55.38% | 50.60% | 72.92% | 29.34% | 54.96% |
| **Binary-Executor** | – | 72.90% | 71.65% | – | – | – | – |
| **Self Feedback (SF)** | 82.22% | 75.69% | 69.94% | 67.52% | 81.03% | 51.73% | **71.36%** |
| **Correctness/Executor (CEF)** | 89.05% | 74.50% | 75.78% | 79.61% | 99.52% | 57.05% | **79.25%** |

### Table 5: Token Complexity vs. Accuracy
> "Token complexity vs. accuracy across methods. CEF: Correctness/Executor Feedback; SF: Self Feedback; NF: No Feedback; DC: Dynamic CheatSheet. Token counts are aggregated across all iterations."

| Dataset | Method | Tokens | Acc (%) |
|---|---|---|---|
| **AIME 2025** | CEF | 108,734 | 84.6% |
| | SF | 108,231 | 83.8% |
| | NF | 106,825 | 74.1% |
| | DC | 11,233 | 74.7% |
| **OmniMath** | CEF | 176,412 | 74.2% |
| | SF | 175,806 | 72.1% |
| | NF | 173,944 | 66.8% |
| | DC | 16,904 | 63.9% |
| **LiveCodeBench (V6)** | CEF | 224,118 | 72.6% |
| | SF | 223,441 | 71.2% |
| | NF | 221,550 | 68.0% |
| | DC | 20,771 | 66.4% |

---

## 4. Dual Feedback & Experience Selection (Table 1)

### Table 1: Dual Feedback and Principled Experience Selection Results (Claude 4.5 Sonnet)
> "Dual feedback combines model feedback with correctness (math) or executor (code) signals. Memory-based methods (DC, SimpleMem) are applied within-task with no cross-task leakage. Acc: best accuracy (%) over 20 iterations; Best R: iteration achieving best performance."

| Setting | AIME 2025 Acc | AIME Best R | LiveBench (Code) Acc | LiveBench Best R | OmniMath Acc | OmniMath Best R |
|---|---|---|---|---|---|---|
| **Dual (Model + Corr/Exec)** | **76.7%** | R19 | **81.2%** | R15 | 73.5% | R17 |
| **Correctness / Executor** | 70.0% | R13 | 78.1% | R15 | **74.5%** | R17 |
| **Binary Executor** | – | – | 71.9% | R13 | – | – |
| **Model Feedback Only** | 60.0% | R6 | 57.8% | R17 | 50.5% | R9 |
| **Model + DC** | 50.0% | R8 | 51.6% | R15 | 46.0% | R10 |
| **Model + SimpleMem** | 56.7% | R6 | 54.7% | R17 | 49.5% | R12 |

---

## 5. Robustness under Spurious Feedback (Table 2)

### Table 2: Spurious Feedback & Selective Majority Voting (SelMV)
> "The best performance over 20 iterations under constant 'correct' or 'incorrect' feedback. Selective majority voting (SelMV) helps LLMs maintain performance. Results are averaged over 3 runs."

| Feedback Setting | AIME 2025 (GPT-5 mini) | AIME 2025 (o4-mini) | GPQA Diamond (GPT-5 mini) | GPQA Diamond (o4-mini) |
|---|---|---|---|---|
| **Self Feedback** | 93.3% | 91.1% | 79.9% | 78.8% |
| **SelMV Self** | 91.1% | 88.9% | 80.4% | 79.5% |
| **All Correct (Spurious)** | 90.0% | 73.3% | 79.3% | 75.8% |
| **SelMV Correct** | 93.3% | 73.3% | 79.3% | 76.3% |
| **All Incorrect (Spurious)** | 91.7% | 83.3% | 79.3% | 72.7% |
| **SelMV Incorrect** | 89.7% | 86.7% | **82.8%** | 77.8% |

---

## 6. Improvement Pattern Breakdown & Judge Criteria

### 6.1 Improvement Attribution Factors (Figure 6 & Appendix F.4)
- **Feedback Fidelity (47.7%)**: 제공된 피드백의 구체적 지적 사항이나 오류 위치를 직접 반영하여 수정한 경우.
- **Specification Recall (코딩 30.0%)**: 문제 지문 및 출력 포맷 제약조건(schema, 'FINAL ANSWER:' wrapper, omitted constraint)을 재인식하여 정렬한 경우.
- **Self Reflection**: 외부 피드백 의존 없이 모델 자체의 내적 논리 재검토 및 오류 수정을 통해 정답으로 전환한 경우.
- **Random / Stochastic**: 명확한 인과관계 없이 표면적 어휘 변경이나 스타일 재배치를 거치며 정답이 된 경우.

### 6.2 Human-GPT Agreement (Table 4)
| Category | Agreement (%) | Cohen’s $\kappa$ |
|---|---|---|
| **Feedback Fidelity** | 84.0% | 0.81 |
| **Self Reflection** | 72.0% | 0.71 |
| **Specification Recall** | 80.0% | 0.78 |
| **Random** | 68.0% | 0.63 |
| **Overall** | **76.0%** | **0.768** |

### 6.3 Base Capacity vs. Learning Ability
- 개선 능력 공식:
  636\Delta_M = rac{S_{\max} - S_{	ext{base}}}{1 - S_{	ext{base}}}636
- Pearson 상관계수 ($):
  - LiveBench (Code):  = 0.97$
  - LiveCodeBench (V6):  = 0.83$
  - EvaLearn:  = 0.37$
  - AIME 2025:  = 0.33$
  - GPQA-Diamond:  = 0.28$
  - OmniMath:  = 0.24$
  - 전체 평균: ** = +0.50* (Base 모델의 추론 능력이 높을수록 피드백 소화 및 개선 역량이 유의미하게 큼).

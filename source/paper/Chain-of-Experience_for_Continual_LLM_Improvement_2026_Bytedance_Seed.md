# Chain-of-Experience for Continual LLM Improvement — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Chain-of-Experience_for_Continual_LLM_Improvement_2026_Bytedance_Seed.md) / 원본: [arXiv:2608.18027](https://arxiv.org/abs/2608.18027)

---

## 1. 논문 개요 및 초록 (Abstract & Problem Definition)

### 1.1 Abstract 발췌
> "Humans continuously learn from experience, whereas conventional large language model (LLM) evaluations ignore the models’ ability to improve through inference-time interaction. In this paper, we study how LLMs learn from iterative experience at test time, a setting we refer to as Chain-of-Experience (CoE), where models accumulate experiential traces through iterative interactions with self or environmental feedback to form a continual improvement loop beyond zero-shot inference. We instantiate CoE with diverse feedback mechanisms, including model self-feedback and environmental signals such as correctness or public coding test pass rates, and evaluate across math, coding, and knowledge domains using 8 LLMs, including GPT-5, Gemini-2.5 Pro, Claude-4.5 Sonnet. Our study shows that leveraging iterative experience consistently outperforms feedback-free baselines, achieving substantial gains with self feedback alone, alongside a 5.6% overall improvement and 19% lower API cost across tasks and models. We further show that combining complementary feedback channels (e.g., model and correctness signals) yields additional gains, and that CoE delivers higher accuracy per token than existing test-time strategies. We observe a positive correlation between LLM base ability and improvement capacity, and show that models remain robust under weak or spurious feedback, with different feedback contributing to distinct improvement aspects and most gains emerging early in the iterations."

---

## 2. Chain-of-Experience (CoE) 정식화 및 피드백 스펙트럼

### 2.1 순차적 의사결정 프로세스 수식화 (Sequential Formulation)

- **기존 질의응답 (Zero-shot QA)**:
  $$a \sim P(A \mid Q)$$

- **환경 피드백 변수 $F$ 도입**:
  $$f \sim P'(F \mid Q, a)$$
  여기서 $P'$는 코드 실행 인터프리터, 에이전트의 내부 월드 모델, 외부 환경 또는 보조 언어 모델에 의해 모델링된다.

- **Chain-of-Experience (CoE) 순차 생성 과정**:
  $$a_t \sim P(a_t \mid Q, e_0, e_1, \dots, e_{t-1})$$
  여기서 $e_i = (a_i, f_i)$는 $i$번째 반복(iteration)에서의 시도($a_i$)와 피드백($f_i$) 쌍으로 구성된 경험(experience) 튜플이다.

- **전체 피드백 이력 조건부 생성**:
  $$a_t \sim P(a_t \mid Q, (a_0, f_0), (a_1, f_1), \dots, (a_{t-1}, f_{t-1}))$$

### 2.2 4단계 피드백 스펙트럼 정의 (Feedback Spectrum)

1. **No Feedback ($f_i = \emptyset$)**:
   $$a_t \sim P(a_t \mid Q, a_0, a_1, \dots, a_{t-1})$$
   외부 신호 없이 모델 자체의 과거 시도 궤적에 대한 내적 반성(self-reflection)에만 의존.
2. **Execution Feedback ($f_i = \mathcal{E}(Q, a_i)$)**:
   인터프리터나 단위 테스트 환경에서 코드를 실행하여 얻는 런타임 오류 로그, 실행 트레이스, 공개 단위 테스트 통과율.
3. **Model Feedback ($f_i = \mathcal{M}_{fb}(Q, a_i)$)**:
   보조 LLM 또는 자기 자신($\mathcal{M}_{fb}$)이 생성하는 자연어 비평(critique), 점수, 또는 구조화된 평가.
4. **Correctness Feedback ($f_i = \mathbf{1}\{a_i \text{ is correct}\} \in \{0, 1\}$)**:
   도메인 검증기(verifier) 또는 오라클이 제공하는 이진 정답 여부 신호 (이론적 상한선 reference 역할).

---

## 3. 주요 벤치마크 실험 결과 (Table 3 & Table 5)

### Table 3: 6개 벤치마크 대상 LLM 평균 성능 비교 (%)
> "Average performance comparison (%) across different LLMs on different datasets. For baselines, ICL, ACE, DC stands for few-shot in-context learning, agentic context engineering, and dynamic cheatsheet, respectively."

| 방법론 (Method) | AIME 2025 | LiveCodeBench (V6) | LiveBench (Code) | OmniMath | GPQA Diamond | EvaLearn | 전체 평균 |
|---|---|---|---|---|---|---|---|
| **ICL ($k \le 20$)** | 71.83% | 62.50% | 65.46% | 53.12% | 78.45% | 40.99% | 62.06% |
| **ACE (Playbook)** | 71.98% | 66.94% | 69.38% | 50.33% | 76.58% | 42.54% | 62.96% |
| **Dynamic CheatSheet (DC)** | 73.33% | 63.59% | 68.58% | 48.64% | 79.56% | 42.68% | 62.73% |
| **No Feedback (NF CoE)** | 77.78% | 72.57% | 60.16% | 65.17% | 80.02% | 44.91% | 66.77% |
| **Reasoning-High** | 69.05% | 70.63% | 55.46% | 61.81% | 76.21% | 39.58% | 62.12% |
| **Reasoning-Low** | 60.48% | 61.03% | 55.38% | 50.60% | 72.92% | 29.34% | 54.96% |
| **Binary-Executor** | – | 72.90% | 71.65% | – | – | – | – |
| **CoE Self Feedback (SF)** | **82.22%** | **75.69%** | **69.94%** | **67.52%** | **81.03%** | **51.73%** | **71.36%** |
| **CoE Correctness/Exec (CEF)** | **89.05%** | **74.50%** | **75.78%** | **79.61%** | **99.52%** | **57.05%** | **79.25%** |

### Table 5: 토큰 복잡도 대비 정확도 효율성 (Token vs Accuracy)
> "Token complexity vs. accuracy across methods. CEF: Correctness/Executor Feedback; SF: Self Feedback; NF: No Feedback; DC: Dynamic CheatSheet. Token counts are aggregated across all iterations."

| 데이터셋 | 방법론 (Method) | 누적 토큰 수 (Tokens) | 정확도 (Acc, %) |
|---|---|---|---|
| **AIME 2025** | CEF (Correctness) | 108,734 | 84.6% |
| | SF (Self Feedback) | 108,231 | 83.8% |
| | NF (No Feedback) | 106,825 | 74.1% |
| | DC (Dynamic CheatSheet) | 11,233 | 74.7% |
| **OmniMath** | CEF (Correctness) | 176,412 | 74.2% |
| | SF (Self Feedback) | 175,806 | 72.1% |
| | NF (No Feedback) | 173,944 | 66.8% |
| | DC (Dynamic CheatSheet) | 16,904 | 63.9% |
| **LiveCodeBench (V6)** | CEF (Executor) | 224,118 | 72.6% |
| | SF (Self Feedback) | 223,441 | 71.2% |
| | NF (No Feedback) | 221,550 | 68.0% |
| | DC (Dynamic CheatSheet) | 20,771 | 66.4% |

---

## 4. 이중 피드백(Dual Feedback) 및 메모리 압축 분석 (Table 1)

### Table 1: Dual Feedback 및 단일 태스크 내 경험 압축 결과 (Claude 4.5 Sonnet)
> "Dual feedback combines model feedback with correctness (math) or executor (code) signals. Memory-based methods (DC, SimpleMem) are applied within-task with no cross-task leakage. Acc: best accuracy (%) over 20 iterations; Best R: iteration achieving best performance."

| 설정 (Setting) | AIME 2025 Acc | AIME Best Round | LiveBench (Code) Acc | LiveBench Best Round | OmniMath Acc | OmniMath Best Round |
|---|---|---|---|---|---|---|
| **Dual (Model + Corr/Exec)** | **76.7%** | Round 19 | **81.2%** | Round 15 | 73.5% | Round 17 |
| **Correctness / Executor** | 70.0% | Round 13 | 78.1% | Round 15 | **74.5%** | Round 17 |
| **Binary Executor** | – | – | 71.9% | Round 13 | – | – |
| **Model Feedback Only** | 60.0% | Round 6 | 57.8% | Round 17 | 50.5% | Round 9 |
| **Model + DC** | 50.0% | Round 8 | 51.6% | Round 15 | 46.0% | Round 10 |
| **Model + SimpleMem** | 56.7% | Round 6 | 54.7% | Round 17 | 49.5% | Round 12 |

---

## 5. 위조 피드백(Spurious Feedback) 및 선택적 다수결(SelMV) (Table 2)

### Table 2: 항구적 정답/오답 위조 피드백 환경 성능 및 SelMV 효과
> "The best performance over 20 iterations under constant 'correct' or 'incorrect' feedback. Selective majority voting (SelMV) helps LLMs maintain performance. Results are averaged over 3 runs."

| 피드백 설정 (Feedback Setting) | AIME 2025 (GPT-5 mini) | AIME 2025 (o4-mini) | GPQA Diamond (GPT-5 mini) | GPQA Diamond (o4-mini) |
|---|---|---|---|---|
| **Self Feedback** | 93.3% | 91.1% | 79.9% | 78.8% |
| **SelMV Self** | 91.1% | 88.9% | 80.4% | 79.5% |
| **All Correct (위조 정답)** | 90.0% | 73.3% | 79.3% | 75.8% |
| **SelMV Correct** | 93.3% | 73.3% | 79.3% | 76.3% |
| **All Incorrect (위조 오답)** | 91.7% | 83.3% | 79.3% | 72.7% |
| **SelMV Incorrect** | 89.7% | 86.7% | **82.8%** | 77.8% |

---

## 6. 모델 개선 원인 분해(Attribution Analysis) 및 판정 기준

### 6.1 개선 원인 4가지 분류 기준 (Figure 6 & Appendix F.4)
- **Feedback Fidelity (47.7%)**: 제공된 피드백의 구체적인 오류 지적이나 단위 테스트 실패 위치를 충실히 반영하여 수정한 경우.
- **Specification Recall (코딩 30.0%)**: 문제 지문의 입출력 제약, 엣지 케이스 처리 규칙, 출력 스키마('FINAL ANSWER:' 래퍼 등)를 재인식하여 정렬한 경우.
- **Self Reflection**: 외부 피드백 의존 없이 모델 자체의 내적 논리 재검토 및 오류 수정을 통해 정답으로 전환한 경우.
- **Random / Stochastic**: 명확한 인과관계 없이 표면적 어휘 변경이나 스타일 재배치를 거치며 정답이 된 경우.

### 6.2 인간-GPT 판정 일치도 (Table 4)
| 카테고리 (Category) | 일치율 (Agreement, %) | Cohen’s $\kappa$ |
|---|---|---|
| **Feedback Fidelity** | 84.0% | 0.81 |
| **Self Reflection** | 72.0% | 0.71 |
| **Specification Recall** | 80.0% | 0.78 |
| **Random** | 68.0% | 0.63 |
| **전체 (Overall)** | **76.0%** | **0.768** |

### 6.3 기본 역량 대비 개선 잠재력 상관계수
- **개선 능력 정의식**:
  $$\Delta_M = \frac{S_{\max} - S_{\text{base}}}{1 - S_{\text{base}}}$$
- **Pearson 상관계수 ($r$)**:
  - LiveBench (Code): $r = 0.97$
  - LiveCodeBench (V6): $r = 0.83$
  - EvaLearn: $r = 0.37$
  - AIME 2025: $r = 0.33$
  - GPQA Diamond: $r = 0.28$
  - OmniMath: $r = 0.24$
  - **전체 벤치마크 평균**: **$r = +0.50$** (Base 모델의 추론 능력이 뛰어날수록 피드백을 소화하여 개선하는 역량이 유의미하게 높음).

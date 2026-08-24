# Chain-of-Experience for Continual LLM Improvement — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_Chain-of-Experience_for_Continual_LLM_Improvement_2026_Bytedance_Seed.md) / 원본: [arXiv:2608.18027](https://arxiv.org/abs/2608.18027)

---

## 1. 논문 개요 및 초록 (Abstract & Problem Definition)

### 1.1 초록 (한국어 번역)
> "인간은 경험을 통해 끊임없이 학습하지만, 기존의 대규모 언어 모델(LLM) 평가는 추론 시점(Inference-time)의 상호작용을 통해 모델이 스스로 개선되는 역량을 간과해 왔다. 본 논문에서는 LLM이 테스트 시점에 반복적인 경험으로부터 어떻게 학습하는지를 규명하며, 이를 **Chain-of-Experience (CoE)** 설정으로 정의한다. CoE에서 모델은 자기 자신 또는 환경 피드백과의 반복적인 상호작용을 통해 경험적 흔적(Experiential Traces)을 축적하고, 이를 통해 기존 Zero-shot 추론을 넘어서는 지속적 개선 루프를 형성한다.
>
> 본 연구에서는 모델 자체의 자기 피드백(Self-feedback)과 정답 여부(Correctness) 신호, 공개 코딩 테스트 통과율과 같은 환경 신호를 포함한 다양한 피드백 메커니즘으로 CoE를 구체화하고, GPT-5, Gemini-2.5 Pro, Claude-4.5 Sonnet을 포함한 8개 최신 LLM을 수학·코딩·지식 도메인에 걸쳐 평가하였다.
> 
> 연구 결과, 반복적 경험을 활용하는 접근법은 피드백이 없는 기준선(Baseline)을 일관되게 능가하며, 자기 피드백만으로도 상당한 성능 향상을 거둠과 동시에 전체 과제 및 모델 전반에서 **5.6%의 종합 성능 개선**과 **19%의 API 호출 비용 절감**을 달성함을 입증하였다. 나아가 상호 보완적인 피드백 채널(예: 모델 비평 신호와 정답 신호)을 결합할 경우 추가적인 성능 이득을 얻을 수 있으며, CoE가 기존 테스트 시점 전략 대비 토큰당 더 높은 정확도를 제공함을 규명하였다.
> 
> 또한 기본 LLM의 추론 능력과 경험 기반 개선 역량 사이에 유의미한 양의 상관관계가 존재함을 관찰하였으며, 모델이 약하거나 위조된(Spurious) 피드백 환경에서도 높은 강건성을 유지하고, 서로 다른 피드백이 모델 개선의 다양한 측면에 기여하며 대부분의 성능 이득이 초기 반복 단계에서 빠르게 발생함을 확인하였다."

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
> 6개 벤치마크 데이터셋에서 다양한 LLM 모델들의 평균 성능 비교(%). 기준선 기법인 ICL, ACE, DC는 각각 퓨샷 문맥 학습, 에이전틱 컨텍스트 엔지니어링, 다이내믹 치트시트를 의미함.

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
> 방법론별 누적 소비 토큰 수 대비 정확도 비교. CEF: 정답/실행기 피드백, SF: 자기 피드백, NF: 무피드백, DC: 다이내믹 치트시트. 토큰 수는 전체 반복 턴의 총합임.

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
> 모델 피드백과 정답(수학) 또는 실행기(코딩) 신호를 결합한 이중 피드백 및 단일 태스크 내 메모리 압축 기법(DC, SimpleMem) 비교 (Claude 4.5 Sonnet). Acc: 20회 반복 중 최고 정확도(%), Best R: 최고 성능을 달성한 라운드.

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
> 항구적으로 일관된 '정답' 또는 '오답' 위조 피드백이 주어지는 환경에서의 20회 반복 최고 성능과 선택적 다수결(SelMV) 적용 효과 (3회 평균).

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

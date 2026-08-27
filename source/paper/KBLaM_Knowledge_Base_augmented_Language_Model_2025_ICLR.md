# KBLaM: Knowledge Base augmented Language Model — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_KBLaM_Knowledge_Base_augmented_Language_Model_2025_ICLR.md) / 원본: [arXiv:2410.10450](https://arxiv.org/abs/2410.10450)

---

## 1. 개요 및 파이프라인 비교 (Overview & Pipeline Comparison)

![Overview of the KBLaM pipeline and comparison with existing approaches](figures/kblam_fig1_pipeline_comparison.png)



### 핵심 요약 (Section 1)
- **KBLaM의 동기**: 대규모 외부 지식을 언어 모델에 통합할 때, RAG는 별도의 검색기(Retriever) 모듈과 생성기 간의 단절 및 검색 오류 전파 문제가 있고, 인컨텍스트 학습(In-Context Learning, ICL)은 컨텍스트 길이에 따른 이차적 계산량()과 메모리 폭증으로 인해 수천 개 이상의 지식 항목을 처리하지 못한다.
- **KBLaM의 해법**: 비정형 문서를 구조화된 지식 베이스(KB) 트리플  형태로 요약한 뒤, 각 트리플을 1개 토큰 크기의 연속 키-값 벡터 쌍인 **지식 토큰(Knowledge Token, )**으로 변환하여 LLM의 각 어텐션 레이어에 **직사각형 어텐션(Rectangular Attention)**으로 직접 주입한다.
- **주요 특징**:
  1. 외부 검색기 없는 완전한 엔드투엔드(End-to-End) 소프트 검색 및 생성.
  2. KB 크기()에 대해 선형적인 연산() 및 메모리 복잡도().
  3. 단일 A100 80GB GPU에서 8K 컨텍스트 윈도우를 가진 8B LLM에 10K(10,000개) 이상의 지식 트리플을 단일 컨텍스트로 통합 가능.
  4. 모델 파라미터 재학습 없이 KB 지식의 동적 추가/수정/삭제 가능 (독립적 인코딩 특성).

---

## 2. 지식 베이스 정의 및 지식 토큰화 (Knowledge Representation & Knowledge Tokens)

![Overview of KBLaM's KB augmentation process](figures/kblam_fig2_architecture_overview.png)

### 지식 베이스 형식 (Section 3 & 4)
외부 문헌 코퍼스로부터 추출된 구조화 지식 베이스는 다음과 같은 개의 트리플 집합으로 정의된다:



### 지식 토큰 생성 (Knowledge Token Formulation)
각 트리플 에 대해 사전학습 문장 인코더 를 사용하여 기본 키 및 값 임베딩을 계산한다:



학습 가능한 선형 어댑터  및 를 통해 인코더 차원 를 LLM의 레이어별 은닉 차원 로 사영하여 최종 지식 토큰을 구성한다:



- 각 지식 토큰 은 LLM 내부 어텐션 토큰 1개의 Key/Value 벡터와 완벽히 동일한 형상을 가지며, 텍스트 문자열 길이에 상관없이 고정된  차원을 유지한다.

---

## 3. 직사각형 어텐션 메커니즘 (Rectangular Attention Mechanism)

### 수식 정의 (Section 4, Eq. 9 & 10)
개 레이어의 디코더 모델에서 번째 어텐션 레이어에 입력되는 프롬프트 토큰 시퀀스 와 개의 지식 토큰 에 대해, 어텐션 출력 는 다음과 같이 계산된다:



여기서 어텐션 로짓은 다음과 같다:



- 는 사전학습 가중치 로부터 초기화되는 학습 가능한 KB 전용 쿼리 헤드이다.
- 프롬프트 토큰들은 모든 지식 토큰들과 이전 프롬프트 토큰들을 참조할 수 있으나, 지식 토큰들은 상호 간에 어텐션을 수행하지 않으므로 어텐션 행렬은  직사각형 형태가 된다.

### 길이 일반화 스케일링 (Attention Score Scaling, Eq. 11)
KB의 크기 이 커질 때 소프트맥스 분모에서 KB 항의 합이 과도하게 비대해져 프롬프트 내부 정보가 손실되는 문제를 방지하기 위해 다음과 같은 상수 시프트를 적용한다:



- 는 학습 시 관측한 최대 트리플 수 (실험에서는 으로 고정). 이는 개 트리플의 소프트맥스 기여도를 으로 재조정하여 임의의 큰 에 대해서도 안정적인 어텐션 밸런스를 보장한다.

---

## 4. 인스트럭션 튜닝 및 합성 데이터셋 (Instruction Tuning with Synthetic Dataset)

![Examples of instruction tuning dataset](figures/kblam_fig12_instruction_tuning_samples.png)

### 학습 목적식 (Section 5, Eq. 12)
동결된 LLM 파라미터  하에서 어댑터 파라미터 만을 최적화한다:



### 합성 데이터 생성 전략
- 어댑터 학습의 목적은 지식을 암기하는 것이 아니라 사전학습 문장 인코더 공간과 LLM 어텐션 공간 간의 **사영 함수(Projection Function)**를 학습하는 것이므로 완전 합성 데이터셋을 사용.
- GPT를 통해 30개 객체 유형(Object Types)과 30개 개념 유형(Idea Types)을 조합하여 45,000개 개체명()과 135,000개 트리플(, , ) 생성. 개체명과 속성값 사이의 상관관계를 배제하여 모델이 사전 지식에 의존하지 않고 KB로부터 정보를 인출하도록 강제.
- **4가지 인스트럭션 유형**:
  1. Simple Q&A: 단일 엔티티의 특정 속성 질의.
  2. Multi-entities Q&A: 복수 엔티티의 다중 속성 비교 및 합성 질의.
  3. Open-ended reasoning Q&A: 속성값을 바탕으로 개방형 추론 요구.
  4. Unanswerable Q&A: KB에 존재하지 않는 정보 질의 시  출력 유도 (환각 억제).

---

## 5. 어텐션 해석 가능성 및 검색 성능 (Interpretability & Retrieval Accuracy)

![Interpretability of KBLaM attention matrix](figures/kblam_fig4_attention_interpretability.png)

![Retrieval accuracy comparison](figures/kblam_fig5_retrieval_accuracy.png)

- **해석 가능성(Figure 4)**: 15번째 레이어(32레이어 Llama-3 8B의 중간층)의 헤드 평균 어텐션 가중치를 분석한 결과, 질문 내의 엔티티 토큰이 KB의 정답 지식 토큰에 매우 강한 어텐션 피크를 형성함을 확인.
- **검색 정확도(Figure 5)**: 별도의 검색 손실함수나 정규화 없이 순수 Q&A 인스트럭션 튜닝만으로도 Top-1 및 Top-5 검색 정확도가 매우 높게 유지됨 (합성 데이터 및 Enron OOD 데이터 모두에서 강건).

---

## 6. 질의응답 추론 성능 및 환각 거부 (Reasoning & Refusal Performance)

![KBLaM reasoning and refusal performance](figures/kblam_fig6_qa_reasoning_and_refusal.png)

- **Q&A 성능 (Figure 6a, 6b)**: 합성 데이터셋에서 In-Context Learning과 대등한 BERT Score 및 GPT-4 평가 점수를 달성하면서도 훨씬 적은 메모리를 소비하며 10,000개 트리플까지 확장.
- **답변 불가 질문에 대한 거부율 (Figure 6c)**: KB에 정답이 없는 경우 거부 응답(Precision/Recall) 평가 시, In-Context Learning보다 과도한 거부(Over-refusal) 현상이 훨씬 완만하게 나타남.

---

## 7. 어블레이션 및 효율성 분석 (Ablation Studies & Efficiency)

![Encoder ablation](figures/kblam_fig7_encoder_ablation.png)
![Knowledge token injection frequency ablation](figures/kblam_fig8_injection_frequency.png)
![Layer embeddings analysis](figures/kblam_fig9_layer_embeddings.png)
![Retrieval vs BM25](figures/kblam_fig10_retrieval_vs_bm25.png)
![Latency and memory vs RAG](figures/kblam_fig11_latency_and_memory_vs_rag.png)

- **인코더 용량 (Figure 7)**: MiniLM(P=384)부터 text-embedding-large(P=3072)까지 인코더 용량이 클수록 Top-1 검색 및 BERT Score가 향상.
- **주입 빈도  (Figure 8)**: 매 레이어() 주입과 3개 레이어마다() 주입 시 성능 차이가 미미하여 계산 효율성을 위해 을 표준으로 채택. 으로 과도하게 줄이면 거부 성능이 급격히 저하.
- **레이어별 표현 특성 (Figure 9)**: 초기 레이어의 지식 토큰은 엔티티 간 분산이 작아 인스트럭션 소프트 프롬프트 역할을 수행하고, 심층 레이어로 갈수록 엔티티 간 분산이 커져 실제 세부 지식을 공급.
- **BM25 대비 견고성 (Figure 10)**: 질문에 엔티티 키워드가 변형된 Perturbed 설정에서 BM25는 정확도가 급락하지만, KBLaM의 임베딩 기반 어텐션은 높은 검색 성능을 유지.
- **RAG 대비 지연 시간 및 메모리 (Figure 11)**: KBLaM은 512개 트리플 전체를 컨텍스트에 유지하면서도 Top-5 개만 검색하는 RAG 대비 Time-to-First-Token(TTFT)과 총 메모리 사용량에서 일관된 우위를 점함.

# Pre-training Limited Memory Language Models with Internal and External Knowledge — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Pre-training_Limited_Memory_Language_Models_with_Internal_and_External_Knowledge_2025_Cornell.md) / 원본: [arXiv:2505.15962](https://arxiv.org/abs/2505.15962) / 코드: [github.com/kilian-group/LMLM](https://github.com/kilian-group/LMLM)

---

## 1. Abstract & Core Motivation

> "Neural language models are black-boxes – both linguistic patterns and factual knowledge are distributed across billions of opaque parameters. This entangled encoding makes it difficult to reliably inspect, verify, or update specific facts. We introduce LIMITED MEMORY LANGUAGE MODELS (LMLM), a new class of language models that externalizes factual knowledge to external database during pre-training rather than memorizing them. Our pre-training approach strategically masks externally retrieved factual values from the training loss, thereby teaching the model to perform targeted lookups rather than relying on memorization in model weights."

![LMLM Concept Comparison](../paper/figures/lmlm_fig1_concept_comparison.png)
*Figure 1: LMLM vs. LLM + RAG 개념 비교. 기존 RAG(좌측)는 내부 파라미터에 사실을 이미 암기한 LLM 위에 외부 지식을 사후 추가하는 반면, LMLM(우측)은 사전학습 단계부터 사실적 지식을 외부 데이터베이스로 오프로딩하여 파라미터에는 언어적 역량(유창성, 문법, 추론)만을 남기고 사실은 필요 시점에 조회하도록 학습한다.*

---

## 2. LMLM Framework Overview

![LMLM Framework Overview](../paper/figures/lmlm_fig2_framework_overview.png)
*Figure 2: LMLM 전체 프레임워크 개요. (좌측) Data Preparation: 사전학습 코퍼스에서 원자적 사실을 추출하여 외부 DB 구축 및 룩업 주석 추가; (중앙) Pre-training: 룩업 주석이 포함된 텍스트로 사전학습을 진행하되 반환값 토큰을 손실에서 마스킹하여 암기 억제; (우측) Inference: 생성 도중 동적으로 룩업 쿼리를 발행하고 데이터베이스에서 값을 조회해 텍스트를 완성.*

---

## 3. Knowledge Extraction & Data Preparation

![Training the Annotator Model](../paper/figures/lmlm_fig3_annotator_training.png)
*Figure 3: ANNOTATOR 모델 훈련 파이프라인. GPT-4o로부터 고품질 시드 주석을 생성한 뒤, 과소적합된 CORRECTOR 모델로 노이즈를 필터링하고, 최종 ANNOTATOR(LLaMA-3.1-8B-Instruct 기반)로 전체 사전학습 코퍼스를 확장 주석 처리.*

### 3-Stage Distillation Pipeline
1. **Seed Annotation**: GPT-4o를 이용해 지식 집약적 문서 $M=1,000$개에 대해 `[dblookup('Entity', 'Relationship') -> Value]` 형식의 주석 생성.
2. **Filtering (CORRECTOR)**: LLaMA-3.1-8B-Instruct를 시드 데이터에 대해 의도적으로 과소적합(2 epoch)시켜 훈련. 포맷 오류나 비정상적으로 구체적인 룩업 호출에 높은 손실이 부여되며, 손실 상위 10%를 제거.
3. **Annotation (ANNOTATOR)**: 정제된 데이터셋으로 LLaMA-3.1-8B-Instruct를 10 epoch 인스트럭션 튜닝하여 ANNOTATOR 구축. 전체 Wikipedia 코퍼스(~3B 토큰)에 적용하여 5,460만 개의 지식 트리플(950만 엔티티, 850만 관계)을 추출.

---

## 4. Formalization of Training Objectives

### Token Categorization
자기회귀 언어 모델 $p_	heta(x) = \prod_{t=1}^T p_	heta(x_t \mid x_{<t})$에서 토큰 $x_t$를 다음과 같이 분류:
- $\mathcal{T}_{	ext{org}}$: 원본 코퍼스의 순수 텍스트 토큰
- $\mathcal{T}_e, \mathcal{T}_r$: 룩업 호출 내 엔티티 및 관계 인자 토큰
- $\mathcal{T}_v$: 데이터베이스에서 인출된 반환값(return value) 토큰
- $\mathcal{T}_{	ext{db}}$: 특수 토큰군 ($\langle|	ext{db\_entity}|angle, \langle|	ext{db\_relationship}|angle, \langle|	ext{db\_return}|angle, \langle|	ext{db\_end}|angle$)

### Masked Pre-training Loss (Equation 1 & 2)
$$\mathcal{L}(	heta) = - \sum_{t=1}^T m_t \log p_	heta(x_t \mid x_{<t})$$
$$m_t = egin{cases} 0, & x_t \in \mathcal{T}_v \cup \{\langle|	ext{db\_end}|angle\} \ 1, & 	ext{otherwise} \end{cases}$$

$$\mathcal{L}(	heta) = - \sum_{t \in \mathcal{T}_{	ext{train}}} \log p_	heta(x_t \mid x_{<t}), \quad 	ext{where } \mathcal{T}_{	ext{train}} = \{t \mid x_t 
otin \mathcal{T}_v \cup \{\langle|	ext{db\_end}|angle\}\}$$

### Evaluation Perplexity Metrics
- **Static (Oracle) PPL**: 모델이 완벽한 룩업 호출을 생성하고 정답을 인출했다고 가정한 하한선.
  $$	ext{PPL}_{	ext{static}} = \exp\left( -rac{1}{|\mathcal{T}_{	ext{org}}|} \sum_{t \in \mathcal{T}_{	ext{org}}} \log p_	heta(x_t \mid x_{<t}) ight)$$
- **Dynamic PPL**: 추론 중 모델이 실시간으로 생성한 룩업 쿼리와 실제 검색 결과를 반영한 Perplexity.
  $$	ext{PPL}_{	ext{dynamic}} = \exp\left( -rac{1}{|\mathcal{T}_{	ext{org}}|} \sum_{t \in \mathcal{T}_{	ext{org}}} \log p_	heta(x_t \mid x_{<t}) ight)$$
- **Normalized PPL**: 룩업 쿼리 생성에 소모된 우도까지 포함하여 평가하되, 원본 텍스트 길이 $|\mathcal{T}_{	ext{org}}|$로 정규화.
  $$	ext{PPL}_{	ext{norm}} = \exp\left( -rac{1}{|\mathcal{T}_{	ext{org}}|} \sum_{t \in \mathcal{T}_{	ext{train}}} \log p_	heta(x_t \mid x_{<t}) ight)$$

---

## 5. Experimental Results

![Results Overview](../paper/figures/lmlm_fig4_results_overview.png)
*Figure 4: 주요 결과 요약. (좌측) 사전학습 중 Validation Perplexity 감소 추이; (중앙) 머신 언러닝(TOFU)에서 유틸리티 손실 없는 완벽한 망각; (우측) 파라미터 규모 대비 사실성 벤치마크(FactScore) 성능 — 소형 LMLM이 수십 배 큰 LLM을 능가.*

### Perplexity Comparison
![Validation Perplexity](../paper/figures/lmlm_fig5_validation_perplexity.png)
*Figure 5: 모델 크기별 Standard vs. LMLM Validation Perplexity 비교.*

| Model Arch & Scale | Standard PPL | LMLM Dynamic PPL | LMLM Normalized PPL | LMLM Static (Oracle) |
|---|---|---|---|---|
| GPT2-124M | 14.1 | **12.2** | 8.5 | 7.7 |
| LLaMA2-176M | 11.8 | **9.9** | 6.7 | 6.3 |
| GPT2-355M | 10.8 | **9.2** | 6.7 | 6.2 |
| LLaMA2-382M | 9.1 | **7.9** | 5.8 | 5.5 |

### Machine Unlearning (TOFU Benchmark)
![Machine Unlearning Evaluation](../paper/figures/lmlm_fig6_machine_unlearning.png)
*Figure 6: TOFU 벤치마크 기반 머신 언러닝 평가. LMLM은 데이터베이스 삭제만으로 Retain Set의 유틸리티 저하 없이 Forget Set의 완벽한 망각(Forget Quality p-value ~ 1.0)을 달성.*

### Factual Precision (FactScore, T-REx, PopQA)
| Model | Model Type | FactScore (%) ↑ | T-REx EM (%) ↑ | PopQA Acc (%) ↑ |
|---|---|---|---|---|
| OpenAI GPT2-124M* | Off-the-shelf | 14.6 | 20.1 | 18.51 |
| GPT2-124M | STANDARD | 10.7 | 41.2 | 18.51 |
| **GPT2-124M** | **LMLM** | **20.6 (+9.9)** | **54.6 (+13.4)** | **49.89 (+31.4)** |
| LLaMA2-176M | STANDARD | 10.1 | 46.3 | 24.59 |
| **LLaMA2-176M** | **LMLM** | **30.6 (+20.5)** | **54.1 (+7.8)** | **49.61 (+25.0)** |
| OpenAI GPT2-355M* | Off-the-shelf | 15.2 | 28.4 | 19.1 |
| GPT2-355M | STANDARD | 14.4 | 44.9 | 21.4 |
| **GPT2-355M** | **LMLM** | **23.9 (+9.5)** | **58.7 (+13.8)** | **52.0 (+30.6)** |
| LLaMA2-382M | STANDARD | 14.0 | 52.0 | 22.7 |
| **LLaMA2-382M** | **LMLM** | **31.9 (+17.9)** | **58.1 (+6.1)** | **50.8 (+28.1)** |
| Pythia-1B* | Off-the-shelf | 21.1 | 47.8 | 19.5 |
| LLaMA2-7B* | Off-the-shelf | 34.0 | 60.5 | 29.2 |
| LLaMA3.1-8B* | Off-the-shelf | 40.3 | 67.3 | 29.4 |

---

## 6. Ablations & Further Analysis

![Loss on Return Values and Offloading Ratios](../paper/figures/lmlm_fig7_8_ablation_curves.png)
*Figure 7 & 8: (좌측) 일반 SFT 대비 LMLM의 사실 반환값 토큰 훈련 손실 유지 추이(암기 방지 검증); (우측) 지식 오프로딩 비율(0%~100%) 증가에 따른 NLU, Perplexity, FactScore 변화.*

### Knowledge Offloading Ratio Trade-off
- 훈련 1 epoch 후 Standard LM 대비 손실 감소폭이 큰 사실(즉, 파라미터로 학습하기 어려운 롱테일 사실)을 우선순위로 오프로딩 비율을 조절.
- 오프로딩 비율이 0%에서 100%로 증가할수록 Perplexity가 지속적으로 개선되고 FactScore가 비례하여 상승하며, 일반 언어 이해(NLU) 능력은 손상되지 않고 유지됨.

### Constrained vs. Unconstrained Query Decoding
![Unconstrained vs Constrained Decoding](../paper/figures/lmlm_fig9_constrained_vs_unconstrained.png)
*Figure 9: 비제약적 임베딩 코사인 검색(좌측) vs. Prefix-Tree(Trie) 기반 제약 생성(우측). Prefix-tree 디코딩은 구조적 정합성을 보장하나 작은 DB에서는 다양성이 제한될 수 있음.*

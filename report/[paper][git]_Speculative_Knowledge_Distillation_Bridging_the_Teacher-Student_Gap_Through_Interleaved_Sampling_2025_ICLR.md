> [paper][git] https://github.com/google-research/google-research/tree/master/speculative_kd · https://arxiv.org/abs/2410.11325

# Speculative Knowledge Distillation (SKD)

> **원본 관계**: both 타입. 논문(arXiv:2410.11325, ICLR 2025)이 방법론·이론·실험의 기반이고, `google-research/google-research` monorepo의 `speculative_kd/` 서브디렉토리가 공식 구현체.
> monorepo 규모가 매우 커(38k★, 수백 프로젝트) **submodule 대신 URL 참조**로 처리(`submodule: null`). 본문에서 코드는 경로 참조, 상세는 스니펫으로 분리.
> 논문 핵심 발췌 → [excerpt](../source/paper/Speculative_Knowledge_Distillation_Bridging_the_Teacher-Student_Gap_Through_Interleaved_Sampling_2025_ICLR.md).

## Overview

LLM 압축을 위한 지식 증류(KD)에서 두 주류 방법 모두 치명적 결함이 있다. **Supervised KD**는 고정 데이터셋으로 학습해 추론 시 학생이 실제로 만드는 분포와 불일치(train-inference mismatch)하고, **On-policy KD**는 학생 자기 생성 샘플로 학습해 mismatch는 해소하지만 학생이 특히 학습 초기에 저품질/OOD 샘플을 만들어 교사의 부정확한 피드백을 유발·초기화에도 민감하다. SKD는 speculative decoding의 "학생 제안 → 교사 검증" 구조를 차용해 이 둘을 동시에 해결한다: 학생이 토큰을 제안하면 교사가 자기 **top-K** 안에 있으면 수락, 아니면 자기 분포에서 재샘플링해 교체한다. 이 interleaved 샘플은 (1) 학생의 추론 분포와 정렬되면서도 (2) 교사가 생성할 법한 고품질 토큰만 남아 on-policy의 저품질 문제를 제거한다. 학습 초기엔 거부가 많아 ≈supervised KD, 후기엔 수락이 많아 ≈on-policy KD로 **자연스럽게 전이**하며, 두 극단을 특수케이스로 일반화한다(K→0 = supervised, K→∞ = on-policy).

## Architecture — SKD 학습 1스텝 전체 흐름

```
              ┌────────────── SKD 학습 1스텝 (per prompt, gradient acc) ──────────────┐
              │                                                                        │
  prompt x ─► │  ① INTERLEAVED SAMPLING  (교사 주도 .generate, 학생 = draft)          │
              │     학생 M_s 가 γ=5 토큰 제안 (ancestral sampling, q)                │
              │     교사 M_t 가 후보 시퀀스 1회 forward (p)                            │
              │     _speculative_kd_sampling:                                          │
              │        학생 토큰 ∈ 교사 top-K?  ─yes─► 수락(학생 토큰 유지)            │
              │                             └─ no ──► 거부→교사 p에서 재샘플 교체    │
              │     n_matches(연속수락) 다음 위치부터 다시 학생이 γ토큰 제안 (반복)   │
              │     ──► interleaved 시퀀스 y  (+ correction_rate=거부율 기록)         │
              │                                                                        │
              │  ② KD 손실 계산  (Eq.1: 토큰별 발산 평균)                              │
              │     학생 forward ─► shift_student_logps (log_softmax)                  │
              │     교사 forward ─► shift_teacher_logps (teacher frozen)              │
              │     loss = D(M_t||M_s) = KL(기본) / JSD / reverse-KL                   │
              │                                                                        │
              │  ③ 학생만 backward → clip → optimizer.step (교사 requires_grad=False) │
              └────────────────────────────────────────────────────────────────────────┘
```

- γ(draft 토큰 수)=5 (`num_assistant_tokens=5`). 논문 Algorithm 1은 γ=1로 단순화 서술.
- 교사가 `.generate()`를 **주도**하되 `assistant_model=학생`을 넘기는 것이 핵심 — 표준 HF speculative decoding 경로를 타되 검증 루틴을 SKD용 top-K 기준으로 교체한 것이다 (`transformers/utils.py` 수정).
- 평가/추론은 일반 greedy 디코딩(학생 단독). SKD는 학습 데이터 생성 방법일 뿐 추론 구조를 바꾸지 않는다. 다만 학습된 학생을 speculative decoding draft로 쓰면 가속(Appendix K).

## Method — SKD 핵심 알고리즘

상세 발췌(수식·Algorithm 1·motivation) → [excerpt](../source/paper/Speculative_Knowledge_Distillation_Bridging_the_Teacher-Student_Gap_Through_Interleaved_Sampling_2025_ICLR.md).

### 수락 기준 (Token Acceptance Criteria)

학생이 위치 i에서 `M_s(·|y_<i,x)`로 샘플링한 토큰 `y_i`가 **교사 분포 `M_t(·|y_<i,x)`의 top-K 토큰 집합에 포함되면 수락**, 아니면 거부. 거부 시 후속 토큰 폐기 후 `y_i ~ M_t(·|y_<i,x)`로 교사에서 재샘플링해 교체. top-K 영감은 top-k 샘플링(Radford 2019)에서 차용.

> **왜 표준 speculative decoding 수락이 아닌가**: 표준은 확률비율 `min(1, p/q)` 수락으로 결국 교사 분포에서 샘플링하게 되어 **supervised KD로 퇴화**. SKD의 top-K 멤버십 판정은 학생 자기 분포를 유지하면서 "교사가 허용할 법한" 토큰만 걸러낸다. 구현 → [acceptance snippet](../source/git/snippets/Speculative_KD_2025_ICLR__acceptance_sampling.md).

### 자연 전이 (Sample Transition) — implicit curriculum

| 학습 단계 | 학생 품질 | 거부율 | SKD 거동 | ≈ 동등 |
|---|---|---|---|---|
| 초기 | 낮음 | 높음 | 대부분 교사 교체 | supervised KD |
| 후기 | 높음 | 낮음 | 대부분 학생 수락 | on-policy KD |

이 전이는 **스케줄이 아니라 교사-학생 분포 갭에 의해 자동**으로 결정된다(거부율 자체가 학생 품질의 함수). 이것이 §5.5 두-단계 ablation(인위적 단계 분할)이 SKD에 크게 열세인 이유다 — 시퀀스 단위 단순 혼합은 토큰 단위 자연 confidence 부재로 data discrepancy 유발.

### 일반화 (특수케이스)

| K 값 | 동작 | 퇴화 대상 |
|---|---|---|
| K→0 | 모두 거부→교사 재샘플 | supervised KD (교사 분포 샘플) |
| K→∞ | 모두 수락 | on-policy KD (학생 샘플 그대로) |
| K∈[5,50] | 적응적 혼합 | **SKD (최적)** |

Appendix B: K∈[5,50] 넓은 범위가 두 baseline 모두 초과; 본 실험은 K=25 고정(탐색 없음).

### 발산 메트릭 D (Eq.1)

토큰 단위 분포 발산의 출력 길이 평균: `D(M_t||M_s)(y|x) = (1/L_y) Σ_i D(M_t(·|y_<i,x) || M_s(·|y_<i,x))`. 구현은 forward KL(`KLDivLoss`) 기본; reverse-KL·JSD도 테스트. 모방학습(DAgger, Ross 2011)의 "샘플 생성을 교사→학생으로 점진 이동" 원칙의 실증적 실현.

### 적용 전제 — 동일 tokenizer/vocabulary 필수

SKD는 speculative decoding 위에 구축되어 **교사·학생이 같은 tokenizer/vocabulary를 공유해야 한다**. 이는 권장이 아니라 **하드 제약**이며, 세 곳에서 강제·전제된다.

| 근거 | 위치 | 내용 |
|---|---|---|
| **코드 하드 게이트** | `transformers/utils.py:1184` (`_validate_assistant`) | `if not self.config.vocab_size == assistant_model.config.vocab_size: raise ValueError("Make sure the main and assistant model use the same tokenizer")` — HF assisted-generation 표준 제약을 SKD가 그대로 상속. vocab size 다르면 즉시 에러. |
| **논문 명시 전제** | Appendix A ("Divergence Metrics") | "Assuming the teacher and student models share the same vocabulary V" — 발산 메트릭 Eq.1(KL) 자체가 같은 vocab 공간 위에서 정의. vocabs가 다르면 토큰 분포 비교 자체가 성립하지 않음. |
| **수락 로직 = token-id 수준** | `_speculative_kd_sampling` (`utils.py:4033`) | 학생이 `multinomial(q)`로 샘플링한 정수 id가 교사 top-K id 집합에 `==` 멤버십 검사(`p_selected_tokens == q_selected_tokens`), 거부 시 교사 vocab에서 `multinomial(p_prime)`으로 id 재샘플. vocab이 다르면 id 공간 자체가 달라 비교·재샘플이 무의미. |

> 본 실험도 **동일 모델族 내 크기만 다른 쌍**(Gemma 7B→2B, Qwen 7B→0.5B, 같은 tokenizer)에서만 검증되었고, **cross-family(이종 tokenizer)는 미테스트**다. 서로 다른 tokenizer를 쓰는 모델 간에 적용하려면 vocab 간 정렬/투영 레이어(토큰 매핑 또는 임베딩 projection)가 필요한데, 이는 speculative decoding 자체의 미해결 연구 영역이며 SKD 코드에는 없다 → 본 분석의 Limitations/Future Work에 기재.

## Code Architecture — 구현체 매핑

```
speculative_kd/
├── train/
│   ├── ddp_skd.py        # 핵심 학습 루프 (kd_type 분기, KD 손실, DDP)
│   ├── run_kd_train.py   # YAML config → accelerate launch 명령 빌더
│   └── train_sft.py      # SFT(교사/학생 초기화용, alignment-handbook 기반)
├── transformers/         # ★ 수정한 HF transformers 4.44.2 파일 (SKD 엔진)
│   ├── candidate_generator.py  # AssistedCandidateGenerator: 학생이 γ토큰 후보+logits 생성
│   └── utils.py                 # _assisted_decoding(루프) + _speculative_kd_sampling(★수락)
├── config/{kd_train.yaml, deepspeed_zero3.yaml, sft/*.yaml}
├── eval/{eval_gsm,eval_mt,eval_summ,eval_math,grader,vali_loss_compute}.py
└── experimental_setup.md, requirements.txt, run.sh
```

### kd_type 변형 (train/ddp_skd.py)

`kd_type` 하나로 논문의 모든 baseline을 재현한다. 학습 루프는 `kd_type`으로 샘플 생성 분기를 갈라 공통 KD 손실로 합친다.

| kd_type | 샘플 출처 | 설명 / 논문 대응 |
|---|---|---|
| `skd` | 교사 `.generate(assistant_model=학생, teacher_k=K)` | **본제안 SKD** — interleaved top-K 수락 |
| `on-policy` | 학생 단독 self-generate | On-policy KD (Agarwal 2024) |
| `supervised_kd` | 정답 시퀀스(`train_src_ref_dict`) | Supervised KD (Sanh 2020), KL 손실 |
| `seq_kd` | 정답 시퀀스 | SeqKD, CE 손실 (SFT가 우세해 본실험 제외) |
| `mixed` | 학생생성 vs 정답 50:50 시퀀스 단위 | ImitKD (Lin 2020) |
| `mixed_train` | 전반 supervised / 후반 on-policy | §5.5 두-단계 ablation |
| `mixed_skd` | SKD vs 정답 확률적 혼합 | SKD+정답 혼합 변형 |

SKD 분기의 핵심: 교사 `.generate()`에 `assistant_model=학생`을 넘겨 **교사 주도 생성 + 학생 draft** 구조로 만들고, `teacher_k=K, teacher_p=P`로 수락 기준 주입, γ=`num_assistant_tokens=5`. 생성 후 `parse_output`이 chat template을 재적용해 로짓 계산용 토큰으로 변환 → 학생·교사 각각 forward → KL 손실. 학습 루프 상세 → [training snippet](../source/git/snippets/Speculative_KD_2025_ICLR__training_loop.md).

### 수정된 transformers — SKD 엔진 (임의 핵심)

HF transformers 4.44.2의 `generation/candidate_generator.py`·`utils.py`를 SKD용으로 덮어쓴다(README 지시: `cp transformers/* .../transformers/generation/`).

- `AssistedCandidateGenerator` (`candidate_generator.py:78`): 학생이 γ개 후보 토큰 + logits 제안. γ는 heuristic schedule로 match율에 따라 ±조정 가능(본설정 constant=5).
- `_assisted_decoding` (`utils.py:3639`): interleaved 생성 메인 루프. 학생 후보 fetch → 교사 1회 forward → `_speculative_kd_sampling` 호출 → `cor_count`/`tot_count` 누적→`correction_rate`(거부율, Appendix L).
- `_speculative_kd_sampling` (`utils.py:4033`): **논문 Algorithm 1 구현체**. 학생 multinomial 샘플 → 교사 top-K 멤버십 수락 → 거부 시 `multinomial(p_prime)` 교체. 상세 → [acceptance snippet](../source/git/snippets/Speculative_KD_2025_ICLR__acceptance_sampling.md).
- `generate()`에 `teacher_k/teacher_p/teacher_k_lower/expected_seq_len` 파라미터 추가 — `expected_seq_len>0`이면 K 선형 감소(Appendix C adaptive K 재현).

### 하이퍼파라미터 (config/kd_train.yaml + experimental_setup.md)

| 항목 | 값 | 비고 |
|---|---|---|
| kd_type / top_k | `skd` / 25 | 탐색 없이 고정 |
| student temp / top_p | 0.5 / 0.5 | 학생 ancestral sampling |
| teacher temp / top_p | **0.2** / 0.5 | 교사 더 focused (저온) |
| distance_metric | `kl` | forward KL (Eq.1) |
| lr / epoch | 1e-5 / 3 (번역 10) | warmup 0.1, dropout 0.1 |
| inp_length | 256(GSM/MT), 1024(Summ/Math) | max = inp + max_new |
| 분산 | DDP + DeepSpeed ZeRO-3, bf16 | 8×A100 80GB, CPU 오프로드 |
| grad_acc_size | 1 | 실질 batch=1/GPU |

## Baseline KD 비교 (§3.1)

| 방법 | 샘플 출처 | train-inference mismatch | 저품질/OOD 위험 | 초기화 민감 |
|---|---|---|---|---|
| Supervised FT / SeqKD | 고정 정답/교사 | O (크게) | — | — |
| Supervised KD | 고정 정답 | O | — | — |
| On-policy KD | 학생 self | X (해소) | **O (초기 심각)** | **O** |
| ImitKD | 학생/정답 50:50(시퀀스) | 부분 | 부분 | 부분 |
| **SKD** | 학생 제안→교사 top-K 필터 | X | **X (교사 교체)** | **X** |

## Experiments & Results

### Setup
- 모델族: Gemma(교사 7B-it → 학생 2B), Qwen2(교사 7B-it → 학생 0.5B). 교사=과제 SFT; 학생 초기화=IT 또는 SFT.
- 과제: Flores-200 Assamese→English 번역(COMET), DialogSum 요약(ROUGE-L), GSM8K 산술추론(정확도), UltraInteract 수학 지시수행(MATH/GSMplus/ASDiv/SVAMP; task-agnostic).
- 데이터: 과제별 ~1K(저자원 100), task-agnostic 1K/10K. 평가=greedy 디코딩.

### Data Availability & Reproducibility

논문 §8은 "All data, model and evaluation metrics are open sourced"라고 선언하지만, **실제 전처리 학습 데이터 접근에는 문제가 있다**. 항목별 공개 여부:

| 항목 | 공개 | 비고 |
|---|---|---|
| 기반 데이터셋 (Flores-200, DialogSum, GSM8K, UltraInteract, MATH/GSMplus/ASDiv/SVAMP) | ✅ | 모두 공개 벤치마크 (HuggingFace 등) |
| 모델 (Gemma, Qwen2) | ✅ | HuggingFace |
| 코드 (KD 학습·평가 전체) | ✅ | `google-research/speculative_kd` |
| 평가 메트릭 (COMET, ROUGE-L) | ✅ | 공개 도구 |
| SFT 교사/학생 체크포인트 | ⚠️ 재현 필요 | 제공된 SFT config + 공개 데이터로 직접 SFT. 체크포인트 자체는 비공개 |
| **전처리 학습 JSON** (`gsm_1k_train.json`, `summ_1k_train.json`, `mt_1k_train.json`, `Math_CoT_train.json` 등) | ❌ **사실상 비공개** | README 데이터 링크가 `drive.google.com/corp/drive/folders/...` — **Google 내부 corp Drive 경로**. 공개 Drive는 `/corp/` 없이 `drive.google.com/drive/folders/...`여야 함. 외부 사용자 접근 불가 |
| 1저자 보조 repo (`xu1998hz/efficient_kd`) | ❌ 404 | README가 experimental_setup 링크로 가리키나 repo가 없거나 private |

> **재현 경로**: 기반 데이터셋 자체는 공개이므로, 논문 Appendix F의 task별 prompt/response 포맷을 참고해 전처리 JSON을 직접 재구성하면 이론적으론 재현 가능하나, 상당한 번거로움이 따른다. 코드(`ddp_skd.py`)의 데이터 로딩 부(`load_dataset("json", data_files="data/{task}_train.json", field="instances")`) 참조.


### Results — Table 1 (IT 학생 초기화, K=25)

| Baseline | Trans(COMET) | Summ(ROUGE-L) | GSM8K(Acc) | Trans·Qwen | Summ·Qwen | GSM8K·Qwen |
|---|---|---|---|---|---|---|
| SFT | 72.5 | 31.7 | 18.7 | 57.6 | 29.2 | 31.8 |
| Supervised KD | 73.3 | 34.1 | 22.5 | 57.5 | 31.3 | 35.0 |
| On-policy KD | **36.1** | 34.1 | 25.3 | 53.0 | 31.2 | 33.9 |
| ImitKD | 74.8 | 34.9 | 26.2 | 55.9 | 30.9 | 33.2 |
| **SKD** | **75.3** | **35.0** | **29.1** | **57.1** | **31.7** | **36.6** |

> On-policy KD가 IT 초기화 번역에서 **붕괴**(COMET 36.1) — 초기화 민감성의 직관적 증거. SKD는 전 과제·전 모델族 일관 우위.

Gemma 7B→2B 기준 SFT 대비: 번역 +41.8%, 요약 +230%, 산술 +160%. 수학 지시수행: MATH +198%, GSMplus +360%.

### Findings & Implications

- **§5.3 초기화 robustness**: on-policy KD는 나쁜 학생 초기화에서 저조·낮은 수준에 갱김; SKD는 IT·SFT 모두 robust. **ImitKD(시퀀스 단위 혼합) < SKD(토큰 단위 혼합)** — 토큰 단위 혼합이 시퀀스 단위보다 우수함 실증.
- **§5.4 저자원(100샘플)**: on-policy 저품질로 약세; SKD > supervised·on-policy 전 과제. SFT 과적합이 post-KD 성능 저하 → **end-to-end SKD(SFT 우회)** 가 저자원에서 유리(§5.6, App. I).
- **§5.5 두-단계 ablation**: 전반 supervised KD + 후반 on-policy가 놀랍게도 둘 중 하나만 쓴 것보다 종종 열세. 시퀀스 단계 단순 혼합은 토큰 단위 자연 confidence 부재로 data discrepancy → **adaptive 토큰 단위 혼합(SKD)의 필수성** 시사.
- **App. C**: adaptive K(감소)는 constant K=25에 못 미침 — 인위적 supervised 강제가 off-policy 토큰 도입해 상해; 모델 자연 전이가 더 낫다.
- **App. K**: SKD 학생을 speculative decoding draft로 사용 → token acceptance 71→85% 향상, 1.2× 가속. (SKD가 학습 도구일 뿐 아니라 추론 가속 draft 품질 향상에도 유효)
- **App. L/M**: 거부율 분석; SKD가 교사 직접 샘플링 대비 **연산 ~50% 절감** 추정(학생 draft가 교사 호출 감소). **App. N**: SKD 최저 validation loss.

## Analysis

### Strengths & Significance
- **이론-실증 정합**: 모방학습(DAgger)의 "교사→학생 점진 이동" 원칙을 speculative sampling으로 자동·암묵적 실현. 스케줄 설계 불필요(거부율 = 학생 품질 함수).
- **일반성**: supervised·on-policy KD를 특수케이스로 통합(K 하나의 스펙트럼). 두 baseline 한계를 직교적으로 모두 해소.
- **robustness**: 학생 초기화/데이터 크기/모델族 무관 일관 우위 → 실용성. 특히 on-policy가 실패하는 저품질 초기화·저자원에서 빛남.
- **토큰 단위 혼합 > 시퀀스 단위**: ImitKD·two-stage ablation 대비 우위로 미세 confidence 기반 혼합의 가치 입증.
- **부수 효과**: 학습된 학생이 speculative decoding draft로 우수 → 학습-추론 가속 모두 기여.

### Limitations
- **동일 tokenizer/vocab 필수**: SKD는 speculative decoding 상속 제약으로 교사·학생 동일 vocab 전제. 코드·논문·수락 로직 3곳에서 강제(상세 → [§적용 전제](#적용-전제--동일-tokenizervocabulary-필수)). 이종 tokenizer cross-family 적용은 미지원.
- **교사 호출 비용**: 매 스텝 교사가 interleaved 생성에 참여(7B 교사) → 학생 단독 학습 대비 비용 증가. 다만 App.M은 교사 직접 샘플링 대비 50% 절감으로 정당화.
- **K 탐색**: 본실험은 K=25 고정. App.B는 [5,50] 넓은 범위 우수를 보이나 과제별 최적 K는 미탐색 — 잠재적 추가 이득 가능.
- **batch=1/GPU (grad_acc=1)**: DDP이지만 효율적 batch가 작아 학습 안정성·처리량 한계. 메모리 제약(교사+학생 동시 상주) 때문.
- **adaptive K 미성숙**: 감소형 adaptive K가 constant보다 열세(App.C) — 더 정교한 적응 기준(top-p 결합 등)은 future work로 남음.
- 단일 γ(=5)만; γ 스케줄링 영향 미상세.

### Future Work / Improvements
- 과제별 K 자동 탐색 / top-p+top-K 결합 수락 기준(App.C 제안).
- 더 큰 γ·가변 γ 스케줄링이 거부율·수렴에 미치는 영향 분석.
- 교사 오프로딩/캐싱으로 비용 추가 절감; larger batch를 위한 메모리 최적화.
- 비-영어·코드·다단계 추론 등 과제 확장; [동일 vocab 전제](#적용-전제--동일-tokenizervocabulary-필수)를 넘어 이종 tokenizer 간 적용(vocab 정렬/투영 레이어) 일반화 — speculative decoding 자체의 미해결 영역.

## References
- 논문: [arXiv:2410.11325](https://arxiv.org/abs/2410.11325) (Xu et al., ICLR 2025)
- 코드: [google-research/google-research/speculative_kd](https://github.com/google-research/google-research/tree/master/speculative_kd)
- 선행: On-policy KD (Agarwal 2024, arXiv:2306.13649), ImitKD (Lin 2020), Supervised KD/DistilBERT (Sanh 2020), Speculative Decoding (Leviathan 2023), DAgger (Ross 2011)
- 본 리포 결합 분석: [Draft-OPD]([paper][git]_Draft-OPD_On-Policy_Distillation_for_Speculative_Draft_Models_2026_arxiv.md) — speculative decoding draft model의 on-policy 증류(SKD와 task-adjacent, 단 SKD는 일반 KD·Draft-OPD는 draft 학습 특화)

> [paper][git] https://github.com/imlrz/InMind.git · https://arxiv.org/abs/2607.24368

# Keep It InMind: Benchmarking the Implicit-Association Blind Spot in Agent Memory

## Summary & Outline

agent memory의 **implicit-association blind spot**(저장된 사실이 간접 질의에 "적용"되지 못하는 구조적 실패)를 정의·측정하는 125-task expert-verified 벤치마크 **InMind**. 세 실패 설명(미저장 / model 지식 부족 / 검색 미surface)을 분리하는 paired control을 두어, “사실을 on-demand로는 최대 100% 회상하면서도 간접 집의 적용은 최대 14.4%”라는 깨끗한 결과를 도출. 실패 원인이 model·저장·표현이 아닌 **query-conditioned retrieval 인터페이스 자체**임을 입증하고, always-in-state probe가 84.0%↔16.0% gap의 대부분(68.8%)을 회복함으로써 **routing**(어떤 사실이 visible state에 남아야 하는가)을 미해결 문제로 지정한다.

**두 원본의 관계**: 본 대상은 paper(벤치마크 정의·이론·실험 결과)와 git(InMind 벤치마크 구현체: dataset + evaluation tooling)을 모두 가짐. 논문이 방법론·진단 설계·실험을 담당하고, git 저장소는 그 프로토콜을 reproducible하게 실행 가능한 코드·데이터로 제공. git은 **memory 시스템 adapter가 아닌 evaluation 인프라**(공정한 timeline 주입 · context 캡처 · 표준 judge payload · 검증)에 집중하며, baseline adapter·pinned dependency·paper-aligned 결과는 release roadmap에 미구현(`[ ]`). 본 문서는 paper 방법론(§Problem–§Experiments)과 git 구현(§Repository Architecture)을 모두 포함.
- 코드 원본: [submodule](../source/git/InMind_imlrz) (`source/git/InMind_imlrz`)
- 상세 발췌 → [paper excerpt](../source/paper/Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv.md)

## Repository Architecture — 코드 구조

```
InMind/                          (git submodule: source/git/InMind_imlrz)
├── benchmark/
│   ├── README.md                # 벤치마크 설계·진단 프로토콜 (논문 §2-3 코드 대응)
│   └── dataset/
│       ├── inmind.jsonl         # ★ 125 task (task_id sparse, 0~225)
│       ├── schema.json          # JSON Schema — task 구조 계약 (additionalProperties:false)
│       └── SHA256SUMS           # dataset 무결성 checksum
├── evaluation/
│   ├── README.md                # reproducible 평가 프로토콜 (_countdown로 non-negotiable 명시)
│   ├── background/
│   │   ├── lme_s_background.jsonl  # ★ 고정 47-session/486-turn LME-s trace (MIT, upstream attribution)
│   │   └── manifest.json        # ★ 주입 상수 (inject-turn 40, session 9, 38 post sessions)
│   ├── prompts/                 # answerer + 4종 judge 프롬프트 (진단 분리의 구현)
│   ├── schema/                  # submission.schema.json — per-task 결과 계약
│   ├── examples/                # 제출 샘플
│   └── scripts/                 # ★ 표준라이브러리 only Python 스크립트 4종
│       ├── build_timeline.py        # target memory → 9번째 session 끝 주입 타임라인 생성
│       ├── build_judge_payloads.py  # 메트릭별 judge payload 조립 (answer-blind 강제)
│       ├── validate_submission.py   # 제출 JSONL 구조·coverage 검증
│       └── validate_release.py      # 아티팩트 무결성 self-check (SHA256/상수/prompt 키워드)
├── skills/evaluate-inmind/SKILL.md  # coding agent용 통합 평가 체크리스트 (누출 방지 contract)
└── assets/                      # 논문 figure
```

전체 동작 흐름 — protocol이 강제하는 평가 파이프라인:

```
 per task (125)                     고정 배경                       memory 시스템 (사용자 구현 adapter)
 ─────────────                      ──────────                       ────────────────────────────────
 inmind.jsonl ─┐                lme_s_background.jsonl ─┐    1. empty per-task state (or copy 8-session prefix bank)
 (target pair, │                (47 session, 486 turn)   │    2. 9번째 session 끝에 target turn pair 주입
 naive+indirect│           ┌──── build_timeline.py ─────┘       (source:"inmind_target")
 query,        │────join───►│  sessions 그룹화 + 주입    ──►  3. 나머지 38 session 정상 update 처리
 explanation)  │            │  (manifest 상수 pin)            4. state freeze/snapshot
               └─queries───►│  queries는 timeline "외부"       5. naive_query ─┐  동일 frozen state에서
                            └─────────────────────────┘       6. query ──────┘  독립 평가 (write-back 금지)

   answerer(gpt-5-mini) ◄── {context} 주입 (answer_system.txt) ── system이 retrieve/route/summarize한 "visible context"
        │                          │
   naive.answer            query.answer + query.context (캡처)
        │                          │
        ▼                          ▼
   judge_naive.txt         build_judge_payloads.py ── 메트릭별 payload 분기
   (context+answer)            ├── target-recall : context만 (answer-blind) → judge_target_recall.txt
                               ├── application  : context+answer → judge_application.txt (양쪽 gate)
                               └── answer-only  : answer만 (context-blind) → judge_answer_only.txt
                                        │
                                   gpt-5-mini 이진 judge  {"score":0|1,"reason":...}
                                        │
                               validate_submission.py (125 coverage/구조) → aggregate (1의 수 / 125)
```

특징:
- **모든 script가 표준 라이브러리 only** — 외부 의존성 없이 어디서나 실행. reproducibility 최우선.
- **adapter는 사용자 책임** — repo는 공정한 timeline 주입·context 캡처·표준 judge payload만 제공. 평가 대상 memory 시스템(A-RAG/xMemory/Mem0 등)의 adapter 코드는 미포함(roadmap `[ ]`).
- **누출 방지가 first-class** — `explanation/entity_*/relation/provenance`는 judge-only ground truth로 memory system/retriever/answerer에 절대 노출 금지. `build_judge_payloads.py`에서만 payload로 조립.

## Key Components (git)

- **dataset 계약 (`schema.json`)**: task는 `additionalProperties:false` 고정 필드(`task_id, domain, user_message, assistant_message, naive_query, query, explanation, provenance` + optional `entity_1/entity_2/relation`). `task_id`는 sparse(0~225 중 125개), row index 아님. `provenance.knowledge_source`에 source URL/chunk trace(77 task에 URL, 12는 human-expert, 36은 original 90-task set 잔존).
- **주입 프로토콜 (`build_timeline.py` + `manifest.json`)**: 논문 "38 further sessions"을 manifest 상수(`post_injection_sessions:38`)로 보증. 9번째 session(인덱스 8) 끝에 target pair append, `source:"inmind_target"` 표기. queries는 timeline 외부 분리(memory ingest 금지). phase A 8-session은 immutable prefix bank로 사전 구축 허용(비용 절감). 상세 → [snippet](../source/git/snippets/Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv__timeline_injection.md)
- **진단 분리 judge (`build_judge_payloads.py` + `prompts/`)**: 메트릭별 payload 매핑이 진단 설계를 구현 — target-recall은 answer 미포함(answer-blind), application은 context_recall AND answer_warning 양쪽 gate, answer-only는 context 미포함(retrieval 기여와 답 consequence 분리). `validate_release.py`가 각 prompt의 진단 보장 문구(`Do NOT judge any answer` 등)를 assertion으로 pin. 상세 → [snippet](../source/git/snippets/Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv__judge_diagnostics.md)
- **무결성 검증 (`validate_release.py`)**: SHA256·turn/session count·주입 상수·prompt 키워드·schema required 필드를 assertion으로 검증. 아티팩트 변조/재정렬/재생성 차단. 실제 실행 시 `OK: 125 tasks; 47 sessions; 486 turns; injection session 9; ... SHA-256 verified`确认.
- **agent skill (`skills/evaluate-inmind/SKILL.md`)**: coding agent에게 benchmark 통합 방법을 지시하는 executable 체크리스트. judge-only field 누출 금지·isolated task state·freeze 시점·write-back 금지 등 non-negotiable contract 명시.

**논문 구조 outline**
1. Introduction — macaron/almond allergy 예시로 blind spot 직관 제시
2. The Blind Spot in Query-Conditioned Memory — Retrieval Hypothesis 정식화 + implicit association 정의
3. The InMind Benchmark — domain分布·task 추출·3단 filtering·평가 프로토콜
4. Experiments: Locating the Failure — “not the model / not storage / not representation” 3단 배제
5. What It Takes to Close the Gap — search harder 불충분 / always-in-state 회복 / routing
6–8. Related Work · Limitations · Conclusion
(부록 9–18: filter 상세, taxonomy, 대표 task, 평가 prompt, hyperparameter, injection protocol, always-in-state baseline, public source 목록)

## Problem & Motivation

- **연구 배경**: language agent가 stateless chatbot이 아닌 persistent assistant로 기대받음. 사용자가 한번 말한 fact가 며칠 뒤 다른 대화·전혀 연결 안 짓는 topic에서도 계속 matter해야 함. 이를 retrieval-based memory(RAG interface)로 해결하려는 것이 지배적 패러다임.
- **풀고자 하는 문제 (implicit-association blind spot)**: retrieval memory의 숨겨진 전제 — “필요한 memory는 query와 닮아 있다” — 가 world knowledge에 의해 위반되는 실패 모드를 정의하고 측정. 예: tree-nut allergy note와 macaron 요청은 알몬드가루라는 bridge로 연결되나 두 텍스트가 token/topic/embedding 어느 곳에서도 닮지 않아 retriever가 신호를 못 봄.
- **기존 접근의 한계**:
  - 기존 memory 시스템(vector/graph/agentic)은 “무엇을 저장·검색하나”를 개선했으나 query-then-retrieve interface 자체는 계승 → relevance 판단을 model(LLM)이 아닌 retriever(similarity)가, model이 memory를 보기 전에 내림.
  - 기존 memory benchmark(LongMemEval, LoCoMo, ImplicitMemBench, LoCoMo-Plus)는 세 실패 설명을 분리 못 함. 특히 연관을 LLM이 저작하면 construction/test에 같은 prior가 적용돼 모델이 이미 만드는 연관만 편향 샘플링.
  - 사용자가 신뢰를 보정하려 테스트하는 상호작용(직접 질문)은 잘 작동하고 memory 의존 상호작용은 거의 chance → **신뢰 역설**: recall에 답하고 macaron을 추천하는 agent가 받을 자격 없는 신뢰 획득.

## Contributions

- **개념**: implicit-association blind spot 정의 + 현 memory 아키텍처가 암묵적 가정하는 **Retrieval Hypothesis**를 최초로 명시·직접 검증.
- **데이터셋 (실증 기여)**: 125-task expert-verified 벤치마크 InMind (113 source-grounded, 12 expert-authored). 세 실패 모드(미저장 / model 지식 부족 / 저장됐으나 미surface)를 분리하는 **paired control**(naive query / in-context control / target recall) 설계. bridge가 공공 문서(FDA/OSHA/CPSC/USCIS 등)에 anchor → model prior가 아닌 expert knowledge 기반 분포.
- **이론·진단**: 6개 vector/graph/agentic memory 시스템 평가 — naive recall 최대 100%인데 indirect application 최대 14.4/16.0%, in-context backbone 84.0%. 8× 차원 embedding도 gap 미회복. 패러다임 속성임을 입증.
- **방향 제시**: always-in-state probe(200-line markdown profile, query-time retrieval 전무)가 68.8%로 gap 대부분 회복 → 실패를 query-conditioned interface에 국한, **routing**을 InMind가 측정하려는 open problem으로 지정. (시스템 제안은 하지 않음 — 의도적 진단.)

## Method

### 핵심 모델 — Retrieval Hypothesis와 그 위반

```
retrieval-based memory:
  ℳ̂ = Retrieve(ℳ; q, θ)        # relevance를 model(LLM) 보기 "전"에 retriever가 결정
  a   = LLM(q, ℳ̂)              # 모델은 ℳ̂만 추론

              query q ──┐
                         ▼
              ┌──────────────────┐    semantic-similarity
   memory ℳ →│  Retrieve(θ)     │──ℳ̂──►  LLM(world knowledge) ──► answer
              └──────────────────┘   (bridge 인식 가능 component)   만
                         ▲
            relevance를 "여기서" 결정 → bridge 신호가 q에 없으면
            결정적 memory는 ℳ̂에서 빠지고 LLM은 영영 못 봄
```

- **implicit association (m,q) 3조건**: (1) Necessity(m이 q 답에 필수) (2) Semantic distance(m,q가 lex/embedding/topical로 멂) (3) Knowledge bridge(m→k→q 인 외부 지식 k 존재, 둘 다에 명시 无). retriever가 건너야 할 semantic-similarity link에 신호를 주지 않는 pair.
- **falsifiable signature**(InMind가 검증): in-context 모델은 간접 집의 잘 답 · retrieval에서 explicit recall은 천장 · indirect application은 붕괴. (난이도/망각/model 약함 어느 설명도 이 패턴 예측 못 함.)

### 평가 프로토콜 — 세 설명을 가르는 설계

```
주입: target memory turn ──► LongMemEval-s 47-session trace "중간" 삽입
                              │  (memory 시스템 정상 동작, 38 session 일상 상호작용 생존)
                              ▼
        ┌───────────── 같은 task에 2회 test ─────────────┐
        │                                                 │
   ① Naive query                                  ② Indirect query
   "네 알레르기 뭐야?" (직접)                    "마카롱 레시피 알려줘" (간접)
        │                                                 │
   [저장 생존 여부]                             ┌───────────┴───────────┐
                                               ▼                       ▼
                                  Target Recall (answer-blind)   Application (context-aware)
                                  context에 fact 있나?            답이 bridge 따르나?
                                  = retrieval이 surface 했나     = model이 fact 보고 적용했나
```

- 분리 목적: Application 실패 시 (a) Target Recall=0 ⇒ **retrieval miss**(fact가 context에 안 닿음), (b) Target Recall=1 & Application=0 ⇒ **reasoning miss**(model이 fact 보고도 bridge 못 침). InMind 결과는 (a)가 압도적.
- 간접 측정 둘 다 이진 GPT-5-mini judge. Target Recall은 답안 미제공(answer-blind)으로 retrieval 품질을 model 추론과 decouple.

### 벤치마크 구축 파이프라인

domain 가중치 ← Anthropic 37,657 personal-guidance 대화 분석. 공공 collection(FDA/OSHA/CPSC/USCIS/CDC 등)에서 3,380 chunk 비례 샘플링(각 chunk source trace 보존). Gemini 2.5 Flash 메인 extractor(+ Claude Sonnet/Opus, DeepSeek V4 Pro)가 chunk마다 (personal fact / naive query / indirect query / source-grounded bridge) record 생성 → 1,000 candidate. **3 content filter**(모두 system 출력 미참조): similarity(BM25+MiniLM, BGE-small-v1.5로 재검증해 InMind만 target이 distractor보다 query에 가깝지 않음) / conflict(단일 배경 trace persona 모순 폐기) / expert verification. → 113 + 12 = **125 task**.
상세 발췌(정식화·정의·filter·대표 task) → [excerpt](../source/paper/Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv.md)

## Experiments & Results

### Benchmark Datasets
- **InMind**: 125 task, 10개 life domain. 113 source-grounded(citable public source), 12 expert-authored, 전부 expert-verified. 주입·간섭 trace는 LongMemEval-s(LME-s, 47-session). health/wellness/safety 치중(실패 판정이 명백한 영역 선별).

### Setup
- **Backbone control**: GPT-5-mini에 target memory + query 직접 in-context → bridge world knowledge 상한.
- **6 retrieval 시스템**: A-RAG(agentic, 최대 15-step), xMemory, Mem0, A-Mem, HippoRAG 2(graph), MemoryOS(hybrid, 이미 profile을 매 답 context에 포함). 각 MiniLM(384-dim) ∥ text-embedding-3-large(3,072-dim). 단발 Naive RAG control(MiniLM/emb3-large/BM25).
- answerer兼judge: GPT-5-mini(이진).

### Results — Main (Table 1, %)

| System (emb) | Naive | Target Recall | Application |
|---|---|---|---|
| **Backbone** (in-context) | – | 100.0 | **84.0** |
| Naive RAG Dense (MiniLM / emb3) | 92.0 / 97.6 | 0.8 / 6.4 | 9.6 / **16.0** |
| Naive RAG BM25 | 53.6 | 3.2 | 9.6 |
| A-RAG (MiniLM / emb3) | 97.6 / 93.6 | 5.6 / 11.2 | 4.8 / 7.2 |
| xMemory (MiniLM / emb3) | 84.8 / 91.2 | 3.2 / 6.4 | 4.8 / 6.4 |
| Mem0 (MiniLM / emb3) | 76.0 / 76.8 | 2.4 / 6.4 | 3.2 / 6.4 |
| A-Mem (MiniLM / emb3) | 99.2 / 100.0 | 2.4 / 12.0 | 6.4 / 9.6 |
| HippoRAG 2 (MiniLM / emb3) | 93.6 / 96.0 | 0.8 / 2.4 | 8.8 / 5.6 |
| MemoryOS (MiniLM / emb3) | 87.2 / 96.8 | 2.4 / 7.2 | 8.0 / **14.4** |

한 줄: **naive recall ≤ 100.0%, indirect application ≤ 16.0%**(memory 시스템 자체 최대 14.4%, Naive RAG보다도 낮음). 시스템 간 분산 < 전체가 backbone(84%)과 떨어진 거리 → blind spot은 구현이 아닌 **paradigm 속성**.

### 결과 — Always-in-State diagnostic (Table 2)

| Method | Naive | Indirect | Δ(Indirect − Backbone) |
|---|---|---|---|
| Backbone (GPT-5-mini) | – | 84.0 | 0.0 |
| Best retrieval config | 97.6 | 16.0 | **−68.0** |
| **Always-in-state** (200-line md, retrieval 전무) | 98.4 | **68.8** | −15.2 |

### Findings & Implications

- **3단 배제 결과 (clean signature)**:
  - *Not the model* — in-context 시 84.0% (Wilson 95% CI [76.6, 89.4]) vs 최고 retrieval [10.6, 23.4]. 84↔16을 가른 건 능력이 아닌 **접근(access)**.
  - *Not storage* — naive 최대 100% → fact write됨·38 session 생존·on-demand 회상 가능. Target Recall이 위치 특정: 간접 query 하 결정 fact가 context 도달 0.8–5.6%(MiniLM)/2.4–12.0%(emb3)에 불과. **저장됐고 쓸 수 있는데 결정적 순간에 부재**.
  - *Not representation* — 8× 차원 embeddings가 모든 시스템 target recall 상승시키나 application gap은 사실상 유지(개별 shift는 ±4–5점 binomial CI 내 noise). embedding이 흡수한 분포적 trace로 bridge 일부 근사하나 약리/법률/종교/발달적 bridge는 co-occurrence로 적힌 적 없음 → 잘못된 방향에서 world model 근사.
- **search harder 불충분**: agentic searcher A-RAG(15-step loop)가 오히려 최하위(4.8/7.2%). 같은 blind 표현에서 나온 query로는 iteration가 search 못 구함; memory store는 query에 닮은 것만 반환하므로 bridge-aware 질문도 아무 stored fact가 안 닮음. bridge hypothesize 자체가 store 확인 전 확률적 추측 → safety가 신뢰성 요하는 곳에서 fix 아닌 open direction.
- **always-in-state가 sufficiency 증명**: model과 memory 사이 query-time retriever가 없다는 설계 변경 한 가지가 vector/graph/agentic search 누적 장치를 능가. retriever는 설계대로 작동 → **Retrieval Hypothesis 자체가 실패 지점**.
- **routing이 open problem**: retrieval(lossless 기록)과 always-visible state(distant bridge 추론)은 complementary, hybrid가 결합하나 criterion이 없음. 어떤 memory가 persistent state를 얻나·어떤 query가 비싼 search를 정당화하나 = routing decision. 현재는 recency/frequency/heat proxy → distant bridge 못 봄. **InMind는 routing decision 자체를 측정하는 첫 benchmark**.

## Analysis

### Strengths & Significance
- **실험 설계의 깔끔함**: 단일 논문 내에서 세 경쟁 설명을 모두 rule-out하는 controlled 구조. 단순한 “시스템 비교”가 아니라 *falsifiable signature*를 세워 검증하는 연역적 설계 — (i) in-context 상한, (ii) naive 생존, (iii) indirect 붕괴 패턴이 3단 배제와 정확히 대응.
- **측정 분리의 정교함**: Target Recall(answer-blind)과 Application(context-aware)을 분리해 retrieval miss와 reasoning miss를 구분. Appendix 14.2 인간 감사(Target Recall 97.0%, Application 85.0%·오류 전부 false positive)로 자동 judge 신뢰성까지 검증. self-preference bias가 gap 제조 불가함을 논리로 방어.
- **construction provenance 차별점**: bridge를 공공 문서(FDA/OSHA 등)에 anchor하고 extractor가 source chunk 외 지식을 못 쓰게 막음 → 기존 LLM-저작 benchmark의 모델-prior 편향(construct/test 동일 prior)을 회피. 113/125 citable source로 감사 가능.
- **범용적 시사점**: 6개 시스템(vector/graph/agentic) 간 분산 < 패러다임과의 거리 라는 결과는 특정 구현 비판이 아닌 **query-conditioned interface 자체**에 대한 구조적 진단 → field가 “이미 작동하는 곳(storage/indexing)”에 쏟은 노력의 방향 재조정 촉구.
- 안전-critical 영역(알레르기/약물/규제)에 치중해 실패를 명백히 채점 가능하게 만든 선택이 현실적 타당성을 높임.

### Limitations
- **저자 인정 한계**: n=125로 몇 %는 sampling noise(결론은 60–70점 효과에 근거). health/wellness/safety 치중 → humor/etiquette/long-term goal/institutional policy 미커버. GPT-5-mini answerer兼judge(self-preference; 독립 judge model이 가장 큰 방법론 gap). bridge factual/uncontested 가정(실제는 probabilistic/jurisdiction-dependent 가능). InMind 최적화 agent의 over-warn risk(음수 control 미구�).
- **분석자 관찰 한계**:
  - always-in-state probe는 *sufficiency*는 보이나 단일 시스템 controlled ablation이 아님(state format·updater도 달라짐)을 저자도 인정 → routing이 실제로 *최적* 경로인지는 추가 ablation 필요.
  - answerer가 GPT-5-mini 단일 → 다른 backbone(오픈 모델·in-context 약한 모델)에서 in-context 상한 84%가 유지되는지 미검증. bridge 지식이 model에 전적으로 의존하므로 약한 모델에선 signature가 달라질 수 있음.
  - 200-line cap profile로 volume 증가 시 fact crowding-out 문제를 스스로 지적 → routing이 “미룰 뿐”이라는 결론이, always-in-state의 scalability 한계를 함께 담고 있어 장기 해법으로서의 한계 명확.
  - routing criterion가 “vertical-specific(의료/재무/coding)”라 명시되나, InMind 자체는 heterogeneous vertical을 평가하지 않으므로 routing 해법의 직접 검증은 후속 과제에 남음.
  - **git 저장소 한계 (분석자)**: repo는 evaluation 인프라(dataset + timeline/judge/validate tooling)만 제공하고, 논문이 평가한 6개 memory 시스템의 **baseline adapter·pinned dependency 버전·paper-aligned 집계 결과는 미포함**(release roadmap `[ ]`). 즉 코드만으로 논문 결과를 그대로 재현할 수 없으며, 사용자가 직접 adapter를 구현해 평가해야 함. provenance도 original 90-task set 잔존 36건은 knowledge_source 누락.

### Future Work / Improvements
- **knowledge-conditioned relevance function** — conclusion이 지목하는 핵심. query와 memory를 함께 본 뒤 relevance를 결정하는 함수(=retriever가 model처럼 추론) 설계; 본 논문은 제안 않고 측정 인프라만 제공.
- **routing 해법 + 음수 control**: over-warning penaly를 채점하는 negative control 추가; hybrid 시스템의 router가 persistent slice + fine-grained store 간 어떻게 분배하는지 측정하는 routing-specific 메트릭.
- 독립 judge model 도입 및 다중 backbone으로 in-context 상한/범용성 재검증.
- 도메인 확장(etch/humor/정책) 및 probabilistic/disputed bridge 처리.

## References
- 논문: [arXiv:2607.24368](https://arxiv.org/abs/2607.24368) · [Project Site](https://keep-it-inmind.github.io/) · [Leaderboard](https://keep-it-inmind.github.io/leaderboard/)
- git 저장소: [github.com/imlrz/InMind](https://github.com/imlrz/InMind) (submodule `source/git/InMind_imlrz`) · paper 발췌: [source/paper](../source/paper/Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv.md)
- 코드 스니펫: [timeline injection](../source/git/snippets/Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv__timeline_injection.md) · [judge diagnostics](../source/git/snippets/Keep_It_InMind_Benchmarking_the_Implicit-Association_Blind_Spot_in_Agent_Memory_2026_arxiv__judge_diagnostics.md)
- 주요 대상 memory 시스템: A-RAG(Du et al. 2026), xMemory(Hu et al. 2026), Mem0(Chhikara et al. 2025), A-Mem(Xu et al. 2025), HippoRAG 2(Gutierrez et al. 2025), MemoryOS(Kang et al. 2025)
- 인접 벤치마크: LongMemEval(Wu et al. 2024), LoCoMo(Maharana et al. 2024), LoCoMo-Plus(Li et al. 2026), ImplicitMemBench(Qin et al. 2026)
- 본 repo 관련 분석: [AgenticSTS](AgenticSTS_A_Bounded-Memory_Testbed_for_Long-Horizon_LLM_Agents_2026_arxiv.md)(bounded-memory contract), [Memora](Memora_A_Harmonic_Memory_Representation_Balancing_Abstraction_and_Specificity_2026_ICML.md)(harmonic memory), [Remember When It Matters](Remember_When_It_Matters_Proactive_Memory_Agent_for_Long-Horizon_Agents_2026_Meta_AI.md)(proactive memory intervention) — 모두 query-conditioned retrieval의 한계를 우회하려는 인접 시도.

> [paper][git] https://github.com/zorazrw/agent-workflow-memory · https://arxiv.org/abs/2409.07429

# Agent Workflow Memory

## Summary & Outline

LM 기반 web agent가 성공한 trajectory에서 LLM prompt 지시로 **반복되는 sub-routine을 잘라내고 example-specific 값을 variable(`{location}`)로 추상화**하여 workflow를 생성, 이를 website별 `.txt` 파일로 저장한 뒤 system message에 append하여 후속 task에 활용. 벡터 DB나 retrieval 시스템 없이 website당 평균 7개의 workflow 텍스트만으로 WebArena 51.1% / Mind2Web 24.6% relative 성공률 향상을 달성. offline(training example 활용)과 online(test query만으로 supervision-free) 모두 지원하며, snowball effect로 단순 workflow 위에 복잡 workflow를 점진적 구축.

```
[추출]  성공 trajectory → GPT-4o prompt 지시 → sub-routine 단위 잘라내기 + variable 추상화
        예: fill('145','New York') → fill('145','{location}')
            전체 task가 아닌 "검색" 부분만 sub-routine으로 분리

[저장]  website별 workflow.txt 파일 (평균 7개, double-line break로 구분)
        별도 DB/인덱스 없음, 그냥 텍스트 파일

[반영]  매 step마다 workflow 텍스트 전체를 system message에 append
        retrieval 없이 전체 주입 (7개라 가능)
        WebArena: sys_msg += workflow_text
        Mind2Web: exemplar(user message)로 token limit 내 포함

[누적]  online: test query가 올 때마다 성공 trajectory → 새 workflow 추출 → 파일 갱신
        단순 workflow가 subgoal이 되어 복잡 workflow에 재사용 (snowball effect)
```

논문 구조:
1. Introduction — human workflow learning 모티베이션, snowball effect
2. AWM — problem statement, workflow representation, induction & utilization (offline/online)
3. Experiments — WebArena (§3.1), Mind2Web (§3.2)
4. Workflow representation ablation — sub-routine/abstract format, text vs code, NL vs HTML
5. Workflow in action space — AWM_AS (high-level function wrapping)
6. Related work
7. Conclusion

## Problem & Motivation

- 연구 배경: LM 기반 web agent는 각 task를 독립적으로 처리하여 과거 성공·실패 경험을 재사용하지 못함. 고정 예시 세트를 in-context learning이나 fine-tuning으로 통합하므로 환경 변화에 대한 robustness 부족
- 풀고자 하는 문제: long-horizon web navigation task에서 재사용 가능한 task workflow를 자동 학습하여 agent의 성공률과 효율성을 향상
- 기존 접근의 한계:
  - 고정 예시 기반 (Synapse, MindAct): example-specific context에 편향되어 cross-task/cross-domain 일반성 부족
  - Human-written workflow (SteP): 도메인별 인적 감독 필요, 확장성 부족
  - 각 task 독립 처리: 과거 경험 비학습, 적응 불가

## Contributions

- **방법론**: Agent Workflow Memory (AWM) — LM 기반 workflow induction + memory integration으로 재사용 가능한 sub-routine을 자동 추출·활용
- **유연성**: offline (training example 활용)과 online (supervision-free) 모두 지원, 추가 annotation 없이 test query만으로 작동
- **실증**: WebArena 51.1% relative SR 향상, Mind2Web 24.6% relative step SR 향상, human workflow 기반 SteP 능가
- **일반성**: cross-task, cross-website, cross-domain에서 train-test distribution gap이 widening될수록 우위 확대 (8.9 → 14.0 absolute points)

## Method

### 전체 구조

```
                     ┌──────────────────────────────────────────────┐
                     │            AWM Pipeline (per website)          │
                     │                                              │
  Test Query q_t ──► │  1. Inference: L(q_t, M, o) → action traj    │
                     │                    │                          │
                     │  2. Evaluate:  L_eval(e_t) ∈ {0, 1}           │
                     │              │          │                     │
                     │           fail(0)    success(1)               │
                     │              │          │                     │
                     │              ▼          ▼                     │
                     │           skip    3. Induce: I(e_t) → {w_t}  │
                     │                         │                     │
                     │              4. Memory: M + {w_t} → M          │
                     │                         │                     │
                     │                    next query q_{t+1}         │
                     └──────────────────────────────────────────────┘

  Offline:  I(E_train) → W_offline  (one-shot, before all tests)
            L(q, M + W_offline, o) → a  (static memory at test time)

  Online:   iterate induce→integrate→utilize per test query (streaming)
            L_eval judges success → only successful trajectories become workflows
```

### Workflow Representation

각 workflow w = (d, P):
- **d** (description): workflow의 high-level goal을 요약한 NL 설명 (예: "find a place by its name")
- **P** (trajectory): step 시퀀스 (p_1, p_2, ...). 각 step은 (1) environment state NL 서술, (2) reasoning, (3) executable action으로 구성

예시:
```
## enter_flight_locations
Given that you are on the flight booking page, this workflow enters the departure
and destination city/airport for your flight.
[link] From Departure Airport -> CLICK
[textbox] Origin City -> TYPE: {your-origin-city}
[link] {best-popup-option} -> CLICK
```

**추상화 핵심**: example-specific value("New York")을 variable("{your-origin-city}")로 대체하여 generality 확보. invariant element(검색 버튼 id 등)은 원래 값 유지.

### Memory Extraction (Workflow Induction)

추출은 AWM의 핵심 메커니즘으로, 성공한 concrete trajectory에서 **반복되는 sub-routine을 찾고 example-specific 값을 variable로 추상화**하여 재사용 가능한 workflow를 생성.

**구체 예시 — map website**:

입력 (2개 성공 trajectory):
```
Task 1: "Show me New York on the map"
  fill('145', 'New York')     ← 검색창 id=145
  click('147')                ← 검색 버튼 id=147

Task 2: "Show me Boston on the map"
  fill('145', 'Boston')
  click('147')
```

처리 과정:
1. LLM evaluator가 success로 판정한 trajectory만 수집
2. `intent_template_id` 기반 deduplication — 같은 template에서 1개만 샘플링
3. instruction + one-shot + examples를 GPT-4o에 전달 (temperature=1.0)

출력 (추상화된 workflow):
```
## Find a place by its name
To find a location on the map, I will search for it.
fill('145', '{location}')     ← "New York"/"Boston" → {location} 추상화
click('147')                   ← invariant id 유지
```

"New York"과 "Boston"이 `{location}`으로 추상화되면서 Chicago, Seattle 등 모든 도시 검색에 재사용 가능. invariant element(button id `145`, `147`)는 값 유지하여 환경 일관성 보장.

**Snowball effect**: 단순 workflow가 subgoal이 되어 점진적 복잡화. 이후 "Tell me the zip code of Boston" task가 들어오면, agent가 "Find a place by its name"의 앞부분을 재사용하고 새 step을 추가하여 "Get the zip code of a place" workflow를 구축. → [Snowball Effect 섹션 참조](#snowball-effect-building-complex-workflows)

### LM-based Induction Module

```
Experience E (successful trajectories)
       │
       ▼
  ┌────────────┐     instruction.txt    one_shot.txt
  │ Formatting │ ──►  + Concrete Examples + Summary Workflows
  └────────────┘              │
       ▲                      ▼
  template_id       ┌─────────────┐
  dedup (n=1)       │  GPT-4o     │  temperature=1.0
                   │  LLM call    │  max_tokens=2048
                    └─────────────┘
                           │
                           ▼
                  Induced Workflows W
                  (double-line break로 분리)
```

instruction prompt 요구사항:
- 반복되는 sub-routine만 추출 (full task가 아닌 finer granularity)
- 최소 2 step 이상
- non-fixed element를 descriptive variable로 대체
- invariant element(id 등)는 값 유지
- 중복/유사 workflow 금지

상세 코드 → [induction_prompt snippet](../source/git/snippets/Agent_Workflow_Memory_2024_arxiv__induction_prompt.md)

### Online vs Offline

| 측면 | Offline | Online |
|------|---------|--------|
| 입력 | training examples (ground-truth) | test queries only |
| Induction 시점 | test 전 one-shot | test 중 streaming (per query) |
| 평가 기준 | ground-truth trajectory | LLM evaluator (Pan et al., 2024) |
| Memory | static W_offline | 누적 {w_1, ..., w_t} |
| Train-test gap 영향 | 영향 받음 | 영향 없음 (test 분포에 직접 적응) |

Online streaming pipeline 상세 코드 → [online_pipeline snippet](../source/git/snippets/Agent_Workflow_Memory_2024_arxiv__online_pipeline.md)

### Memory Retrieval & Integration

AWM은 workflow 수가 website당 평균 7.3개로 적으므로, **기본 모드에서는 retrieval 없이 전체 workflow를 context에 주입**합니다. 두 벤치마크에서 주입 위치만 다름:

- **WebArena**: workflow 텍스트를 system message 끝에 append. 매 step마다 workflow 파일을 읽어 `sys_msg += workflow_text`로 결합.
- **Mind2Web**: workflow 텍스트를 exemplar(user message)의 첫 번째 항목으로 배정. concrete example들과 함께 token limit 내에서 최대한 많이 포함.

**대안: FAISS semantic retrieval** (`mind2web/workflow/retrieve.py`):
workflow 수가 많을 때를 대비한 optional 검색 모드. 논문 메인 실험에서는 미사용이지만 cross-website/cross-domain 설정에서 `random` 모드와 함께 활용.

```
1. 각 workflow의 name + docstring을 embedding으로 변환
   예: "Find a place by its name\nTo find a location on the map..."
   → OpenAIEmbeddings(text-embedding-ada-002)

2. FAISS 인덱스 구축

3. test query로 similarity search
   query: "Tell me the zip code of Boston"
   → top-k: "Get the zip code of a place" (score 높음)
             "Find a place by its name" (score 중간)
             ...

4. top-k workflow만 output 파일에 저장 → agent context에 주입
```

세 가지 모드 비교:

| 모드 | 동작 | 사용 실험 |
|------|------|-----------|
| 전체 주입 | website 그룹의 모든 workflow를 context에 포함 | WebArena online, Mind2Web cross-task |
| random | 여러 website의 workflow 중 임의 선택 | Mind2Web cross-website/domain (AWM_offline baseline) |
| semantic | FAISS embedding similarity로 top-k 선택 | 대안 모드 (논문 메인 실험 미사용) |

상세 코드 → [memory_integration snippet](../source/git/snippets/Agent_Workflow_Memory_2024_arxiv__memory_integration.md)

### Snowball Effect (Building Complex Workflows)

```
Example 1: "Show me {location} on the map"
    → induces: "Find a place by its name"
      fill('145', {location}) → click('147')

Example 10: "Tell me the zip code of {location}"
    → adopts earlier workflow's first steps
    → adds: send_msg_to_user("The zip code is {zip-code}")
    → induces: "Get the zip code of a place" (builds on simpler workflow)

Example 20: "Decide if I can drive from A to B in given time"
    → builds on "Find a place" + "Get zip code" patterns
    → induces even more complex workflow
```

단순 workflow가 subgoal 역할을 하여, 후속 task에서 simpler workflow의 앞부분을 재사용하고 확장하는 방식으로 점진적 복잡화.

## Experiments & Results

### Benchmark Datasets

AWM은 web navigation agent가 자연어 지시를 받아 실제 웹사이트에서 행동을 수행하는 두 벤치마크에서 평가합니다. 두 벤치마크는 평가 방식과 data 구성이 크게 다릅니다.

#### WebArena — 실행 기반 평가 (실제 웹사이트 환경)

**구조**: 실제 동작하는 5개 웹사이트 환경(자체 호스팅)에서 agent가 브라우저를 조작. 각 task는 자연어 instruction + 시작 URL + 정답 기준으로 구성.

```
5개 website:
  shopping (e-commerce)  │ shopping_admin (CMS 관리자)  │ reddit (소셜 포럼)
  gitlab (개발 협업)      │ map (OpenStreetMap)
812개 task, 각 task는 intent_template에서 변수 치환으로 생성
```

**Task 데이터 예시** (`config_files/test.json`):
```json
{
  "sites": ["map"],
  "task_id": 2,
  "intent_template": "Show me {{location}} on the map",
  "instantiation_dict": {"location": "Yale University"},
  "intent": "Show me Yale University on the map",
  "eval": {
    "eval_types": ["string_match"],
    "reference_answers": {"exact_match": "06511"}
  },
  "intent_template_id": 12
}
```

하나의 template("Show me {{location}} on the map")에서 여러 location으로 다수 task가 생성됨. `intent_template_id`로 같은 template끼리 묶을 수 있어, AWM은 dedup에 이를 활용.

**Agent 행동**: 매 step마다 웹페이지의 accessibility tree를 관측하고, `click('147')`, `fill('145', 'Yale University')`, `send_msg_to_user("06511")`, `stop()` 같은 programmatic action을 생성. 실제 브라우저에서 실행되어 상태가 변함.

**평가**: agent가 `send_msg_to_user()`로 보낸 최종 응답이 reference answer와 일치하면 success. step 수도 측정 (적을수록 효율적). test-only 데이터이므로 training example이 없어 **online AWM만 적용**.

#### Mind2Web — step-wise 평가 (정적 HTML 스냅샷)

**구조**: 200+ 실제 웹사이트에서 수집한 HTML 스냅샷과 human annotation. 실제 브라우저가 아닌 정적 페이지에서 step별로 정답 element를 맞히는 방식.

```
3가지 generalization split:
  cross-task:     동일 website 내 다른 task (train 있음 → offline 가능)
  cross-website:  같은 domain의 다른 website (예: apple → bestbuy)
  cross-domain:   완전히 다른 domain (예: macays shopping → reddit social)
```

**Task 데이터 예시**:
```json
{
  "confirmed_task": "Find flights from Seattle to New York on June 5th",
  "website": "delta",
  "domain": "Travel",
  "subdomain": "Airlines",
  "actions": [
    {
      "pos_candidates": [{"backend_node_id": "a12", "rank": 0}],
      "neg_candidates": [...],
      "operation": {"op": "CLICK", "value": ""},
      "action_reprs": "[link] From Departure Airport -> CLICK"
    },
    {
      "operation": {"op": "TYPE", "value": "Seattle"},
      "action_reprs": "[textbox] Origin City -> TYPE: Seattle"
    },
    ...
  ]
}
```

각 step마다: (1) HTML에서 정답 element(pos_candidate)를 agent가 선택했는지 → **element accuracy**, (2) action 종류(CLICK/TYPE/SELECT)와 value가 맞는지 → **action F1**, 두 개 모두 맞으면 → **step success rate**, task의 모든 step이 성공 → **task success rate**.

**차이점 요약**:

| | WebArena | Mind2Web |
|---|---------|----------|
| 환경 | 실제 동적 웹사이트 (실행 기반) | 정적 HTML 스냅샷 (예측 기반) |
| Action | `click('147')` program 실행 → 페이지 변화 | `CLICK [element_id]` 정답 비교만 |
| 평가 | 최종 응답 string match (success/fail) | 매 step별 element + action 정확도 |
| Train data | 없음 (test-only) | cross-task split에 train 존재 |
| AWM 모드 | online only | offline + online |
| Workflow 표현 | `click('147')` ( 실제 element id) | `[link] From Departure -> CLICK` (element type + label) |

### Setup

- Model: GPT-4 (gpt-4-0613), temperature 0.0
- WebArena: BrowserGym framework, accessibility tree 표현, online AWM only (test-only 데이터)
- Mind2Web: offline + online, GPT-3.5-turbo 및 GPT-4
- Evaluation: WebArena는 execution-based success rate + step count; Mind2Web는 element accuracy, action F1, step SR, task SR
- Baselines: BrowserGym (autonomous), SteP (human workflow), AutoEval (eval+refine), MindAct, Synapse, CogAgent

### Results

**WebArena (Table 1)**:

| Method | Total SR | Shopping | CMS | Reddit | GitLab | Maps | # Steps |
|--------|----------|----------|-----|--------|--------|------|---------|
| *SteP (human wf) | 33.0 | 37.0 | 24.0 | 59.0 | 32.0 | 30.0 | - |
| BrowserGym_ax-tree | 20.2 | 25.5 | 18.1 | 23.5 | 15.0 | 17.2 | 31.9 |
| **AWM** | **35.5** | **50.9** | **31.8** | **43.3** | 5.9 | **30.8** | **29.1** |

- +51.1% relative SR over BrowserGym baseline
- 인적 감독 없이 SteP(human workflow)를 +7.6% relative로 능가
- 모든 website에서 개선 (11.8–30.7 absolute points), GitLab만 예외적으로 5.9% (template 부족 추정)
- 평균 2.0 step 감소, AutoEval 대비 40.8 step 감소

**Mind2Web Cross-task (Table 3, offline)**:

| Method | Elem Acc | Action F1 | Step SR | SR |
|--------|----------|-----------|---------|-----|
| MindAct_3.5 | 20.3 | 56.6 | 17.4 | 0.8 |
| Synapse_3.5 | 34.0 | 52.8 | 30.6 | 2.8 |
| **AWM_3.5** | **39.0** | **56.6** | **34.6** | **2.8** |
| MindAct_4 | 41.6 | 60.6 | 36.2 | 2.0 |
| **AWM_4** | **50.6** | 57.3 | **45.1** | **4.8** |

- GPT-4 기준 +24.6% relative step SR, +140% relative task SR
- Synapse(concrete examples) 대비 +5.0 element accuracy → abstract workflow가 element selection bias 감소
- Action F1은 MindAct 대비 약간 낮음 (workflow guideline이 특정 action으로 편향시키는 부작용)

**Cross-website / Cross-domain (Table 4, online)**:

| Method | Cross-Task Step SR | Cross-Website Step SR | Cross-Domain Step SR |
|--------|--------------------|-----------------------|----------------------|
| MindAct | 36.2 | 30.1 | 18.6 |
| AWM_offline | 45.1 | 33.7 | 32.6 |
| AWM_online | 43.6 | **33.9** | **35.5** |

- distribution gap widening 시 AWM_online 우위 확대: +7.4 → +3.8 → +16.9 absolute points
- AWM_online은 training data 의존성이 없어 domain gap에 영향 없음

**Workflow Quality (Table 10)**:

| Metric | WebArena | Mind2Web |
|--------|----------|----------|
| # Workflows per website | 7.4 | 7.3 |
| Coverage | — | 0.40 |
| Function Overlap | 0.08 | 0.20 |
| Utility Rate | 0.94 | 0.91 |

### How AWM Improves at Test Time — 구체 시나리오

**WebArena map website online 시나리오** (test query 40개가 순차 들어옴):

```
Query 1: "Show me Yale University on the map"
  └─ agent가 처음부터 action 탐색 (workflow 없음)
     fill('145','Yale University') → click('147') → send_msg_to_user(...)
  └─ AutoEval 판정: success ✓
  └─ workflow 추출: "Find a place by its name"
       fill('145','{location}') → click('147')
  └─ workflow/map.txt 갱신

Query 3: "Show me Pittsburgh Airport on the map"
  └─ system message에 "Find a place by its name" workflow 포함
  └─ agent가 workflow를 참고하여 바로 fill('145','Pittsburgh Airport') → click('147')
  └─ 이전보다 빠르고 정확하게 성공 ✓

Query 10: "Tell me the zip code of Yale University"
  └─ agent가 "Find a place by its name"의 앞부분 재사용 + 새 step 추가:
     fill('145','Yale University') → click('147') → send_msg_to_user("06511")
  └─ success ✓ → "Get the zip code of a place" workflow 추출

Query 20: "How long does it take to drive from CMU to Cold Stone Creamery?"
  └─ "Find a place" + directions workflow 모두 memory에 있음
  └─ agent가 directions 패턴(click('149') → fill from/to → select_option → click('171')) 참고
  └─ 성공률 30.8% (baseline 17.2% → +13.6 absolute)
```

→ 0-40 example 구간에 rapid learning curve, 이후 안정화 (Figure 5). 40개만 지나면 baseline 대비 22.5 point gap 형성 (Figure 1).

**Mind2Web cross-task offline 시나리오** (training example에서 workflow 미리 추출):

```
[Training] delta 항공 웹사이트의 training examples:
  "Find flights from Seattle to New York on June 5th"
  "Find my trip with confirmation number SFTBAO"
  → GPT-4o가 공통 sub-routine 추출:

  ## enter_flight_locations         ← training에서 추출한 workflow
  [link] From Departure Airport -> CLICK
  [textbox] Origin City -> TYPE: {your-origin-city}
  [link] {best-popup-option} -> CLICK
  [link] To Destination Airport -> CLICK
  [textbox] Destination City -> TYPE: {your-destination-city}
  [link] {best-popup-option} -> CLICK

[Test] "Check all available one way flights from Manhattan to Philadelphia"
  └─ workflow가 agent memory에 포함됨
  └─ agent가 enter_flight_locations workflow 참고 → {your-origin-city}에 Manhattan 대입
  └─ element accuracy 향상: 50.6% (MindAct 41.6% → +9.0 absolute)
  └─ concrete example 대신 abstract variable이므로 element 선택 편향 없음
```

### Findings & Implications

1. **소량 데이터 학습**: 약 40 example만으로 rapid learning phase 완료, 이후 안정화 (Figure 5). 수십 개 task만으로 충분한 workflow 확보
2. **Cross-template 일반화**: 동일 template의 변형이 아닌 서로 다른 template 간에도 workflow가 전이됨 (Table 2)
3. **Abstract > Concrete**: abstract workflow가 concrete example보다 element selection bias를 줄여 더 높은 정확도. variable화된 표현이 여러 task에 걸쳐 유연하게 적용
4. **Online > Offline for unseen domains**: training-required offline보다 online이 unseen website/domain에서 더 강함. train-test distribution gap이 클수록 online의 우위가 확대
5. **Format 영향 최소**: code vs text format 간 큰 차이 없음. NL state description이 HTML보다 효과적. HTML 추가 시 오히려 context length 증가 + irrelevant elements로 성능 저하

## Code Architecture

> 원본: [submodule](../source/git/agent-workflow-memory_zorazrw/) · 스니펫: [induction](../source/git/snippets/Agent_Workflow_Memory_2024_arxiv__induction_prompt.md), [pipeline](../source/git/snippets/Agent_Workflow_Memory_2024_arxiv__online_pipeline.md), [integration](../source/git/snippets/Agent_Workflow_Memory_2024_arxiv__memory_integration.md)

논문의 방법론을 그대로 구현한 연구용 코드. 두 벤치마크(WebArena, Mind2Web)가 구조가 달라 별도 디렉토리로 분리되어 있으며, 공통 패턴은 동일: **pipeline.py가 3-step 루프(inference → eval → induction)를 subprocess로 순차 실행**.

```
agent-workflow-memory/
├── webarena/                    # WebArena 구현
│   ├── pipeline.py              # online streaming 루프 (3-step per task)
│   ├── run.py                   # agent inference (BrowserGym)
│   ├── induce_prompt.py         # LM-based workflow induction ★
│   ├── induce_rule.py           # rule-based workflow induction (대안)
│   ├── prompt/
│   │   ├── instruction.txt      # induction 지시 프롬프트
│   │   └── one_shot.txt         # one-shot 예시
│   ├── workflow/                # 생성된 workflow 메모리 (.txt)
│   │   ├── shopping.txt
│   │   ├── map.txt
│   │   └── ...
│   ├── agents/legacy/
│   │   ├── agent.py             # GenericAgent (workflow 주입 지점)
│   │   └── dynamic_prompting.py # prompt 조립 (Flags, PromptElement)
│   └── autoeval/                # LLM-based trajectory 평가
│       └── prompts.py           # 성공/실패 판정 프롬프트
│
└── mind2web/                    # Mind2Web 구현
    ├── pipeline.py              # offline/online orchestration
    ├── offline_induction.py     # training example → workflow
    ├── online_induction.py      # test trajectory → workflow
    ├── memory.py                # agent memory + per-step eval ★
    ├── run_mind2web.py          # inference entry point
    ├── prompt/
    │   ├── instruction_action.txt
    │   └── one_shot_action.txt
    ├── workflow/
    │   └── retrieve.py          # FAISS semantic retrieval (대안)
    └── utils/
        ├── data.py              # examples formatting, workflow filtering
        └── env.py               # DOM 처리, observation/action 추출, F1
```

### 핵심 컴포넌트

#### 1. Pipeline (`webarena/pipeline.py`, `mind2web/pipeline.py`)

3-step 루프를 subprocess로 실행하는 orchestrator. 각 step이 독립 프로세스로 실행되며, workflow 파일(`workflow/{website}.txt`)이 shared state 역할.

```
for each test query q_t:
  Step 1: run.py (inference)     → results/ trajectory 저장
  Step 2: autoeval (evaluation)  → {model}_autoeval.json 판정 결과 저장
  Step 3: induce_prompt.py       → results/의 모든 성공 trajectory 재읽어
                                   workflow 재유도 → workflow/{website}.txt 갱신
```

Mind2Web은 batch 단위(`--induce_steps`, 기본 1)로 처리하며, offline 모드는 induction 후 inference만 한 번 실행.

#### 2. Workflow Induction (`induce_prompt.py` / `offline_induction.py`)

paper §2.3의 LM-based induction module I(E) → W를 구현. 핵심 동작:

- **성공 trajectory 수집**: WebArena는 `autoeval` 또는 `gt`(ground-truth reward) 기준. Mind2Web은 모든 결과를 그대로 전달.
- **Template dedup (WebArena only)**: `intent_template_id`로 grouping 후 template당 1개 샘플링. Mind2Web은 dedup 없음.
- **Prompt 구성**: `instruction.txt` + `one_shot.txt` + formatted examples를 결합해 GPT-4o 호출 (temperature=1.0).
- **출력**: workflow 텍스트를 `\n\n`로 분리된 블록들로 저장. WebArena는 utility workflow 2개(`click('id')` 사용법, `select_option` 사용법)를 append.

→ 상세 코드 → [induction snippet](../source/git/snippets/Agent_Workflow_Memory_2024_arxiv__induction_prompt.md)

#### 3. Workflow Memory Integration

 두 벤치마크에서 주입 방식이 다름:

- **WebArena** (`agents/legacy/agent.py`): `GenericAgent.get_action()`에서 매 step마다 `workflow_path` 파일을 읽어 system message에 append. 별도 retrieval 없음.
- **Mind2Web** (`memory.py`): `get_exemplars()`가 workflow text를 첫 번째 exemplar(user message)로 배정. concrete examples와 함께 token limit 내에서 최대한 포함.

→ 상세 코드 → [memory_integration snippet](../source/git/snippets/Agent_Workflow_Memory_2024_arxiv__memory_integration.md)

#### 4. AutoEval (`webarena/autoeval/`)

Online AWM의 핵심: trajectory 성공 여부를 LLM이 판정. `build_text_eval_prompt()`가 user intent + action history + final page state를 GPT-4o에 전달하여 "success" or "failure" 반환. 이 판정이 workflow induction의 게이트 역할 (성공한 trajectory만 workflow로 추출).

#### 5. Workflow Retrieval (`mind2web/workflow/retrieve.py`)

대안 모드: workflow 수가 많을 때 FAISS embedding(text-embedding-ada-002)으로 semantic search. workflow의 `name + docstring`을 임베딩하여 test query와 similarity match → top-k 선택. **논문 메인 실험에서는 미사용** (website당 7개라 전체 주입이 더 효과적).

#### 6. Rule-based Induction (`webarena/induce_rule.py`)

LM 없이 trajectory 자체를 workflow로 저장하는 대안. action sequence signature(`click('227')_click('1843')`) 기반 dedup. 논문 §4.1 ablation에서 LM-based와 비교 (WebArena 0.1 gap, Mind2Web LM이 +2.8 우위).

### 코드 vs 논문 대응

| 논문 개념 | 코드 구현 | 파일 |
|-----------|-----------|------|
| Induction module I(E)→W | `induce_prompt.py` (LM), `induce_rule.py` (rule) | `webarena/` |
| Workflow representation w=(d,P) | `.txt` 파일 내 `## name\ndescription\n actions` 블록 | `workflow/*.txt` |
| Memory M + W → M_w | system message append | `agents/legacy/agent.py:116` |
| L_eval (online eval) | AutoEval GPT-4o 판정 | `autoeval/prompts.py` |
| Online streaming | pipeline.py 3-step 루프 (inference→eval→induce) | `webarena/pipeline.py` |
| Offline I(E_train)→W | `offline_induction.py` (train data로 one-shot) | `mind2web/` |
| Template dedup | `intent_template_id` grouping (WebArena meta) | `induce_prompt.py:127` |

## Analysis

### Strengths & Significance

- **Supervision-free online 학습**: test query만으로 작동하므로 추가 annotation이나 training data 불필요. 실제 deployment 환경에 적합
- **Snowball effect**: 단순 workflow가 subgoal이 되어 점진적 복잡화. 인간의 학습 과정과 유사한 메커니즘
- **추상화 설계**: example-specific context를 variable로 대체하여 cross-task/cross-domain 일반성 확보. concrete example의 편향 문제 해결
- **경량성**: website당 7개 workflow로 94% utility rate 달성. 별도 retrieval 시스템 없이 전체 workflow를 context에 포함
- **인적 감독 불필요**: human-written workflow(SteP, 14개) 대비 더 많은 양과 질의 workflow를 자동 생성

### Limitations

1. **GitLab 저조**: 5.9% SR로 유일한 underperforming domain. task template 부족 또는 workflow 부적합 추정되나 명확한 원인 분석 부족
2. **Noisy online workflow**: online 모드에서 LLM evaluator가 잘못 success로 판단한 trajectory에서 부정확한 workflow가 유도될 수 있음. offline 대비 cross-task 성능 약간 저하 (43.6 vs 45.1 Step SR)
3. **Action F1 저하**: workflow guideline이 특정 action으로 편향시켜, 상황에 따라 workflow에서 벗어나야 할 때 대응 어려움 (Table 3, MindAct 대비 Action F1 감소)
4. **AWM_AS 한계**: workflow를 action space에 추가해도 agent가 18.5% task에서만 호출. dynamic environment change(예: popup option 선택)에 취약 (Figure 7)
5. **단일 모델 의존**: GPT-4o로 induction, GPT-4로 inference. 모델 간 일관성 가정. evaluation model의 정확도에 성능 의존
6. **정적 workflow memory**: workflow가 overwrite 방식으로 갱신되어, 이전 workflow의 effective version을 추적하지 않음. 부정확한 workflow가 메모리에 잔류 가능

### Future Work / Improvements

- **Dynamic workflow execution**: AWM_AS에서 real-time state access와 dynamic execution loop를 통한 popup/dynamic content 대응 (논문에서 제안)
- **Workflow 품질 관리**: function overlap이나 utility rate 기반으로 저사용 workflow를 자동 가지치기하는 메커니즘
- **Multi-modal workflow**: 시각 정보를 활용한 workflow 추출 및 활용 (VisualWebArena 확장)
- **Workflow verification**: 유도된 workflow의 정확성을 환경에서 사전 검증하는 메커니즘 (noisy online workflow 문제 해결)
- **Offline+Online 통합 개선**: 현재 AWM_off+on이 additive 효과를 내지 못함(offline이 online 품질을 저하). 호환성 개선 필요

## References

- Paper: [arXiv:2409.07429](https://arxiv.org/abs/2409.07429)
- Code: [github.com/zorazrw/agent-workflow-memory](https://github.com/zorazrw/agent-workflow-memory)
- WebArena: [Zhou et al., 2024](https://openreview.net/forum?id=oKn9c6ytLx)
- Mind2Web: [Deng et al., 2023](https://openreview.net/forum?id=kiYqbO3wqw)
- SteP (human workflows): [Sodhi et al., 2023](https://arxiv.org/abs/2310.03720)
- AutoEval: [Pan et al., 2024](https://arxiv.org/abs/2404.06474)
- Synapse (trajectory exemplar): [Zheng et al., 2024](https://openreview.net/forum?id=Pc8AU1aF5e)

> [paper][git] https://github.com/addxai/ReflectWorld.git · https://arxiv.org/abs/2607.09759

# ReflectWorld-MM: An Entity-Oriented Multimodal Memory System for Open-Ended Video Streams

## Summary & Outline

**한 줄 요약:** 카메라가 계속 켜져 있는 환경(착용형 글래스, 로봇, 스마트폰)에서 끊임없이 들어오는 영상을 보고, 그 속에 등장하는 **사람과 사물을 하나의 고유 대상(entity)** 으로 묶어 경험을 축적하여 언제든 질문에 답하는 멀티모달 기억 시스템. 영상 한 편을 이해하는 모델이 아니라 "어제 누가 나타났는지", "상황이 어떻게 바뀌었는지"를 회상하는 조수의 메모리를 만드는 연구. 기억을 모델 안이 아니라 외부 DB에 구조화해 저장하며, 인간 기억 이론을 따라 **사건 기억(episodic, "무슨 일이 있었나") · 의미 기억(semantic, "무엇이 사실인가") · 절차 기억(procedural, "무엇을 해야 하나")** 세 종류로 나누되 모두 entity 중심으로 조직하는 것이 핵심. 6개 long-video·lifelong-memory 벤치마크 전부 최고 성능(SOTA), 특히 "그 사람이 누구인지"에 달린 질문에서 큰 폭 우위. 자동 구축 기억이 사람이 직접 주석 단 결과보다 나은 것도 입증. OpenClaw 조수 런타임에 plugin으로 실 배포.

**이 논문이 기존과 다른 3가지 핵심 아이디어:**
1. **과거가 현재를 "볼 때" 참여한다** — 영상을 의미 단위로 자른 한 조각(segment)을 해석하기 전에, 이미 알고 있는 맥락(현재 사건 상태, 장면 종류, 해당 entity의 과거 기록)을 미리 주입. 기존은 저장해 두고 질문이 올 때만 과거를 꺼내 보지만, 여기서는 보는 순간부터 과거가 해석을 돕는다.
2. **"누가 누구인지" 결정은 단일 resolver만** — 재식별기(re-IDer, 같은 사람/사물이 다시 나타났는지 얼굴·몸 모양으로 비교해 점수가 붙은 후보를 제안하는 모듈)는 후보만 내놓고, **별도의 판정기(resolver) 하나**만 최종 identity 판정·기록을 담당. re-IDer가 한 번 잘못 맞춰도 그것이 곧바로 기억에 "이 사람 = A"로 못 박히지 않아, 단일 오류가 전체 기억을 오염시키지 못하고 모든 판정이 추적 가능(auditable).
3. **기억이 살아 움직인다(추가 전용이 아니다)** — 기존 multimodal agent는 한 번 쓴 기억을 수정/삭제하지 못하고 계속 추가만(append-only) 하지만, 이 시스템은 새 증거가 쌓이면 의미 기억의 기존 사실을 **정정(Update)/삭제(Delete)/강화(reinforce)** 할 수 있음. 반복 확인된 사실은 0~1 중요도 점수(importance score)가 올라가 안정되고, 새 사실은 점수가 낮아 쉽게 정정되는 구조.

**논문 구조 outline:**
1. Introduction — 연구 동기, 기존 한계, 세 설계 원칙, 기여 요약
2. Related Work — ① streaming/long-video memory ② language agent memory ③ entity-centric lifelong multimodal memory
3. ReflectWorld-MM — problem setup → perception front-end → hierarchical episodic memory(3-level) → evolving entity-centric semantic memory → procedural memory → retrieval & system realization
4. Experiments — 6개 벤치마크 설정 · main results · answer efficiency(memory 품질) · ablation · qualitative 분석 · real-world operation
5. Conclusion
6. Appendix A–G — A 수치 출처 · B 벤치마크 세부 · C 구현 상세 · D agent/memory 접근면 · E stream gateway · F dashboard · G 질적 traces

상세 발췌 → [excerpt](../source/paper/ReflectWorld-MM_An_Entity-Oriented_Multimodal_Memory_System_for_Open-Ended_Video_Streams_2026_Rightly_Robotics.md)

## Problem & Motivation

- **연구 배경**: 카메라가 착용형 글래스·로봇·스마트폰 어디든 존재. 한 clip을 이해하는 video understanding 모델을 넘어, "어제 누가 나타났는지"·"상황이 어떻게 변했는지"를 회상할 수 있는 assistant memory 필요.
- **Task**: open-ended multimodal stream(long-form video)의 continual한 long-term memory 구축 및 질의. stream은 arbitrary length, 끝 없음. 임의 시각 T에 자연어 query q에 답해야 함. **per-step cost(한 segment를 처리하는 데 드는 시간·계산량)가 시스템을 얼마나 오래 돌렸는지(총 누적 stream 길이)와 무관하게 일정**해야 함 — 즉 1주일째든 1년째든 한 단계 처리 비용은 같아야 하며, 과거를 매번 전부 다시 읽는 식이면 안 됨. 이것이 memory를 외부 DB에 구조화하는 근본 이유.
- **기존 접근의 한계**:
  1. **Streaming/long-video models**(MovieChat, MA-LMM, Flash-VStream, VideoLLM-online, ReKV): memory를 model 내부(token/feature/kv-cache)에 frame·token 단위로 보관. bounded video에만 효과적이고 stream 길어지면 degrade. content-agnostic(엔티티가 아닌 frame 기반).
  2. **Text-only agent memory**(MemGPT, Generative Agents, A-MEM, Mem0, MemoryBank): cognitive 분해(episodic/semantic/procedural)는 부합하나, video stream 자체를 perceive하지 못하고 persistent visual entity를 resolve하지 않음.
  3. **M3-Agent**(entity-centric graph + multi-turn retrieval): frequency 가중치 투표 기반 append-only라 사실 revise/delete 불가.
  4. **WorldMM**(multi-scale episodic/semantic/visual): temporal scale 수동 설정, entity identity 미유지 → identity 질문 불가.
  5. 둘 다 episodic + evolving semantic + procedural를 단일 아키텍처로 통합 못하며, 임의 live stream에서 동작하는 완전 서비스로 구현 안 됨.

## Contributions

- **방법론 기여**: human memory theory(Tulving 1985) 기반으로 entity-oriented multimodal memory 통합 아키텍처 제안. episodic(entity/trace/schema 3-level) + evolving entity-centric semantic(Add/Update/Delete) + procedural(user rule). M3-Agent(append-only)·WorldMM(identity 미유지)의 빈틈을 메움.
- **메커니즘 기여**: "과거가 현재를 본다"는 perception-time context 참여. bounded working memory + per-camera scene context + entity history + agent-steered adaptive perception. storage 후 query 시에만 과거를 쓰는 기존 대비 근본적 차이.
- **실증 기여**: 6개 long-video·lifelong-memory benchmark 전부 SOTA. entity-centric M3-bench에서 M3-Agent 대비 +10.4(web)·+9.1(robot) pp. EgoLife-QA는 자동 memory로 manual annotation(45.5) 초과 달성(46.8).
- **시스템 기여**: DB-backed service + video-source adapter + agent-facing API + dashboard + OpenClaw plugin. 임의 video stream ingest·임의 agentic system query 가능. open-ended real-world operation 최초 통합 구현. 코드 오픈소스.

## Method

### 전체 Architecture

```
   ┌─ live stream (RTSP / file / USB / HTTP / smartphone)  ─┐
   │                                                        │
   ▼                                                        │
┌──────────────────┐  activity-coherent  ┌────────────────────────────────────────────┐
│ Stream Gateway   │ ──segment(s)──────► │  Perception Front-End                       │
│ normalize source │   +transcript       │                                             │
│ split segment    │                     │  ┌──── Detect(s)          ─ local evidence   │
└──────────────────┘                     │  │     ReID(D)             ─ scored cand.    │
                                         │  │     Resolver(only ID write)              │
                                         │  │                                           │
                                         │  ├──── Context assembly (3 layers)          │
                                         │  │     ① working memory (event state)       │
                                         │  │     ② per-camera scene (routine/baseline)│
                                         │  │     ③ entity history (re-ID hit)         │
                                         │  │                                           │
                                         │  ├──── Steer(s, c)  ─ agent scene            │
                                         │  │     understanding injection (allowlist,  │
                                         │  │     exit condition)                       │
                                         │  │                                           │
                                         │  └──── VLM(s, prompt) → O anchored to D,Ev  │
                                         │        Resolve(O, Ev) → E                   │
                                         └───────────────┬─────────────────────────────┘
                                                          │  entity-resolved observation
                                                          ▼
       ┌──────────────────────────────────────────────────────────────────────┐
       │  Hierarchical Long-Term Memory (externalized, indexed DB)            │
       │                                                                      │
       │  ┌─ Episodic ──────────────────────────────────────────────────────┐ │
       │  │  entity level ─→ trace level ─→ schema level  (3 abstraction)   │ │
       │  └──────────────────────────────────────────────────────────────────┘│
       │  ┌─ Semantic (entity-centric, evolving) ── Consolidate(e) every N=5 ┐│
       │  │     Add / Update / Delete + reinforce  (importance w∈[0,1])     ││
       │  └──────────────────────────────────────────────────────────────────┘│
       │  ┌─ Procedural (user rule) ─ semantic signal → deterministic notify ┐│
       │  └──────────────────────────────────────────────────────────────────┘│
       └─────────────────────────────┬────────────────────────────────────────┘
                                     │  tool-based retrieval (level selectable)
                                     ▼
                          ┌────────────────────────────────┐
                          │   QA / Acting Agent (GPT-5)    │
                          │   --- benchmark agent          │
                          │   --- OpenClaw plugin(deploy)  │
                          └────────────────────────────────┘
                                     │
                          shared entity ID ↔ level link │ importance in ranking
```

### 3 design principles (이해의 열쇠)

| 원칙 | 의미 | 효과 |
|------|------|------|
| evidence vs decision 분리 | detector/re-IDer는 scored evidence만; resolver 1개만 identity write | 단일 noisy match가 memory 부패 차단, 모든 identity 결정 auditable |
| time scale/abstraction 계층화 | flat log 대신 3-level episodic + semantic + procedural | 질문이 원하는 granularity로 retrieve 가능, long history 탐색 bound |
| 과거가 현재 인식에 참여 | segment 해석 전 context 주입(perception-time use of past) | isolated description 회피, 장기 stream에서 narrative continuity |

### Perception Front-End (상세)

stream gateway가 임의 source(RTSP/RTSPS, file, USB webcam, HTTP, smartphone via WebRTC bridge)를 정규화하고 visual motion + voice activity로 segment 분할. 각 segment는 frame과 transcript(speaker diarization 가능) 보유. open-ended stream 가정(끝 없음).

네 부분 동작 흐름:
1. **Entity recognition**: segment 내 person/object를 detect하고 기존 entity gallery(face+body dual appearance for person)에 re-ID. re-ID는 scored candidates만 제안.
2. **Resolver**: scored evidence를 받아 최종 identity 결정하고, 유일한 identity writer. unsafe evidence는 표시만 되고 gallery/name/identity write 금지. cross-segment continuity는 short-term memory와 weak pending evidence가 담당(tracker 아님).
3. **Context-enhanced perception**(핵심 차별점): VLM이 isolated 상태로 segment를 해석하지 않음. segment prompt가 3 layer로 강화:
   - bounded working memory(rolling event summary ~100 words, 최근 3 segment summary, active entities, ≤8 tracked targets).
   - per-camera semantic context(scene type, routines, anomaly baseline; LLM이 주기 갱신).
   - entity history(re-ID hit 시 해당 entity의 최근 episodic+semantic record retrieve).
4. **Agent-steered adaptive perception**: 고위 agent가 scene description·focus target·active rule을 prompt에 주입. controller(deterministic frame-level + semantic segment-level policy)가 segment 분석 여부/풍부도 결정. static·silent는 저렴, 정보 풍부/rule-relevant는 풍부. hard guards(첫 segment, speech 포함, audio 미변환)는 무조건 분석. detection anchoring + ungrounded identity target 방지.

### Hierarchical Episodic Memory (3단 추상화)

사건 기억(episodic memory)은 "무슨 일이 있었는가"를 기록하며, 단순한 시간순 로그가 아니라 **추상화 단계가 3단**인 계층 구조. 사람의 자서전적 기억 구조(Conway & Pleydell-Pearce 2000)를 따라, **거친(상위) 기억이 세밀한(하위) 사건을 가리키는 색인** 역할을 하는 방식. 이 계층화의 핵심 동기는 긴 영상에서 질문이 필요로 하는 상세도로 검색할 수 있게 하고, 긴 과거를 뒤질 때 반드시 봐야 하는 세밀 항목 수에 상한을 두는 것.

| 단계 | 단위 | 내용 | 역할 |
|------|------|------|------|
| **Entity** (가장 세밀) | segment × entity | 특정 사람/사물의 모습·행동·상호작용. 고유 entity ID로 저장 | "특정 시각에 특정 사람이 무엇을 했는지" 회상 |
| **Trace** (중간) | segment | 한 사건의 요약 + 관련 entity 목록. entity 단위 관찰로 역링크 | 기본 회상 단위. 같은 시각 여러 entity 관찰을 한 사건으로 묶어 결합 |
| **Schema** (가장 거침) | 여러 segment → 장(chapter) | 사건이 닫히면 누적하고, 개수 예산 B 또는 시간 간격 도달 시 장 요약으로 통합 | 긴 과거에서 세밀 trace 탐색 범위에 상한. 작은 색인으로 긴 이력을 탐색 |

**예시로 보는 3단 계층 구조** (Alice의 1주 vlog):

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Schema 단계 (가장 거침) — 여러 사건을 묶은 "장(chapter)" 요약       │
 │                                                                     │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │ "Alice의 1주일: 카페에서 작업 + 암벽등반/헬스로 운동"          │  │
 │  │  (Day1~7 사건들을 통합한 한 줄 요약 — 긴 과거를 한눈에)       │  │
 │  └──────────────┬───────────────────────────┬────────────────────┘  │
 │       색인 ↓                                ↓ 색인                   │
 ├─────────────────┼───────────────────────────┼───────────────────────┤
 │  Trace 단계 (중간) — segment 1개 = 사건 1건 요약                    │
 │                                                                     │
 │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
 │  │ Day1 카페    │  │ Day3 카페    │  │ Day5 암벽등반│  │ Day7 헬스  │  │
 │  │ "Alice,     │  │ "Alice,     │  │ "Alice,     │  │ "Alice,   │  │
 │  │  노트북 작업"│  │  노트북 작업"│  │  클라이밍"  │  │  헬스장"   │  │
 │  │ 참여: E1    │  │ 참여: E1    │  │ 참여: E1    │  │ 참여: E1  │  │
 │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
 │    역링크↓               ↓역링크          ↓역링크         ↓역링크   │
 ├──────────┼──────────────┼───────────────┼───────────────┼─────────┤
 │  Entity 단계 (가장 세밀) — segment × entity = 개별 모습·행동         │
 │                                                                     │
 │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐          │
 │  │E1 @ Day1  │ │E1 @ Day3  │ │E1 @ Day5  │ │E1 @ Day7  │  ← 모두   │
 │  │카페,       │ │카페,       │ │암벽등반장, │ │헬스장,     │   같은    │
 │  │노트북,작업 │ │노트북,작업 │ │클라이밍    │ │운동        │   E1 ID  │
 │  └───────────┘ └───────────┘ └───────────┘ └───────────┘          │
 └─────────────────────────────────────────────────────────────────────┘

 검색 진입 예시:
   "Alice는 보통 어디서 시간을 보내?"  → Schema에서 흐름 파악 → 답
   "화요일에 무슨 일이 있었나?"        → Trace(Day3)로 진입 → 사건 요약
   "Day5 암벽등반 때 Alice가 뭘 했나?" → Entity(E1@Day5)로 진입 → 상세 행동
```

**왜 계층인가?** schema 없이 모든 entity 관찰만 쌓으면, 1년 된 stream에서 "이번 주 흐름"을 물었을 때 수만 개의 세밀 관찰을 전부 뒤져야 한다(탐색 비용이 stream 길이에 비례). schema가 며칠을 한 문장으로 묶어주면, 먼저 schema에서 관련 장을 찾고 → 거기서 trace 몇 개만 → 필요 시 entity 관찰로 내려가는 식으로 탐색 범위가 좁아진다. 위→아래로 갈수록 세밀해지고, 아래→위로 갈수록 요약되는 양방향 구조.

동작 흐름상 이 3단은 기록(쓰기) 시점에 자동 연결된다. segment가 처리될 때마다 (1) 해석된 각 entity마다 entity 단위 관찰이 기록되고, (2) 그 segment 전체를 요약한 trace가 기록되면서 관련 entity 관찰에 역링크를 걸며, (3) 사건이 닫히면 schema 단계에 누적되다가 임계치(개수 예산 B 또는 시간 간격) 도달 시 장(chapter)으로 통합된다. 검색 시에는 agent가 단계를 선택할 수 있어, "이 사람의 최근 행동"은 entity 단계, "이 시각에 무슨 일이"는 trace 단계, "이 주간 흐름"은 schema 단계로 들어가는 식으로 질문에 맞춰 진입. 공유 entity ID가 단계 사이를 연결하기 때문에 사건에서 참여 entity로, entity에서 자기 과거로 자유롭게 이동 가능. ablation(Figure 3)에서 schema 제거 시 M3-robot 37.4 → 33.6으로 가장 큰 하락이 발생 → 거친 기억이 세밀 사건을 색인하는 구조가 긴 과거 탐색 비용을 잡는 핵심 역할임을 입증.

### Evolving Entity-Centric Semantic Memory

episodic이 "사건의 기록"이라면, semantic memory는 그 사건들에서 증류한 **entity별 durable knowledge**(attributes, habits, relations). 한 사람이 반복해서 카페에 가는 관찰이 누적되면 "이 사람은 카페에서 일하는 습관이 있다"는 사실로 consolidate되는 식. 이 부분이 M3-Agent(append-only, frequency 가중치 투표만으로 update)와 가장 크게 대비되는 기여.

**consolidation trigger**: 각 entity에 N=5개의 신규 관측이 쌓일 때마다 trigger. consolidator는 해당 entity의 prior episodic record와 기존 semantic record를 함께 읽고, 4가지 edit decision 중 하나를 내린다:
- **Add** — 새로 나타난 사실 추가.
- **Update** — 기존 사실을 revise(stale 정정). 반례가 나타나면 과거 사실을 고쳐 쓴다.
- **Delete** — 더 이상 지원되지 않는 사실 제거. 사라진 상황·취소된 속성을 버린다.
- (no change)

즉 새 증거가 기존 사실을 reinforce(강화)하거나 revise(정정)하거나 delete(제거)할 수 있다. M3-Agent가 쓰면 끝인 append-only와 달리 메모리가 살아 움직이도록 만드는 부분.

**Importance score** `w ∈ [0, 1]` — 각 semantic fact이 반복 확인될 때마다 커지며, 1에 점근적(asymptotic)으로 수렴:

```
w ← w + (1 − w)·γ,   γ ∈ (0, 1)        ... Eq.(1)
```

| 항목 | 값/규칙 |
|------|---------|
| importance update 공식 | `w ← w + (1 − w)·γ`, γ=0.2 (기본) |
| 동작 | 반복 증거 → 1에 asymptotic, 신규 fact는 revise 용이 |
| ranking 참여 | well-confirmed fact(w 큰 값)가 retrieval 우선 |
| identity fact | maximum importance로 write, 자동 update/delete에서 보호 |
| Update/Delete | 기존 semantic fact ID를 반드시 지정, 미검증 target edit은 drop |
| consolidation scope | per-entity counter는 persistent·per-user, 현재 segment는 self-evidence로 제외 |
| prior history 없는 entity | merge skip(초기 entity는 consolidate 대상 아님) |

이 공식의 직관: 처음 추가된 사실은 w가 작아 revise되기 쉽지만(새 정보라 검증 부족), 여러 번 재확인되면 w가 커져 안정적인 사실로 자리잡는다. identity 사실(이름 등)은 maximum importance로 보호되어 자동 정정/삭제에서 제외된다. per-entity 관측 카운터는 persistent하여 stream이 길어져도 정확히 5번째 관측마다 consolidate가 trigger되며, 현재 segment는 자기 증거(self-evidence)로 제외되어 circular update를 방지. Update/Delete는 항상 기존 semantic fact ID를 지정해야 하고 없는 target에 대한 edit은 drop되어 ghost 사실이 생기지 않는다.

### Procedural Memory & Proactive Response

마지막 메모리 종류는 사용자 rule과 preference를 저장하는 procedural memory. 예: "Notify me when the dog is on the sofa". semantic memory가 "무엇이 사실인가"라면, procedural memory는 "무엇을 해야 하는가"를 담당. 핵심 설계 결정 두 가지:

**(1) stream을 한 번만 분석**: rule은 perception prompt에 surface되어 VLM이 segment를 분석할 때 함께 고려한다. 별도 rule-matching pass를 위해 다시 stream을 읽지 않는다. 이는 open-ended stream에서 per-segment cost를 bound하려는 원칙의 연장선.

**(2) rule match는 trigger가 아니라 evidence**: VLM이 rule을 직접 실행(notify 전송)하지 않고 **semantic signal**만 emit. 예를 들어 "이 상황이 위험해 보인다" 또는 "이 segment가 이 rule에 매칭되는 것 같다"는 signal. 이 signal은 **deterministic resolver**로 전달되고, resolver가 per-camera notification policy와 최근 전송 기록(중복 방지)을 함께 고려해 "무엇을 보낼지, 보낼지 말지"를 최종 결정. 이 구조는 evidence/decision 분리 원칙을 procedural 영역까지 일관되게 적용한 것. rule이 즉시 action을 강제하면 같은 상황이 반복될 때마다 알림이 중복되거나 짧은 순간의 false positive가 trigger되어 spurious notification이 쏟아질 수 있는데, resolver가 dedup·cooldown·policy를 담당해 반복/허위 알림을 차단한다. Figure 6의 사례에서 dog가 소파 근처 → 점프 → 눕는 visual 증거가 연속으로 관찰되지만, 시스템은 각 단계를 signal로 변환한 뒤 resolver에서 rule·policy·cooldown과 조합해 최종 한 번의 notification을 발생시킨다.

### Per-step cost를 일정하게 유지하는 메커니즘

문제 설정(Problem & Motivation)에서 "1주일째든 1년째든 한 segment 처리 비용은 같아야 하며, 과거를 매번 전부 다시 읽으면 안 됨"이라고 했다. 시스템이 이를 어떻게 달성하는지 정리하면 다섯 가지가 얽혀 있다:

```
stream이 길어질수록 누적되는 것 vs 한 번 처리에 쓰는 고정 비용
──────────────────────────────────────────────────────────
 늘어나는 것 (전체 저장량)         고정인 것 (한 segment 처리 비용)
 ─────────────────────────         ─────────────────────────────
 DB에 쌓이는 전체 기록              ① working memory 한도
 entity gallery                     ② 계층 색인(schema → trace)
 semantic fact 수                    ③ consolidate는 N=5 주기
                                    ④ 검색은 subset만
                                    ⑤ live 모드 critical-only 즉시 commit
```

① **작업 기억(working memory) 한도** — 한 segment를 해석할 때 쓰는 context가 고정 한도 안에 갇힌다: 사건 요약 ~100단어, 최근 3 segment 요약, 활성 entity, 추적 대상 ≤8개. 대상은 연속 미검출 시 버리고, entity는 수분 비활성 시 잊고, 사건은 최대 segment 수 초과 시 교체(rototate). 즉 "과거 전부"를 보지 않고 고정 크기 창만 본다.

② **계층 구조로 탐색 범위 상한** — episodic memory의 3단 계층(schema → trace → entity, 위 다이어그램 참조) 덕에, 긴 과거를 뒤질 때 전체 entity 관찰을 다 보는 게 아니라 schema 몇 개 → 관련 trace 몇 개 → 필요한 entity 관찰만. 1년 stream에서도 schema를 먼저 보고 들어가므로 탐색 비용이 stream 길이에 비례하지 않는다. ablation에서 schema 제거 시 정확도 가장 큰 하락(37.4→33.6) → 이 상한이 실제 탐색 비용을 잡는 핵심.

③ **의미 기억 통합(consolidation)은 N=5 주기** — entity별로 신규 관측 5개 쌓일 때만 semantic 통합이 trigger. 매 segment마다 하는 게 아니라 5단위로. per-entity 카운터는 persistent하지만 단계당 작업량은 고정.

④ **검색은 subset에만 attend** — 질의 시 full stream이 아니라 검색해서 나온 부분집합(retrieved subset)에만 agent가 attend. importance 점수가 순위를 매겨 well-confirmed 사실 우선.

⑤ **live 실행 모드는 critical-only 즉시 commit** — `live_latency` 모드에서 identity-critical 필드(사건 경계, identity, 매칭 rule, 요약)만 per-user lock 하 즉시 commit하고 다음 segment를 release. enrichment 필드는 frozen critical state에 후속 reconcile. 한 segment가 다음 segment를 막지 않는다.

> **대조**: ReKV 같은 streaming-cache 방식은 stream 길이에 비례하는 kv-cache state를 매번 읽어야 해서 per-step cost가 길어질수록 커진다. ReflectWorld-MM은 위 다섯 가지로 per-segment 작업 상태와 질의 context를 bounded로 유지 → 무한 실행 가능. 논문 원문: "The total store naturally grows as the system remembers more, but the per-segment working state and query context remain bounded."

### Retrieval & System Realization

- **Retrieval**: tool call로 외부 store 질의. agent가 memory level 선택(entity obs / traces / schemas / semantic facts / procedural rules). shared entity ID가 level을 연결(event→participants, entity→history). importance가 ranking에 참여 → well-confirmed fact 우선. full stream이 아닌 retrieved subset에 attend.
- **Persistence**: indexed DB(per-collection: episodic 3 level, entity-centric/per-camera semantic, procedural rule, face/body gallery, reserved visual-embedding). model context가 아님. 총 store는 성장하나 per-segment working state와 query context는 **bounded** → 무한 실행. ReKV(streaming-cache)가 stream 길이 비례 state인 것과 대조.
- **Two execution modes**:
  - `offline_quality` — deterministic benchmark용.
  - `live_latency` — identity-critical 필드(event boundary, identity, matched rules, summary)를 per-user lock 하 즉시 commit → 다음 segment release; enrichment 필드는 frozen critical state에 후속 reconcile.
- **Local perception utilities**(ONNX, bounded signal utility): YOLO26m(object detect, 필수), RetinaFace-ResNet50(face box/landmark, 필수), ArcFace-MobileFaceNet(face embed, 필수), CLIP-ReID ViT-B(body embed, 선택), RTMPose-S(pose, 선택), Moonshine Tiny int8 STT(cloud 대체 가능).
- **Deployment via OpenClaw**: OpenClaw assistant runtime용 plugin. tool-based contract로 capture control / perception / memory query / notification policy / per-camera context 노출. host assistant가 새 source 시청, 상황 질의, long-term memory 조회, rule 설정을 ordinary tool call로 수행. benchmark agent와 deployed assistant가 동일 memory interface 사용(주변 agent만 다름).

### 전체 동작 예시 — 1주일 Vlog 시나리오 (Write & Read Walkthrough)

논문 Figure 4의 시나리오를 바탕으로, re-ID(재식별)가 무엇인지부터 메모리 추출(write)과 검색(read) 과정을 한 번에 보여주는 예시.

> **re-ID(재식별)란?** 영상에서 어떤 사람/사물이 이전에도 나왔던 **같은 대상**인지 알아내는 과정. 어떤 사람이 화면에서 잠시 사라졌다가 다시 나타나면 "새 사람"이 아니라 "아까 그 사람"으로 인식해야, 그 사람에 대한 경험이 같은 기록에 누적될 수 있다. 이 시스템에서는 얼굴·몸 모양을 비교해 점수가 붙은 후보를 내놓고, resolver가 최종 판정을 내린다.

**시나리오**: 어떤 사람(Alice)이 1주일 동안 여러 날에 걸쳐 vlog에 등장. 카페에서 노트북으로 작업하는 모습이 며칠 반복되고, 나중에 암벽등반·헬스장에도 가는 모습이 관찰됨.

#### Write(메모리 추출) 과정 — Algorithm 1 단계별

```
Day 1 — Alice가 카페에 등장 (처음 보는 사람)
─────────────────────────────────────────────
① 스트림 → gateway가 activity 단위로 segment 분할
② Detect(segment): "사람 있음" 검출
③ ReID(D): gallery에 일치하는 기존 entity 없음 → "새 entity 후보 (score 0.0)"
④ Resolver: "새 entity" 판정 → entity E1 생성, gallery에 Alice 얼굴/몸 등록
⑤ Context assembly:
     working memory: (첫 등장, 빈 이력)
     scene context: "카페"
     entity history: (E1 신규, 이력 없음)
⑥ Agent steer: "사람과 활동에 집중"
⑦ VLM 해석(prompt + context): "한 여성이 카페에서 노트북으로 작업 중"
   → entity observation O1: E1 - 카페, 노트북, 작업
⑧ Resolve(O1, Ev): E1 확정
⑨ Write:
     episodic/entity level  → E1@Day1: "카페에서 노트북 작업"
     episodic/trace level   → "Day1 카페 segment: E1 참여, 노트북 작업"
     working memory 갱신      → active entity: E1
  (E1 관측 카운트 = 1, N=5 미만 → consolidation 미trigger)

Day 3 — Alice가 다시 카페에 등장 (re-ID 발생)
─────────────────────────────────────────────
② Detect: "사람 있음"
③ ReID(D): gallery 비교 → "기존 E1 후보 (face score 0.91, body score 0.85)"
   ★ 이것이 re-ID — Day 1에 본 사람과 같은 사람임을 점수로 제안
④ Resolver: score 0.91+0.85 확인 → "E1 = 아까 그 사람" 판정
   (잘못된 낮은 점수 후보가 있어도 resolver가 gatekeeper 역할)
⑤ Context assembly:
     working memory: (현재 카페 event 진행 중)
     scene context: "카페" (routine: 작업 자주 발생)
     entity history: E1의 Day1 관측 retrieve → "이 사람은 이전에도 카페에서 작업함"
     ★ 과거가 현재 인식에 참여 — VLM이 "또 왔네, 같은 활동"으로 해석
⑥-⑦ VLM 해석: "E1(Alice)이 카페에서 다시 노트북 작업 중"
⑨ Write:
     episodic/entity level  → E1@Day3: "카페, 노트북 작업"
     episodic/trace level   → "Day3 카페 segment: E1, 노트бук 작업"
  (E1 관측 카운트 = 2)

Day 5 — Alice가 암벽등반장에 등장
────────────────────────────────
③ ReID(D): gallery 비교 → "E1 후보 (face score 0.78, 다른 옷·장소)"
④ Resolver: 0.78이 임계 이상 → E1 판정 (다른 장소·복장이어도 같은 사람)
⑤ Context assembly:
     entity history: E1의 Day1/Day3 관측 → "이 사람은 보통 카페에서 작업하는 사람"
⑥-⑦ VLM 해석(context enhanced): "E1이 오늘은 암벽등반을 하고 있음"
⑨ Write:
     episodic/entity level  → E1@Day5: "암벽등반장, 클라이밍"
     episodic/trace level   → "Day5 암벽등반 segment: E1, 클라이밍"
  (E1 관측 카운트 = 3)

Day 7 — 관측 카운트 5 도달, semantic consolidation trigger ★
────────────────────────────────────────────────────────────
E1의 관측이 5개 누적 (Day1, 3, 5, 7 카페 2회 + 운동 2회 + 기타 1회)
→ consolidator가 E1의 prior episodic + 기존 semantic record를 읽음:

  edit decision 과정:
  ┌ "카페에서 작업"이 Day1, Day3, Day7에서 반복 관찰됨
  │   → Add: "E1은 카페에서 작업하는 습관이 있다" (w=0.2, 새 사실)
  │   (이후 반복 확인될 때마다 w ← w + (1-w)×0.2 로 증가 → 안정화)
  │
  ├ "암벽등반"이 Day5에서 관찰됨
  │   → Add: "E1은 암벽등반을 한다" (w=0.2)
  │
  └ Day7 헬스장 관찰 → "운동 습관" 정정:
      기존 "암벽등반을 한다"를 Update → "E1은 운동을 즐긴다(암벽등반, 헬스)" (w↑)

  ★ append-only가 아님 — 기존 사실이 Update/강화됨. M3-Agent라면 그냥 추가만 됨.

episodic/schema level:
  Day1~7의 사건들이 하나의 chapter로 consolidate → "E1의 일주일: 카페 작업 + 운동"
  (이 coarse 기억이 fine-grained trace를 index → 긴 이력 탐색 bound)
```

#### Read(메모리 검색) 과정 — 질의 응답

```
질문: "Alice는 보통 어디서 시간을 보내?"
──────────────────────────────────────
① Agent(GPT-5)가 memory tool 호출 — level 선택: semantic facts
   (왜 semantic? "보통"이라는 단어 → 습관/패턴 질문 → entity별 지식 필요)
② 검색: entity E1로 filter, semantic facts retrieve
   결과:
     "E1은 카페에서 작업하는 습관이 있다"  (w=0.8, 반복 확인됨 → 높은 순위)
     "E1은 운동을 즐긴다(암벽등반, 헬스)"  (w=0.4, 덜 확인됨 → 낮은 순위)
③ importance가 ranking에 참여 → w=0.8인 카페 사실이 우선
④ Agent가 memory에서 답 조립: "주로 카페에서 노트북 작업을 하고, 운동도 즐깁니다."
⑤ video fallback 필요? → No (4.6~6.8%만 fallback)
   ★ memory가 source video를 다시 보지 않아도 답할 수 있는 evidence를 저장

만약 질문이 "화요일에 무슨 일이 있었나?" 였다면:
  → agent가 trace level 또는 schema level 선택
  → shared entity ID로 event 참여자 추적, 시간순 이웃 get
  → 해당 날짜의 trace retrieve → 답
```

이 예시가 보여주는 것: (1) re-ID는 같은 사람을 여러 날에 걸쳐 같은 entity로 묶는 과정, (2) write는 segment → entity/trace/schema 3-level episodic + N=5마다 semantic consolidation으로 이어지며, (3) read는 질문 종류에 따라 적절한 memory level을 선택해 검색하고 importance가 순위를 매긴다.

## Experiments & Results

### Benchmark Datasets

general + egocentric + entity-centric 혼합 6종:
- **VideoMME-Long** (Fu et al. 2025) — 일반 long video MC, ~1시간.
- **LVBench** (Wang et al. 2025a) — 일반 long video MC, ~2시간.
- **HippoVlog** (Lin et al. 2025) — ~1000 MC, audiovisual vlog, modality 간 memory 형성·연상.
- **EgoLife-QA** (Yang et al. 2025) — 1주 egocentric, identity-dependent life question. sub-task: EntityLog(EL), EventRecall(ER), HabitInsight(HI), RelationMap(RM), TaskMaster(TM).
- **M3-bench-robot / M3-bench-web** (Long et al. 2025) — egocentric/online video open-ended QA, multi-hop/cross-modal/person-centric reasoning. 평균 ~30분.

### Setup

- **Baselines**: M3-Agent(Long 2025), WorldMM(Yeo 2025) — 두 최강 multimodal memory agent. GPT-5(frontier model reference).
- **Implementation**: GPT-5-mini(perception, memory extraction, semantic consolidation), GPT-5(query answering agent). person re-ID = face+body. 모든 memory item은 OpenAI text-embedding-3-small(1,536 dim) 임베딩. vector DB 기반 store.
- **Metric**: MC는 option matching, open-ended(M3-bench)는 GPT-5-mini judge(원 GPT-4o judge 사용 불가).
- **Provenance**(Appendix A): 공정 비교 위해 M3-Agent(EgoLife-QA, M3-bench)·WorldMM(VideoMME-L)는 공식 코드 re-run. GPT-5는 EgoLife-QA·M3-bench 직접 run; VideoMME-L·LVBench·HippoVlog는 WorldMM paper 인용.

### Results — Entity-sensitive benchmarks (Table 1)

| Method | EgoLife-QA | M3-robot | M3-web |
|-------|-----------|----------|--------|
| Gemini-1.5-Pro | 36.9 | 8.0 | 23.2 |
| GPT-5 | 42.6 | 34.7 | 53.9 |
| MovieChat | – | 11.2 | 12.6 |
| Flash-VStream | – | 19.4 | 23.6 |
| M3-Agent | 30.8 | 28.3 | 45.6 |
| **ReflectWorld-MM** | **46.8** | **37.4** | **56.0** |

- EgoLife 원래 paper: manual annotated identity description으로 45.5 → ReflectWorld-MM 자동 memory로 46.8 (자동 memory가 manual annotation 초과).
- WorldMM은 entity identity 미유지로 해당 benchmark 미보고.

### Results — General long-video benchmarks (Table 2)

| Method | VideoMME-L | LVBench | HippoVlog |
|-------|-----------|---------|-----------|
| GPT-5 | 74.3 | 60.4 | 75.7 |
| HippoMM | 41.6 | 38.2 | 71.9 |
| M3-Agent | 55.3 | 49.3 | 65.5 |
| WorldMM | 73.8 | 61.9 | 78.3 |
| **ReflectWorld-MM** | **76.9** | **69.4** | **80.9** |

GPT-5 reference 대비 개선: VideoMME +2.6, LVBench +9.0(video 길어질수록 확대).

### Answer Efficiency (Table 3) — Memory Quality

| Benchmark | Method | Acc. | Avg Tok | Vid. fallback |
|-----------|--------|------|---------|---------------|
| EgoLife-QA | M3-Agent | 30.8 | 13k | 0.0% |
| EgoLife-QA | **ReflectWorld-MM** | **46.8** | 55k | **4.6%** |
| VideoMME-L | M3-Agent | 55.3 | 8.5k | 0.0% |
| VideoMME-L | WorldMM | 73.8 | 56k | 34.0% |
| VideoMME-L | **ReflectWorld-MM** | **76.9** | 43k | **6.8%** |

의미: high accuracy + rare fallback → memory가 answerable evidence 저장(stored index가 아님). M3-Agent는 compact but 누락(0% fallback, 저정확도). WorldMM은 strong but 34% fallback(index=source video 재필요).

### Ablation (Figure 3 — response-time)

memory 재구축 비용 회피 위해 동일 memory 구축 후 질의 agent에서 한 component 차단. 중복 evidence로 drop은 component 가치의 **lower bound**.
- **w/o entity association**(entity-linked retrieval 차단)
- **w/o schema**(coarse event index 제거) — M3-robot 37.4 → 33.6 (최대 drop)
- **w/o semantic**(distilled entity knowledge 은폐)
→ 각 component gain 실제 효과 확인(answer model 효과 아님).

### Findings & Implications

- **stream 길어질수록 GPT-5 reference 대비 격차 확대**(VideoMME +2.6 → LVBench +9.0) → memory 구축이 긴 영상에서 더 큰 가치.
- **자동 memory가 manual annotation 초과**(EgoLife 46.8 vs 45.5) → entity resolver가 사람 주석 수준 이상.
- **answer efficiency가 memory "완성도"의 증거**: fallback 비율이 4.6/6.8%로 낮다는 것은 source video 없이 저장된 memory로 답한다는 뜻. index가 아닌 evidence 저장 입증.
- **schema memory가 가장 큰 drop(−3.8) 기여** → long history에서 coarse index의navigate 가치. semantic과 entity association도 일관 기여.

## Analysis

### Strengths & Significance

- **과거가 현재를 본다(perception-time context use)**: storage 후 query 시에만 과거를 쓰는 기존 multimodal agent·streaming cache와 근본적으로 다른 설계. 장시간 stream에서 cumulative context로 perception 자체가 향상되는 snowball 구조.
- **evolving semantic**(Add/Update/Delete + importance Eq.1): append-only M3-Agent의 한계를 넘어, 사실이 stale해지면 revise/삭제되며 반복 증거로 안정화. cognitive memory theory + graph memory(HippoRAG, Zep)의 장점을 streaming multimodal로 확장.
- **evidence/decision 분리 + 단일 resolver**: 단일 noisy re-ID match가 memory를 부패시키지 않고 모든 identity 결정이 auditable. 안전·디버깅 관점에서 실배포에 유리.
- **bounded per-segment state + indexed DB persistence**: ReKV처럼 stream 길이에 비례하는 state 없이 무한 실행 가능.
- **완전 service 구현**(gateway + adapter + API + dashboard + OpenClaw plugin): 임의 source·agent 호환. 다른 multimodal memory 연구가 offline pipeline인 것과 달리 real-world deploy 가능.
- **통합 검증**: 6개 benchmark 전부 SOTA + answer efficiency로 "저장된 evidence" 입증 + ablation으로 component별 기여 분리. 실험 설계 탄탄.

### Limitations

- **저자 인정 한계**: M3-bench 절대 정확도가 모든 시스템에서 여전히 낮음(56.0/37.4) → benchmark가 far from solved. long-form video entity reasoning 자체가 도전 과제로 남음.
- **외부 의존**: perception/consolidation = GPT-5-mini, query agent = GPT-5, embedding = OpenAI text-embedding-3-small. 비용·latency·data privacy 측면에서 cloud 의존도 높음. local ONNX utility는 보조 신호만 제공(reasoning 모델이 아님).
- **judge 변경**: M3-bench를 원 GPT-4o judge 대신 GPT-5-mini judge로 re-run. baseline 중 일부는 원 GPT-4o judge 인용값과 혼재(Appendix A) → 완전히 동일 protocol 비교 약간 불완전.
- **re-ID gallery가 entity basis**: person/object detection의 오류·gallery 누적 drift가 장기 실행에서 어떻게 영향하는지에 대한 스터디 부족. resolver가 auditable하다는 점은 완화 요소.
- **WorldMM 미보고**: EgoLife-QA·M3-bench에서 WorldMM 직접 비교 누락(WorldMM이 entity identity 미유지라는 정당한 사유이나, SOTA 비교 폭은 다소 좁음).
- **분석자 관찰**: procedural memory의 notification 정책이 fixed policy benchmark이며, 실질 rule 충돌·우선순위 시나리오 실험은 부족. agent steering의 per-event budget이 "소형"으로만 서술되어 구체적 예산·latency cost 공개 미흡.

### Future Work / Improvements

- **다양한 VLM backbone**: GPT-5 family 외 local open VLM으로 교체 가능성. cost·privacy 개선.
- **반복 rule 충돌 해결**: 복수 procedural rule 간 우선순위·충돌 해소 메커니즘 및 평가 추가.
- **re-ID drift 모니터링**: 장기 실행 시 gallery 변화·잘못된 identity merge 탐지 정량화.
- **schema memory 자동 생성 budget B 자동 조정**: video 특성에 따른 adaptive budget.
- **다중 카메라 entity merge**: per-camera에서 cross-camera global entity graph로 확장(현재는 per-camera context only).
- **procedural memory의 skill 승격**: 반복 행동 패턴을 procedural rule에 한정하지 않고, agentic-memory·AgenticSTS 식으로 "skill" layer로 일반화하는 방향.

## Git Repository 분석

> 논문의 오픈소스 구현체(`https://github.com/addxai/ReflectWorld.git`)를 분석. 논문의 메서드가 실제 코드로 어떻게 구현되었는지, paper와 git의 대응을 정리.
>
> submodule 경로: `source/git/ReflectWorld_addxai/` · 언어: TypeScript(Node 22, npm workspaces) + Python(FastAPI memory service)

### Overview

ReflectWorld는 **TypeScript 모노레포 + Python 메모리 서비스**로 구현된 "live world memory runtime"입니다. 영상 source → segment 분할 → perception(VLM+로컬 검출) → 메모리 저장 → agent/dashboard 질의까지 전체 파이프라인이 하나의 실행 가능한 서비스로 구현되어 있습니다. 논문이 방법론을 제시하면, 코드는 그것을 실제 카메라 환경에서 돌아가게 만드는 엔지니어링 실현입니다.

```
 ┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐     ┌─────────────┐
 │  packages/cam   │────▶│  packages/percept │────▶│  services/mem    │────▶│  Dashboard  │
 │  Stream Gateway │     │  Perception       │     │  (Python FastAPI)│     │  + Agent    │
 │  go2rtc 분할    │     │  VLM+re-ID+context│     │  Qdrant 9 coll.  │     │  QA tools   │
 └─────────────────┘     └───────────────────┘     └──────────────────┘     └─────────────┘
         │                       │                        │
         │ motion+VAD segment    │ working memory         │ semantic consolidation
         │ cut (CompositeSeg)    │ + semantic context     │ (N=5, Add/Update/Delete)
         │                       │ + agent steering       │ importance Eq.(1)
         ▼                       ▼                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  packages/reflectworld — orchestrator: VisionController, MemoryWriteScheduler,│
 │  PostProcessor(notifier), PipelineStore(SQLite), CLI, Dashboard server        │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### 패키지 구조

| 패키지 | 언어 | 역할 | 논문 대응 |
|--------|------|------|-----------|
| `packages/shared` | TS | 공유 타입·이벤트 버스(monitorBus). 모든 패키지 의존 | — |
| `packages/cam` | TS | Stream gateway. go2rtc로 임의 source 정규화, motion+VAD로 segment 분할 | §3.2 streaming ingestion |
| `packages/percept` | TS | Perception front-end. YOLO 검출→RetinaFace→ArcFace/CLIP-ReID(query-only)→DetectionRegistry(D{id} grounding)→VLM subject proposal→identity binding→live-latency commit→keyframe→postprocess | §3.2 perception front-end 전체 |
| `services/mem` | Python | 메모리 엔진. FastAPI + Qdrant. 9 collection으로 3-level episodic + semantic consolidation + procedural + ReID gallery 관리 | §3.3-3.5 + §3.6 persistence |
| `packages/mem` | TS | memory service의 TS HTTP 클라이언트 + OpenClaw plugin. tool schema(search/get/delete/add_note/get_narrative) | §3.6 retrieval tool |
| `packages/reflectworld` | TS | 오케스트레이터. VisionController(agent steering), MemoryWriteScheduler(직렬 write+retry), PostProcessor(알림), PipelineStore(SQLite), Dashboard server, CLI | §3.6 system realization |
| `packages/host` | TS | dev/demo용 최소 OpenClaw host. tool HTTP API(`POST /api/tool/:name`) 노출 | 배포 런타임 |
| `packages/dashboard` | TS(React) | UI. WS monitor 이벤트 + HTTP API로 segment timeline, entity panel, memory panel, chat | Figure 7 dashboard |
| `skills/reflectworld` | MD | OpenClaw skill descriptor. 4 tool(cam/perception/memory/act)의 언제 쓸지 라우팅 | §3.6 OpenClaw plugin |

### 핵심 코드 대응 — 논문 메서드 ↔ 코드 구현

#### 1. Segment 분할 (§3.2)

`packages/cam/src/segmenter/composite.ts` — `CompositeSegmenter`: motion+voice+entity 협력 세그멘테이션. idle/active 상태머신, hard min/max duration + frame cap, segment 길어질수록 짧은 무음도 cut하는 adaptive VAD silence threshold. `CamPipeline`(`packages/cam/src/pipeline.ts`)이 frame+PCM을 누적하고 segmenter cut 신호 시 `buildSegment` → `onSegment`(perception 전달).

#### 2. Perception Pipeline (§3.2)

`packages/percept/src/pipeline.ts` — `analyze()`: 10단계 staged pipeline. (1) WorkingMemory event 할당 → (2) SemanticContext load → (3) Strategy 선택(L0=VLM only / L1=VLM+detection+re-ID) → (4) STT 병렬 → (5) L1 detection: per-frame YOLO + RetinaFace + ArcFace/CLIP-ReID **query-only**(gallery write는 별도 `ReIdAccumulator`가 batch 처리, re-IDer는 identity commit 안 함=evidence/decision 분리 원칙 코드 구현) → (6) D{id} set-of-mark grounding → (7) VLM subject proposal(context-enhanced prompt) → (8) identity/name binding → (9) live-latency streaming-critical commit → (10) keyframe save + postprocess.

#### 3. Context-enhanced Perception (§3.2 핵심 차별점)

3 layer context가 prompt에 조립되는 구현:
- **Working memory** — `packages/percept/src/working-memory.ts`: per-camera 고정 한도. event_timeout 30s, entity_ttl 300s(5분), max_event_segments 18. rolling event summary, 최근 3 segment summary, active entities, subject targets(≤8, 연속 미검출 시 drop) 관리.
- **Semantic context** — `packages/percept/src/semantic-context.ts`: per-camera persistent profile. bootstrap(15 segment 후) / stable-phase(일일) 갱신. 100 segment마다 `onAgentConsult` callback → agent steering hook.
- **Entity history** — re-ID hit 시 해당 entity의 최근 episodic+semantic record를 memory service에서 retrieve해 prompt에 주입.

→ 스니펫: [agent steering & working memory](../source/git/snippets/ReflectWorld-MM_2026_Rightly_Robotics__agent_steering_working_memory.md)

#### 4. Agent Steering (§3.2)

`packages/reflectworld/src/vision-controller/vision-controller.ts` — `VisionController`:
- `evaluateFrame()`: frame-level 정책. detection+motion+VAD 신호로 VLM 분석 여부 gate. hold/cooldown, voice-never-dropped 보장.
- `evaluateSegment()`: segment-level 정책. (1) agent가 설정한 exit condition 확인 → (2) segment policy rules 실행(cooldown) → (3) override 상태 갱신(절대 5분 timeout) → **(4) Agent consultation**: cooldown(~1분 media time) + event budget(N events마다 강제) + trigger conditions 확인 → `shouldConsultAgent=true` 시 host agent(OpenClaw)에 consult. **reply는 allowlist제한**(scene description, focus targets, security rules) + 명시 exit condition 필수.

※ 동일 segment를 VLM이 다시 보는 게 아니라 — 한 segment에서 VLM은 1회 실행되지만, agent의 steering이 prompt를 조작해 해석을 실질 재구성. 그리고 그 이해가 후속 segment prompt에 누적 주입됨(Figure 5 task-adaptive extraction의 구현).

#### 5. Hierarchical Episodic Memory 3-level (§3.3)

`services/mem/src/qdrant_store.py` — 9 collection 중 episodic 3-level:
- `episodic_entity`(1536d) — per-entity per-segment observation (finest)
- `episodic_trace`(1536d) — segment-level summary (middle)
- `episodic_schema`(1536d) — chapter-level narrative summary (coarsest). `NarrativeMemory`(`packages/percept/src/narrative-memory.ts`)가 event 종료 시 누적 → N events 또는 시간 간격 도달 시 LLM으로 chapter 요약.

#### 6. Evolving Semantic Memory — Add/Update/Delete + Importance (§3.4)

`services/mem/src/memory_manager.py` — `_process_entity_semantic_memory` + `_merge_entity_semantic_memory`:
- `ENTITY_SEMANTIC_UPDATE_INTERVAL = 5` — entity별 관측 5회마다 trigger.
- LLM이 `SEMANTIC_PATTERN_ANALYSIS_PROMPT`로 Add/Update/Delete/NONE 결정.
- `compute_importance(old, γ=0.2)` = `old + (1-old)*0.2` — 논문 Eq.(1) 구현.
- **identity fact protection**: `category="identity"`인 사실은 importance=1.0, LLM UPDATE/DELETE에서 skip.
- identity가 아닌 Update/Delete는 기존 semantic fact ID(임시 정수 매핑) 지정 필수.

→ 스니펫: [memory collections & consolidation](../source/git/snippets/ReflectWorld-MM_2026_Rightly_Robotics__memory_collections_consolidation.md)

#### 7. Procedural Memory (§3.5)

`services/mem/src/memory_manager.py` — `_create_procedural_memory`: user rule을 LLM이 summarize(카메라 scope면 raw) → `procedural` collection. `packages/percept/src/notification-resolver.ts` — VLM semantic signal + per-camera policy + 최근 전송 기록으로 deterministic 알림 결정(dedup·cooldown).

#### 8. Retrieval & OpenClaw Deployment (§3.6)

- `packages/mem/src/client.ts` — `ReflectWorldClient`: memory service HTTP 클라이언트. `search`(collection 그룹 alias: episodic/semantic/all), `getMemories`(importance desc 정렬), reID search/upsert.
- `skills/reflectworld/SKILL.md` — 4 tool(cam/perception/memory/act) 라우팅: 과거 질문→memory search, 현재 상황→snapshot, 카메라 추가→add_source, 알림→act add_rule.
- `packages/host` — dev/demo용 `POST /api/tool/:name` HTTP API. dashboard chat은 `DashboardChatController`가 agent-policy + memory-search grounding으로 답.

### Per-step cost 유지 메커니즘 (코드 확인)

논문 §3.1의 "per-segment working state and query context remain bounded"가 코드로 확인됨:
- **Working memory 고정 한도**: `max_event_segments=18`, `entity_ttl_sec=300`, target ≤8, 최근 3 segment summary — stream 길어져도 한 segment 처리 시 context 크기 변화 없음.
- **계층 탐색**: episodic_schema가 trace를 index → 전체 entity observation 전부 안 봄.
- **consolidation N=5 주기**: 매 segment가 아니라 5단위.
- **live-latency 모드**: `pipeline/live-latency.ts`가 identity-critical 필드(event boundary, identity, matched rules, summary)만 per-user lock 하 즉시 commit → 다음 segment가 막히지 않음.

## References

- 논문: [arXiv:2607.09759](https://arxiv.org/abs/2607.09759) (v2, 14 Jul 2026)
- 코드: [github.com/addxai/ReflectWorld](https://github.com/addxai/ReflectWorld.git) (submodule: `source/git/ReflectWorld_addxai/`)
- 비교 대상: M3-Agent [arXiv:2508.09736](https://arxiv.org/abs/2508.09736) · WorldMM [arXiv:2512.02425](https://arxiv.org/abs/2512.02425)
- 배포 runtime: OpenClaw(본 repo 분석 → [OpenClaw]([git]_openclaw_openclaw.md))
- 관련 본 repo 분석: [AgenticSTS](../report/[paper]_AgenticSTS_A_Bounded-Memory_Testbed_for_Long-Horizon_LLM_Agents_2026_arxiv.md) · [ABot-AgentOS](../report/[paper]_ABot-AgentOS_A_General_Robotic_Agent_OS_with_Lifelong_Multi-modal_Memory_2026_Alibaba.md) · [Memora](../report/[paper][git]_Memora_A_Harmonic_Memory_Representation_Balancing_Abstraction_and_Specificity_2026_ICML.md)

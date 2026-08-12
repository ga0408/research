# ReflectWorld-MM: An Entity-Oriented Multimodal Memory System for Open-Ended Video Streams — 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_ReflectWorld-MM_An_Entity-Oriented_Multimodal_Memory_System_for_Open-Ended_Video_Streams_2026_Rightly_Robotics.md) / 원본: [arXiv:2607.09759](https://arxiv.org/abs/2607.09759)
>
> (※ `paper문서명`은 `report/` 폴더의 분석 문서명)

## Abstract & Problem Setup

**Task**: open-ended video stream의 long-term multimodal memory. 캡처 장치(착용형 글래스·로봇·스마트폰)가 끊임없이 보는 세계를 연속으로 관측하고, 그 경험을 축적하여 언제든 질의·판단에 활용해야 하는 assistant 메모리 시스템. 한 video clip을 이해하는 모델과 달리, "어제 누가 나타났는지"·"상황이 어떻게 변했는지"를 회상할 수 있어야 함.

**기존 접근의 한계**:
1. Streaming/long-video 모델(MovieChat, MA-LMM, Flash-VStream, VideoLLM-online, ReKV): memory를 model 내부(context/kv-cache/sparse token bank)에 frame·token 단위로 구성. content-agnostic하고 bounded video에만 강함.
2. Text-only agent memory(MemGPT, Generative Agents, A-MEM, Mem0, MemoryBank): episodic/semantic/procedural 분해는 cognitive theory에 부합하지만, video stream을 perceive하지 않고 persistent visual entity를 resolve하지 않음.
3. 가장 가까운 선행 — M3-Agent(entity-centric multimodal graph + multi-turn retrieval)와 WorldMM(multi-scale episodic/semantic/visual memory):
   - M3-Agent: frequency 가중치 투표 기반 append-only 업데이트 → 사실이 revise/delete되지 않음.
   - WorldMM: temporal scale을 dataset마다 수동 설정, entity identity 유지 안 함 → "그 사람이 누구인지"에 달린 질문 불가.
   - 둘 다 multi-scale episodic + evolving entity semantic + procedural를 단일 아키텍처로 통합하지 못하며, 임의 live stream에서 동작하는 완전한 서비스로 구현되지 않음.

**Three design principles**:
1. **evidence vs decision 분리** — detector/re-IDer는 scored evidence만 제공, single resolver가 identity 결정을 독점. 오류가 국소적·auditable.
2. **time-scale/abstraction level 기반 계층 구성** — flat log가 아닌 계층 메모리.
3. **과거가 현재 인식에 참여** — segment 해석 전에 축적된 context로 prompt를 enhance하여 isolated description 회피.

## Algorithm 1 — Online memory construction

```
Require: stream S; consolidation interval N; schema budget B
 1: init short-term memory W ← ∅, store M ← ∅
 2: for each segment s in Segment(S) do
 3:    D ← Detect(s); Ev ← ReID(D)            {local evidence}
 4:    c ← Context(W, M, Ev)                  {working + scene + entity history}
 5:    (ℓ, prompt) ← Steer(s, c)              {agent-steered, context-enhanced}
 6:    O ← VLM(s, prompt, ℓ) anchored to D, Ev
 7:    E ← Resolve(O, Ev)                     {evidence → identity}
 8:    W ← UpdateShortTerm(W, O, E)           {narrative continuity}
 9:    M.WriteEntity(E); M.WriteTrace(O, E)
10:    for all entity e updated in E do
11:       if Count(e) mod N = 0 then
12:          M.Consolidate(e)                 {Add/Update/Delete + reinforce}
13:       end if
14:    end for
15:    if event closed and (#events ≥ B or interval elapsed) then
16:       M.WriteSchema(chapter)
17:    end if
18: end for
```

## 3.2 Perception Front-End

**Streaming ingestion**: 공통 gateway로 임의 live source(RTSP/RTSPS 네트워크 카메라, 로컬 파일, USB 웹캠, HTTP stream, 스마트폰)를 정규화. visual motion + voice activity로 activity-coherent segment로 분할, transcript(가능 시 speaker diarization) 부착. open-ended stream 가정(끝 없음).

**Entity recognition**: segment 내 person/object detect → 기존 entity gallery와 비교 re-ID. person은 face+body appearance dual evidence. 단, re-ID는 scored candidates만 제안(scored evidence 원칙). **resolver 단독**이 identity 쓰기 권한 보유 → 단일 noisy match가 memory를 부패시키지 않고 모든 identity 결정이 auditable.

**Context-enhanced perception** (핵심 차별점): VLM이 segment를 isolated 상태로 해석하지 않음. segment prompt가 3 layer context로 강화됨:
1. **bounded working memory** — 현재 event state 유지, within-event narrative continuity 제공 (잠시 frame 이탈한 사람도 동일 actor로 읽힘). rolling event summary(~100 words), 최근 3 segment summary, active entities, ≤8 tracked targets. target은 연속 미검출 시 드롭, entity는 수분 inactivity timeout, event는 max segment 초과 시 rotate.
2. **per-camera semantic context** — scene type, typical routines, anomaly baseline. LLM이 주기적으로 관측 누적 시 갱신. scene/routine directive로 prompt에 render.
3. **entity history** — re-ID hit 시 해당 entity의 최근 episodic+semantic record를 retrieve해 prompt에 추가. 현재 frame을 "누가·무엇이 있는지"와 함께 이해.

과거를 저장 후 query 시에만 참조하는 기존 multimodal agent와의 핵심 차이.

**Agent-steered adaptive perception**: 고위 agent가 scene understanding을 prompt에 주입 — scene description, focus target, active user rule. controller는 deterministic frame-level policy + semantic segment-level policy로 segment 분석 여부/풍부도 결정. static/silent segment는 저렴하게, 정보 풍부/rule-relevant는 풍부하게. hard guards(첫 segment, speech 포함 segment, audio 미변환 segment)는 무조건 분석. detection anchoring + ungrounded identity target 방지 정책(Appendix C).

## 3.3 Hierarchical Episodic Memory — 3 abstraction levels

자서전적 기억 구조(Conway & Pleydell-Pearce 2000) 따름. coarse memory가 fine event를 index.

| Level | 단위 | 내용 | 역할 |
|-------|------|------|------|
| **Entity** (finest) | segment×entity | 특정 entity의 appearance, behavior, interaction. persistent entity ID로 keyed | event-specific knowledge — "특정 시각에 특정 entity가 무엇을 했는지" 회상 |
| **Trace** (middle) | segment | event 요약 + 관련 entity list → entity-level observation에 back-link | default recall 단위. 동일 시각 다수 entity observation 결합 |
| **Schema** (coarsest) | many segments → chapter | event 닫힘 시 누적, event budget B 또는 시간 간격 도달 시 chapter-level summary로 consolidate | long history에서 fine trace 탐색 범위 bound. 작은 인덱스로 긴 이력 navigate |

## 3.4 Evolving Entity-Centric Semantic Memory

entity별 durable knowledge(attributes, habits, relations)를 증류. **N=5 신규 관측마다 consolidation** trigger. consolidator가 prior episodic+semantic record를 읽어 4가지 edit decision 중 하나 내림:
- **Add** — 새 사실 추가
- **Update** — 기존 사실 정정(revise)
- **Delete** — 더 이상 지원되지 않는 사실 제거
- (no change)

→ append-only M3-Agent와의 본질적 차이: 새 증거가 기존 사실 reinforce/revise/delete 가능.

**Importance score** w∈[0,1] — fact가 재확인될 때 증가 (asymptotic toward 1):

```
w ← w + (1 − w)·γ,   γ ∈ (0,1)        ... Eq.(1)
```

- γ = 0.2 (default, Appendix C).
- 반복 증거 → 사실 안정화, 신규 fact는 revise 용이. ranking에 참여해 well-confirmed fact 우선.
- identity fact는 maximum importance로 write되며 automatic update/delete에서 보호.
- per-entity counter는 persistent·per-user scope, 현재 segment는 self-evidence로 제외, prior history 없는 entity는 merge skip.
- Update/Delete는 기존 semantic fact ID를 지정해야 하며, 미검증 target edit은 drop.

## 3.5 Procedural Memory & Proactive Response

사용자 rule/preference 저장("Notify me when the dog is on the sofa"). rule이 perception prompt에 surface되어 stream을 한 번만 분석. VLM이 rule 직접 실행하지 않고 **semantic signal**(situation risk 성격, 어떤 rule에 매치되는지)만 emit. 별개 **deterministic resolver**가 이 신호를 per-camera notification policy + 최근 전송 기록과 결합해 "무엇을 보낼지" 결정. rule match는 decision을 위한 evidence(forece trigger 아님) → 반복/허위 알림 방지.

## 3.6 Retrieval & System Realization

**Retrieval**: 외부 store에서 tool call로 질의. agent가 memory level 선택(entity observations / traces / schemas / semantic facts / procedural rules). shared entity identifier가 level 연결 → event→participants, entity→history 복구 가능. importance score가 ranking 참여. agent는 full stream이 아닌 retrieved subset에 attend.

**System realization**:
- memory는 model context가 아닌 indexed DB에 persist.
- 총 store는 성장하나 per-segment working state와 query context는 bounded → 무한 실행 가능. ReKV(streaming-cache)가 stream 길이에 비례하는 state를 갖는 것과 대조.
- 공통 interface: capture / perception / memory query / notification policy / context management tool. benchmark agent와 OpenClaw deployment가 동일 interface 공유.

**Two execution modes**(streaming commit, Appendix C):
- `offline_quality` — deterministic benchmark용.
- `live_latency` — identity-critical 필드(event boundary, identity, matched rules, summary)를 per-user lock 하 즉시 commit → 다음 segment release. enrichment 필드는 frozen critical state에 후속 reconcile.

**Local perception utilities**(ONNX 변환, Table 7): YOLO26m(객체 검출, 필수), RetinaFace-ResNet50(얼굴 박스/랜드마크, 필수), ArcFace-MobileFaceNet(얼굴 임베딩, 필수), CLIP-ReID ViT-B(몸 임베딩, 선택), RTMPose-S(pose, 선택), Moonshine Tiny int8(STT, 선택·cloud 대체 가능). 이들은 reasoning 모델이 아닌 bounded signal utility.

**Deployment via OpenClaw**: OpenClaw assistant runtime용 plugin으로 package. tool-based contract로 capture control / perception / memory query / notification policy / per-camera context 노출. host assistant가 새 source 시청, "무슨 일이 일어나는지" 질의, long-term memory 조회, rule 설정을 ordinary tool call로 수행 (기저 시스템 변경 없음). benchmark agent와 deployed assistant가 동일 memory interface 사용(주변 agent만 다름).

## 4 Experiments — Six benchmarks

**Benchmarks**(general + egocentric + entity-centric 혼합):
- **VideoMME-Long** (Fu 2025) — 일반 long video MC, 최대 ~1시간.
- **LVBench** (Wang 2025a) — 일반 long video MC, 최대 ~2시간.
- **HippoVlog** (Lin 2025) — ~1000 MC questions over audiovisual vlog, modality 간 memory 형성·연상.
- **EgoLife-QA** (Yang 2025) — 1주 egocentric 기록, identity-dependent life question. sub-task: EntityLog(EL), EventRecall(ER), HabitInsight(HI), RelationMap(RM), TaskMaster(TM).
- **M3-bench-robot** (Long 2025) — egocentric video open-ended QA, multi-hop/cross-modal/person-centric reasoning. 평균 ~30분.
- **M3-bench-web** — online video split, 동일 성격.

**Implementation**: GPT-5-mini(memory extraction, semantic consolidation, VLM perception), GPT-5(query answering agent). person re-ID = face+body appearance. 모든 memory item은 OpenAI text-embedding-3-small(1,536 dim)로 임베딩. vector DB 기반 store(episodic 3 level, entity-centric/per-camera semantic, procedural rule, face/body gallery 분리 collection). M3-bench judge는 GPT-5-mini(원래 GPT-4o judge 사용 불가). 공정 비교 위해 M3-Agent(EgoLife-QA, M3-bench), WorldMM(VideoMME-L)는 공식 코드로 re-run. GPT-5 reference는 EgoLife-QA, M3-bench 직접 run; VideoMME-L, LVBench, HippoVlog는 WorldMM paper 인용.

### Results (Table 1 — entity-sensitive benchmarks)

| Method | EgoLife-QA | M3-robot | M3-web |
|-------|-----------|----------|--------|
| Qwen2.5-VL-7B | – | 3.4 | 14.9 |
| Gemini-1.5-Pro | 36.9 | 8.0 | 23.2 |
| GPT-5 | 42.6 | 34.7 | 53.9 |
| MovieChat | – | 11.2 | 12.6 |
| Flash-VStream | – | 19.4 | 23.6 |
| M3-Agent | 30.8 | 28.3 | 45.6 |
| **ReflectWorld-MM** | **46.8** | **37.4** | **56.0** |

- EgoLife 원래 paper는 manually annotated identity description로 45.5; ReflectWorld-MM은 자동 구축 memory로 46.8 달성.
- WorldMM은 entity identity 미유지로 EgoLife-QA/M3-bench 미보고.

### Results (Table 2 — general long-video benchmarks)

| Method | VideoMME-L | LVBench | HippoVlog |
|-------|-----------|---------|-----------|
| GPT-5 | 74.3 | 60.4 | 75.7 |
| MovieChat/VideoChat-Flash/Video-RTS 등 | <55 | <40 | <60 |
| HippoMM | 41.6 | 38.2 | 71.9 |
| M3-Agent | 55.3 | 49.3 | 65.5 |
| WorldMM | 73.8 | 61.9 | 78.3 |
| **ReflectWorld-MM** | **76.9** | **69.4** | **80.9** |

GPT-5 reference 대비: VideoMME +2.6, LVBench +9.0 (video 길어질수록 개선 확대).

### Answer Efficiency (Table 3) — memory quality

검증 질문: memory가 실제로 유용한지(JSON), 단순 sparse/empty이거나 fallback 남용은 아닌지. accuracy, average answer tokens, video fallback 비율 측정.

| Benchmark | Method | Acc. | Tok. | Vid. fallback |
-----------|--------|------|------|---------------|
| EgoLife-QA | M3-Agent | 30.8 | 13k | 0.0% |
| EgoLife-QA | ReflectWorld-MM | 46.8 | 55k | 4.6% |
| VideoMME-L | M3-Agent | 55.3 | 8.5k | 0.0% |
| VideoMME-L | WorldMM | 73.8 | 56k | 34.0% |
| VideoMME-L | ReflectWorld-MM | 76.9 | 43k | 6.8% |

- ReflectWorld-MM: high accuracy + rare fallback(4.6%, 6.8%) → memory가 answerable evidence 저장(stored index가 아님).
- M3-Agent: 적은 token·0% fallback but 낮은 정확도 → compact trace가 evidence 누락.
- WorldMM: 강하지만 34% fallback → memory가 index 역할, source video 재필요.

### Ablation (Figure 3, 응답 시간 기준 — memory 재구축 비용 회피)

동일 memory 구축 후 질의 agent에서 한 component 차단. 중복 evidence로 인해 drop은 component 가치의 **lower bound**.
- **w/o entity association** (entity-linked retrieval 차단)
- **w/o schema** (coarse event index 제거) — M3-robot 37.4 → 33.6 (최대 drop)
- **w/o semantic** (distilled entity knowledge 은폐)
→ 각 component gain이 실제 효과임 확인. 단순 answer model 효과 아님.

### Consolidation & Importance params (Appendix C)

| 파라미터 | 값 |
|---------|-----|
| consolidation interval N | 5 (entity 신규 관측 단위) |
| importance growth rate γ | 0.2 |
| importance cap | 1.0 |
| identity fact importance | maximum (자동 update/delete에서 보호) |
| working memory event summary cap | ~100 words |
| working memory recent segment summaries | 3 |
| working memory tracked targets | ≤8 |
| target drop | 연속 미검출 후 |
| entity inactivity timeout | 수분 |
| agent steering cooldown | ~1분 media time |
| per-event steering budget | 소형 |
| steering reply | allowlist(scene description, focus targets, security rules) + 명시 exit condition |

## 4.5 Qualitative Analysis (case studies — write side)

1. **Entity-centric semantic memory 진화** (Figure 4) — 1주 vlog. 서로 다른 날의 관측이 동일 entity에 link → 식당/카페가 work/study routine, 볼더링/헬스가 exercise habit으로 consolidate. entity identity가 retrieval key 그 이상: longitudinal evidence 축적 단위.
2. **Task-adaptive extraction** (Figure 5) — coffee-making scene. scene이 coffee 준비로 인식되면 후속 perception에 주입 → dripper setup, extraction, adjustment, brew check가 동일 workflow 단계로 해석. 고정 field 추출이 아닌 누적 scene context가 의미 결정.
3. **Procedural memory & proactive response** (Figure 6) — "Notify me when the dog is on the sofa" rule. dog 근처 → 점프 → 소파 위 누움의 visual evidence가 action trigger 자체가 아님. semantic signal로 변환 → rule·policy·dedup·cooldown과 resolve.

## 4.6 Real-World Operation

완전 service 동작. capture layer가 임의 video stream(network camera, webcam, local file, HTTP stream, smartphone)을 공통 gateway로 받아 fixed duration 가정 없이 open-ended 처리. memory는 indexed DB persist → downstream agentic system 동일 backend attach. dashboard(Figure 7)는 live interview에서 video source+chat panel, per-segment moments(timestamp), entity panel(semantic memory, sightings, activity) 제공. benchmark agent, dashboard, OpenClaw agent 모두 동일 deployed memory backend 사용.

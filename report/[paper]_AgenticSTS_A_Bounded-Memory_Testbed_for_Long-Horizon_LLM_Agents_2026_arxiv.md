> [paper] https://arxiv.org/abs/2607.02255

# AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents

## Summary & Outline

**한 줄 요약:** Long-horizon LLM agent가 매 결정을 내릴 때 이전 행동 기록을 prompt에 누적하지 않고, 5개 typed layer(L1–L5)에서 필요한 정보만 선택해 매번 새 user message를 조립하는 **bounded-memory contract**(prompt 크기가 run 길이와 무관하게 항상 일정하도록, 각 결정이 볼 정보를 규칙으로 정해둔 구조)를 제안. 각 layer를 독립적으로 켜고 끄어 효과를 분리 측정하는 **ablation**(분리 실험)으로 L5 skill layer 활성화 시 3/10→6/10 승률 향상(directional)을 입증했으며, 누적 append 방식 경쟁 agent 대비 66–90× token 효율. 298 trajectory의 재현 가능한 아카이브 release.

**논문 구조 outline:**
1. Introduction — memory as a contract, bounded contract 제안
2. Related Work — loop/context engineering, prompt-history agents, structured memory, skill libraries, game testbeds
3. The Slay the Spire 2 Testbed — 게임의 4가지 testbed property, ascension ladder, scoring
4. Architecture: Per-Decision Typed Retrieval — 5개 typed layer, routing, combat truncation, L4/L5 상세
5. Experimental Methodology — 5-condition decomposition, cross-backbone probe, auto-mode ladder, statistical protocol
6. Results — difficulty calibration, within-harness ablation, ascension ladder, template skills/transfer/episodes
7. Comparison with Open-Source Accumulating-Context Agents — STS2MCP, CharTyr 비교
8. Discussion — interpretation, implications for loop engineering
9. Conclusion + Limitations

## Problem & Motivation

- **연구 배경:** Long-horizon LLM agent에서 기존 방식(ReAct/Reflexion)은 매 결정마다 이전의 관찰·도구 호출·자기 비판을 prompt에 **그대로 누적 append**함. 접근은 쉽지만, 모든 기록이 한 prompt에 뒤섞여 "어떤 memory component가 효과가 있었는지"를 분리해서 말할 수 없는 "jumbled mixture(뒤섞인 덩어리)"가 됨. 본 논문은 memory를 "저장소"가 아니라 **"각 결정 시점에 LLM에게 무엇을 보여줄 것인가를 설계자가 통제하는 규칙(contract)"**으로 재구성함.
- **풀고자 하는 문제:** Long-horizon benchmark가 memory interface를 bounded, inspectable, reusable하게 만들 수 있는가? Context growth를 implicit default로 두는 대신, memory stage를 typed, bounded, **ablatable**(각 component를 독립적으로 켜고 끌 수 있는) contract로 정식화할 수 있는가?
- **기존 접근의 한계:**
  - **Prompt-history / replay agent (ReAct, Reflexion):** transcript가 run 길이에 비례해 growth → context overflow, attention dilution. 단일 component 효과 분리 불가
  - **Structured memory (MemGPT, Mem0, MemoryOS, GAM 등):** raw history에서 external store로 정보 이동은 하지만, 대부분 dialogue/QA 평가에 국한. action policy에 retrieval feed하는 stochastic environment 부재
  - **Skill-library agent (Voyager, SkillsBench, SkillOS 등):** skill library는 있지만 typed memory + skill + game policy를 **joint로 ablate**(예: "skill만 끄고 memory는 켠 상태에서 승률 측정")할 수 있는 구조적 contract가 부재. 모든 것이 prompt history에 뒤섞여 "skill layer만의 기여"를 분리 측정할 수 없음
  - **Game testbed (Crafter, NetHack, BALROG 등):** pixel-rendered 또는 code complexity로 인해 LLM-readable rule space 부재. closed-rule + text-readable 한 long-horizon testbed 필요

> **"ablatable contract가 부재하다"는 것의 의미:** ablation = 특정 component만 켜고 끄면서 그 효과를 분리 측정하는 실험. 기존 agent는 memory, skill, rules가 모두 prompt history에 뒤섞여 있어서 "skill만 껐을 때 승률이 어떻게 변하나?"를 실험할 수 없음. 본 논문의 5개 typed layer는 각 layer를 명명된 slot으로 분리하여, L4(memory)나 L5(skill)를 on/off만으로 독립적 ablation이 가능한 "contract(구조적 규칙)"을 제공함.

## Contributions

1. **방법론 기여 — 구조 설계:** 매 결정마다 이전 기록을 누적하지 않고 L1–L5 layer에서 필요한 정보만 선택해 prompt를 조립하는 bounded-memory contract를 설계. 각 layer를 독립적으로 on/off(ablation)할 수 있어 "skill layer만 효과가 있나?" "memory layer만 효과가 있나?"를 분리해서 측정 가능
2. **실증 기여 — 실험으로 검증:** A0 난이도에서 가장 큰 차이가 L5 skill layer 활성화와 일치 (3/10→6/10, 통계적으로는 확정 아닌 directional). 같은 구조를 3개 모델(Gemini, Qwen, DeepSeek)에 적용해 ablation이 모델에 관계없이 작동함을 확인. postrun memory가 켜진 stream은 A6–A8 고난도까지 도달, 꺼진 stream은 A2–A4에서 정지
3. **데이터셋/도구 기여 — 재현 가능한 아카이브:** 298개 run 기록, 각 run의 조건 태그(L4/L5 on/off 등), L4/L5 저장소 스냅샷, 결정 시점의 prompt 기록, 통계 분석 스크립트를 release. 다른 연구자가 동일한 환경에서 다른 memory 구조를 비교 실험할 수 있음

> 핵심 주장은 contract 내부의 **layer separability**에 관한 것. matched accumulating-context comparison은 future work로 명시.

## Method

### 전체 아키텍처: Bounded-Memory Contract

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Bounded-Memory Contract Architecture                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Per-Decision Composition (no raw transcript append)                  │
│                                                                        │
│   ┌──────────────────────────────────────────────────────────┐        │
│   │  Decision d at state s_d                                 │        │
│   │                                                          │        │
│   │  u_d = π(L1, L2(s_d), L3(s_d), L4(s_d), L5(s_d))        │        │
│   │                                                          │        │
│   │  → ⟨sys, u_d⟩ sent to LLM                                │        │
│   └──────────────────────────────────────────────────────────┘        │
│                          │                                             │
│                          ▼                                             │
│   ┌──────────────────────────────────────────────────────────┐        │
│   │  Five Typed Knowledge Layers                             │        │
│   │                                                          │        │
│   │  L1 (protocol)     ── fixed, always on ──────────────┐   │        │
│   │  L2 (schema)       ── fixed, always on ────────────┐ │   │        │
│   │  L3 (rules)        ── refresh, filter ───────────┐ │ │   │        │
│   │  L4 (episodes)     ── postrun, off/on ────────┐  │ │ │   │        │
│   │  L5 (skills)       ── gated, off/A/B ──────┐  │ │ │ │   │        │
│   │                                              │  │ │ │ │   │        │
│   │  Mutability: L1,L2 (fixed) → L3 (refresh)   │  │ │ │ │   │        │
│   │              → L4 (postrun) → L5 (gated)     │  │ │ │ │   │        │
│   └──────────────────────────────────────────────┘  │ │ │ │   │        │
│                                                     │ │ │ │   │        │
│   ┌──────────────────────────────────────────────────┘ │ │ │   │        │
│   │  Routing & Dispatch                               │ │ │   │        │
│   │  ┌──────────┐ ┌────────────┐ ┌────────┐ ┌───────┐ │ │ │   │        │
│   │  │ fast     │ │ strategic  │ │analysis│ │evolve │ │ │ │   │        │
│   │  │ (trivial │ │ (ordinary  │ │(postrun│ │(skill │ │ │ │   │        │
│   │  │  combat) │ │  decision) │ │ memory)│ │ distil│ │ │ │   │        │
│   │  └──────────┘ └────────────┘ └────────┘ └───────┘ │ │ │   │        │
│   └───────────────────────────────────────────────────┘ │ │   │        │
│                                                         │ │   │        │
│   Combat: max 3 messages/round (combat_start, ok,       │ │   │        │
│           latest user state) — earlier rounds → typed   │ │   │        │
│           state summary, NOT appended to transcript     │ │   │        │
│                                                         │ │   │        │
│   Result: median 67 strategic LLM calls/run             │ │   │        │
│           (~500 additional decisions mechanically       │ │   │        │
│            resolved or routed to fast tier)             │ │   │        │
└─────────────────────────────────────────────────────────┘─┘   ┘        │
└──────────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙: Memory = Bounded Typed Contract

이 논문의 핵심 통찰은 memory를 "얼마나 많은 history가 들어가느냐"의 문제에서 **"어떤 typed evidence가 선택되느냐"**의 문제로 전환한 것. 매 결정마다 raw transcript를 append 하는 대신, 5개 typed layer에서 retrieve 한 slice들로 fresh user message를 조립.

#### Contract는 누가 정하는가: 인간 설계자 vs Agent

> **Contract 자체는 인간(논문 작성자)이 정한 고정 구조.** agent는 contract 안에서 자기 영역(L4/L5)의 content만 확장할 수 있음. contract 자체를 바꿀 수는 없음.

| | Contract 구조 (인간이 정함) | Content (누가 채우는가) |
|---|---|---|
| **L1** | system prompt의 역할·구조 — **immutable** | 인간이 작성, agent 수정 불가 |
| **L2** | 결정 타입별 schema·legal action format — **immutable** | 인간이 작성, agent 수정 불가 |
| **L3** | card/relic/monster 데이터 형식 — patch 단위 | 인간이 작성, game patch 시 업데이트 |
| **L4** | 키 구조 (char×asc×act×enemy), 저장 형식 | **agent** (analysis tier)가 postrun에 전투 요약 추출 |
| **L5** | trigger/policy 형식, A/B 검증, 4-level write gate | **agent** (evolution tier)가 mistake-driven으로 skill 제안 |

> **prompt evolution 실패 맥락:** 처음에는 agent가 L1/L2(prompt)도 수정할 수 있게 했음. 하지만 postrun LLM이 제안한 prompt 수정 33/33이 전부 A/B 검증에서 실패(주로 decision-schema 위반). "agent가 contract 구조를 바꾸는 것"은 시도했다가 실패한 것. 이후 L1/L2를 immutable로 고정하고, agent 수정 영역을 L4/L5 content로만 제한. → **구조(contract)는 인간이 고정하고, 내용(L4/L5)만 agent가 gate 아래에서 채움.**

이 설계가 주는 4가지 evaluation handle:

1. **horizon growth capped (prompt 크기 상한):** 각 layer에 slot budget(top-k)이 있어서, run이 길어져도 prompt 크기가 늘어나지 않음. 1번째 결정이나 100번째 결정이나 prompt 크기가 항상 ~5k token으로 동일. 기존 방식은 run 길이에 비례해 ~500k까지 증가.
2. **retrieved evidence labeled (정출처 라벨링):** prompt에 들어가는 모든 정보가 "이건 L3 rules에서 온 것", "이건 L4 episodes에서 온 것"으로 layer별 라벨이 붙어 있어서, 어느 layer가 이 결정에 기여했는지 추적 가능. 기존 방식은 모든 기록이 섞여 있어 출처를 알 수 없음.
3. **L4/L5 toggle (component on/off):** prompt 전체를 다시 작성할 필요 없이, L4(episodes)나 L5(skills)를 flag 하나로 on/off 가능. 예: `--no-skills` 플래그만 주면 L5를 끈 상태로 동일한 게임을 실행해서 "skill이 있을 때 vs 없을 때" 승률을 비교할 수 있음.
4. **condition tag (실험 조건 관리):** 각 run에 조건 태그(L4 on/off, L5 mode A/B, backbone 종류 등)가 붙어서, 동일한 harness에서 다른 contract 설정을 비교 가능. 298 trajectory가 모두 condition tag와 함께 release되어 community가 재현·비교 가능.

### 게임 배경: Slay the Spire 2 구조

> L4/L5 lifecycle을 이해하려면 게임의 구조를 먼저 알아야 함. 본 논문에서는 게임 용어를 번역 없이 그대로 사용.

```
Slay the Spire 2 — run 구조
═══════════════════════════════════════════════════════════
Run (한 판)
├─ Act 1 ─── 10~15개 node (combat / elite / event / shop / rest / treasure)
│   └─ Act 1 Boss
├─ Act 2 ─── 10~15개 node
│   └─ Act 2 Boss
└─ Act 3 ─── 10~15개 node
    └─ Act 3 Boss  ← 격파 시 victory

  ↑ defeat (HP 0) 시 run 종료

Combat (전투)
├─ Round 1: enemy intent 공개 → hand에서 card 선택 → 실행
├─ Round 2: ...
└─ Round N: enemy HP 0 시 combat 승리 (run 계속)

  ↑ combat 패배 = HP 0 = run defeat (즉시 run 종료)
```

| 용어 | 의미 | 예시 |
|------|------|------|
| **run** | 완전한 한 게임. Act 1-3 보스를 차례로 격파하면 victory, 어느 시점이든 HP가 0이 되면 defeat | "10번 run 했는데 6번 승리" = 6/10 win rate |
| **combat** | run 내의 개별 전투. 전투에서 HP를 잃을 수 있지만, HP가 0이 아니면 run은 계속됨 | "Act 1 Fuzzy Wurm combat에서 18 HP 잃음" — combat은 졌지만 run은 계속 |
| **round** | combat 내의 한 턴. enemy intent가 공개되고 hand에서 card를 선택 | "R3에서 18 데미지 incoming, Weak 쓸지 말지 결정" |
| **act** | run의 큰 단계. Act 1 → 2 → 3으로 진행, 각 act 끝에 boss | "Act 1에서는 공격 카드 위주로 build" |
| **postrun** | run 하나가 victory 또는 defeat로 종료된 후. L4/L5 추출이 일어나는 시점 | "postrun에 각 combat을 분석해서 L4에 요약 저장" |

**핵심:** 한 run 안에는 ~15-20개의 combat이 있음. 각 combat에서 HP를 잃을 수 있고, HP가 0이 되면 run이 종료됨(defeat). L4와 L5는 **run이 끝난 후(postrun)**에 그 run 안의 각 combat을 분석해서 추출됨. combat 패배(=run defeat) 시에만 추출되는 것이 아님.

### Five Typed Knowledge Layers

| Layer | Store | 내용 | Mutability | Write | Ablation |
|-------|-------|------|------------|-------|----------|
| **L1** | protocol | state type별 role·protocol template | immutable | fixed | always on |
| **L2** | schema | combat/deckbuilding/map/event/intermission 결정별 schema + legal action format | immutable | fixed | always on |
| **L3** | rules | card(576), relic(293), monster(115), encounter(87), event(66) 데이터 | refresh | patch 단위 | filter |
| **L4** | episodes | postrun summary (char × ascension × act × enemy class) — case-based recall | postrun | run 후 작성 | off/on |
| **L5** | skills | triggered strategic guide — trigger + prose policy + 4-level write gate | gated | mistake-driven 또는 template | off/A/B |

#### Layer별 구체 예시 (Slay the Spire 2 — Silent)

각 layer가 매 결정마다 어떤 typed slice를 retrieve하여 fresh user message에 주입하는지, **combat 결정**(Fuzzy Wurm Crawler R3, incoming 18 damage)을 예시로 보면:

**L1 — Protocol (항상 on, cached)**
```
You are playing Silent in Slay the Spire 2.
Role: analyze combat state → output JSON action plan.
Core principles:
  - HP is a resource, not a score.
  - Front-load damage: kill enemies fast.
  - Read intents: Buff → offense; Attack → balance block/damage.
  - Play 0-cost first, skills before attacks, biggest attacks last.
```
→ 4개 정적 system prompt(COMBAT / COMBAT_BOSS / DECKBUILD / STRATEGIC) 중 상황에 맞는 것 선택. cacheable, run 길이와 무관하게 고정.

**L2 — Schema (항상 on, strictness toggle)**
```json
{
  "state_type": "monster",
  "legal_actions": [
    "play_card(card_index, target_index)",
    "use_potion(potion_index, target_index)",
    "end_turn()"
  ],
  "state": {
    "hand": [{"name":"Neutralize","cost":1,"damage":3},
             {"name":"Defend","cost":1,"block":5},
             {"name":"Strike","cost":1,"damage":6}],
    "intents": [{"name":"Fuzzy Wurm","intent":"Attack 9×2","value":18}],
    "player": {"hp":55,"block":0,"energy":3},
    "turn": 3
  }
}
```
→ 결정 타입(combat / deckbuilding / map / event / intermission)별로 legal action format + state schema가 고정. strictness toggle로 prompt만 on/off 가능.

**L3 — Rules (filterable, patch 단위 refresh)**
```
[Cards]
  Neutralize — Cost 1, Deal 3 dmg, Apply 1 Weak
  Defend     — Cost 1, Gain 5 block
  Strike     — Cost 1, Deal 6 dmg
[Monster]
  Fuzzy Wurm Crawler — HP 42-48
    R1: Buff (Strength +2)
    R2: Attack 7
    R3: Attack 9×2 = 18  ← burst window
    R4: Buff (recovery)
```
→ card / relic / monster / encounter / event의 enumerable rule data. game patch 시 refresh. filter ablation으로 특정 category 제외 가능.

**L4 — Episodes (off/on, postrun 작성, **multi-session**)**

> L4는 단일 run 내의 round 요약이 아님. **여러 run에 걸쳐 누적**되는 cross-session memory. run이 종료된 후(postrun) analysis tier 모델이 그 run 안의 각 combat을 요약해서 저장하며, 키는 (character × ascension × act × enemy class). 새 run에서 같은 enemy를 만나면 과거 run들의 요약을 retrieve해옴. victory든 defeat든 상관없이 **매 run마다** 추출됨.

```
[Silent × A0 × Act 1 × Fuzzy Wurm Crawler — 3개 run에서 누적됨]
Episode 1 (run 3, victory): combat에서 5 HP만 잃음
  R3: 18 incoming; hand had Weak+Block → took 5 dmg (clean)
Episode 2 (run 5, victory): combat에서 12 HP 잃음
  R3: 18 incoming; no Weak, pure block → took 12 dmg (bad)
Episode 3 (run 7, defeat): 이 combat에서 사망 (run 종료)
  R3: 18 incoming; greedy attack, no block → died (disaster)

Summary: R3 is burst window.
  can_apply_weak → <5 HP loss.
  no Weak → 12+ HP loss or death.
```
→ run이 끝날 때마다 postrun extraction으로 저장. victory든 defeat든 상관없이 매 run마다 각 combat의 요약이 저장됨. **새 run을 시작할 때** 과거 run들의 요약을 retrieve. 단일 run 내의 round 요약이 아니라, 여러 run에 걸친 cross-session case-based recall. raw game log가 아닌 **요약된 패턴**만 저장.

**L5 — Skills (off/A/B, gated trigger)**
```
Skill: "Weak-before-block on burst round"
Trigger:
  enemy_names: ["Fuzzy Wurm Crawler"]
  threat_levels: ["high", "lethal"]
  requires_hand_capabilities: ["can_apply_weak"]
Policy:
  At high-threat round (15+ incoming) with Weak available:
  play Weak FIRST (reduces incoming ~25%),
  then maximize block. Skip greedy attacks.
  Expected: 18 → ~5 damage taken.
```
→ trigger가 현재 상황과 match될 때만 주입. mistake-driven discovery(combat loss → A/B 검증 → 4-level write gate) 또는 stub-template authoring(Mode B)으로 populate.

#### Per-Decision Composition 예시

위 예시의 R3 결정에서 fresh user message `u_d` 조립:

```
sys  = L1 COMBAT system prompt (role + principles)        ← cached, ~2k tokens

u_d  = L2: combat schema + 현재 hand/intent/HP/energy     ← 실시간 state
     + L3: Fuzzy Wurm card data + Neutralize/Defend rules  ← enemy filter
     + L4: 3 episodes의 R3 요약 (burst window 패턴)        ← off/on toggle
     + L5: "Weak-before-block" skill (trigger match)       ← off/A/B toggle

→ sys + u_d 로 LLM 호출 (~5k tokens, run 길이와 무관하게 bounded)
```

accumulating-context agent라면 이 결정이 run의 67번째 strategic call일 때 이전 66개 결정의 raw transcript가 모두 prompt에 append되어 ~500k tokens에 도달하지만, bounded contract는 항상 ~5k로 고정.

**L4와 L5의 차이:** L4는 specific episode의 case-based recall("이전 run에서 Fuzzy Wurm R3가 18 데미지였고, Weak 쓰면 5만 닳았다"), L5는 general scenario-class tactic("high-threat burst round에서 Weak 가지고 있으면 Weak 먼저 쳐라"). Raw StS2 log는 similarity RAG로 사용하지 않음 — 비슷해 보이는 state가 card order, relic combo, route history에 따라 전략적 의미가 크게 다를 수 있기 때문.

### L4 · L5 Lifecycle: 추출 → 저장 → 검색 → 주입

L4와 L5는 **둘 다 run 종료 후(postrun)에 추출**되며, 둘 다 run 안의 **각 combat 단위**로 분석됨. 핵심 차이는 저장하는 것의 성격 — L4는 "무슨 일이 있었나(사례 회상)", L5는 "무엇을 해야 하나(일반화 전술)". 두 layer의 lifecycle을 비교하면:

```
                    L4 (Episodic Memory)                    L5 (Skill Library)
                    ─────────────────                       ──────────────────
[추출 시점]         run 종료 후 (postrun)                   run 종료 후 (postrun)
                    매 run마다, victory/defeat 무관           매 run마다, 각 combat의 loss_ratio가
                    analysis tier 모델이 각 combat 요약      baseline보다 나쁘면 evolution tier가
                                                            실수 감지 → candidate skill 제안
                    ↓                                       ↓ (또는 Mode B 템플릿 채움)
[저장 키]           (char × ascension × act × enemy)        (trigger condition)
                    여러 run에 걸쳐 누적                     trigger가 match되는 상황에만 발동
                    ↓                                       ↓
[검색 시점]         새 run에서 같은 enemy 만났을 때          매 결정마다 trigger match 체크
                    동일 키의 과거 run 요약 retrieve         현재 threat/hand/enemy에 match하는 skill만
                    ↓                                       ↓
[주입]              L4 slot으로 user message에 삽입           L5 slot으로 user message에 삽입
                    (off/on toggle 가능)                     (off/A/B toggle 가능)

[저장 내용]         "무슨 일이 있었나"                      "무엇을 해야 하나"
                    (factual recall)                         (generalized tactic)
```

> **오해 정정:** L5가 run이 fail(defeat)했을 때만 추출되는 것이 아님. run에서 victory해도 각 combat에서 HP를 너무 많이 잃었다면(loss_ratio가 baseline 대비 나쁘면) 실수로 감지되어 skill candidate가 제안됨. 예: victory한 run이라도 Act 1 Fuzzy Wurm combat에서 18 HP를 잃었는데 baseline이 5 HP면, "왜 5 HP가 아니라 18 HP를 잃었나?"를 분석해서 skill을 제안.

#### L4 추출 상세 (postrun)
```
[run 진행 중]  ← 전투(combat)별로 tracker가 round 단위로 기록 (hand, intent, damage, outcome)
        ↓
[run 종료: victory 또는 defeat]
        ↓
analysis tier 모델 호출 (Gemini 3.1 Pro 등)
  입력: 이 run 안의 모든 전투 기록 (CombatTracker에 누적된 round 데이터)
  출력: 각 combat별 요약 — "Fuzzy Wurm R3가 burst window, Weak 쓰면 5 HP만 닳음"
  주: victory든 defeat든 상관없이 모든 combat의 요약을 추출
        ↓
저장: (Silent × A0 × Act1 × Fuzzy Wurm) 키로 memory/v2/combat_episodes.jsonl에 append
  여러 run에 걸쳐 동일 키에 누적됨
        ↓
다음 run에서 Fuzzy Wurm 만남 → 동일 키로 검색 → 과거 run들의 episode 요약 retrieve → prompt에 주입
```

#### L5 추출 상세 (mistake-driven discovery)
```
[run 종료 후 postrun]  ← 각 combat의 실수(mistake) 여부를 판정
        ↓
실수 감지 trigger — 2가지 baseline과 비교:

  loss_ratio = 전투에서 잃은 HP / 전투 직전 HP
    예: 55 HP → 37 HP = 18 HP loss → loss_ratio = 18/55 = 0.33

  Baseline A (per-enemy): 같은 enemy의 과거 episode들의 loss_ratio 중앙값
    조건: 과거에 같은 enemy와 3판 이상 싸운 데이터 필요
    예: Fuzzy Wurm 과거 중앙값 = 5/55 = 0.09

  Baseline B (per-act×type): 같은 act + combat_type + character의 최근 pool 평균
    조건: pool에 3개 이상 episode 필요
    예: Act 1 monster Silent 평균 = 8/55 = 0.15

  ┌─ 실수 판정 조건 ──────────────────────────────────────────┐
  │ loss_ratio > Baseline A + delta  OR                        │
  │ loss_ratio > Baseline B + delta                             │
  │                                                             │
  │ delta (combat_type별): monster +10%, elite +15%, boss +20% │
  │ → harder combat일수록 variance가 크므로 threshold 완화       │
  │                                                             │
  │ 둘 다 baseline 데이터 부족(3판 미만) → 실수 아님 (건너뜀)    │
  └──────────────────────────────────────────────────────────┘

  예: 0.33 > 0.09 + 0.10 = 0.19  ✓ → 실수 감지!
        ↓
evolution tier 모델이 candidate skill 제안
  "Weak-before-block on burst round" (trigger + policy)
  + expected_correction: "Weak 먼저 치고 block"
  + mistake_round_indices: [3] (R3이 실수 round)
        ↓
pre-write A/B 검증 — "이 skill이 있었으면 어땠을까?"를 실제로 실행해봄:

  ┌─ A (원래 결정) ──────────────────────────────────────────┐
  │ run log에서 R3 round의 원래 prompt를 그대로 가져옴         │
  │ (fetch_prompt_a: log에서 llm_call_seq로 원본 prompt 복원)  │
  │ → A 결정: "greedy attack, no block" → 18 HP loss          │
  └──────────────────────────────────────────────────────────┘
  ┌─ B (skill 주입 후 resample × 3) ─────────────────────────┐
  │ 같은 prompt에 candidate skill을 주입 (inject_candidate)   │
  │ → 3번 독립적으로 resample (B=3, parallel)                  │
  │ → B 결정 3개:                                             │
  │   Sample 1: "Weak → Defend → end_turn" (5 HP loss)         │
  │   Sample 2: "Weak → Defend+ → end_turn" (3 HP loss)        │
  │   Sample 3: "Defend → Defend → end_turn" (8 HP loss)       │
  └──────────────────────────────────────────────────────────┘
  ┌─ Judge (analysis tier 모델이 A vs B 비교) ───────────────┐
  │ A(원래 결정)와 B(skill 주입 결정 3개)를 비교               │
  │ → verdict: skill_helps / skill_unclear / skill_harmful     │
  │ → hit_count: B 중 expected_correction를 따른 개수 (0~3)    │
  └──────────────────────────────────────────────────────────┘
  ┌─ 통과 조건 (strict) ─────────────────────────────────────┐
  │ 1. zero-harmful: 어느 round도 skill_harmful이 아니어야 함  │
  │ 2. 2/3 이상 hit: sum(hit_count) >= ceil(total × 2/3)      │
  │ → 실패 시 skill 폐기 (library에 저장 안 됨)                │
  └──────────────────────────────────────────────────────────┘
        ↓
4-level write gate (통과한 candidate만):
  1. cosine 유사도 → 기존 skill과 의미 중복? (중복 시 merge)
  2. Jaccard → token 겹침? (중복 시 merge)
  3. LLM judge → 품질/유용성 판단
  4. optional reap → 추가 검증
  → 대부분 reject 또는 기존 skill에 merge (새 skill 추가 보수적)
        ↓
통과한 skill만 library에 저장 (skills.json)
        ↓
다음 run에서 trigger match 시 → L5 slot으로 주입
```

#### L5 추출 상세 (Mode B — stub-template authoring)
```
5개 빈 템플릿 (combat / boss / deckbuilding / map / intermission)
  → character-specific 내용으로 채움
  → namespace isolation, library lock, warn-only validator 하에 작성
  → Mode A(인간 작성 seed)와 동일한 6/10 fixed-A0 달성
  → "누가 썼는가(prose source)가 아니라 skill layer의 존재 자체가 효과"
```

### Routing and Combat Truncation

Dispatcher가 4개 model tier로 결정을 routing: fast(trivial combat), strategic(ordinary), analysis(postrun memory extraction), evolution(skill distillation). 4개 static system prompt는 cacheable, per-run state는 user message에 배치.

Combat만 local conversation object를 가지며, round당 최대 3 message (combat_start, ok, latest user state). 이전 round는 typed state로 요약, transcript에 append 안 함. 이를 통해 run당 median 67 strategic LLM call로 억제 (~500개 추가 결정은 fast tier 또는 mechanical handler로 처리).

상세 발췌 → [excerpt](../source/paper/AgenticSTS_A_Bounded-Memory_Testbed_for_Long-Horizon_LLM_Agents_2026_arxiv.md)

## Experiments & Results

### Benchmark / Testbed

- **Slay the Spire 2:** closed-rule stochastic deck-building roguelike. 4가지 testbed property:
  - P1: closed, enumerable, LLM-readable rule space (text record로 표현)
  - P2: empirically long horizon (median ~80 min, 67 strategic LLM call, ~500 additional decision)
  - P3: multi-axis stochasticity (card draw, shuffle, reward, map, relic, elite/event placement, Ascension modifier)
  - P4: state-conditioned combat math (hand + enemy intent + block timing + effect로 계산; web-like recall 불충분)
- **Ascension ladder:** A0(최저)–A10(최고). higher Ascension이 modifier를 stack하여 전략 변화.
- **외부 calibration:** AGI-Eval에서 5개 frontier LLM configuration 모두 A0 victory 0건. Mega Crit developer report: A0 human win rate 16% (240M community run 기준). → hard but not saturated

### Setup

- **Backbone:** Gemini 3.1 Pro (main), Qwen 3.6 27B, DeepSeek V4 Pro (cross-backbone probe)
- **Character:** Silent (단일 character, typed substrate self-consistency 유지)
- **Difficulty:** A0 (fixed ablation), A0–A8 (auto-mode ladder)
- **Metric:** win rate (Wilson 95% CI), derived analysis score (Eq. 1: 100 if victory, floor + (52/3)·bosses otherwise), 5,000-bootstrap 95% CI
- **Balanced subset:** fixed-A0 5 condition × 10 completed game = 50 game. 나머지 trajectory는 diagnostic stream

### Results

**Fixed-A0 ablation (Table 2):**

| Cell | L5 | L4 | Win | Score | Wilson 95% CI |
|------|-----|-----|------|-------|---------------|
| No scaffold | – | – | 3/10 | 70.4 | [10.8, 60.3] |
| Prompt only | – | – | 4/10 | 69.6 | [17, 69] |
| Hand skills (A) | A | – | 6/10 | 85.5 | [31.3, 83.2] |
| Template skills (B) | B | – | 6/10 | 83.3 | [31.3, 83.2] |
| Skills+episodes | A | ✓ | 6/10 | 82.1 | [31.3, 83.2] |

- 가장 큰 observed difference: no-scaffold(3/10) → skill-scaffolded(6/10), ∆_L5 = +2/10
- Fisher exact test 3/10 vs 6/10: **p ≈ 0.37** (directional, NOT statistically significant)
- Pooled scaffolded vs unscaffolded (18/30 vs 7/20): p ≈ 0.148 (역시 non-significant)
- L4 at A0 is saturated: Mode A (no L4) = full-frozen (with L4) = 6/10

**Auto-mode ascension ladder (Figure 5):**

| Stream | Max Attempted | Max Won |
|--------|---------------|---------|
| baseline-strict (frozen) | A2 | A1 |
| prompt-only (frozen) | A3 | A3 |
| mode-a (frozen) | A4 | A3 |
| Mode B (self-evolve, postrun-active) | A6 | A6 |
| full+postrun (Phase 3) | A8 | A7 |

postrun-active memory stream이 A6–A8 도달, no-postrun stream은 A2–A4 정지.

**Cross-backbone transfer (Table 3):**

| Backbone | Wins | Score | Δ% | Floor |
|----------|------|-------|-----|-------|
| Qwen 3.6-27B | 0/5→0/5 | 14.6→26.9 | +84.5% | 17→33 |
| DeepSeek V4-Pro | 0/5→0/5 | 41.3→33.8 | −18.1% | 37→33 |
| Gemini 3.1-Pro | 3/10→6/10 | 70.4→82.1 | +16.6% | 48→48† |

Gemini-trained stack이 Qwen에서는 score 향상, DeepSeek에서는 score 하락. transfer는 empirical property이지 premise가 아님.

**Competitor comparison (§7, Figure 6–8):**

| Agent | Win | Score | min/floor | Fresh tokens/score pt |
|-------|-----|-------|-----------|----------------------|
| Ours (full-frozen) | 6/10 | 82.1 | 2.3 | ~6.4k |
| Ours (baseline-strict) | 3/10 | 70.4 | 2.4 | ~6.7k |
| STS2MCP | 0/5 | 21.1 | 9.9 | ~422.3k |
| CharTyr | 0/5 | 5.6 | 8.5 | ~570.7k |

- Accumulating-context agent의 per-call prompt: ~9k → **500k tokens** (single run, decision ~1,100)
- Bounded contract: **~5k median, constant** (run 길이에 무관)
- Token 효율: 경쟁 agent가 **66–90× more fresh tokens/score pt**; raw ingested context >450×; absurd upper bound 시에도 ≥7× gap
- 주의: shipped system 비교이지 memory contract variable의 controlled ablation이 아님 (game patch, routing, thinking effort, decision batching, prompt cadence 등이 다름)

### Findings & Implications

1. **Memory interface를 evaluation 대상으로 만들 수 있음:** bounded typed contract는 fixed-A0 win을 support하고, within-harness difference를 L5 skill layer에 locate하며, fixed-difficulty performance를 ladder endpoint와 분리.
2. **Attribution tractability:** typed slot 분리로 gain을 "more context"가 아닌 특정 layer에 trace 가능.
3. **Interface design decoupled from accumulating state:** 동일 evaluation surface가 non-game agentic task로 portable.
4. **Loop engineering의 empirical complement:** bounded contract는 closed-rule, turn-based agent loop의 memory stage에 대한 concrete, measurable design point. 정성적 loop-engineering guidance에 대한 empirical 보완.
5. **Skill layer 존재 자체가 효과:** Mode B(template) = Mode A(human-authored) = 6/10 → prose source가 아닌 skill layer의 존재가 차이.

## Analysis

### Strengths & Significance

- **Memory as contract framing:** memory를 "저장·검색" 문제가 아닌 "각 결정이 무엇을 볼 수 있는가"의 contract로 재구성한 것이 가장 큰 지적 기여. accumulating-context와 bounded contract를 명확히 구분하고, 각각의 token growth pattern을 정량화.
- **Layer separability:** 5개 typed layer가 named slot으로 분리되어, prompt strictness, rule retrieval, episode, skill을 독립적으로 toggle 가능. 이는 raw prompt-history setup이 감추는 4가지 evaluation handle을 explicit하게 만듦.
- **적절한 난이도의 testbed 선택:** Slay the Spire 2는 closed-rule + text-readable이면서 long-horizon + multi-axis stochastic. AGI-Eval 0 wins / human 16%로 "hard but not saturated" 검증. pixel-rendered game(Crafter)이나 code-complex game(NetHack)의 한계를 회피.
- **통계적 정직성:** sample size 한계를 명시적으로 인정 (Fisher exact p≈0.37, Wilson CI overlap). "directional rather than statistically decisive"로 표현. 과도한 클레임을 회피.
- **재현 가능한 release:** 298 trajectory + condition tag + SHA-anchored snapshot + prompt record + analysis script. community가 동일 harness에서 alternative contract를 비교할 수 있는 infrastructure 제공.
- **Competitor 비교의 공정성:** leak audit, 동일 backbone, 동일 game line, 동일 machine, 동일 denominator rule을 명시. "shipped system 비교이지 controlled ablation이 아님"을 명확히 선언.

### Limitations

- **Sample size:** balanced 50-game subset. 최근 LLM-agent game benchmark(Voyager 3 trial, BALROG 10–25 seed) 수준이지만, finer-grained equivalence test에는 부족. 저자가 명시적으로 인정.
- **No matched accumulating-context variant:** same-codebase accumulating-context cell가 없어, bounded contract 자체가 accumulating-context보다 우수한지에 대한 controlled comparison이 missing. 이는 future work로 명시.
- **Single character (Silent):** typed substrate self-consistency를 위해 단일 character 사용. cross-character run은 L3/L4/L5 repopulation 필요.
- **Single game:** Slay the Spire 2에 특화. continuous/streaming control loop, visual input, multi-agent, online human correction, model-internal fine-tuning, cross-game transfer은 deliberate non-target.
- **Training-free:** 모델 fine-tuning 없이 prompted setup만. skill invention이 아닌 template filling (Mode B)으로 측정.
- **Author-curated template/seed:** stub template과 expert seed skill이 author-curated. fully autonomous skill invention과는 다름.
- **경쟁 agent 비교의 한계:** STS2MCP, CharTyr은 community project이지 tuned baseline이 아님. CharTyr의 패배가 partly interface error. game patch(0.103.1 vs 0.103.3) 차이 존재.

### Future Work / Improvements

- **Same-codebase accumulating-context cell 추가:** 동일 codebase, condition tag, scoring script, frozen store를 공유하는 accumulating-context variant를 one more row로 추가 → bounded contract vs accumulating-context의 controlled comparison.
- **더 큰 sample size:** finer-grained equivalence test, smooth backbone-transfer curve를 위한 추가 run.
- **Cross-character 확장:** L3/L4/L5를 새 character에 repopulate하여 동일 harness 사용.
- **Cross-game transfer:** bounded contract를 다른 closed-rule game으로 portable.
- **Autonomous skill invention:** Mode B template filling을 넘어 fully autonomous skill creation.
- **Continuous/streaming loop 적용:** turn-based가 아닌 continuous control loop에 bounded contract 적용 가능성 탐색.

## References

- Project page / Code: https://github.com/AlayaLab/AgenticSTS
- Data: https://huggingface.co/datasets/ShandaAI/AgenticSTS-trajectories
- arXiv: https://arxiv.org/abs/2607.02255
- AGI-Eval StS2 benchmark [1]: zero A0 victories across 5 frontier-model configurations
- Mega Crit A0 human win rate [23]: 16% across 240M community runs
- STS2MCP [11]: https://github.com/Gennadiyev/STS2MCP
- CharTyr [6]: https://github.com/CharTyr/STS2-Agent
- Voyager [36]: skill library pioneer
- MemGPT [25], Mem0 [7]: structured memory
- ReAct [47], Reflexion [28]: prompt-history agents
- Anthropic context engineering [2]: loop engineering qualitative guidance

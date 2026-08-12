# AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents — 핵심 발췌

> 출처: [분석 문서](../../report/[paper]_AgenticSTS_A_Bounded-Memory_Testbed_for_Long-Horizon_LLM_Agents_2026_arxiv.md) / 원본: [arXiv:2607.02255](https://arxiv.org/abs/2607.02255)

## Memory as a Contract (§1, §4)

> "Memory for a long-horizon LLM agent is a contract about what each future decision is allowed to see."

두 가지 contract:
- **Accumulating-context (prompt-history):** 과거 observation, tool call, reflection 을 모두 다음 prompt 에 append → 접근은 쉽지만 단일 memory component 의 효과를 분리하기 어려움
- **Bounded contract (본 논문):** 매 decision마다 typed retrieval 로 fresh user message 를 조립. raw cross-decision transcript 를 append 하지 않음 → prompt 가 run 길이에 무관하게 bounded, 각 layer 를 독립적으로 ablate 가능

## Per-Decision Compositional Context (§4.1, Eq. 2)

decision d at state s_d 에서 engine 은 L1–L5 에서 retrieve 하여 fresh user message 구성:

```
u_d = π(L1, L2(s_d), L3(s_d), L4(s_d), L5(s_d))
```

- sys + u_d 로 LLM 호출
- bounded typed summaries, per-run Strategic Thread, same-decision repair retries 는 허용
- unbounded transcript growth 는 disallow

**Bounded context 크기:** O(|sys| + s_thread + Σ_i k_i · s_i) — raw transcript 의 worst-case Ω(d · s̄) growth 와 대조적

## Five Typed Knowledge Layers (§4.2, Table 6)

| Layer | Store | Key | Write | Ablation |
|-------|-------|-----|-------|----------|
| **L1** | protocol | state | fixed | always |
| **L2** | schema | decision | fixed | always |
| **L3** | rules | entities | refresh | filter |
| **L4** | episodes | char/A/act | postrun | off/on |
| **L5** | skills | trigger | gated | off/A/B |

- **L1:** immutable role 및 protocol template (state type 별)
- **L2:** immutable schema — combat, deckbuilding, map, event, intermission 결정별 legal action format 포함
- **L3:** enumerable rule data — cards (576), relics (293), monsters (115), encounters (87), events (66). patch 단위 refresh
- **L4:** postrun summary (character × ascension × act × enemy class) — case-based recall
- **L5:** triggered strategic guide. trigger + prose policy + four-level write gate. SoK notation S=(C,π,T,R) 에서 (C,π) 에 해당

Raw StS2 logs 는 similarity RAG 로 사용하지 않음 — 비슷해 보이는 state 가 card order, relic combo, route history 에 따라 전략적 의미가 크게 다를 수 있음

## Routing and Combat Truncation (§4.3)

4개 model tier 로 dispatch:
- **fast:** trivial combat plan
- **strategic:** ordinary decision
- **analysis:** postrun memory extraction
- **evolution:** skill distillation

Combat 만 local conversation object 를 가지며, round 당 최대 3 message (combat_start, ok, latest user state). 이전 round 는 typed state 로 요약, transcript 에 append 안 함. 결과: run 당 median 67 strategic LLM call.

## Skill Discovery (§4.5)

### Mistake-driven discovery (self-evolve)
1. combat loss 를 per-enemy baseline 과 비교
2. pre-write A/B check (B=3 resample, strict 2/3 + zero-harmful)
3. four-level write gate 적용:

| Gate | 역할 |
|------|------|
| cosine | semantic similarity 중복 검사 |
| Jaccard | token overlap 중복 검사 |
| LLM judge | 품질·유용성 판단 |
| optional reap | 추가 검증 |

대부분의 candidate 는 reject 또는 merge 됨

### Stub-template-filled authoring (Mode B)
5개 character-parametric template (combat, boss, deckbuilding, map, intermission) 을 namespace isolation, library lock, warn-only validator 하에 채움. Mode B 가 human-authored Mode A seed library 와 동일한 6/10 fixed-A0 point estimate 달성 → skill layer 의 존재 자체가 효과이지, prose source 가 효과가 아님을 분리

## Scoring Formula (§3.2, Eq. 1)

```
s = 100                                     if victory
  = floor + (52/3) · bosses                 otherwise
```

- bosses: 0 (floor<18), 1 (floor<34), 2 (otherwise), 3 (victory)
- 52/3 coefficient: 3 cleared bosses = 52점 (대략 mid-Act-3 floor reach)
- ±10% perturbation 으로 score-based qualitative comparison 검증; win-rate claim 은 이 score scale 에 무관

## Fixed-A0 Ablation (§5.1, §6.2, Table 2)

5-cell decomposition (N=10/cell, frozen stores at SHA 1888a62):

| Cell | L5 | L4 | Win | Score |
|------|-----|-----|------|-------|
| No scaffold (baseline-strict) | – | – | 3/10 | 70.4 |
| Prompt only | – | – | 4/10 | 69.6 |
| Hand skills (Mode A) | A | – | 6/10 | 85.5 |
| Template skills (Mode B) | B | – | 6/10 | 83.3 |
| Skills+episodes (full-frozen) | A | ✓ | 6/10 | 82.1 |

- Wilson 95% CI: [10.8, 60.3] / [17, 69] / [31.3, 83.2] for 3/10, 4/10, 6/10
- Fisher exact test 3/10 vs 6/10: p ≈ 0.37 (directional, not significant)
- Pooled scaffolded vs unscaffolded (18/30 vs 7/20): p ≈ 0.148
- ∆_L5 = +2/10 (largest observed difference), ∆_prompt = +1/10
- L4 at A0 is saturated: Mode A (no L4) = full-frozen (with L4) = 6/10

## Cross-Backbone Transfer (§6.4, Table 3)

Gemini-trained L4+L5 stack 을 다른 backbone 에 transfer (N=5/cell, Gemini N=10):

| Backbone | Wins | Score | Δ% | Floor |
|----------|------|-------|-----|-------|
| Qwen 3.6-27B | 0/5→0/5 | 14.6→26.9 | +84.5% | 17→33 |
| DeepSeek V4-Pro | 0/5→0/5 | 41.3→33.8 | −18.1% | 37→33 |
| Gemini 3.1-Pro | 3/10→6/10 | 70.4→82.1 | +16.6% | 48→48† |

Transfer 는 empirical property 이지 premise 가 아님 — backbone 마다 다름

## Auto-Mode Ascension Ladder (§6.3, Figure 5)

| Stream | Max Attempted | Max Won |
|--------|---------------|---------|
| baseline-strict (frozen) | A2 | A1 |
| prompt-only (frozen) | A3 | A3 |
| mode-a (L5 seeds, frozen) | A4 | A3 |
| Mode B (self-evolve, postrun-active) | A6 | A6 |
| full+postrun (Phase 3) | A8 | A7 |

postrun-active memory → A6–A8 도달, no-postrun → A2–A4 정지

## Competitor Comparison (§7, Figure 6–8)

Accumulating-context agent (STS2MCP, CharTyr) 와 비교 (동일 backbone: gemini-3.1-pro-preview, A0, Silent):

| Agent | Win | Score | Wall-clock min/floor | Fresh tokens/score pt |
|-------|-----|-------|----------------------|----------------------|
| Ours (full-frozen) | 6/10 | 82.1 | 2.3 | ~6.4k |
| Ours (baseline-strict) | 3/10 | 70.4 | 2.4 | ~6.7k |
| STS2MCP | 0/5 | 21.1 | 9.9 | ~422.3k |
| CharTyr | 0/5 | 5.6 | 8.5 | ~570.7k |

- Accumulating agent 의 per-call prompt: ~9k → 500k tokens within single run (decision ~1100)
- Bounded contract: ~5k median, constant
- Token 효율: 66–90× more fresh tokens per score point for competitors; raw ingested context >450×; absurd upper bound pricing 시에도 ≥7× gap

## Statistical Protocol (§5.3)

- Win rate: Wilson 95% CI (Eq. 4)
- Score: 5,000-bootstrap 95% CI
- Pooled scaffolded row: exact Clopper-Pearson interval
- Balanced 50-game comparison (first 10 completed games per condition by start time)
- Stream separation: fixed-A0, backbone, ladder, archive — never pooled

## Slay the Spire 2 as Testbed — Four Properties (§3.1)

| Property | 설명 |
|----------|------|
| P1: Closed, enumerable, LLM-readable rule space | 576 cards, 293 relics, 115 monsters, 87 encounters, 66 events. pixel 이 아닌 text record 로 표현 가능 |
| P2: Empirically long horizon | median ~80 min, 67 LLM strategic calls, ~500 additional per-run decisions |
| P3: Multi-axis stochasticity | card draw, shuffle, rewards, map paths, relic effects, elite/event placement, Ascension modifiers |
| P4: State-conditioned combat math | hand contents + enemy intent + block timing + effects (vulnerable, weak, strength, dexterity) 로부터 계산 필요; web-like recall 보다 state-conditioned calculation 이 중요 |

## Release Contents (§3.3, §5.4)

298 completed trajectories:
- condition tags (fixed-A0, cross-backbone, ladder)
- SHA-anchored L4+L5 frozen snapshots
- decision-time prompt records
- Wilson/bootstrap analysis scripts
- per-trajectory: target/reached Ascension, outcome, wall-clock, LLM-call counts, condition tag, memory/scaffold setting

# Agent Workflow Memory -- 핵심 발췌

> 출처: [분석 문서](../../report/[paper][git]_Agent_Workflow_Memory_2024_arxiv.md) / 원본: [arXiv:2409.07429](https://arxiv.org/abs/2409.07429)

## Problem Statement

LM 기반 agent가 web navigation 과제를 해결할 때, 각 task를 독립적으로 처리하여 과거 성공·실패 경험으로부터 학습하지 못함. 기존 방법은 고정된 예시 세트를 in-context learning 또는 학습으로 통합하므로, task 맥락이나 환경 변화에 대한 robustness가 부족함.

인간은 과거 경험에서 재사용 가능한 task routine(workflow)을 추상화하여 future activity에 활용함(Chi et al., 1981). AWM은 이 메커니즘을 agent에 구현하는 것이 목표.

형식화: agent는 LM backbone L과 text memory M을 가짐. task q가 주어지면 observe-act loop를 반복: 상태 s_i에서 관측 o_i를 얻고, L(q, M, o_i) -> a_i로 action 생성. action 실행으로 T(s_i, a_i) -> s_{i+1}. STOP 또는 최대 step 도달 시 종료.

완료된 task는 experience e = (q, P)를 형성하며, trajectory P는 step들의 시퀀스. 각 step p = (o, a). 목표는 experience 집합 E로부터 workflow W = {w}를 유도하는 것: I(E) -> W. 유도된 workflow는 memory M에 추가되어 subsequent task-solving에 활용.

## Workflow Representation

Workflow는 두 component로 구성:
1. **Workflow description d**: workflow의 high-level goal을 요약한 NL 설명. experience instruction에서 heuristically 추출하거나 LM으로 요약.
2. **Workflow trajectory** (p_1, p_2, ...): d를 완료하기 위한 step 시퀀스. 각 step p는 세 부분:
   - (1) 현재 environment state의 NL 서술 (예: "Order {id} is shown")
   - (2) 관측 기반 action 결정 reasoning (예: "Order {id} is found, I will now terminate the task.")
   - (3) executable program 형태의 action (예: `stop()`)

## LM-based Workflow Induction

Induction module I는 past experience E = {e_i}를 입력으로 workflow 집합 W = {(d_j, P_j)}를 생성: I(E) -> W.

핵심 설계:
- task instruction("Buy dry cat food on Amazon...")과 달리, **finer granularity의 sub-task**("search for a product on Amazon")를 추출하도록 prompt
- example-specific value("dry cat food")를 **abstract variable**("{product-name}")로 대체하여 generality 향상
- double-line break 기준으로 workflow를 분리하여 개별 저장

Induction prompt (논문 Appendix A.1):
> Given a list of web navigation tasks, your task is to extract the common workflows.
> Each given task contains a natural language instruction, and a series of actions to solve the task.
> You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow.
> Each workflow should be a commonly reused sub-routine of the tasks. Do not generate similar or overlapping workflows.
> Each workflow should have at least two steps. Represent the non-fixed elements (input text, button strings) with descriptive variable names as shown in the example.

## Offline Scenario

추가 canonical experience(인간 annotation 또는 model synthesis)가 available한 경우:
1. training examples를 single prompt에 concatenation → LM으로 workflow 집합 W_offline 유도
2. 유도된 workflow를 모두 memory에 통합: M + W_offline -> M_w
3. test 시 동일한 W_offline으로 모든 test 해결: L(q, M + W_offline, o_i) -> a_i

## Online Scenario (supervision-free)

test query만으로 작동. test query를 streaming으로 처리:
1. t-th test instruction q_t에 대해 action trajectory 생성 → experience e_t = (q_t, {p_t})
2. LM evaluation model로 binary 판정: L_eval(e_t) ∈ {0, 1}
3. 성공(1)으로 판정되면 workflow로 변환: I(e_t) -> {w_t}
4. memory 업데이트: M + {w_t} -> M (t+1-th instruction에 사용)
5. 모든 test 처리까지 반복 (induce → integrate → utilize 루프)

## WebArena 실험

812 web navigation task, 5 website (e-commerce, social forum, CMS, GitLab, map). GPT-4 (gpt-4-0613), temperature 0.0. WebArena는 test-only이므로 online AWM만 적용.

| Method | Total SR | # Steps |
|--------|----------|---------|
| *SteP (human workflows) | 33.0 | - |
| BrowserGym_ax-tree | 20.2 | 31.9 |
| AWM (ours) | **35.5** | **29.1** |

- BrowserGym 대비 +12.0 absolute / +51.1% relative
- human workflow 기반 SteP 대비 +7.6% relative (추가 인적 감독 없이)
- 평균 2.0 step 감소 (효율적 trajectory)
- 0-40 example 구간에서 rapid learning, 이후 안정화 (snowball effect)

## Mind2Web 실험

Cross-task, cross-website, cross-domain generalization 평가. GPT-3.5-turbo 및 GPT-4 사용.

Cross-task (offline):
| Method | Elem Acc | Action F1 | Step SR | SR |
|--------|----------|-----------|---------|-----|
| MindAct_4 | 41.6 | 60.6 | 36.2 | 2.0 |
| AWM_4 (offline) | **50.6** | 57.3 | **45.1** | **4.8** |

- +24.6% relative step SR 향상
- Synapse(retrieved examples) 대비 +5.0 element accuracy → abstract workflow가 concrete example보다 element selection bias 감소

Cross-website / Cross-domain (online):
| Method | Cross-Website Step SR | Cross-Domain Step SR |
|--------|----------------------|---------------------|
| MindAct | 30.1 | 18.6 |
| AWM_online | **33.9** | **35.5** |

- train-test distribution gap이 widening될수록 AWM_online의 우위가 확대 (8.9 → 14.0 absolute points)
- AWM_online은 training data에 의존하지 않아 domain gap 영향 없음

## Ablation Studies

### Workflow format (§4.1-4.2)
- LM-based vs rule-based induction: WebArena에서 0.1 gap (거의 동등), Mind2Web에서 LM이 +2.8 우위
- Code format vs text format: 큰 차이 없음 (둘 다 유효)
- NL state description vs HTML: NL이 더 효과적, HTML 추가 시 오히려 성능 저하 (context length 증가 + irrelevant elements)

### Action space expansion (§5, AWM_AS)
- workflow를 high-level function으로 wrapping하여 action space에 추가
- Step SR +1.3 향상이지만 agent가 workflow action을 18.5% task에서만 호출
- dynamic environment change(예: popup airport 선택)에 취약

## Workflow Quality Metrics (Appendix A.3)

| Metric | WebArena | Mind2Web |
|--------|----------|----------|
| # Workflows | 7.4 | 7.3 |
| Coverage | - | 0.40 |
| Function Overlap | 0.08 | 0.20 |
| Utility Rate | 0.94 | 0.91 |

- website당 평균 7.3-7.4개 workflow로 효율적
- WebArena 94% test example이 workflow 활용
- function overlap 0.08-0.20으로 workflow 간 중복 최소화

## Example Workflows (Appendix A.2)

WebArena map example:
```
## map: Calculate Travel Time and Distance
To calculate travel time and distance between two locations, I will use the directions feature.
fill('158', 'FROM LOCATION')
fill('163', 'TO LOCATION')
select_option('166', 'MODE OF TRANSPORTATION')
click('171')
send_msg_to_user('The distance between FROM LOCATION and TO LOCATION is DISTANCE and the estimated travel time is TIME.')
```

Mind2Web travel example:
```
## enter_flight_locations
Given that you are on the flight booking page, this workflow enters the departure and destination city/airport.
[link] From Departure Airport or City Your Origin -> CLICK
[textbox] Origin City or Airport -> TYPE: {your-origin-city}
[link] {best-popup-option} -> CLICK
[link] To Destination Airport or City Your Destination -> CLICK
[textbox] Destination City or Airport -> TYPE: {your-destination-city}
[link] {best-popup-option} -> CLICK
```

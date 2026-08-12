# dreaming — Hermes 자기개선 루프 (background review fork)

> 출처: [분석 문서](../../../report/[git]_hermes-agent_NousResearch.md) / submodule: `source/git/hermes-agent_NousResearch`

## 설명

Hermes는 openclaw식 light/REM/deep 페이즈 + 점수 승격 모델이 없다. 대신 **"턴 종료 후 포크된 에이전트가 자신의 대화를 재생하며 무엇을 기억/학습할지 결정"**하는 직관적 자기개선 루프가 dreaming에 해당. `agent/background_review.py`.

### 트리거 — 턴 카운트 기반 케이던스(heartbeat)

`mem/agent_init.py`에서 기본 10. memory는 **사용자 턴 카운트**, skill은 **툴 반복(iteration) 카운트**로 별개 측정.

```python
# agent/agent_init.py
agent._memory_nudge_interval = 10   # config: memory.nudge_interval — 10 user turns 마다 memory review
agent._skill_nudge_interval  = 10   # config: skills.creation_nudge_interval — 10 iters 마다 skill review

# agent/turn_context.py — 매 턴 시작
agent._turns_since_memory += 1
if agent._memory_nudge_interval > 0 and "memory" in agent.valid_tool_names \
        and agent._memory_store:
    if agent._turns_since_memory >= agent._memory_nudge_interval:
        should_review_memory = True; agent._turns_since_memory = 0

# agent/turn_finalizer.py:454 — 턴 종료 시 skill 트리거(현 턴의 툴 iteration 수 기준)
if agent._skill_nudge_interval > 0 and agent._iters_since_skill >= agent._skill_nudge_interval \
        and "skill_manage" in agent.valid_tool_names:
    _should_review_skills = True; agent._iters_since_skill = 0
```

포크는 **응답 전달 이후**에 실행 → 사용자 작업과 모델 주의력 경쟁 안 함.

```python
# agent/turn_finalizer.py:472
if final_response and not interrupted and (_should_review_memory or _should_review_skills):
    agent._spawn_background_review(messages_snapshot=list(messages),
                                   review_memory=_should_review_memory,
                                   review_skills=_should_review_skills)
```

### 포크 구성 — 부모 런타임 상속 + cache warm + 강력한 격리

단일 daemon 스레드에서 forked `AIAgent`. 부모의 provider/model/creds/시스템 프롬프트를 **그대로** 물려받아 **같은 prefix cache** hit(≈26% 비용 절감, #25322). 다른 모델 라우팅 시 캐시가 차가워지므로 compact **digest** 재생.

```python
# agent/background_review.py — _run_review_in_thread
_rt = _resolve_review_runtime(agent)             # 기본: 부모 런타임(routed=False, warm cache)
_routed = bool(_rt.get("routed"))
review_agent = AIAgent(
    model=_rt.get("model") or agent.model, max_iterations=16, quiet_mode=True,
    provider=_rt.get("provider") or agent.provider, api_mode=_rt.get("api_mode"),
    base_url=_rt.get("base_url"), api_key=_rt.get("api_key"),
    credential_pool=getattr(agent, "_credential_pool", None),
    parent_session_id=agent.session_id,
    enabled_toolsets=getattr(agent, "enabled_toolsets", None),   # tools[] byte-동일(Anthropic 캐시 키 포함)
    skip_memory=True)                          # 외부 memory provider/harness 누출 차단
review_agent._memory_write_origin = "background_review"   # ← 프로바이넌스
review_agent._memory_write_context = "background_review"
review_agent._skip_mcp_refresh = True          # between-turn MCP 리프레시 → 캐시 키 깨짐 방지
review_agent._memory_store = agent._memory_store   # builtin MEMORY.md/USER.md는 공유(디스크 쓰기 유효)
review_agent._persist_disabled = True          # ★ 부모 session_id 공유 → state.db에 harness 턴 주입 차단
review_agent._session_db = None; review_agent._session_json_enabled = False
review_agent.suppress_status_output = True
review_agent._end_session_on_close = False     # 부모 세션 조기 종료 방지
review_agent.compression_enabled = False       # 압축 레이스/부모 회전 방지
if not _routed:
    review_agent._cached_system_prompt = agent._cached_system_prompt  # warm cache 핀
    review_agent.session_start = agent.session_start
review_agent.session_id = agent.session_id     # 캐시 warm 위해 공유(단, _persist_disabled로 쓰기 차단)
```

### 툴 화이트리스트 — memory/skill 만 허용

부모 toolset 스키마는 그대로(캐시 키) 두되, **dispatch 단계**에서 화이트리스트 통과한 툴만 실행. memory가 비활성 프로파일엔 memory 툴도 제거(#54937).

```python
review_toolsets = ["skills"]
if review_agent._memory_enabled or review_agent._user_profile_enabled:
    review_toolsets.insert(0, "memory")
review_whitelist = {t["function"]["name"] for t in get_tool_definitions(enabled_toolsets=review_toolsets, quiet_mode=True)}
set_thread_tool_whitelist(review_whitelist,
    deny_msg_fmt="Background review denied non-whitelisted tool: {tool_name}. Only memory/skill tools are allowed.")
# 위험 명령 자동 deny 콜백 설치(input() 데드락 방지, #15216)
```

### 리뷰 프롬프트 — memory vs skill 역할 분리

- **memory**: "사용자가 자신에 대해 드러낸 것(persona·선호·기대). 기억할 가치가 있으면 memory 툴로 저장."
- **skill**: "ACTIVE — 대부분 세션은 최소 1개 skill 업데이트를 만든다. 아무것도 안 하는 패스는 학습 기회 낭비." 우선순위: (1) 현재 로드된 skill 패치 → (2) 기존 umbrella 패치 → (3) support 파일(references/templates/scripts) 추가 → (4) 새 class-level umbrella 생성. **금지 캡처**: 환경 의존 실패, 도구 부정 주장("X tool broken"), 세션 일시적 에러, 일회성 작업.

```python
# _COMBINED_REVIEW_PROMPT 발췌
"**Memory**: who the user is ... **Skills**: how to do this class of task. Be ACTIVE ...
 User-preference embedding: when the user complains about how you handled a task, update the skill
 that governs that task — memory alone isn't enough.
 Do NOT capture: environment-dependent failures, negative claims about tools,
 session-specific transient errors, one-off task narratives."
```

### 라우팅 시 digest 재생

```python
def _digest_history(messages_snapshot, tail=24):
    # 최근 tail 메시지는 verbatim, 오래된 턴은 "USER: .../ASSISTANT[tools: ...]" 한 줄 요약 →
    # 단일 synthetic user-role digest. cold-write 토큰 최소화(다른 모델 라우팅 시에만)
```

### 출력 요약 + 사용자 통지

포크의 tool 결과에서 성공 action만 추출(`memory`/`skill_manage` notify_tools, prior_snapshot에 있는 건 stale 재노출 방지 #14944) → `💾 Self-improvement review: Memory ➕ ... · Skill 'X' patched: ...` 형태로 `_safe_print` + `background_review_callback`.

> 이 루프가 곧 "단기 기억(현 턴) → 장기 기억(memory.md)/절차적 기억(skill)으로의 통합(dreaming)". 단, 별도 점수 모델/게이트 없이 LLM이 매 회 전체 재생 후 판단. 백그라운드 병렬(daemon thread)이라 사용자 응답 지연 0.

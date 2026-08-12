# self_evolving — Hermes 스킬 자기진화 (curator + skill_manage)

> 출처: [분석 문서](../../../report/[git]_hermes-agent_NousResearch.md) / submodule: `source/git/hermes-agent_NousResearch`

## 설명

Hermes의 self-evolving은 두 메커니즘의 결합: **(A) background review**가 매 ~10턴마다 skill을 **생성/패치**(절차적 기억 축적), **(B) curator**가 주기적(기본 7일)으로 에이전트 생성 skill을 **통합/가지치기/보관**하여 라이브러리를 class-level umbrella 형태로 정돈. 둘 다 **agent-created provenance** 스킬에만 작용.

### A. skill_manage 툴 — 즉석 진화 API

`tools/skill_manager_tool.py`. 백그라운드 리뷰 포크가 `skill_manage(action=create/patch/edit/write_file/remove_file)`로 skill을 직접 수정. 신규 skill은 `~/.hermes/skills/`에 생성.

```python
# tools/skill_manager_tool.py — 외부 소유 skill 쓰기 거부(autonomous curator 보호)
def _refuse_autonomous_writes_to_externally_owned_skills(...):
    from tools.skill_provenance import is_background_review
    if is_background_review():   # 백그라운드 리뷰 포크만 curator 관리 대상 스킬 수정 가능
        ...

# create 시 agent-created 마킹 — 포그라운드 skill_manage(create)는 사용자 주도로 간주(마킹 안 함)
if background_self_improvement_review_fork:
    from tools.skill_usage import bump_patch, forget, mark_agent_created
```

### B. provenance — agent-created 식별 (ContextVar)

`tools/skill_provenance.py`. curator가 다룰 수 있는 스킬을 구분. 휴먼/번들/허브 스킬은 절대 건드리지 않음.

```python
# tools/skill_provenance.py — 전체 78라인
"""Skill write-origin provenance — agent-sediment skill writes vs foreground user-directed writes 구분.
 The curator only consolidates/prunes skills it autonomously created via the
 background self-improvement review fork."""
# ContextVar; "background_review" — the self-improvement review fork; only skills
# created through that path opt into curator management.
```

보호 대상(bundled/hub)과 `mark_agent_created`로 opt-in한 스킬만 curator 후보. 핵심 built-in(`plan` 등)은 영구 보호.

### C. curator — 비활성 트리거 + 상태 머신

`agent/curator.py`. **cron 데몬 없이 비활성 트리거**: 게이트웨이 하우스키핑 루프(`gateway/run.py:20189`, `CURATOR_EVERY`마다 폴링)와 CLI(`cli.py:13257`)에서 `maybe_run_curator` 호출. `interval_hours`(기본 7일) 경과 + `min_idle_hours`(기본 2h) 유휴 시 실행. 첫 실행은 1주일 연기(즉시 변이 방지).

```python
# agent/curator.py
DEFAULT_INTERVAL_HOURS = 24*7     # 7 days
DEFAULT_MIN_IDLE_HOURS = 2
DEFAULT_STALE_AFTER_DAYS = 30
DEFAULT_ARCHIVE_AFTER_DAYS = 90
DEFAULT_CONSOLIDATE = False        # ★ LLM umbrella-building 패스는 OFF 기본. 결정적 가지치기만 ON

def should_run_now(now=None):
    if not is_enabled() or is_paused(): return False
    last = _parse_iso(state.get("last_run_at"))
    if last is None:               # 신규 설치: last_run_at=now seed 후 1주일 연기
        state["last_run_at"] = now.isoformat(); save_state(state); return False
    return (now - last) >= timedelta(hours=get_interval_hours())

# gateway/run.py 하우스키핑 루프
if tick_count % CURATOR_EVERY == 0:
    maybe_run_curator(idle_for_seconds=float("inf"), on_summary=...)  # 폴링만, 실제 작업은 interval 마다
```

### D. 결정적 상태 전이 (LLM 없음, pure function)

`apply_automatic_transitions()`: 활동 타임스탬프 기반 active→stale→archived. **절대 삭제 안 함(보관만)**. pinned/cron 참조 스킬은 우회. use_count==0 신규 스킬은 stale_floor 전까지 보호(부재 증거≠증거).

```python
def apply_automatic_transitions(now=None):
    stale_cutoff   = now - timedelta(days=get_stale_after_days())     # 30d
    archive_cutoff = now - timedelta(days=get_archive_after_days())  # 90d
    cron_referenced = _cron_referenced_skills()
    for row in _u.agent_created_report():
        if row.get("pinned") or name in cron_referenced: continue
        anchor = last_activity or created_at or now
        never_used = int(row.get("use_count",0))==0
        if never_used and anchor > stale_cutoff:           # 신규 미사용 → 보호
            if current==STALE: set_state(ACTIVE); reactivate++; continue
        if anchor <= archive_cutoff and current != ARCHIVED: archive_skill(name); archived++
        elif anchor <= stale_cutoff and current == ACTIVE:  set_state(STALE); marked_stale++
        elif anchor > stale_cutoff and current == STALE:    set_state(ACTIVE); reactivated++
```

### E. LLM umbrella 통합 (opt-in, `curator.consolidate: true`)

별도 forked AIAgent가 `CURATOR_REVIEW_PROMPT`로 스킬 라이브러리를 **class-level umbrella**로 통합. "수백 개 좁은 세션별 스킬 = 라이브러리 실패. 하나의 넓은 umbrella + subsection/support 파일이 검색성에서 우위." 하드 룰: 외부/bundled/hub/pinned/`plan`/cron 참조 스킬 금지, **삭제 금지(보관만)**, usage count로 통합 스킵 금지(내용 기반 판단).

통합 3방식: (1) 기존 umbrella에 patch(섹션 추가) (2) support 파일 이동(`references/`/`templates/`/`scripts/`) (3) 새 umbrella 생성 후 형제 흡수. 구조화 YAML 출력(`consolidations:[{from,into}]`/`prunings:[{name}]`) → `cron/jobs.py:rewrite_skill_refs`가 cron 잡의 스킬 참조를 umbrella로 자동 재작성(통합 후 잡이 깨지지 않게).

```python
# agent/curator.py:603 — 통합 vs 가지치기 분류
def _classify_removed_skills(removed, ...):
    # 흡수된(umbrella에 content 이동) = "consolidated"(into: Y)
    # 단순 보관 = "pruned"
    return {"consolidated": [...], "pruned": [...]}
```

### F. 보관/복구 + 백업

보관 = `~/.hermes/skills/.archive/` 이동(복구 가능). `hermes curator restore`. `agent/curator_backup.py`가 실행 전 tar.gz 스냅샷(`curator.backup`). 상태는 `~/.hermes/skills/.curator_state`(last_run_at/paused/run_count).

### G. 학습 그래프 시각화 (부가)

`agent/learning_graph.py` + `learning_graph_render.py`: skill 노드 + 메모리 카드 + 관계 엣지를 그래프로 렌더(`metadata.hermes.related_skills`, 토큰 교집합 점수로 memory↔skill 엣지). `agent/curator.py`의 통합과 보조—중복/관계 가시화.

> self-evolving = "background review가 skill을 만들고 → curator가 그것들을 class-level umbrella로 정돈/가지치기"는 폐쇄 루프. 사용자 입력 없이 진화하되 **삭제 불가·보관만·provenance 기반 보호**로 안전망. 단, 점수 모델/게이트 없이 LLM이 매 실행 전체 라이브러리 재검토 후 판단.

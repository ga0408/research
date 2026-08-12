# bootstrap — Hermes 부트스트랩 메모리

> 출처: [분석 문서](../../../report/[git]_hermes-agent_NousResearch.md) / submodule: `source/git/hermes-agent_NousResearch`

## 설명

Hermes는 openclaw식 **최초실행 identity 시드 메모리**가 없다. builtin `MEMORY.md`/`USER.md`는 **빈 상태로 시작**하여 background review가 사용자가 자신을 드러낼 때마다 점진적으로 채운다. 즉 "bootstrap = 빈 슬레이트 + 자기개선 루프가 점진적 시드" 모델.

### 애플리케이션 부트스트랩 ≠ 메모리 부트스트랩

`hermes_bootstrap.py`(195라인)는 venv/Python 환경 구성(앱 부트스트랩)이지 메모리 시드가 아님. `agent/onboarding.py`는 설정 마법사·환영 메시지 처리.

### 메모리 초기화 — 빈 슬레이트

```python
# agent/agent_init.py:1327 — MemoryStore 생성. 시드 주입 없음.
agent._memory_store = None
...
from tools.memory_tool import MemoryStore
agent._memory_store = MemoryStore(memory_char_limit=..., user_char_limit=...)
agent._memory_store.load_from_disk()   # MEMORY.md/USER.md 읽기(없으면 빈 리스트)

# tools/memory_tool.py — 파일 없으면 빈 상태
def _read_file(path):
    if not path.exists(): return []
    raw = path.read_text(...)
    if not raw.strip(): return []
    return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
```

### 정적 identity = 시스템 프롬프트(시드 역할)

"에이전트가 누구인지"는 메모리가 아닌 **시스템 프롬프트의 stable tier**에 고정(`DEFAULT_AGENT_IDENTITY`, `HERMES_AGENT_HELP_GUIDANCE`, MEMORY/SESSION_SEARCH/SKILLS GUIDANCE). 이것이 세션 전체 byte-stable 캐시 prefix. 동적 identity(사용자에 대한 모델)는 `USER.md`가 점진적 축적.

```python
# agent/system_prompt.py — stable_parts(캐시됨, 세션 내 불변)
stable_parts.append(DEFAULT_AGENT_IDENTITY)        # ← 정적 identity 시드
stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)
tool_guidance = [MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE, ...]
# agent/system_prompt.py:460 — volatile_parts(세션 단위 갱신, 세션 내 frozen snapshot)
mem_block  = agent._memory_store.format_for_system_prompt("memory")  # 빈이면 None(생략)
user_block = agent._memory_store.format_for_system_prompt("user")
```

### 점진적 시드 = background review

첫 세션엔 USER.md가 비어 있지만, 10턴마다 background review가 `USER.md`에 사용자 persona/선호/기대를 add → 이후 세션부터 frozen snapshot에 포함되어 자동 주입. bootstrap 예산 보호개념은 없지만, 문자수 한도(2200/1375) 자체가 무한증식 방지 역할.

> 매핑: Hermes의 "bootstrap memory" = (정적 시스템 프롬프트 identity) + (빈 builtin + background review 점진적 축적). 별도 bootstrap 레이어/계층 구분 없이 모든 자기모델링이 동일 `USER.md`/`MEMORY.md` + external provider로 수렴.

> [git] https://github.com/NousResearch/hermes-agent.git

# Hermes Agent

> 분석 대상: `NousResearch/hermes-agent` (submodule HEAD 기준). Python 3.11, ~2,929개 `.py`(대형 god-file: `cli.py` 748KB·`gateway/run.py` 20933라인·`hermes_state.py` 6459라인·`run_agent.py` 6016라인). 자체 `AGENTS.md`가 71KB로 설계 원칙을 상세히 기술. "The self-improving AI agent" — 학습 루프 내장이 핵심 차별점.

## Overview

Hermes는 단일 **AIAgent 코어**(`run_agent.py` + `agent/conversation_loop.py`)를 4개 표면(CLI / 메시징 게이트웨이 / Ink TUI / Electron Desktop)에 재사용하는 개인 AI 에이전트. 20+ 메시징 채널(Telegram·Discord·Slack·WhatsApp·Signal·Matrix·iMessage·Teams…) 인바운드를 게이트웨이가 단일 에이전트 루프로 수렴시킨다. 6개 터미널 백엔드(local·Docker·SSH·Singularity·Modal·Daytona)로 어디서든 구동.

**모든 설계 결정을 지배하는 두 원칙**(자체 AGENTS.md):
1. **턴별 프롬프트 캐시가 신성** — 긴 대화는 매 턴 캐시된 prefix 재사용. 과거 컨텍스트 변경·도구셋 교체·시스템 프롬프트 재빌드는 캐시를 무효화해 비용을 배가. 유일 예외=컨텍스트 압축.
2. **코어는 좁은 허리, 능력은 가장자리** — 모든 모델 툴은 매 API 호출에 전송되므로 코어 툴 추가 기준은 매우 높음. 신기능은 CLI명령+스킬 → service-gated 툴 → 플러그인 → MCP 서버 → (최후) 코어 툴 순([Footprint Ladder]).

자기개선의 3축(readme): "Agent-curated memory with periodic nudges / Autonomous skill creation after complex tasks / Skills self-improve during use / FTS5 session search".

## Architecture — 전체 동작 flow

```
   ┌───────────────── 표면(SEND 방향은 생략, INBOUND 표시) ─────────────────┐
   │ CLI(cli.py HermesCLI)  TUI(Ink→tui_gateway JSON-RPC)  Desktop(Electron) │
   │ Messaging gateway(gateway/run.py) ← 20+ platforms/platforms/*.py         │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │ user message (+ /skill 전개)
                                      ▼
   agent/conversation_loop.py:523  run_conversation(user_message, history)
                                      │ turn_context.py: 세션 키/캐시/스킬 스냅샷,
                                      │   memory nudge 카운트(_turns_since_memory++),
                                      │   skill nudge 카운트(_iters_since_skill)
                                      ▼
   ┌───────────────  에이전트 루프 (동기, conversation_loop.py:638) ──────────┐
   │ while (api_call<max_iterations and iteration_budget.remaining>0)         │
   │       or _budget_grace_call:                                             │
   │   if _interrupt_requested: break              ← 사용자 중단/리다이렉트   │
   │   iteration_budget.consume() / _budget_grace_call                        │
   │   ── build_turn_context: system_prompt(3-tier) + tools[] + memory prefetch│
   │       • stable_parts(캐시): identity·도구가이드·skills_prompt·env hints   │
   │       • context_parts: system_message·AGENTS.md/CLAUDE.md (cwd)          │
   │       • volatile_parts: MEMORY/USER frozen snapshot·외부 provider block   │
   │         · date-only 타임스탬프(byte-stable)                              │
   │   ── client.chat.completions.create(model, messages, tools=tool_schemas) │
   │        ★ tool_schemas는 세션 중 불변(캐시)                               │
   │   if response.tool_calls:                                               │
   │       for tc: handle_function_call(tc.name, tc.args)  ← model_tools.py:1019│
   │           • plugin pre/post_tool_call 훅 / write_approval 게이트          │
   │           • terminal/browser/file/search/memory/skill_manage/...          │
   │           • memory 쓰기 → notify_memory_tool_write(외부 provider 미러)    │
   │         messages.append(tool_result)  ★ tool 결과는 messages_fts 자동 색인│
   │         api_call_count++; continue                                      │
   │   else: final_response = response.content; break                         │
   └────────────────────────────────────────────┬─────────────────────────────┘
                                                │
   turn_finalizer.py: 턴 종료 정리 + 트리거 발화
      ├── _sync_external_memory_for_turn (MemoryManager.sync_all 백그라운드 직렬)
      ├── cadence 체크: should_review_memory(턴 카운트≥10) / _should_review_skills(iter≥10)
      └── if (final_response and not interrupted) and (review_mem or review_skill):
            ╔══════════════ 백그라운드 daemon thread ══════════════╗
            ║ background_review.py _run_review_in_thread            ║  ← DREAMING
            ║  forked AIAgent(부모 runtime 상속, skip_memory=True,   ║
            ║    _persist_disabled=True, 화이트리스트: memory/skill) ║
            ║  부모 cached_system_prompt 핀 → 같은 prefix cache hit  ║
            ║  대화 snapshot 재생(동일모델=full, 라우팅=digest)       ║
            ║  → "무엇을 memory/skill에 저장·패치할까?" self-prompt   ║
            ║  → memory tool(MEMORY.md/USER.md 쓰기) + skill_manage   ║
            ║    (create/patch/write_file)  ← SELF-EVOLVING(즉석)    ║
            ║  💾 Self-improvement review: ... 요약 통지             ║
            ╚════════════════════════════════════════════════════════╝
                                                │
   ┌─────────────── 백그라운드/주기 프로세스(사용자 입력 없음) ───────────────┐
   │ ① Cron 스케줄러 (cron/scheduler.py 틱 루프)                              │
   │    • 스케줄: "30m"/"every 2h"/cron식/ISO one-shot  · 3분 하드 인터럽트   │
   │    • cron 세션: skip_memory=True(메모리 provider 미실행)                 │
   │ ② Curator (agent/curator.py) — 게이트웨이 하우스키핑 루프가 폴링          │
   │    • interval_hours(기본 7d) 경과 시 maybe_run_curator                    │
   │    • apply_automatic_transitions(결정적: active→stale→archived, 삭제X)   │
   │    • consolidate 옵션 시 LLM umbrella 통합 → SELF-EVOLVING(구조 정돈)    │
   │ ③ Memory sync 워커 (MemoryManager 단일 ThreadPoolExecutor max_workers=1) │
   │    턴 N 쓰기 → 턴 N+1 순서 보장(직렬)                                     │
   └──────────────────────────────────────────────────────────────────────────┘
```

**진입점**: CLI `hermes`(`hermes` 쉘 스크립트→`cli.py`) / 게이트웨이 `run.py` / TUI `hermes --tui` / `mcp_serve.py`. 코어 루프는 모두 `agent/conversation_loop.py:638`의 동기 `while`로 수렴. 크래시 영속은 `hermes_state.py` SessionDB(SQLite)가 담당.

## System Prompt 구성

`agent/system_prompt.py:assemble_system_prompt`가 **3개 정렬 티어**로 조립(캐시 안정성이 핵심):

| Tier | 내용 | 캐시 |
|---|---|---|
| **stable_parts** | `DEFAULT_AGENT_IDENTITY`(정적 자아), `HERMES_AGENT_HELP_GUIDANCE`, task-completion·parallel-tool-call 가이드, **MEMORY/SESSION_SEARCH/SKILLS/KANBAN tool guidance**, `skills_prompt`(활성 스킬), env hints, 모델별 실행 가이드 | 세션+대화 전체 byte-stable |
| **context_parts** | `system_message`(호출자 제공), **context files**(`AGENTS.md`/`CLAUDE.md`/`SOUL.md` in cwd) | cwd 변경 시만 |
| **volatile_parts** | **MEMORY/USER frozen snapshot**(`MemoryStore.format_for_system_prompt`), 외부 provider `system_prompt_block`, `Conversation started: <date-only>` | 세션 단위 갱신, **세션 내는 frozen** |

- **캐시 경계/결정론**: 타임스탬프를 **date-only**로(분 단위 정밀도면 매 리빌드마다 KV 무효화, PR #20451). 메모리는 frozen snapshot이라 세션 중 쓰기가 시스템 프롬프트를 바꾸지 않음.
- **툴 설명 vs API 스키마**: `get_tool_definitions()`(`model_tools.py:279`)가 도구 카탈로그 생성. 스키마 설명은 모델 토큰에 들어가므로 `description`은 간결하게, 크로스툴 참조는 동적 post-processing으로만(codex 백엔드 거부 방지).
- `/skill`은 **시스템 프롬프트가 아닌 user 메시지**로 주입(캐시 보존). 캐시를 깨는 슬래시명령은 기본 **지연 무효화**(다음 세션 적용), `--now` 플래그 옵션.

## Tools & Skills

**코어 툴**(`toolsets.py:_HERMES_CORE_TOOLS`): `terminal·read_file·web_search·browser_navigate·memory·session_search·skill_manage·delegate_task·clarify·todo` 등 — 매 API 호출마다 전송되는 최소 번들. 나머지는 27개 toolset(browser/file/search/vision/tts/kanban/...)로 그룹화, 플랫폼별 `tools.<platform>.enabled/disabled`로 제어.

| 핵심 툴 | 파일 | 역할 |
|---|---|---|
| `terminal` | `tools/terminal_tool.py` | 실제 셸 실행, 6 백엔드(local/docker/ssh/modal/daytona/singularity), 위험명령→`tools/approval.py` 승인 게이트 |
| `memory` | `tools/memory_tool.py` | builtin `MEMORY.md`/`USER.md` add/replace/remove/batch(→ [memory_extract](../source/git/snippets/hermes-agent_NousResearch__memory_extract.md)) |
| `session_search` | `tools/session_search_tool.py` | 과거 대화 FTS5 BM25 회상(3모드, → [memory_search](../source/git/snippets/hermes-agent_NousResearch__memory_search.md)) |
| `skill_manage` | `tools/skill_manager_tool.py` | 스킬 create/patch/edit/write_file/remove_file(자기진화 API) |
| `delegate_task` | `tools/delegate_tool.py` | subagent 분할(leaf/orchestrator, batch 병렬, `max_concurrent_children`=3) |
| `cronjob` | `tools/cronjob_tools.py` | 스케줄 잡 생성/편집 |

**스킬(SKILL.md)**: `skills/`(번들) + `optional-skills/`(설치형, autonomous-ai-agents/blockchain/devops/email/health/mlops/research/security…). frontmatter `metadata.hermes.*`로 태그/카테고리/관계/설정. **agentskills.io** 오픈표준 호환. `skills/config` 의존성은 setup 시 프롬프트·load 시 주입.

**sandbox 정책**: 게이트웨이 두 단계 가드(base adapter `_pending_messages` 큐 + runner가 `/stop`·`/new`·`/approve` 인터셉트) — 실행 중 명령은 양쪽 모두 bypass해야 inline dispatch. 위험 쓰기는 `tools/write_approval.py` 게이트(`MEMORY`/터미널/파일): allow/block/stage(승인 대기). background review 포크는 위험명령 **자동 deny**(input() 데드락 방지).

**훅 라이프사이클**: 일반 플러그인(`hermes_cli/plugins.py`)이 `pre/post_tool_call`·`pre/post_llm_call`·`on_session_start/end` 등록. `model_tools.py`가 툴 전후, `run_agent.py`가 세션 라이프사이클에서 호출.

## Cron / 스케줄링

`cron/jobs.py`(잡 스토어) + `cron/scheduler.py`(틱 루프, 3750라인). 잡 생성: `cronjob` 툴 / `hermes cron <verb>` / `/cron`.

- **스케줄 문법**: duration(`30m`/`2h`/`1d`) · "every"구(`every 2h`/`every monday 9am`) · 5필드 cron식(`0 9 * * *`) · ISO 원샷(`2026-06-01T09:00:00Z`).
- **잡 필드**: `skills`(사전 로드), `model`/`provider` 오버라이드, `script`(프롼프트에 stdout 주입, `no_agent=True`면 스크립트=잡 전체), `context_from`(잡A 출력→잡B 프롬프트 체인), `workdir`, 다중플랫폼 delivery.
- **틱 루프**: `CronScheduler` provider가 60초 폴링, catchup window(잡 주기의 절반, 120s–2h clamp), grace window(원샷 120s), `~/.hermes/cron/.tick.lock`로 다중프로세스 중복 방지.
- **isolated 실행**: 잡마다 독립 cron 세션. **3분 하드 인터럽트**(런어웨이 루프 방지). **`skip_memory=True` 기본**(cron 시스템 프롼프트가 사용자 표현을 오염). delivery는 게이트웨이 세션에 미러링 **안 함**(롤 교대 안정성).
- **에이전트 도구 인터페이스**: `cronjob` 툴로 생성/편집/일시정지/실행.

## Memory 관리 (특별 상세)

### 아키텍처

3계층 분리(저장 백엔드·테이블 스키마):

| 계층 | 백엔드 | 스키마 | 매니저 |
|---|---|---|---|
| **builtin curated** | 파일 `memories/MEMORY.md`·`USER.md` | `§`-구분 엔트리 리스트, char limit 2200/1375 | `tools/memory_tool.py:MemoryStore` |
| **external provider** | Honcho/Mem0/Hindsight/Supermemory/Byterover/Holographic/Openviking/RetainDB(**한 번에 1개**) | provider 자체 | `agent/memory_manager.py:MemoryManager` → `MemoryProvider` ABC(`agent/memory_provider.py`) |
| **session DB** | SQLite (`state.db`) | `sessions`·`messages`·`messages_fts`(FTS5)·`messages_fts_trigram`(CJK) | `hermes_state.py:SessionDB` |

`MemoryManager`(`agent/memory_manager.py:353`)가 builtin + 최대 1개 외부 provider를 오케스트레이션. 외부 provider 2개 등록 시 거부(툴 스키마 bloth/충돌 방지). 핵심 툴명(clarify/delegate_task 등) 그림자 금지. lifecycle: `initialize`→`system_prompt_block`→`prefetch`→`sync_turn`→`on_session_end`(추출)→`on_pre_compress`→`shutdown`. 단일 백그라운드 워커(`DaemonThreadPoolExecutor max_workers=1`)로 provider 쓰기 직렬화.

> **주의**: Hermes 코어 자체엔 **벡터 임베딩/KNN/BM25 하이브리드 퓨전이 없다**. 벡터 검색이 필요하면 external provider(mem0/honcho 등)가 자체 구현. builtin은 큐레이션 텍스트, 과거 회상은 SQLite FTS5.

### Memory 추출(쓰기) 로직  → [snippet](../source/git/snippets/hermes-agent_NousResearch__memory_extract.md)

```
 ┌── 경로 A: 사실 미러링 (builtin 쓰기를 외부에 통지) ──────────────────┐
 │  memory 툴(add/replace/remove/batch)                                  │
 │    → MemoryStore.* → MEMORY.md/USER.md (atomic temp+rename+.lock)    │
 │    → notify_memory_tool_write                                         │
 │        ① 성공+커밋(staged❌)만  ② add/replace/remove만  ③ 프로바이넌스 │
 │    → on_memory_write ── builtin SKIP ── 외부 provider에 통지           │
 │      (provider 자율 구현, default no-op · 단방향 builtin→external만)  │
 └──────────────────────────────────────────────────────────────────────┘

 ┌── 경로 B: 대화 인제스트 (매 턴, MEMORY.md와 무관) ──────────────────┐
 │  턴 종료 → MemoryManager.sync_all (백그라운드 직렬, 단일 워커)        │
 │    → provider.sync_turn(user, assistant, messages=전체대화)           │
 │    → provider가 자체 임베딩/저장                                       │
 └──────────────────────────────────────────────────────────────────────┘
            두 경로는 저장 대상이 달라 병렬·보완 관계 (동기화 아님)
```

| 쓰기 경로 | 대상 | 트리거 | 방식 | 안전장치 |
|---|---|---|---|---|
| **builtin** | `MEMORY.md`/`USER.md` | 에이전트 `memory` 툴 호출 | `add`/`replace`/`remove`(부분문자열 매칭)·`apply_batch`(최종 예산 원자적 all-or-nothing) | 위협패턴 strict 스캔·외부 drift 감지·3회 초과 시 terminal skip |
| **경로 A 미러링** | 외부 provider 저장소 | builtin 쓰기 성공 후 | `notify_memory_tool_write`→`on_memory_write` 통지 | 성공+커밋+변이액션만·provider 자율(default no-op)·단방향 |
| **경로 B 대화 인제스트** | 외부 provider 저장소 | 매 턴 종료 | `sync_all`→`sync_turn`(user+assistant+messages) | 백그라운드 직렬(턴 순서 보장)·/skill 전개 제거 |
| **세션 종료/압축** | 외부 provider | `/reset`·게이트웨이 만료·압축 직전 | `on_session_end`·`on_pre_compress`(LLM 요약 추출) | end→switch 단일 직렬 태스크(#16454) |

쓰기 = atomic temp+rename(`os.replace`) + 별도 `.lock` 파일(fcntl/msvcrt). 턴당 3회 초과 consolidation 실패 시 terminal "save skipped" → 사이드이펙트가 응답 블록 방지(#42405). 프로바이넌스: `write_origin`(`assistant_tool`/`background_review`)·`execution_context`·`session_id`.

### Memory 검색 로직  → [snippet](../source/git/snippets/hermes-agent_NousResearch__memory_search.md)

| 계층 | 검색 방식 | 트리거 | 특징 |
|---|---|---|---|
| **builtin** | **검색 없음** — frozen snapshot 자동 주입 | 매 턴 시스템 프롬프트 조립 시 | 세션 시작 시 고정, 세션 중 불변(캐시 안정) |
| **session_search** | SQLite FTS5 BM25 | 모델이 `query`로 툴 호출(온디맨드) | `ORDER BY rank`·`snippet()` 하이라이트·CJK 3자+ trigram·cron 세션 강등·lineage 중복제거·±5 메시지 윈도우 |
| **외부 provider** | provider 자체(임베딩/벡터 등) | `prefetch()` 매 API 호출 전 | `<memory-context>` 펜스 + 시스템 노트로 주입·스트리밍 scrubber가 청크 경계 누출 방지 |

### Bootstrap 메모리  → [snippet](../source/git/snippets/hermes-agent_NousResearch__bootstrap.md)

| 항목 | 내용 |
|---|---|
| identity 시드 | **없음** — 빈 슬레이트 시작 |
| 정적 identity | 시스템 프롬프트 stable tier(`DEFAULT_AGENT_IDENTITY`) |
| 동적 identity | `USER.md`가 background review로 점진적 축적(10턴마다) |
| `hermes_bootstrap.py` | venv 환경 구성(앱 부트스트랩)이지 메모리 시드가 아님 |

### Dreaming / 통합  → [snippet](../source/git/snippets/hermes-agent_NousResearch__dreaming.md)

```
 매 ~10턴 종료 후 (memory: 사용자 턴 카운트 / skill: 툴 iteration 카운트)
       │ final_response && !interrupted
       ▼
 daemon thread에서 forked AIAgent 생성
  · 부모 runtime(model/provider/creds) 상속 → 같은 prefix cache hit (≈26% 비용절감)
  · 부모 cached_system_prompt 핀 (동일 모델 시만; 라우팅 시 자체 빌드)
       │
       ▼
 강력 격리 설정
  · skip_memory=True      (외부 provider 누출 차단)
  · _persist_disabled=True (state.db에 harness 턴 주입 차단)
  · compression_enabled=False (압축 레이스/부모 회전 방지)
  · _end_session_on_close=False (부모 세션 조기 종료 방지)
  · 툴 화이트리스트: memory + skills 만 (나머지 runtime deny)
  · 위험명령 자동 deny (input() 데드락 방지)
       │
       ▼
 대화 snapshot 재생
  · 동일 모델 → full replay (warm cache) / 라우팅 → compact digest (cold cache)
  → "무엇을 memory/skill에 저장·패치할까?" 자기 질문
       │
       ▼
 memory 툴 → MEMORY.md/USER.md 쓰기  +  skill_manage → 스킬 create/patch/write_file
  → 💾 Self-improvement review: ... 요약 통지
```

| 단계 | 내용 |
|---|---|
| **트리거** | memory: `_turns_since_memory >= 10` / skill: `_iters_since_skill >= 10` |
| **포크** | 부모 런타임+캐시 상속, 응답 전달 후 실행(지연 0) |
| **격리** | skip_memory·_persist_disabled·compression=off·화이트리스트·auto-deny |
| **판단** | LLM이 재생 후 memory(사용자 모델링) vs skill(절차적 지식) 역할 분리 |
| **출력** | tool 쓰기 + 사용자 요약 통지 |

> openclaw식 light/REM/deep 페이즈 + 가중치 점수 승격 모델은 **없음**. LLM이 매 회 전체 재생 후 판단하는 직관적 루프.

### 인출 주입

| 계층 | 주입 방식 | 시점 |
|---|---|---|
| **builtin** | 시스템 프롬프트 volatile tier에 frozen snapshot **자동 주입** | 매 턴(캐시 안정) |
| **외부 provider** | `prefetch()` 결과 → `<memory-context>` 펜스 + 시스템 노트 | 매 API 호출 전 |
| **session_search** | 도구 기반 온디맨드(모델이 `query` 호출) → `>>>`/`<<<` 하이라이트 스니펫 | 모델 판단 시 |

## Self-Evolving (스킬 자기진화)  → [snippet](../source/git/snippets/hermes-agent_NousResearch__self_evolving.md)

```
 (A) 즉석 진화 — background review 포크 (~10턴마다)
     skill_manage(create/patch/edit/write_file/remove_file)
     → 신규 스킬 생성 · 기존 패치 · support 파일 추가 · 새 umbrella 생성
     provenance: agent-created 마킹 (ContextVar로 bundled/hub 스킬 보호)
                        │
                        │ 누적
                        ▼
 (B) 구조 정돈 — curator (기본 7일마다, 비활성 트리거)
     ① 결정적 전이 (LLM 없음): 30d 미사용→stale · 90d→archived (삭제❌, 보관만)
     ② LLM umbrella 통합 (opt-in): 겹치는 스킬→umbrella 흡수 · cron 잡 참조 자동 재작성
     ③ 백업: tar.gz 스냅샷 (hermes curator restore로 복구)
```

| 단계 | 주체 | 액션 | 안전장치 |
|---|---|---|---|
| **(A) 즉석 진화** | background review 포크 | `skill_manage`(create/patch/edit/write_file/remove_file) → 패치 우선순위: 로드된 스킬→기존 umbrella→support 파일→새 umbrella | **금지 캡처**: 환경 의존 실패·도구 부정 주장·일시적 에러·일회성 작업 (자기제약 경화 방지) |
| **(B)-① 결정적 전이** | curator(LLM 없음) | `apply_automatic_transitions`: 30d→stale·90d→archived | **삭제 금지**(보관만, 복구 가능)·pinned/cron 참조 스킬 우회·use=0 신규 스킬은 stale_floor 전까지 보호 |
| **(B)-② LLM 통합** | curator forked AIAgent(opt-in) | umbrella 흡수 3방식: 기존 patch / support 파일 이동 / 새 umbrella 생성. YAML(`consolidations`/`prunings`) 출력 | 기본 OFF(비용)·`rewrite_skill_refs`로 cron 잡 참조 자동 재작성 |
| **provenance 보호** | `skill_provenance.py` | **agent-created** 스킬만 curator 대상 | bundled·hub·external 스킬 절대 건드림❌·핵심 built-in(`plan`) 영구 보호 |
| **백업/복구** | `curator_backup.py` | 실행 전 tar.gz 스냅샷·`hermes curator restore` | 상태 `~/.hermes/skills/.curator_state` |

> 점수 모델/게이트 없이 LLM이 매 실행 전체 라이브러리 재검토 후 판단. 통합은 기본 OFF, 결정적 가지치기만 ON.

## Autonomy & Safety (보조 항목)

- **목표/계획 자기생성**: 사용자 입력 없는 자율 행동은 cron 잡(스케줄된 자연어 작업) + background review/curator(자기개선). 별도 heartbeat·goal 도구·planner는 없음 — 자율성은 "학습 루프"에 집중, "자율 목표 수행"은 cron에 위임.
- **루프 종료/제어**: `max_iterations`(기본 90) + `iteration_budget`(`agent/iteration_budget.py`) + `_budget_grace_call`(1회 추가 기회). 도구 루프는 interrupt 체크 매 iteration. cron은 **3분 하드 인터럽트**. background review `max_iterations=16`.
- **오류 복구/자가치유**: 400/인코딩/압축 실패 시 retry 경로(`conversation_loop.py`·`conversation_compression.py`). rate-limit breaker(확인빈 계정 버킷). FTS5 손상 시 `'rebuild'`→drop→VACUUM 자가복구(`hermes_state.py`). memory 외부 drift 감지 시 쓰기 거부+백업.
- **인간개입/승인 게이트**: `tools/write_approval.py`(MEMORY/터미널/파일 allow/block/stage) + `tools/approval.py`(위험명령). `/approve`·`/deny`. background review는 위험명령 자동 deny.
- **컨텍스트/토큰 관리**: `agent/conversation_compression.py` — 컨텍스트 한계 도달 시 **유일한 캐시 무효화 허용 경로**. 회전(부모→자식 세션). 압축 전 provider `on_pre_compress`로 통찰 추출 보존. `trajectory_compressor.py`(68KB)는 학습용 트라젝토리 압축.
- **상태 영속/복원**: `hermes_state.py` SessionDB(SQLite)가 `sessions`/`messages` 영속. 메시지는 FTS5 트리거로 자동 색인. checkpoint 매 iteration 스냅샷(dedup). `/resume`·`/branch`·`/reset`·`/new`·압축 모두 세션 회전(provider `on_session_switch`로 per-session 캐시 갱신). 크래시 후 세션 복구. profile별 `HERMES_HOME` 격리(각각 독립 config/메모리/스킬/게이트웨이).

## Analysis

**장점**
- **캐시 불변성 설계가 일관적** — frozen snapshot 메모리·byte-stable 시스템 프롬프트·date-only 타임스탬프·/skill user-주입·세션 중 도구셋 불변. 장기 대화 비용 최소화(foreground review fork가 같은 캐시 hit).
- **자기개선 루프가 폐쇄적** — background review(생성/패치) → curator(통합/가지치기)가 사용자 개입 없이 스킬 라이브러리를 진화시키되, provenance 보호·삭제금지(보관만)·점진적 시드로 안전망 구축.
- **코어 좁은 허리 원칙** — 모든 툴이 매 호출 전송됨을 자각하고 Footprint Ladder로 능력을 가장자리(스킬/플러그인/MCP)로 밀어냄. 메모리도 builtin+1외부 provider로 스키마 bloth 방지.
- **견고한 운영** — 6 터미널 백엔드·20+ 채널·profile 격리·cron 3분 하드한계·SQLite 자가복구·위협패턴 strict 스캔. 실사용 이슈(#번호)가 코드 주석에 광범위 문서화.

**단점/한계**
- **god-file 부채** — `cli.py`(748KB)·`gateway/run.py`(20933라인)·`run_agent.py`·`hermes_state.py`가 거대. 자체 AGENTS.md도 리팩토링을 장려하나 진입 장벽.
- **dreaming이 단순** — 점수 모델/페이즈/게이트 없이 LLM 전체 재생 매 판단. openclaw식 정교한 승격/그림자 트라이얼 대비 비용-품질 tradeoff가 LLM 성능에 직접 의존. 통합은 기본 OFF(비용 우려).
- **builtin 메모리 정성적** — char limit 2200/1375의 큐레이션 텍스트. 벡터 검색·대규모 회상은 external provider에 의존(설치 필요). 자체 회상은 FTS5 BM25(의미 검색 아님).
- **단일 외부 provider 제한** — 한 번에 1개만. 멀티 백엔드 결합 불가(설계적 트레이드오프).

**적용 가능성**: 개인 비서·메시징 연동 에이전트에 최적. 캐시 안정 설계와 background review 패턴은 장기 대화 에이전트 일반에 차용 가치 있음. 자율 목표 수행(heartbeat planner)보다는 "사용자 주도 + 학습 누적" 모델.

## References
- repo: https://github.com/NousResearch/hermes-agent
- docs: https://hermes-agent.nousresearch.com/docs/
- 자체 `AGENTS.md`(71KB, 설계 원칙·Footprint Ladder·스킨/플러그인/스킬 표준 상세)
- Honcho(dialectic user modeling): https://github.com/plastic-labs/honcho
- agentskills.io 오픈 스킬 표준
- 관련 주제: `topics/agent_framework.md`, `topics/agent_memory_framework.md`(추가 시)

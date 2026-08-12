# Bootstrap Memory (최초 실행 identity 시드)

> 출처: [분석 문서](../../../report/[git]_openclaw_openclaw.md) / submodule: `source/git/openclaw_openclaw`

## 설명
"Bootstrap memory"는 memory-core 기능이 아니라 **core agent-workspace의 최초실행 identity 의식**(`src/agents/`). 새 워크스페이스에 `AGENTS/SOUL/TOOLS/IDENTITY/USER/HEARTBEAT/BOOTSTRAP.md`를 시드하고 자유형 대화로 이름/성격/바이브/이모지를 정해 `IDENTITY.md/USER.md/SOUL.md`에 기록한 뒤 `BOOTSTRAP.md`는 삭제해 1회만 실행. memory-core는 인출 가이드 주입 + dreaming 승격이 bootstrap 주입 cap을 넘지 않게 `MEMORY.md`를 예산 컴팩션하는 역할만 담당.

## 코드

### 부트스트랩 핸드오프 프롬프트 — `src/agents/bootstrap-prompt.ts`
```ts
/** 전체 BOOTSTRAP.md 워크플로우 핸드오프 (정상 케이스) */
export function buildFullBootstrapPromptLines(params: {
  readLine: string;
  firstReplyLine: string;
}): string[] {
  return [
    params.readLine,
    "If this run can complete the BOOTSTRAP.md workflow, do so.",
    "If it cannot, explain the blocker briefly, continue with any bootstrap steps that are still possible here, and offer the simplest next step.",
    "Do not pretend bootstrap is complete when it is not.",
    "Do not use a generic first greeting or reply normally until after you have handled BOOTSTRAP.md.",
    params.firstReplyLine,
  ];
}

/** 제약 핸드오프 (워크스페이스 접근 불가 등) */
export function buildLimitedBootstrapPromptLines(params: { introLine; nextStepLine }): string[] {
  return [
    params.introLine,
    "Do not claim bootstrap is complete, and do not use a generic first greeting.",
    "Briefly explain the limitation, continue only with any bootstrap steps that are still safely possible here, and offer the simplest next step.",
    params.nextStepLine,
  ];
}
// 주의: BOOTSTRAP.md는 시스템 프롬프트가 아니라 사용자 프롬프트로 주입 → read 도구를 안 쓰는 모델도 의식 수행 가능
// workspace.ts: DEFAULT_BOOTSTRAP_FILENAME="BOOTSTRAP.md"; ensureAgentWorkspace가 시드, filterCompletedWorkspaceBootstrapFile이 setup 완료시 BOOTSTRAP.md 제거
```

### MEMORY.md 예산 컴팩션 — `extensions/memory-core/src/memory-budget.ts`
```ts
// 문제: dreaming 승격이 MEMORY.md를 무한히 키워 bootstrap 주입 cap(~12KB)을 넘으면 세션 부트스트랩이 끊김(#73691)
// 전략: 가장 오래된 자동 승격 섹션(날짜순)을 예산 안에 들어올 때까지 삭제. 사용자 작성 콘텐츠는 무조건 보존.
const PROMOTION_SECTION_HEADING_RE = /^## Promoted From Short-Term Memory \(([^)]+)\)\s*$/;
export const DEFAULT_MEMORY_FILE_MAX_CHARS = 10_000; // bootstrap 주입 cap 미만 유지
const WRITE_OVERHEAD_RESERVE = 21; // 헤더+트레일링 뉴라인 오버헤드

export function compactMemoryForBudget({ existingMemory, newSection, budgetChars }) {
  const effectiveBudget = Math.max(0, budgetChars - WRITE_OVERHEAD_RESERVE);
  if (existingMemory.length + newSection.length <= effectiveBudget) return { compacted: existingMemory, droppedDates: [] };
  // 메모리를 블록(보존/승격)으로 파싱 → 승격 블록을 날짜 오름차순 정렬 → 예산 들어올 때까지 오래된 것부터 드랍
  const promotionEntries = blocks.filter(승격).toSorted((a,b) => a.date.localeCompare(b.date));
  for (const entry of promotionEntries) {
    if (projectedExistingSize + newSection.length <= effectiveBudget) break;
    droppedIndices.add(entry.index); droppedDates.push(entry.date);
    projectedExistingSize -= entry.length - blockSeparatorCost;
  }
  return { compacted: joinBlocks(remaining), droppedDates };
}
```

### 메모리 계층(bootstrap vs episodic vs MEMORY.md)
```
- BOOTSTRAP.md  : 1회성 identity 시드 → 완료후 삭제. 산출물은 IDENTITY/SOUL/USER.md(템플릿/성격)
- memory/YYYY-MM-DD.md : 작업/일화 계층(일일 노트). 검색용 인덱싱, 매 턴 주입 X
- MEMORY.md     : 증류된 장기 팩층. 메인 세션 시작시 주입(그룹/공유 세션 제외). dreaming deep가 여기에 승격
- memory-core prompt-section.ts: memory_search/memory_get 가용시 인출 가이드("이전 작업 언급 전 memory_search 먼저")를 시스템 프롬프트에 주입. BOOTSTRAP.md 자체는 주입 안함
- compaction flush(flush-plan.ts): 압축전 모델 턴이 memory/YYYY-MM-DD.md에만 쓰게 강제, MEMORY.md/DREAMS.md/SOUL.md/TOOLS.md/AGENTS.md는 읽기전용 취급
```

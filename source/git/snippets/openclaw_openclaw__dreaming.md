# Dreaming (단기 → 장기 메모리 승격)

> 출처: [분석 문서](../../../report/[git]_openclaw_openclaw.md) / submodule: `source/git/openclaw_openclaw`

## 설명
`extensions/memory-core`의 오프라인 메모리 통합 시스템. cron 기반(기본 `0 3 * * *`)으로 light → REM → deep 3페이즈 스윕을 실행. light/REM은 일기(`DREAMS.md` 관리 블록)만 작성하고, deep만 6개 가중 신호로 단기 후보를 평가해 `MEMORY.md`에 승격(append). 사용자가 요청한 "dreaming 과정"의 핵심.

## 코드

### 승격 가중치 — `extensions/memory-core/src/short-term-promotion.ts:103`
```ts
const DEFAULT_PROMOTION_WEIGHTS: PromotionWeights = {
  frequency: 0.24,     // 로그 빈도: log1p(signalCount)/log1p(10)
  relevance: 0.30,     // 평균 검색 점수: totalScore/signalCount
  diversity: 0.15,     // max(uniqueQueries, recallDays)/5
  recency: 0.15,       // 시간 감쇠(halfLifeDays)
  consolidation: 0.10, // max(consolidationComponent, groundedCount/3)
  conceptual: 0.06,    // conceptTags.length/6 (concept-vocabulary.ts)
};
```

### 점수 산정 + 3단계 게이트 — `short-term-promotion.ts:1855`
```ts
// 게이트 1: signalCount >= minRecallCount
if (signalCount < minRecallCount) continue;
const avgScore = clampScore(entry.totalScore / Math.max(1, signalCount));
const frequency = clampScore(Math.log1p(signalCount) / Math.log1p(10));
const uniqueQueries = entry.queryHashes?.length ?? 0;
const contextDiversity = Math.max(uniqueQueries, entry.recallDays?.length ?? 0);
// 게이트 2: contextDiversity >= minUniqueQueries
if (contextDiversity < minUniqueQueries) continue;
const diversity = clampScore(contextDiversity / 5);
const ageDays = ...;
if (maxAgeDays >= 0 && ageDays > maxAgeDays) continue;
const recency = clampScore(calculateRecencyComponent(ageDays, halfLifeDays));
const consolidation = Math.max(calculateConsolidationComponent(recallDays), clampScore(groundedCount / 3));
const conceptual = calculateConceptualComponent(conceptTags);
// 페이즈 강화 부스트(light/REM 신호, 최대 0.06)
const phaseBoost = calculatePhaseSignalBoost(phaseSignals.entries[entry.key], nowMs);
const score =
  weights.frequency * frequency +
  weights.relevance * avgScore +
  weights.diversity * diversity +
  weights.recency * recency +
  weights.consolidation * consolidation +
  weights.conceptual * conceptual +
  phaseBoost;
// 게이트 3: score >= minScore
if (score < minScore) continue;
candidates.push({ ..., score: clampScore(score), components: {...} });
```

### 스윕 페이즈 조율 — `extensions/memory-core/src/dreaming-phases.ts:1889`
```ts
export async function runDreamingSweepPhases(params) {
  // 1) Light: 최근 memory/YYYY-MM-DD.md + 세션 트랜스크립트를 단기 recall 저장소에 적재(스니펫 280자, 점수 0.62/0.58). DREAMS.md "## Light Sleep" 블록 + 일기.
  if (light.enabled && light.limit > 0) await runLightDreaming(...);
  // 2) REM: light가 스테이징한 항목 선호, 컨셉 태그로 패턴/반복 아이디어 반영. "## REM Sleep" 블록 + 일기. (MEMORY.md 안 씀)
  if (rem.enabled && rem.limit > 0) await runRemDreaming(...);
}
// 3) Deep는 dreaming.ts에서: rankShortTermPromotionCandidates(위 점수화) → applyShortTermPromotions(MEMORY.md에 "## Promoted From Short-Term Memory (DATE)" 섹션 append)
//    승격시 compactMemoryForBudget으로 MEMORY.md ≤ 10KB 유지(bootstrap 주입 cap 보호), 프로모션 마커로 중복 승격 방지, promotedAt 스탬프
```

### 트리거/스케줄 — `extensions/memory-core/src/dreaming.ts`
```
registerShortTermPromotionDreaming → gateway_start 훅 → reconcileShortTermDreamingCronJob
  관리 cron 1개 생성: sessionTarget="isolated", wakeMode="now", payload agentTurn + DREAMING_SYSTEM_EVENT_TEXT 토큰, delivery.mode="none"
  기본 0 3 * * * (매일 03:00). 레거시 light/REM per-phase cron은 통합 컨트롤러로 마이그레이션/삭제.
실행: before_agent_reply 훅에서 trigger in {heartbeat,cron} + 시스템 이벤트 토큰 매치 → runShortTermDreamingPromotionIfTriggered
  → 워크스페이스별 runDreamingSweepPhases(light→REM) → repairShortTermPromotionArtifacts → rankShortTermPromotionCandidates(deep 점수화) → applyShortTermPromotions → writeDeepDreamingReport → generateAndAppendDreamNarrative(서브에이전트 1인칭 일기, 80~180단어, 60s 타임아웃)
상태: SQLite plugin KV 네임스페이스(short-term-recall/phase-signals/meta/locks/ingestion), 워크스페이스 경로 sha256로 키링, 5만 엔트리 한계
출력: memory/.dreams/ (머신) + DREAMS.md (관리 블록, 마커 <!-- openclaw:dreaming:{light|rem|deep}:start/end -->) + memory/dreaming/<phase>/YYYY-MM-DD.md (휴먼)
그림자 트라이얼(dreaming-shadow-trial.ts): report-only, MEMORY.md에 직접 쓰지 않음. helpful=소폭 부스트, neutral=defer, harmful=reject
```

# Memory Search (검색/재순위 로직)

> 출처: [분석 문서](../../../report/[git]_openclaw_openclaw.md) / submodule: `source/git/openclaw_openclaw`

## 설명
builtin memory 엔진의 검색 파이프라인. `MemoryIndexManager.search()`가 쿼리 임베딩 → 벡터 KNN(sqlite-vec `vec0`) + FTS5 BM25 키워드 검색을 병렬로 수행하고, `mergeHybridResults`가 가중 합 + 시간 감쇠 + MMR(다양성 재순위)로 최종 점수를 산출. 사용자가 요청한 "memory 검색 로직"의 핵심.

## 코드

### search() 메인 흐름 — `extensions/memory-core/src/memory/manager.ts:612`
```ts
async search(query, opts?) {
  // 1) provider 가용성 게이트 (required 모드: 실패시 fail-closed)
  if (this.providerRequirement.mode === "required") {
    await this.ensureProviderInitialized();
    this.assertRequiredProviderAvailable("search");
  }
  // 2) 인덱스 비어있으면 강제 sync(bootstrap) — 첫 검색이 빈 결과 방지
  let hasIndexedContent = this.hasIndexedContent();
  if (!hasIndexedContent) { await this.sync({ reason: "search", force: true }); ... }
  // 3) preflight: 빈 쿼리/인덱스 없으면 즉시 []
  const preflight = resolveMemorySearchPreflight({ query, hasIndexedContent });
  if (!preflight.shouldSearch) return [];
  // 4) 인덱스 identity(stale/mismatch)면 [] — 오래된 인덱스 안 씀
  const indexIdentity = this.refreshIndexIdentityDirty(...);
  if (indexIdentity.status !== "valid") return [];
  // 5) 후보 예산 = min(200, maxResults * candidateMultiplier) (기본 6*4=24)
  const candidates = Math.min(200, Math.max(1, Math.floor(maxResults * hybrid.candidateMultiplier)));

  // 6a) FTS-only 모드(provider 없음): 키워드 검색 → 시간감쇠 → selectScoredResults
  if (!this.provider) { ... return this.selectScoredResults(...); }

  // 6b) 하이브리드: 키워드(FTS) + 쿼리 임베딩 → 벡터 KNN
  let keywordResults = await loadKeywordResults();           // FTS BM25
  const queryVec = await this.embedQueryWithRetry(cleaned);  // 쿼리 임베딩 (실패시 fallback provider→재임베딩, 그래도 안되면 keyword-only)
  const vectorResults = hasVector ? await this.searchVector(queryVec, candidates, ...) : [];

  // 7) 퓨전 + 재순위
  const merged = await this.mergeHybridResults({
    vector: vectorResults, keyword: keywordResults,
    vectorWeight: hybrid.vectorWeight,   // 기본 0.7
    textWeight: hybrid.textWeight,        // 기본 0.3
    mmr: hybrid.mmr, temporalDecay: hybrid.temporalDecay,
  });
  const strict = merged.filter((e) => e.score >= minScore);  // 기본 minScore 0.35
  if (strict.length > 0 || keywordResults.length === 0) return strict.slice(0, maxResults);
  // 8) 완화 패스: strict 비어있고 키워드 히트만 있으면 BM25 정규화 결과를 minScore 0으로 보존
  return this.selectScoredResults(merged-only-keyword, maxResults, minScore, 0);
}
```

### 벡터 KNN (sqlite-vec) — `extensions/memory-core/src/memory/manager-search.ts:141`
```ts
// native KNN: vec0 인덱스로 O(log N + k). 점수 = 1 - cosine_dist
const candidateLimit = Math.min(params.limit * VECTOR_KNN_OVERSAMPLE_FACTOR /*8*/, MAX_VECTOR_KNN_K /*4096*/);
rows = db.prepare(
  `SELECT c.id, c.path, c.text, c.source,
          vec_distance_cosine(v.embedding, ?) AS dist
     FROM ${vectorTable} v
     JOIN memory_index_chunks c ON c.id = v.id
    WHERE v.embedding MATCH ? AND k = ? AND ${vectorModelFilter}${sourceFilter}
    ORDER BY dist ASC LIMIT ?`
).all(qBlob, qBlob, candidateLimit, ...models, ...sourceParams, limit);
// 후보가 부족하면 vectorCount까지 widen; 그래도 모자라면 bounded scan(256행 배치, setImmediate yield)으로 cosineSimilarity 직접 계산
```

### 하이브리드 퓨전 — `extensions/memory-core/src/memory/hybrid.ts:52`
```ts
export async function mergeHybridResults(params) {
  // id 기준으로 vector + keyword 병합 (한쪽에만 있으면 부재 점수 0)
  const merged = Array.from(byId.values()).map((entry) => ({
    // 가중 선형 결합
    score: params.vectorWeight * entry.vectorScore + params.textWeight * entry.textScore,
    vectorScore: entry.vectorScore, textScore: entry.textScore, ...
  }));
  // 시간 감쇠: memory/YYYY-MM-DD.md 만 적용, MEMORY.md/비날짜파일은 evergreen
  const decayed = await applyTemporalDecayToHybridResults({ results: merged, ... });
  const sorted = decayed.toSorted((a, b) => b.score - a.score);
  if (mmrConfig.enabled) return applyMMRToHybridResults(sorted, mmrConfig); // 기본 OFF
  return sorted;
}

// BM25 rank → [0,1] 점수
export function bm25RankToScore(rank) {
  if (rank < 0) { const relevance = -rank; return relevance / (1 + relevance); }
  return 1 / (1 + rank);
}
```

### MMR 재순위 — `extensions/memory-core/src/memory/mmr.ts`
```ts
// MMR = λ * relevance - (1-λ) * max_jaccard_similarity_to_selected  (Carbonell & Goldstein 1998)
export const DEFAULT_MMR_CONFIG = { enabled: false, lambda: 0.7 }; // 0=최대다양성, 1=최대관련성
export function computeMMRScore(relevance, maxSimilarity, lambda) {
  return lambda * relevance - (1 - lambda) * maxSimilarity;
}
// 점수 min-max 정규화 → [0,1] 후, 가장 높은 항목부터 시작해 매 슬롯마다 MMR 최대화 항목 선택
// 유사도는 CJK 인식 토큰화 후 Jaccard. 동점은 원래 score로 타이브레이크.
```

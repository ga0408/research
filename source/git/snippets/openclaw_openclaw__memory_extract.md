# Memory Extract (색인/쓰기 로직)

> 출처: [분석 문서](../../../report/[git]_openclaw_openclaw.md) / submodule: `source/git/openclaw_openclaw`

## 설명
builtin memory 엔진의 쓰기(추출/색인) 파이프라인. 파일 감시·세션 트랜스크립트 업데이트·타이머·세션 시작·검색 부트스트랩이 `sync()`를 트리거하고, `runSync`가 마크다운을 청킹(400토큰/80오버랩) → 임베딩(provider 배치, 재시도/분할/캐시) → `writeChunks`로 SQLite 3개 테이블(chunks/vec/fts)에 원자적으로 기록. 사용자가 요청한 "memory 추출 로직"의 핵심.

## 코드

### writeChunks — `extensions/memory-core/src/memory/manager-embedding-ops.ts:750`
```ts
// 청크+임베딩을 인덱스에 기록. 하나의 runSqliteImmediateTransactionSync로 원자적.
private writeChunks(entry, source, model, chunks, embeddings, vectorReady): void {
  const now = Date.now();
  runSqliteImmediateTransactionSync(this.db, () => {
    this.clearIndexedFileData(entry.path, source);   // 기존 행 삭제(path+source 단위)
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = embeddings[i] ?? [];
      // chunk id = hash(source:path:startLine:endLine:chunkHash:model)
      const id = hashText(`${source}:${entry.path}:${chunk.startLine}:${chunk.endLine}:${chunk.hash}:${model}`);
      // 1) memory_index_chunks (본문 + 임베딩 JSON 스캔 폴백용)
      this.db.prepare(
        `INSERT INTO memory_index_chunks (id,path,source,start_line,end_line,hash,model,text,embedding,updated_at)
         VALUES (?,?,?,?,?,?,?,?,?,?)
         ON CONFLICT(id) DO UPDATE SET hash=excluded.hash, model=excluded.model, text=excluded.text,
           embedding=excluded.embedding, updated_at=excluded.updated_at`
      ).run(id, entry.path, source, chunk.startLine, chunk.endLine, chunk.hash, model, chunk.text, JSON.stringify(embedding), now);
      // 2) memory_index_chunks_vec (vec0 가상테이블, KNN용) — vectorReady && embedding.length>0 일때만
      if (vectorReady && embedding.length > 0) {
        replaceMemoryVectorRow({ db: this.db, tableName: VECTOR_TABLE, id, embedding });
      }
      // 3) memory_index_chunks_fts (FTS5, BM25) — 모델 무관(model-agnostic, #48300)
      if (this.fts.enabled && this.fts.available) {
        this.db.prepare(
          `INSERT INTO ${FTS_TABLE} (text,id,path,source,model,start_line,end_line) VALUES (?,?,?,?,?,?,?)`
        ).run(chunk.text, id, entry.path, source, model, chunk.startLine, chunk.endLine);
      }
    }
    this.upsertFileRecord(entry, source); // memory_index_sources (hash/mtime/size)
  });
}
```

### 청킹 + 임베딩 — `manager-embedding-ops.ts:819` `prepareIndexEntry`
```ts
// 마크다운: chunkMarkdown(content, settings.chunking) // 기본 400토큰, 80 오버랩
// → buildMemoryEmbeddingBatches(missing, EMBEDDING_BATCH_MAX_TOKENS=8000) 배치 분할
// → provider.embedBatch (watchdog 타임아웃: 원격 2분/로컬 10분)
// 재시도 정책: 3회, 지수 백오프(500ms~8s). 전송 실패시 배치 분할 onSplit. 2회 실패시 인라인 폴백
// 캐시: memory_embedding_cache (provider,model,provider_key,hash) 키 → 기존 벡터 재사용. LRU 트리밋.
// 멀티모달(이미지/오디오): gemini-embedding-2-preview 등 → 단일 구조화 청크
```

### sync 트리거 — `extensions/memory-core/src/memory/manager-sync-ops.ts`
```
트리거(모두 sync({reason})로 수렴):
- 파일 감시: chokidar + native fs.watch, watchDebounceMs(1.5s) 디바운스, markDirty
- 세션 트랜스크립트: onSessionTranscriptUpdate, 5s 디바운스 + 바이트/메시지 델타 게이트
- interval: sync.intervalMinutes>0 일때 setInterval
- 세션 시작: warmSession + ensureSessionStartupCatchup (온디스크 vs 인덱스 diff)
- 검색 부트스트랩: 인덱스 비어있으면 search()가 강제 sync(force:true)
- CLI: openclaw memory index --force

runSync 결정:
 indexIdentity(missing/mismatched) 또는 memoryFullRetryDirty → runInPlaceReindex(그림자DB 빌드 후 원자적 테이블 교체)
 그 외 → syncMemoryFiles + syncSessionFiles (증분) 또는 source-wide 배치(배치 provider)
 리인덱스 잠금: <dbPath>.reindex-lock.sqlite (BEGIN EXCLUSIVE, 별도 커넥션 → 읽기 차지 않음)
```

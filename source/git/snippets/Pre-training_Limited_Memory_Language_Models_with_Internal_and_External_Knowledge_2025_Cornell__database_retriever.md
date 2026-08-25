# LMLM Knowledge Database & Top-K Retriever

> 출처: [분석 문서](../../../report/[paper][git]_Pre-training_Limited_Memory_Language_Models_with_Internal_and_External_Knowledge_2025_Cornell.md) / submodule: `source/git/LMLM_kilian-group/src/lmlm/database/database_manager.py` 및 `topk_retriever.py`

## 설명

LMLM의 외부 지식베이스 관리 및 인출 메커니즘:
1. `DatabaseManager`: 주석 데이터셋에서 `(entity, relation, return_value)` 트리플을 추출하여 JSON 형식으로 직렬화 및 관리.
2. `TopkRetriever`: Dense 임베딩 기반 검색기. `sentence-transformers/all-MiniLM-L6-v2` 모델로 `"{entity} {relation}"` 텍스트를 인코딩하고, FAISS `IndexFlatIP` 인덱스를 통해 코사인 유사도 검색을 수행.
3. 임계값(`threshold=0.6`) 이상의 최고 유사도를 가진 트리플의 사실 반환값을 반환. 만약 검색 실패 시 지정된 fallback policy(`top1_anyway`, `unknown`, `regenerate_query`)를 통해 안전하게 처리.

## 코드 스니펫

```python
# src/lmlm/database/topk_retriever.py

class TopkRetriever:
    def __init__(self, database, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=5, threshold=0.6, ...):
        self.database = database
        self.top_k = top_k if top_k else 5
        self.default_threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device).half().eval()
        self.index = None
        self.id_to_triplet = {}
        # FAISS 인덱스 로드 또는 빌드...

    def _build_index(self):
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        texts = [f"{self._normalize_text(ent)} {self._normalize_text(rel)}" for ent, rel, _ in self.database]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        ids = np.arange(len(self.database))
        self.index.add_with_ids(embeddings, ids)
        self.id_to_triplet = {i: triplet for i, triplet in enumerate(self.database)}

    def retrieve_top_k(self, entity: str, relation: str, threshold: Optional[float] = None) -> List[str]:
        query_text = f"{self._normalize_text(entity)} {self._normalize_text(relation)}"
        query_embedding = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)

        distances, indices = self.index.search(query_embedding, self.top_k)
        th = threshold if threshold is not None else self.default_threshold

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.id_to_triplet and dist >= th:
                triplet = self.id_to_triplet[idx]
                results.append((triplet[0], triplet[1], triplet[2], float(dist)))

        results.sort(key=lambda x: x[-1], reverse=True)
        return [r[2] for r in results]

# src/lmlm/database/database_manager.py

class DatabaseManager:
    def retrieve_from_database(self, prompt: str, threshold: Optional[float] = None):
        pattern_lst = [
            r"\[dblookup\('((?:[^'\\]|\\.)+)',\s*'((?:[^'\\]|\\.)+)'\)\s*->",
            r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->",
            r"<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>"
        ]
        matches = {tuple(match) for pattern in pattern_lst for match in re.findall(pattern, prompt)}

        if not matches:
            raise DatabaseLookupError(f"No valid dblookup pattern found: {prompt}", "no_match_found")
        if len(matches) > 1:
            raise DatabaseLookupError(f"Multiple dblookup matches found: {matches}", "multiple_matches")

        entity, relationship = matches.pop()
        self.init_topk_retriever()
        results = self.topk_retriever.retrieve_top_k(entity, relationship, threshold=threshold)

        if not results:
            raise DatabaseLookupError(f"No retrieval results for entity='{entity}', rel='{relationship}'", "no_retrieval_data_found")

        return results[0]
```

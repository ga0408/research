# retrieve.py + agent: 검색·저장 도구 및 에이전트 검색

> 출처: [PageIndex_VectifyAI.md](../../../report/[git]_PageIndex_VectifyAI.md) / `source/git/PageIndex_VectifyAI/pageindex/retrieve.py`, `examples/agentic_vectorless_rag_demo.py`

## 설명

PageIndex의 검색은 vector DB 대신 3개의 tool function을 LLM agent에게 제공하여, agent가 트리 구조를 "읽고" reasoning으로 관련 페이지를 찾는 방식. `get_document_structure()`가 트리 인덱스(text 제외, 토큰 절약)를 제공하고, agent가 이를 보고 `get_page_content()`로 필요한 페이지만 가져온다.

## 코드

```python
# retrieve.py:81-137  검색 도구 3종

def get_document(documents: dict, doc_id: str) -> str:
    """문서 메타데이터 반환: doc_name, doc_description, type, page_count/line_count."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    result = {
        'doc_id': doc_id,
        'doc_name': doc_info.get('doc_name', ''),
        'doc_description': doc_info.get('doc_description', ''),
        'type': doc_info.get('type', ''),
        'status': 'completed',
    }
    if doc_info.get('type') == 'pdf':
        result['page_count'] = _count_pages(doc_info)
    else:
        result['line_count'] = doc_info.get('line_count', 0)
    return json.dumps(result)


def get_document_structure(documents: dict, doc_id: str) -> str:
    """트리 구조 JSON 반환 (text 필드 제거하여 토큰 절약).
    agent는 이 트리를 보고 reasoning으로 관련 페이지 범위를 식별."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    structure = doc_info.get('structure', [])
    structure_no_text = remove_fields(structure, fields=['text'])  # text 제거
    return json.dumps(structure_no_text, ensure_ascii=False)


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """특정 페이지의 원본 텍스트 반환.
    pages 형식: '5-7' (범위), '3,8' (복수), '12' (단일)
    PDF: 1-indexed 물리적 페이지 번호
    Markdown: line_num 기준"""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    page_nums = _parse_pages(pages)  # '5-7' → [5,6,7], '3,8' → [3,8]
    if doc_info.get('type') == 'pdf':
        content = _get_pdf_page_content(doc_info, page_nums)
    else:
        content = _get_md_page_content(doc_info, page_nums)
    return json.dumps(content, ensure_ascii=False)


# _get_pdf_page_content: 캐시 우선, 폴백으로 PDF 직접 읽기
def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    cached_pages = doc_info.get('pages')
    if cached_pages:
        page_map = {p['page']: p['content'] for p in cached_pages}
        return [{'page': p, 'content': page_map[p]} for p in page_nums if p in page_map]
    path = doc_info['path']
    with open(path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [{'page': p, 'content': pdf_reader.pages[p - 1].extract_text() or ''}
                for p in valid_pages]


# client.py:55-130  인덱싱 + 워크스페이스 저장
class PageIndexClient:
    def index(self, file_path: str, mode: str = "auto") -> str:
        """문서 인덱싱 → doc_id 반환.
        PDF: page_index() 호출 → 트리 빌드, per-page 텍스트 추출 후 캐시.
        MD:  md_to_tree() 호출 → 헤더 기반 트리 빌드."""
        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            result = page_index(doc=file_path, model=self.model,
                                if_add_node_summary='yes', if_add_node_text='yes',
                                if_add_node_id='yes', if_add_doc_description='yes')
            # per-page 텍스트 추출 (검색 시 원본 PDF 불필요)
            pages = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({'page': i, 'content': page.extract_text() or ''})
            self.documents[doc_id] = {
                'id': doc_id, 'type': 'pdf', 'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'page_count': len(pages),
                'structure': result['structure'],
                'pages': pages,  # 캐시된 페이지 텍스트
            }
        # ...
        if self.workspace:
            self._save_doc(doc_id)  # {doc_id}.json 으로 영구 저장
        return doc_id

    def _save_doc(self, doc_id: str):
        """워크스페이스에 JSON 저장. text 필드는 pages와 중복이므로 제거.
        저장 후 메모리에서 structure/pages 제거 (lazy-load)."""
        doc = self.documents[doc_id].copy()
        if doc.get('structure') and doc.get('type') == 'pdf':
            doc['structure'] = remove_fields(doc['structure'], fields=['text'])
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        self.documents[doc_id].pop('structure', None)  # lazy-load
        self.documents[doc_id].pop('pages', None)


# agentic_vectorless_rag_demo.py:55-88  에이전트 검색 설정
AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm status and page/line count.
- Call get_document_structure() to identify relevant page ranges.
- Call get_page_content(pages="5-7") with tight ranges; never fetch the whole document.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise.
"""

def query_agent(client: PageIndexClient, doc_id: str, prompt: str, verbose: bool = False) -> str:
    @function_tool
    def get_document() -> str:
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure() -> str:
        """Get the document's full tree structure (without text) to find relevant sections."""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(pages: str) -> str:
        """Use tight ranges: '5-7', '3,8', '12'."""
        return client.get_page_content(doc_id, pages)

    agent = Agent(
        name="PageIndex",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[get_document, get_document_structure, get_page_content],
        model=client.retrieve_model,
    )
    # Runner.run_streamed(agent, prompt) 로 실행
```

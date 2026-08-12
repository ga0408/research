# page_offset 계산 + page grouping: 인덱스 매핑 핵심

> 출처: [PageIndex_VectifyAI.md](../../../report/[git]_PageIndex_VectifyAI.md) / `source/git/PageIndex_VectifyAI/pageindex/page_index.py`, `utils.py`

## 설명

TOC에 있는 논리적 페이지 번호와 PDF의 물리적 페이지 인덱스 사이의 offset을 계산하는 로직. 여러 matching pair에서 가장 빈도 높은 offset을 선택한다. 또한 페이지를 토큰 한도 내로 그룹화하여 LLM 컨텍스트 창에 맞추는 로직.

## 코드

```python
# page_index.py:379-422  page offset 계산 및 적용
def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    """TOC의 논리적 페이지 번호와 LLM이 추출한 물리적 인덱스를
    같은 title끼리 매칭하여 pair 리스트 생성."""
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),       # TOC에 적힌 논리적 번호
                        'physical_index': physical_index      # LLM이 찾은 실제 위치
                    })
    return pairs

def calculate_page_offset(pairs):
    """각 pair의 (physical_index - page) 차이를 계산하고
    가장 빈도 높은 차이값을 offset으로 반환."""
    differences = []
    for pair in pairs:
        try:
            difference = pair['physical_index'] - pair['page']
            differences.append(difference)
        except (KeyError, TypeError):
            continue
    if not differences:
        return None
    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1
    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]
    return most_common

def add_page_offset_to_toc_json(data, offset):
    """모든 TOC 항목의 논리적 page 번호에 offset을 더해
    physical_index로 변환."""
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']
    return data


# page_index.py:426-459  토큰 한도 기반 페이지 그룹화
def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):
    """페이지들을 max_tokens 이하의 그룹으로 병합.
    그룹 경계에는 overlap_page 만큼 겹침을 두어 문맥 단절 방지."""
    num_tokens = sum(token_lengths)
    if num_tokens <= max_tokens:
        return ["".join(page_contents)]  # 전체가 한 그룹

    subsets = []
    current_subset = []
    current_token_count = 0
    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)

    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:
            subsets.append(''.join(current_subset))
            # overlap: 이전 페이지부터 다시 시작
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        current_subset.append(page_content)
        current_token_count += page_tokens

    if current_subset:
        subsets.append(''.join(current_subset))
    return subsets


# utils.py:387-410  PDF 페이지 텍스트 + 토큰 수 추출
def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))  # (text, token_count) 튜플
        return page_list
    elif pdf_parser == "PyMuPDF":
        # BytesIO 및 파일 경로 모두 지원
        doc = pymupdf.open(stream=pdf_path, filetype="pdf") if isinstance(pdf_path, BytesIO) else pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
```

# tree_parser + meta_processor: 인덱스 빌드 오케스트레이션

> 출처: [PageIndex_VectifyAI.md](../../../report/[git]_PageIndex_VectifyAI.md) / `source/git/PageIndex_VectifyAI/pageindex/page_index.py`

## 설명

PageIndex 인덱싱의 핵심 오케스트레이션. `tree_parser`는 TOC 존재 여부와 페이지 번호 포함 여부에 따라 3가지 모드를 선택하고, `meta_processor`는 해당 모드로 트리를 빌드한 뒤 검증·수정 루프를 실행한다. 정확도가 60% 미만이면 더 단순한 모드로 fallback 한다.

## 코드

```python
# page_index.py:1029-1063  tree_parser — 모드 분기 + 후처리
async def tree_parser(page_list, opt, doc=None, logger=None):
    check_toc_result = check_toc(page_list, opt)

    if check_toc_result.get("toc_content") and check_toc_result["page_index_given_in_toc"] == "yes":
        toc_with_page_number = await meta_processor(
            page_list, mode='process_toc_with_page_numbers',
            toc_content=check_toc_result['toc_content'],
            toc_page_list=check_toc_result['toc_page_list'], opt=opt, logger=logger)
    else:
        toc_with_page_number = await meta_processor(
            page_list, mode='process_no_toc', opt=opt, logger=logger)

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(
        toc_with_page_number, page_list, model=opt.model, logger=logger)

    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    toc_tree = post_processing(valid_toc_items, len(page_list))

    # 대형 노드 재귀 분할
    tasks = [process_large_node_recursively(node, page_list, opt, logger=logger) for node in toc_tree]
    await asyncio.gather(*tasks)
    return toc_tree


# page_index.py:959-997  meta_processor — 빌드 + 검증 + 수정 루프
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None,
                         start_index=1, opt=None, logger=None):
    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = process_toc_with_page_numbers(
            toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num,
            model=opt.model, logger=logger)
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = process_toc_no_page_numbers(
            toc_content, toc_page_list, page_list, model=opt.model, logger=logger)
    else:
        toc_with_page_number = process_no_toc(
            page_list, start_index=start_index, model=opt.model, logger=logger)

    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number, len(page_list), start_index=start_index, logger=logger)

    accuracy, incorrect_results = await verify_toc(
        page_list, toc_with_page_number, start_index=start_index, model=opt.model)

    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(
            toc_with_page_number, page_list, incorrect_results,
            start_index=start_index, max_attempts=3, model=opt.model, logger=logger)
        return toc_with_page_number
    else:
        # fallback: 더 단순한 모드로 재시도
        if mode == 'process_toc_with_page_numbers':
            return await meta_processor(page_list, mode='process_toc_no_page_numbers', ...)
        elif mode == 'process_toc_no_page_numbers':
            return await meta_processor(page_list, mode='process_no_toc', ...)
        else:
            raise Exception('Processing failed')


# page_index.py:1000-1027  process_large_node_recursively — 대형 노드 재귀 분할
async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])

    # 페이지 수 AND 토큰 수 둘 다 임계값 초과시 자식 트리 추출
    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node \
       and token_num >= opt.max_token_num_each_node:
        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc',
                                             start_index=node['start_index'], opt=opt, logger=logger)
        node_toc_tree = await check_title_appearance_in_start_concurrent(
            node_toc_tree, page_list, model=opt.model, logger=logger)

        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]

        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']

    # 자식 노드들에 대해 재귀 (병렬)
    if 'nodes' in node and node['nodes']:
        tasks = [process_large_node_recursively(child_node, page_list, opt, logger=logger)
                 for child_node in node['nodes']]
        await asyncio.gather(*tasks)
    return node
```

# Workflow Memory Integration

> 출처: [분석 문서](../../../report/[paper][git]_Agent_Workflow_Memory_2024_arxiv.md) / submodule: `source/git/agent-workflow-memory_zorazrw/webarena/agents/legacy/agent.py`, `mind2web/memory.py`

## 설명

유도된 workflow가 agent의 memory에 통합되어 action 생성에 활용되는 방식. 두 벤치마크에서 서로 다른 통합 방식을 사용:

- **WebArena**: workflow 텍스트를 system message에 직접 append. 매 step마다 workflow 파일을 읽어 system prompt 뒤에 추가.
- **Mind2Web**: workflow 텍스트를 user message(exemplar)의 첫 번째 항목으로 배정. concrete example과 함께 token limit 내에서 최대한 많이 포함.

두 방식 모두 workflow를 추가적인 context로 제공하며, agent가 action 생성 시 이를 참고하도록 유도함. 명시적 retrieval 없이 전체 workflow를 memory에 포함(website당 평균 7개 workflow로 크기가 작음).

## 코드

### WebArena: System Message Append

```python
# webarena/agents/legacy/agent.py — GenericAgent.get_action()

def get_action(self, obs):
    self.obs_history.append(obs)

    # construct main prompt (observation + history + action space + think)
    main_prompt = dynamic_prompting.MainPrompt(
        obs_history=self.obs_history,
        actions=self.actions,
        memories=self.memories,
        thoughts=self.thoughts,
        flags=self.flags,
    )
    prompt = dynamic_prompting.fit_tokens(main_prompt, max_prompt_tokens=...)

    # base system message
    sys_msg = dynamic_prompting.SystemPrompt().prompt
    # ★ workflow memory injection: workflow file을 읽어 system message에 append
    if self.flags.workflow_path is not None:
        sys_msg += '\n\n' + open(self.flags.workflow_path).read()

    chat_messages = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=prompt),
    ]

    ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
    self.actions.append(ans_dict["action"])
    return ans_dict["action"], ans_dict
```

### Mind2Web: Exemplar-based Memory

```python
# mind2web/memory.py — get_exemplars()

def get_exemplars(args):
    """Workflow memory + concrete examples를 prompt context로 구성."""
    memory = []
    # ★ workflow text를 첫 번째 exemplar로 배정
    workflow_text = open(args.workflow_path, 'r').read().strip()
    if len(workflow_text):
        memory = [[{"role": "user", "content": workflow_text}]]

    # concrete examples (website/domain 필터링)
    with open(f"{args.memory_path}/exemplars.json") as f:
        concrete_examples = json.load(f)
    # hierarchy: website > subdomain > domain 순으로 필터링
    if any([args.website in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [cex for cex in concrete_examples
                             if all(tag in cex[0]["specifier"]
                                    for tag in [args.domain, args.subdomain, args.website])]
    elif any([args.subdomain in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [cex for cex in concrete_examples
                             if all(tag in cex[0]["specifier"]
                                    for tag in [args.domain, args.subdomain])]

    memory += random.sample(concrete_examples, min(args.retrieve_top_k, len(concrete_examples)))
    return memory

# eval_sample()에서 token limit 내에서 exemplar들 포함
def eval_sample(task_id, args, sample):
    exemplars = get_exemplars(args)
    sys_message = [{"role": "system", "content": "...action space..."}]

    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        # construct query from observation + previous actions
        query = [...]  # task instruction + trajectory

        # token limit 내에서 최대한 많은 exemplar 포함
        demo_message = []
        for e_id, e in enumerate(exemplars):
            total_tokens = num_tokens_from_messages(sys_message + demo_message + e + query, args.model)
            if total_tokens > MAX_TOKENS[args.model]:
                break  # token limit 초과 시 남은 exemplar 생략
            demo_message.extend(e)

        message = sys_message + demo_message + query
        response = generate_response(messages=message, model=args.model, ...)
```

# LM-based Workflow Induction

> 출처: [분석 문서](../../../report/[paper][git]_Agent_Workflow_Memory_2024_arxiv.md) / submodule: `source/git/agent-workflow-memory_zorazrw/webarena/induce_prompt.py`

## 설명

AWM의 핵심 모듈: agent의 성공한 trajectory로부터 LM(GPT-4o)을 통해 재사용 가능한 workflow를 추출하는 로직. 성공 기준으로 두 가지를 지원:
- `gt`: ground-truth reward (`summary_info.json["cum_reward"]`)
- `autoeval`: LLM evaluator의 판정 (`{model}_autoeval.json[0]["rm"]`)

주요 단계:
1. 성공한 trajectory 수집 → blank-line separated log에서 think/action pair 추출
2. invalid step 제거 (click/fill의 arg가 string-formatted integer가 아닌 경우, scroll/noop 제거)
3. task template 기반 deduplication (template_id별 1개 샘플링)
4. formatting: `Query: ... Actions: <think>...</think><action>...</action>` 형식
5. instruction + one-shot + examples를 GPT-4o에 전달 (temperature=1.0, max_tokens=2048)
6. 출력에 utility workflow 2개(click id 사용법, select_option 사용법) append

## 코드

```python
# webarena/induce_prompt.py 핵심 로직

def remove_invalid_steps(actions):
    """click/fill의 arg가 유효한 string-formatted integer인지 검증."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            try:
                if type(eval(arg)) == str and type(eval(arg[1:-1])) == int:
                    valid_actions.append(a)
            except: continue
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
        elif "scroll(" in a or "noop(" in a:
            continue
        else:
            valid_actions.append(a)
    return valid_actions

def format_examples(examples):
    """examples를 prompt 형식으로 변환."""
    formatted = []
    for ex in examples:
        trajectory = format_trajectory(ex["think_list"], ex["action_list"])
        formatted.append(f"Query: {ex['query']}\nActions:\n{trajectory}")
    return '\n\n'.join(["## Concrete Examples"] + formatted + ["## Summary Workflows"])

def llm_generate(examples, args):
    """instruction + one-shot + examples를 결합해 GPT-4o로 workflow 유도."""
    prompt = format_examples(examples)
    prompt = '\n\n'.join([args.INSTRUCTION, args.ONE_SHOT, prompt])
    response = client.chat.completions.create(
        model=args.model,           # default: gpt-4o
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,            # diversity를 위한 high temperature
        max_tokens=2048,
    )
    return response.choices[0].message.content

# main() 핵심 흐름
# 1. 성공한 trajectory 수집 (criteria = "autoeval" or "gt")
# 2. template_id 기반 deduplication → template당 1개 샘플
template_dict = {}
for f in file_dirs:
    config = json.load(open(f"config_files/{task_id}.json"))
    template_id = config["intent_template_id"]
    think_list, action_list = extract_think_and_action(log_path)
    template_dict.setdefault(template_id, []).append(wdict)
selected_examples = random_group_sample(template_dict, n=1)

# 3. GPT-4o로 workflow 유도
workflows = llm_generate(selected_examples, args)
# 4. utility workflow append
workflows += "\n\nclick('id') # input string id value for all actions\n\nselect_option('id', 'value')"
# 5. workflow memory 파일에 덮어쓰기
with open(args.output_path, 'w') as fw:
    fw.write(workflows)
```

## Induction Prompt (prompt/instruction.txt)

```
Given a list of web navigation tasks, your task is to extract the common workflows to solve these tasks.
Each given task contains a natural language instruction, and a series of actions to solve the task.
You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow.
Each workflow should be a commonly-reused sub-routine of the tasks. Do not generate similar or overlapping workflows.
Each workflow should have at least two steps. Represent the non-fixed elements (input text, button strings) with descriptive variable names as shown in the example.
Keep the values of invariant elements, e.g., id of "Search" or "Customers", as they will share and stay invariant across tasks.
Try to generate as many workflows that can cover all the tasks in the input list.
```

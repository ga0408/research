# Online Streaming Pipeline

> 출처: [분석 문서](../../../report/[paper][git]_Agent_Workflow_Memory_2024_arxiv.md) / submodule: `source/git/agent-workflow-memory_zorazrw/webarena/pipeline.py`, `mind2web/pipeline.py`

## 설명

AWM online 모드의 streaming 루프: test query를 순차적으로 처리하며, 각 task마다 (1) inference → (2) evaluation → (3) workflow induction의 3-step을 실행. workflow memory 파일이 매 iteration 갱신되며 이전 task에서 학습한 workflow가 다음 task에 활용됨 (snowball effect).

**WebArena**: task ID를 website별로 필터링 후 순차 처리. 각 step을 subprocess로 실행.
**Mind2Web**: examples를 batch(`induce_steps`개)로 처리. 매 batch 후 online_induction.py 실행.

## 코드

### WebArena Online Pipeline

```python
# webarena/pipeline.py

def main():
    # website별 task ID 수집
    config_files = [f for f in os.listdir("config_files") if f.endswith(".json")]
    config_list = [json.load(open(f"config_files/{f}")) for f in config_files]
    task_ids = [c["task_id"] for c in config_list if c["sites"][0] == args.website]

    for tid in task_ids[args.start_index:args.end_index]:
        # step 1: inference with current workflow memory
        Popen(["python", "run.py",
               "--task", f"webarena.{tid}",
               "--workflow_path", f"workflow/{args.website}.txt"]).wait()

        # step 2: LLM-based evaluation
        Popen(["python", "-m", "autoeval.evaluate_trajectory",
               "--result_dir", f"results/webarena.{tid}"]).wait()

        # step 3: re-induce workflows from ALL successful results so far
        Popen(["python", "induce_prompt.py",
               "--result_dir", "results",
               "--output_path", f"workflow/{args.website}.txt"]).wait()
```

**핵심**: `induce_prompt.py`는 `results/` 디렉토리의 모든 성공 trajectory를 재읽어 workflow를 재유도하므로, workflow memory는 누적된 모든 경험을 반영. 매 iteration마다 overwrite 방식으로 갱신.

### Mind2Web Online Pipeline

```python
# mind2web/pipeline.py

def online():
    samples = load_json(args.data_dir, args.benchmark)
    samples = [s for s in samples if s["website"] == args.website]
    n = len(samples)

    for i in range(0, n, args.induce_steps):  # default induce_steps=1
        j = min(n, i + args.induce_steps)

        # step 1: batch inference
        Popen(["python", "run_mind2web.py",
               "--workflow_path", args.workflow_path,
               "--start_idx", f"{i}", "--end_idx", f"{j}"]).wait()

        # step 2: induce workflows from past results (if not last batch)
        if (j + 1) < len(samples):
            Popen(["python", "online_induction.py",
                   "--results_dir", args.results_dir,
                   "--output_path", args.workflow_path]).wait()
```

### Mind2Web Online Induction

```python
# mind2web/online_induction.py

def main():
    # load past results and format as examples
    result_files = [os.path.join(args.results_dir, f) for f in os.listdir(args.results_dir)]
    result_list = [get_trajectory(rf) for rf in result_files]
    examples = [{"confirmed_task": s["confirmed_task"],
                 "action_reprs": [step["env"] + '\n' + step["action"] for step in r]}
                for r, s in zip(result_list, samples)]

    # prompt GPT for workflow induction
    prompt = format_examples(examples, args.prefix, args.suffix)
    prompt = '\n\n'.join([INSTRUCTION, ONE_SHOT, f"Website: {domain}, {subdomain}, {website}\n{prompt}"])
    response = client.chat.completions.create(
        model=args.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,  # 0.0
    ).choices[0].message.content

    workflows = filter_workflows(response, args.website)
    with open(args.output_path, 'w') as fw:
        fw.write(workflows)
```

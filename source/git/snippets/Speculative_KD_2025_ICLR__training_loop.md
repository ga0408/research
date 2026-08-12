# SKD 학습 루프 — 샘플 생성 → KD 손실

> 출처: [분석 문서](../../../report/[paper][git]_Speculative_Knowledge_Distillation_Bridging_the_Teacher-Student_Gap_Through_Interleaved_Sampling_2025_ICLR.md) / 원본: `speculative_kd/train/ddp_skd.py`
> (monorepo URL 참조: https://github.com/google-research/google-research/tree/master/speculative_kd)

## 설명

`ddp_skd.py`의 학습 루프는 `kd_type`으로 7가지 변형을 지원한다. SKD 핵심은 `kd_type == "skd"` 분기: **교사 모델이 `.generate()`를 주도하되 학생을 `assistant_model`(draft)로 넘겨** interleaved 샘플을 만든 뒤, 그 시퀀스에 대해 교사·학생 logprob 간 KL/JSD/reverse-KL 손실을 계산한다.

- `skd`: 교사 `.generate(assistant_model=student, teacher_k=K, teacher_p=P, num_assistant_tokens=5, ...)` — `num_assistant_tokens=5`(γ=5)로 학생이 5토큰 제안 → `_speculative_kd_sampling` 검증/교체 반복.
- `on-policy`: 학생이 단독으로 self-generate → KD 손실 (교사는 로짓 계산용만).
- `supervised_kd` / `seq_kd`: 정답 시퀀스(`train_src_ref_dict`) 사용; seq_kd는 CE.
- `mixed` (ImitKD): 시퀀스 단위로 확률적 혼합 (학생 생성 vs 정답).
- `mixed_train` (§5.5 두 단계 ablation): 전반 supervised / 후반 on-policy.
- `mixed_skd`: SKD vs 정답 확률적 혼합.

손실 함수: `nn.KLDivLoss(log_target=True)`(기본 forward KL), `JSD`, `reverse_kl`. 교사는 frozen(`requires_grad=False`), 학생만 backward. DDP+DeepSpeed ZeRO-3, bf16, 8×A100.

## 코드: SKD 샘플 생성 분기 + KD 손실

```python
# speculative_kd/train/ddp_skd.py (SKD 매설, 발췌)

# --- 학생(assistant)을 draft로 구성: γ=5, ancestral sampling ON ---
if kd_type == "skd" or kd_type == "mixed_skd":
    assistant_model.module.generation_config.num_assistant_tokens = 5
    assistant_model.tokenizer = tokenizer
    assistant_model.num_beams = 1          # disable top-k
    assistant_model.do_sample = True       # enable sampling
    assistant_model.top_p = student_top_p
    assistant_model.temperature = student_temperature
    assistant_model.module.generation_config.num_assistant_tokens_schedule = "constant"

# --- 손실 함수 선택 ---
if kd_type == "seq_kd":
    loss_funct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    loss_type = "ce"
else:
    loss_type = "non-ce"
    if distance_metric == "kl":
        loss_funct = nn.KLDivLoss(reduction="batchmean", log_target=True)
    elif distance_metric == "reverse_kl":
        loss_funct = reverse_kl()
    elif distance_metric == "jsd":
        loss_funct = JSD()

# --- per-prompt 샘플 생성 (no_grad) ---
for prompt in prompts:
    with torch.no_grad():
        ...
        elif kd_type == "skd":
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], ...,
                add_generation_prompt=True,
            ).to("cuda")
            # 교사가 generate 주도, 학생은 assistant_model(draft). teacher_k로 top-K 수락 적용.
            final_outputs = model.module.generate(
                **inputs,
                assistant_model=assistant_model.module,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                expected_seq_len=expected_seq_len,
                num_beams=1,
                teacher_k=top_k,
                teacher_p=top_p,
                stop_strings=end_of_string_ls,
                tokenizer=tokenizer,
                temperature=teacher_temperature,
                top_p=teacher_top_p,
                return_dict_in_generate=True,
            )
            # final_outputs.correction_rate  ← 거부율(Appendix L)

    # 생성물 → 토큰 재인코딩 (parse_output: chat template 재적용)
    if kd_type == "skd":  # (on-policy/mixed 포함)
        gen_output = tokenizer.batch_decode(final_outputs.sequences, skip_special_tokens=True)
        final_outputs, new_str = parse_output(
            gen_output, assistant_checkpoint, prompt, end_of_string_ls, max_length, tokenizer)

    # --- KD 손실: 학생/교사 logprob 비교 ---
    student_outputs = assistant_model(**final_outputs)
    shift_student_logps = (
        student_outputs.logits[0, inputs.input_ids.shape[1]:]
        .log_softmax(dim=-1).contiguous()
    )
    final_outputs = final_outputs.to(model.device)
    teacher_outputs = model(**final_outputs)
    shift_teacher_logps = (
        teacher_outputs.logits[0, inputs.input_ids.shape[1]:]
        .log_softmax(dim=-1).contiguous()
    )
    # KL(기본) / reverse_kl / jsd  —  loss = D(Mt || Ms)(y|x)
    if distance_metric in ("kl", "reverse_kl", "jsd"):
        loss = loss_funct(shift_student_logps, shift_teacher_logps)

    loss = loss / len(prompts)
    batch_loss += loss
    accelerator.backward(loss)

# accumulation step 마다 clip + optimizer.step + scheduler.step
grad_norm = torch.nn.utils.clip_grad_norm_(assistant_model.parameters(), max_grad_norm)
optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
```

## JSD / reverse-KL 구현

```python
# speculative_kd/train/ddp_skd.py:60
class JSD(nn.Module):
    def forward(self, log_p, log_q):
        p = torch.exp(log_p); q = torch.exp(log_q)
        m = 0.5 * (p + q + 1e-12)
        return 0.5 * (F.kl_div(log_p, torch.log(m), reduction="batchmean", log_target=True)
                      + F.kl_div(log_q, torch.log(m), reduction="batchmean", log_target=True))

class reverse_kl(nn.Module):
    # q, p are logprobs
    def forward(self, log_p, log_q):
        return F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
```

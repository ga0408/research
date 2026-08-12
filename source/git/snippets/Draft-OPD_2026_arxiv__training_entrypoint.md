# Training Entrypoint & Hyperparameters

> 출처: [분석 문서](../../../report/[paper][git]_Draft-OPD_On-Policy_Distillation_for_Speculative_Draft_Models_2026_arxiv.md) / submodule 경로: `source/git/Draft-OPD_bingyang-lei/verl/examples/on_policy_distillation_trainer/`

## 설명

Draft-OPD 학습의 공식 진입점 스크립트 두 단계 구조. `run_qwen_gsm8k_forward-ins.sh`(래퍼) → `run_qwen_gsm8k.sh`(본체) → `verl.trainer.main_ppo`(Hydra config). verl 프레임워크 위에 DFlash on-policy distillation을 얹어 학생(draft)/교사(target) 양쪽 GPU를 구성한다.

핵심 파라미터 매핑(논문 ↔ 코드):
- `λ_acc=λ_rej=1` → `stream_weight=1.0`, `rejected_draft_stream_weight=1.0`
- `γ=0.8` position decay → `REJECTED_DRAFT_POSITION_DECAY=0.8`, `REJECTED_DRAFT_POSITION_DECAY_ENABLED=True`
- reverse KL on rejected → `REJECTED_DRAFT_USE_REVERSE_KL=True`
- supervised(정책경사 아님) → `USE_POLICY_GRADIENT=False`, `loss_mode=k3`
- block size → DFlash config `DFLASH_LM_HEAD_CHUNK_SIZE=512`
- "Random Anchors" ablation → `RANDOM_RESPONSE_ANCHOR_ENABLED`/`RANDOM_RESPONSE_ANCHOR_SEED=42`

학생은 `actor_rollout_ref`, 교사는 `distillation.teacher_models`로 별도 SGLang rollout 엔진. 학생 쪽 rollout에 `speculative_algorithm=DFLASH` + `speculative_draft_model_path`로 speculative decoding를 켜두어, rollout 자체가 target-assisted speculative 검증이 되도록 설계.

## 코드

```bash
# verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh  (래퍼)
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k3}
REJECTED_DRAFT_USE_REVERSE_KL=${REJECTED_DRAFT_USE_REVERSE_KL:-True}
REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-True}
REJECTED_DRAFT_POSITION_DECAY=${REJECTED_DRAFT_POSITION_DECAY:-0.8}
RANDOM_RESPONSE_ANCHOR_ENABLED=${RANDOM_RESPONSE_ANCHOR_ENABLED:-False}
LR=${LR:-3e-4}
stream_weight=${stream_weight:-1.0}
rejected_draft_stream_weight=${rejected_draft_stream_weight:-1.0}

exec "${SCRIPT_DIR}/run_qwen_gsm8k.sh" \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}" \
    distillation.distillation_loss.rejected_draft_use_reverse_kl="${REJECTED_DRAFT_USE_REVERSE_KL}" \
    distillation.distillation_loss.response_stream_weight="${stream_weight}" \
    distillation.distillation_loss.rejected_draft_stream_weight="${rejected_draft_stream_weight}" \
    distillation.distillation_loss.rejected_draft_position_decay_enabled="${REJECTED_DRAFT_POSITION_DECAY_ENABLED}" \
    distillation.distillation_loss.rejected_draft_position_decay="${REJECTED_DRAFT_POSITION_DECAY}" \
    "$@"

# --- 본체 run_qwen_gsm8k.sh 핵심 설정 ---
USE_POLICY_GRADIENT=False                 # 감독 KL 직접 역전파 (k3)
DISTILLATION_LOSS_MODE="k3"               # sampled KL estimator
ROLLOUT_TEMPERATURE=0.0                   # rollout은 결정론
TEACHER_TEMPERATURE=1.0                   # 교사 log-prob는 T=1 (원 분포)
REJECTED_DRAFT_POSITION_DECAY=0.9         # (본체 기본값; 래퍼가 0.8로 오버라이드)
DFLASH_RESPONSE_ANCHOR_STRIDE=1           # 모든 anchor 계산
MAX_RESPONSE_LENGTH=4096                  # thinking train 길이 상한 (eval은 8192)

# 학생(드래프트) 구성: DFlash 합성 모델
MODEL=(
  +actor_rollout_ref.model.override_config.verl_composed_dflash_student=True
  +actor_rollout_ref.model.override_config.verl_dflash_draft_model_path="$DRAFT_MODEL_PATH"
  +actor_rollout_ref.model.override_config.verl_dflash_response_anchor_stride=...
  +actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_enabled=...
)

# 교사 구성: 별도 SGLang 추론 엔진
DISTILLATION=(
  distillation.enabled=True
  distillation.teacher_models.teacher_model.inference.name=sglang
  distillation.teacher_models.teacher_model.inference.temperature=$TEACHER_TEMPERATURE
  distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
  distillation.distillation_loss.response_stream_weight=$stream_weight
  distillation.distillation_loss.rejected_draft_stream_weight=$rejected_draft_stream_weight
  distillation.distillation_loss.rejected_draft_position_decay_enabled=$REJECTED_DRAFT_POSITION_DECAY_ENABLED
)

# 학생 추론(rollout) 자체를 DFlash speculative decoding로 설정
ROLLOUT=(
  +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_algorithm=DFLASH
  +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_draft_model_path="$DRAFT_MODEL_PATH"
)

python3 -m verl.trainer.main_ppo --config-path=config --config-name='ppo_trainer.yaml' "${DATA[@]}" ...
```

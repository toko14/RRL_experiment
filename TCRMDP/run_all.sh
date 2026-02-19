#!/usr/bin/env bash
set -euo pipefail

DEVICE="cuda:0"

run_sac_ablation_walker () {
  for s in 4; do
    time WANDB_MODE=disabled python src/main_ablation_sac.py \
      --experiment_name "W_ablation_SAC" \
      --env_name "Walker" \
      --nb_uncertainty_dim 1 \
      --seed "${s}" \
      --output_dir "/home/toko/デスクトップ/RRL_exp/Walker/W_dim1_SAC_seed${s}_0101" \
      --device "${DEVICE}" \
      --alpha_ent 0.2
  done
}

run_td3_ablation_walker () {
  for s in 2 3 4; do
    time WANDB_MODE=disabled python src/main_ablation_td3.py \
      --experiment_name "HC_ablation_TD3" \
      --env_name "Walker" \
      --nb_uncertainty_dim 1 \
      --seed "${s}" \
      --output_dir "/home/toko/デスクトップ/RRL_exp/Walker/W_dim1_TD3_seed${s}_0101" \
      --device "${DEVICE}"
  done
}

run_dr_halfcheetah () {
  for s in 8 9; do
    WANDB_MODE=disabled time python src/main_dr.py \
      --experiment_name "HC_1D_dr_seed${s}" \
      --env_name "HalfCheetah" \
      --nb_uncertainty_dim 1 \
      --seed "${s}" \
      --output_dir "/home/toko/デスクトップ/RRL_exp/HC/HC_dim1_DR_seed${s}_0101" \
      --max_steps 2000000 \
      --start_steps 100000 \
      --device "${DEVICE}" \
      --track True
  done
}

run_sac_ablation_walker
run_td3_ablation_walker
run_dr_halfcheetah
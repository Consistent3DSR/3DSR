#!/bin/bash

gpu=0
factors=2
scene="bicycle"
num_inference_steps=50  # Make sure this is defined correctly

exp_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting/outputs/mip-splatting-multiresolution/load_DS_$((factors * 4))/train_SD_SR_proposed_DS_${factors}_NO_GS_w_img_encode_timesteps_total_${num_inference_steps}_steps_1110_debug_no_scaling"
output_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting/outputs/my_data/original_setting/input_DS_$((factors * 4))"

export NUM_ITERS=50;
export CONSEC_TIMESTEPS=1;

# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_proposed_stable_sr.py \
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_proposed_2025.py \
    -s /fs/nexus-projects/dyn3Dscene/Codes/data/SR/SD/${scene} \
    -m ${output_dir}/${scene} \
    -r $factors \
    --port $((6009 + gpu)) \
    --eval \
    --output_folder ${exp_dir} \
    --load_pretrain \
    --kernel_size 0.1 \
    --config /fs/nexus-projects/dyn3Dscene/Codes/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml \
   --ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/stablesr_turbo.ckpt \
   --init-img ./inputs/test/ \
   --outdir ./outputs_0207/ \
   --ddpm_steps 4 \
   --dec_w 0.5 \
   --seed 42 \
   --n_samples 1 \
   --vqgan_ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/vqgan_cfw_00011.ckpt \
   --colorfix_type wavelet \
   --upscale 4 \
    --train_tiny
    
    
    
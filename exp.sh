#!/bin/bash

gpu=0
factors=2

# Scenes to process
scenes=("fern")  # Example: "fern" "flower" "fortress" "horns" "orchids" "room" "trex"

# Dataset paths
dataset="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data"
dataset_gt="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data"

# Output path
output_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/llff/orig/input_DS_$((factors * 4))"

# Learning rate weights to loop through
lr_weights=("1.3" "1.5" "2")

# Number of iterations
export NUM_ITERS=5000

# Loop through each scene
for scene in "${scenes[@]}"; do
    for lr_weight in "${lr_weights[@]}"; do
        echo "------------------------------------------------------"
        echo "🚀 Processing scene: $scene with learning rate weight: ${lr_weight}"
        echo "------------------------------------------------------"

        # Experiment variation
        variation="fidelity_wt_1_lr_wt_${lr_weight}_iter_${NUM_ITERS}_0306"

        # Experiment directory (inside loop to change with each iteration)
        exp_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/llff/proposed/load_DS_$((factors * 4))/train_proposed_DS_$((factors))_${variation}"

        # Training
        echo "🛠️  Starting Training: $scene with lr weight: ${lr_weight} ..."
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_proposed_2025.py \
            -s ${dataset}/${scene} \
            -m ${output_dir}/${scene} \
            -r $factors \
            --port $((6000 + gpu)) \
            --eval \
            --output_folder ${exp_dir} \
            --load_pretrain \
            --kernel_size 0.1 \
            --config /fs/nexus-projects/dyn3Dscene/Codes/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml \
            --ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/stablesr_turbo.ckpt \
            --init-img ${dataset}/${scene}/images_$((factors * 4)) \
            --outdir ${exp_dir}/${scene}/outputs_proposed_load_DS_$((factors * 4))/ \
            --ddpm_steps 4 \
            --dec_w 0.5 \
            --seed 42 \
            --n_samples 1 \
            --vqgan_ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/vqgan_cfw_00011.ckpt \
            --colorfix_type wavelet \
            --upscale 4 \
            --fidelity_train \
            --wt_lr ${lr_weight} \
            --densify_end $((NUM_ITERS / 2))

        # Rendering at multiple iterations
        for iteration in 5000 10000 15000 20000 35000 37000; do
            echo "🎨 Rendering scene: $scene at iteration ${iteration}"
            OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py \
                -m ${exp_dir}/${scene} -r $((factors)) --data_device cpu --skip_train --iteration ${iteration}
        done

        # Evaluation
        echo "📊 Evaluating scene: $scene"
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python metrics.py \
            -m ${exp_dir}/${scene} -g ${dataset_gt}/${scene}

        echo "✅ Completed processing for scene: $scene with learning rate weight: ${lr_weight}"
        echo "------------------------------------------------------"
    done  # End of lr_weights loop
done  # End of scenes loop
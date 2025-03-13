#!/bin/bash

gpu=0
factors=2
# scenes=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
# scenes=("room" "stump" "treehill")
scenes=("kitchen")

# scenes=("bonsai" "counter" "flowers" "garden" "kitchen" "stump" "treehill")  # List of scenes

# Loaded pretrained path
output_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_$((factors * 4))"

# Experiment variation
variation="fidelity_wt_1_iter_5000_stop_densify_2500_0304"

# Proposed method directory
exp_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/mip-splatting-multiresolution/load_DS_$((factors * 4))/train_proposed_DS_$((factors))_${variation}"

# Dataset paths
dataset="/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"
dataset_gt="/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"

export NUM_ITERS=5000

# Loop through each scene
for scene in "${scenes[@]}"; do
    echo "------------------------------------------------------"
    echo "🚀 Processing scene: $scene"
    echo "📂 Experiment directory: ${exp_dir}/${scene}"
    echo "------------------------------------------------------"

    # Training
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_proposed_2025.py \
        -s ${dataset}/${scene} \
        -m ${output_dir}/${scene} \
        -r $factors \
        --port $((6009 + gpu)) \
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
        --wt_lr 1 \
        --densify_end 2500

    # Rendering
    echo "🎨 Rendering scene: $scene"
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py -m ${exp_dir}/${scene} -r $((factors)) --data_device cpu --skip_train --iteration 20000
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py -m ${exp_dir}/${scene} -r $((factors)) --data_device cpu --skip_train

    # Evaluation
    echo "📊 Evaluating scene: $scene"
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python metrics.py -m ${exp_dir}/${scene} -g ${dataset_gt}/${scene}

    echo "✅ Completed processing for scene: $scene"
    echo "------------------------------------------------------"
done
#!/bin/bash

gpu=0
factors=2
scenes=(
   "bicycle"
#    "flowers"
#    "garden"
#    "stump"
#    "treehill"
#    "room"
#    "counter"
#    "kitchen"
#    "bonsai"
)
# Scenes to process
# scenes=(
#     # "DrJohnson"
#     # "Aqua"
#     # "Bedroom"
#     # "Boats"
#     # "Bridge"
#     # "CreepyAttic"
#     # "Hugo-1"
#     # "Library"
#     # "Museum-1"
#     # "Museum-2"
#     # "NightSnow"
#     # "Playroom"
#     # "Ponche"
#     # "SaintAnne"
#     "Shed"
#     # "Street-10"
#     # "Tree-18" 
#     # "Yellowhouse-12"
#     )  # Example: 
# scenes=(
#     # "fern" 
#     # "flower" 
#     # "fortress" 
#     "horns" 
#     # "leaves"
#     # "orchids" 
#     # "room" 
#     # "trex"
# )

# scenes=(
#     # "Family"
#     "Horse"    
#     "M60"
#     "Playground"
#     "Train"
#     "Truck"
#     "Lighthouse"
# )

# Dataset paths
dataset_name="mipnerf360"
# dataset_name="nerf_llff_data"
# dataset_name="deep_blending"
# dataset_name="TanksAndTempleBG"
# dataset="/fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}"
# dataset_gt="/fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}"
dataset="/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"
dataset_gt="/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"

# Pretrained model output path
# dataset_name="llff"
output_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/orig/input_DS_$((factors * 4))"
# output_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting/input_DS_$((factors * 4))"

# Learning rate weights to loop through
# lr_weights=("1.3" "1.5" "2")
lr_weights=("1")

# Number of iterations
export NUM_ITERS=5000

# Loop through each scene
for scene in "${scenes[@]}"; do
    for lr_weight in "${lr_weights[@]}"; do
        echo "------------------------------------------------------"
        echo "🚀 Processing scene: $scene with learning rate weight: ${lr_weight}"
        echo "------------------------------------------------------"

        # Render LR model to x4 resolution for training views
        # OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py -m ${output_dir}/${scene} -r ${factors} --data_device cpu --skip_test

        # # Experiment variation
        variation="iteration_8000"

        # Experiment directory (inside loop to change with each iteration)
        # exp_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/load_DS_$((factors * 4))/train_proposed_DS_$((factors))_${variation}"
        # exp_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/proposed/no_pretrain/train_proposed_DS_$((factors))_${variation}"
        exp_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/ours/mipnerf360/load_DS_8/train_proposed_DS_2_1015"

        # Training - no pretrained
        echo "🛠️  Starting Training: $scene with lr weight: ${lr_weight} ..."
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_proposed_2025.py \
            -s ${dataset}/${scene} \
            -m ${exp_dir}/${scene} \
            -r $factors \
            --port $((6000 + gpu)) \
            --eval \
            --output_folder ${exp_dir} \
            --kernel_size 0.1 \
            --config /fs/nexus-projects/dyn3Dscene/Codes/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml \
            --ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/stablesr_turbo.ckpt \
            --init-img ${dataset}/${scene}/images_$((factors * 4)) \
            --outdir ${exp_dir}/${scene} \
            --ddpm_steps 4 \
            --dec_w 0.5 \
            --seed 42 \
            --n_samples 2 \
            --vqgan_ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/vqgan_cfw_00011.ckpt \
            --colorfix_type wavelet \
            --upscale 4 \
            --fidelity_train \
            --wt_lr ${lr_weight} \
            --densify_end $((NUM_ITERS / 2))

        # Training -- load pretrain
        # echo "🛠️  Starting Training: $scene with lr weight: ${lr_weight} ..."
        # OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_proposed_2025.py \
        #     -s ${dataset}/${scene} \
        #     -m ${output_dir}/${scene} \
        #     -r $factors \
        #     --port $((6000 + gpu)) \
        #     --eval \
        #     --output_folder ${exp_dir} \
        #     --load_pretrain \
        #     --kernel_size 0.1 \
        #     --config /fs/nexus-projects/dyn3Dscene/Codes/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml \
        #     --ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/stablesr_turbo.ckpt \
        #     --init-img ${dataset}/${scene}/images_$((factors * 4)) \
        #     --outdir ${exp_dir}/${scene}/outputs_proposed_load_DS_$((factors * 4))/ \
        #     --ddpm_steps 4 \
        #     --dec_w 0.5 \
        #     --seed 42 \
        #     --n_samples 2 \
        #     --vqgan_ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/vqgan_cfw_00011.ckpt \
        #     --colorfix_type wavelet \
        #     --upscale 4 \
        #     --fidelity_train \
        #     --wt_lr ${lr_weight} \
        #     --densify_end $((NUM_ITERS / 2))

        # Rendering at multiple iterations
        # for iteration in 5000 10000 15000 20000; do
        for iteration in 30000; do
            echo "🎨 Rendering scene: $scene at iteration ${iteration}"
            OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py \
                -m ${exp_dir}/${scene} -r $((factors)) --data_device cpu --skip_train --iteration ${iteration}
            
            OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py \
                -m ${exp_dir}/${scene} -r $((factors)) --data_device cpu --skip_train --iteration ${iteration} --vis_video
        done

        # Evaluation
        echo "📊 Evaluating scene: $scene"
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python metrics.py \
            -m ${exp_dir}/${scene} -g ${dataset_gt}/${scene}

        echo "✅ Completed processing for scene: $scene with learning rate weight: ${lr_weight}"

        # # Render interpolated frames: Ours
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path  "$exp_dir/$scene" --video --iteration 15000

        echo "✅ Finished rendering | method: ours | scene: $scene with DS_scale: $DS_scale"
        echo "------------------------------------------------------"
    done  # End of lr_weights loop
done  # End of scenes loop
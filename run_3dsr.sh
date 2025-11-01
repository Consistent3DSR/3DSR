#!/bin/bash
######################################################################
# User-configurable parameters
######################################################################
dataset_name="mipnerf360" #choose from [mipnerf360, llff]
# dataset_path="/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"
dataset_path="path/to/your/dataset"
# GPU ID
gpu=0
# HR resolution downscale factor
HR_factor=4
# Number of GS training iterations for each diffusion step
GS_iters=1000
# Pretrained LR model path
output_dir="./outputs/LR_pretrained/input_DS_$((HR_factor * 4))"
# Define 3DSR experiment directory    
exp_dir="./outputs/${dataset_name}/load_DS_$((HR_factor * 4))"
######################################################################

# Mip-NeRF 360 scenes
scenes=(
   "bicycle"
   "flowers"
   "garden"
   "stump"
   "treehill"
   "room"
   "counter"
   "kitchen"
   "bonsai"
)

# LLFF scenes
# scenes=(
#     "fern" 
#     "flower" 
#     "fortress" 
#     "horns" 
#     "leaves"
#     "orchids" 
#     "room" 
#     "trex"
# )

# Loop through each scene
for scene in "${scenes[@]}"; do    
    #################
    # Train LR model
    #################
    export NUM_ITERS=30000
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_3dsr.py \
        -s ${dataset_path}/${scene} \
        -m ${output_dir}/${scene} \
        --eval \
        -r $((HR_factor * 4)) \
        --port $((6000 + gpu)) \
        --kernel_size 0.1 \
        --output_folder ${output_dir}/${scene} \
        --original

    # Render test set
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py \
        -m ${output_dir}/${scene} \
        -r $((HR_factor * 4)) \
        --data_device cpu \
        --skip_train

    #################
    # Training 3DSR
    #################    
    echo "🛠️  Starting Training: $scene ..."
    export NUM_ITERS=${GS_iters}
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_3dsr.py \
        -s ${dataset_path}/${scene} \
        -m ${output_dir}/${scene} \
        -r $HR_factor \
        --port $((6000 + gpu)) \
        --eval \
        --output_folder ${exp_dir} \
        --load_pretrain \
        --kernel_size 0.1 \
        --config ./configs/stableSRNew/v2-finetune_text_T_512.yaml \
        --ckpt ./third_parties/weights/stablesr_turbo.ckpt \
        --init-img ${dataset_path}/${scene}/images_$((HR_factor * 4)) \
        --outdir ${exp_dir}/${scene} \
        --ddpm_steps 4 \
        --dec_w 0.5 \
        --seed 42 \
        --n_samples 2 \
        --vqgan_ckpt ./third_parties/weights/vqgan_cfw_00011.ckpt \
        --colorfix_type wavelet \
        --upscale 4 \
        --fidelity_train_en \
        --wt_lr 1 \
        --densify_end $((NUM_ITERS / 2))
    
    #################
    # Rendering 
    #################
    echo "🎨 Rendering scene: $scene at iteration ${iteration}"
    # Render test set
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py \
        -m ${exp_dir}/${scene} \
        -r $((HR_factor)) \
        --data_device cpu \
        --skip_train
    # Render video
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py \
        -m ${exp_dir}/${scene} \
        -r $((HR_factor)) \
        --data_device cpu \
        --skip_train \
        --vis_video

    #################
    # Evaluation
    #################
    echo "📊 Evaluating scene: $scene"
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python metrics.py \
        -m ${exp_dir}/${scene} -g ${dataset_path}/${scene}

    echo "✅ Completed processing for scene: $scene with learning rate weight: ${lr_weight}"
    echo "✅ Finished rendering | method: ours | scene: $scene with DS_scale: $DS_scale"
    echo "------------------------------------------------------"    
done
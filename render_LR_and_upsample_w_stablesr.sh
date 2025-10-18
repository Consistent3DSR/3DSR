#!/bin/bash

# Downsampling scales
DS_scales=(4)

# List of scenes
scenes=(
    # "Aqua"
    "Bedroom"
    "Boats"
    "Bridge"
    "CreepyAttic"
    "Hugo-1"
    "Library"
    "Museum-1"
    "Museum-2"
    "NightSnow"
    "Playroom"
    "Ponche"
    "SaintAnne"
    "Shed"    
    "Tree-18"
    "Yellowhouse-12"
    "Street-10"
)
# scenes=(
#     bicycle
#     bonsai
#     counter     
#     flowers    
#     garden   
#     kitchen  
#     room
#     stump
#     treehill
# )
# scenes=(
#     # fern
#     # flower
#     # fortress
#     trex
#     # horns
#     # leaves
#     # orchids
#     # room
# )
dataset_name="deep_blending"
# Loop through each scene and DS_scale
for scene in "${scenes[@]}"; do
    for DS_scale in "${DS_scales[@]}"; do

        echo "🚀 Processing scene: $scene with DS_scale: $DS_scale"

        # echo "🎨 Rendering LR model: $scene"
        base_path=/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/orig
        # # Render the LR model
        /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
            --model_path "$base_path/input_DS_${DS_scale}/$scene" --video

        # # Define output directory
        outdir="/fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/${dataset_name}/${scene}/video/input_DS_${DS_scale}_x4"

        # Ensure output directory exists
        mkdir -p "$outdir"

        # Run Python script
        python /fs/nexus-projects/dyn3Dscene/Codes/StableSR/scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
            --config /fs/nexus-projects/dyn3Dscene/Codes/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml \
            --ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/stablesr_turbo.ckpt \
            --init-img "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/${dataset_name}/orig/input_DS_${DS_scales}/${scene}/rendering/DS_${DS_scales}/ours" \
            --outdir "$outdir" \
            --ddpm_steps 4 \
            --dec_w 0.5 \
            --seed 42 \
            --n_samples 1 \
            --vqgan_ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/vqgan_cfw_00011.ckpt \
            --colorfix_type wavelet \
            --upscale 4
        echo "------------------------------------------------------"

        python /fs/nexus-projects/dyn3Dscene/Codes/utils/write_video.py \
            --input_folder ${outdir} \
            --output_file ${outdir}/video.mp4 \
            --fps 15
    done
done

echo "🎉 All scenes processed successfully!"

#     fern
#     flower
#     fortress
# #     trex
#     horns
#     leaves
#     orchids
#     room
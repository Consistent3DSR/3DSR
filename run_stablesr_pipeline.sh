#!/bin/bash

# Define scenes, sets, and downsampling factors
# scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
# scenes=(
    
#     "Aqua"
#     "Bedroom"
#     "Boats"
#     "Bridge"
#     "CreepyAttic"
#     "DrJohnson"
#     "Hugo-1"
#     "Library"
#     "Museum-1"
#     "Museum-2"
#     "NightSnow"
#     "Playroom"
#     "Ponche"
#     "SaintAnne"
#     "Shed"
#     "Street-10"
#     "Tree-18"
# )
# scenes=(
#     # "Library"
#     # "Shed"
#     # "Street-10"
#     # "Yellowhouse-12"
# )
# scenes=(
#     "Family"
#     "Horse"    
#     "M60"
#     "Playground"
#     "Train"
#     "Truck"
#     "Lighthouse"
# )
scenes=(
    "bicycle"
    # "bonsai"
    # "counter"
    # "flowers"
    # "garden"
    # "kitchen"
    # "room"
    # "stump"
    # "treehill"
)
factors=(8)  # Different downsampling factors
dataset_name="mipnerf360"
# dataset_name="deep_blending"
# dataset_name="TanksAndTempleBG"
# dataset_folder="/fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}"
dataset_folder="/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize"

# Loop through each scene
for scene in "${scenes[@]}"; do
    echo "test!!!"
    for factor in "${factors[@]}"; do
        echo "🚀 Processing scene: ${scene}, set: ${set}, downsampling factor: ${factor}"

        # Input image folder
        img_folder="${dataset_folder}/${scene}/images_${factor}"

        # Define output directory
        # outdir="/fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/consistent_noise/${dataset_name}/${scene}/images"
        outdir="/fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/${dataset_name}/${scene}/images_$((factors/4))"

        # Ensure output directory exists
        mkdir -p "$outdir"

        # Run Python script
        python /fs/nexus-projects/dyn3Dscene/Codes/StableSR/scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
            --config /fs/nexus-projects/dyn3Dscene/Codes/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml \
            --ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/stablesr_turbo.ckpt \
            --init-img "${img_folder}" \
            --outdir "$outdir" \
            --ddpm_steps 4 \
            --dec_w 0.5 \
            --seed 42 \
            --n_samples 1 \
            --vqgan_ckpt /fs/nexus-projects/dyn3Dscene/Codes/StableSR/weights/vqgan_cfw_00011.ckpt \
            --colorfix_type wavelet \
            --upscale 4 \
            --consistent_noise

        echo "✅ Completed: ${scene}, set: ${set}, factor: ${factor}"
    
    done
    
    # cp -r "${dataset_folder}/${scene}/sparse" "/fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/consistent_noise/${dataset_name}/${scene}"
    # cp -r "${dataset_folder}/${scene}/sparse" "/fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/${dataset_name}/${scene}"

    # mkdir -p "${dataset_folder}/${scene}/sparse/0"
    # mv ${dataset_folder}/${scene}/sparse/*.txt ${dataset_folder}/${scene}/sparse/0

done

echo "All scenes processed successfully!"
echo "=============================================================================="
echo "Start training mipnerf360 with scene ${scene} and factor ${factor} for StableSR!"

# sh run_train_orig.sh
python scripts/run_mipnerf360_stablesr.py
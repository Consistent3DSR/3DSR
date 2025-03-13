#!/bin/bash

# Downsampling scales
DS_scales=(16 4)

# List of scenes
scenes=(
    bicycle
    bonsai
    counter     
    flowers    
    garden   
    kitchen  
    room
    stump
    treehill      
)

# Loop through each scene and DS_scale
for scene in "${scenes[@]}"; do
    for DS_scale in "${DS_scales[@]}"; do

        echo "🚀 Processing scene: $scene with DS_scale: $DS_scale"

        # StableSR
        base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting"
        
        echo "🎨 Rendering StableSR: $scene"
        /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
            --model_path "$base_path/input_DS_$((DS_scale / 4))/$scene" --video_yt

        # Proposed method base path
        base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/mip-splatting-multiresolution"

        # Set variation based on DS_scale
        if [[ $DS_scale -eq 16 ]]; then
            variation="train_proposed_DS_4_fidelity_wt_1_iter_5000_stop_densify_2500_0304"
        else
            variation="train_proposed_DS_2_full_dataset_fidelity_wt_1_iter_5000_0303"
        fi

        echo "🎨 Rendering Proposed: $scene"
        /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
            --model_path "$base_path/load_DS_$DS_scale/$variation/$scene" --video_yt

        echo "✅ Finished rendering $scene with DS_scale: $DS_scale"
        echo "------------------------------------------------------"

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
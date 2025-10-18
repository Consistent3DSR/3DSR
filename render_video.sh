#!/bin/bash

# Downsampling scales
DS_scales=(4)

# List of scenes
scenes=(
    # "Aqua"
    # "Bedroom"
    # "Boats"
    # "Bridge"
    # "CreepyAttic"
    # "Hugo-1"
    # "Library"
    # "Museum-1"
    # "Museum-2"
    # "NightSnow"
    "Playroom"
    # "Ponche"
    # "SaintAnne"
    # "Shed"
    # "Street-10"
    # "Tree-18"
    # "Yellowhouse-12"
)
# scenes=(
#     # bicycle
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

# Loop through each scene and DS_scale
for scene in "${scenes[@]}"; do
    for DS_scale in "${DS_scales[@]}"; do

        echo "🚀 Processing scene: $scene with DS_scale: $DS_scale"

        # StableSR
        # base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/llff/StableSR"
                    
        
        # echo "🎨 Rendering StableSR: $scene"
        # # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        # #     --model_path "$base_path/input_DS_$((DS_scale / 4))/$scene" --video_yt
        
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path "$base_path/input_DS_$((DS_scale / 4))/$scene" --vis_video

        # Proposed method base path
        # base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/mip-splatting-multiresolution"
        base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/deep_blending/proposed"
        # base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/llff/proposed"
        # Set variation based on DS_scale
        # if [[ $DS_scale -eq 16 ]]; then
        #     variation="train_proposed_DS_4_full_dataset_fidelity_wt_1_iter_5000_0303"
        #     # train_proposed_DS_4_fidelity_wt_1_iter_5000_stop_densify_2500_0304"
        # else
        #     variation="train_proposed_DS_2_full_dataset_fidelity_wt_1_iter_5000_0303"
        # fi
        variation="train_proposed_DS_1_fidelity_wt_1_lr_wt_1_iter_5000_0306_0427"

        echo "🎨 Rendering Proposed: $scene"
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path "$base_path/load_DS_$DS_scale/$variation/$scene" --video_yt
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path "$base_path/load_DS_$DS_scale/$variation/$scene" --vis_video
        
        
        # # Render interpolated frames: Ours
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path  "$base_path/load_DS_$DS_scale/$variation/$scene" --video

        # echo "✅ Finished rendering | method: ours | scene: $scene with DS_scale: $DS_scale"


        # Render interpolated frames: proposed
        variation="train_proposed_DS_1_fidelity_wt_1_lr_wt_1_iter_5000_0306_0427"
        base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/deep_blending/proposed"        
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path  "$base_path/load_DS_$DS_scale/$variation/$scene" --vis_video --iteration 15000

        # echo "✅ Finished rendering | method: GT | scene: $scene with DS_scale: $DS_scale"

        # Render interpolated frames: Ours
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path  "$base_path/load_DS_$DS_scale/$variation/$scene" --video --iteration 15000

        # echo "✅ Finished rendering | method: ours | scene: $scene with DS_scale: $DS_scale"

        # Render interpolated frames: StableSR
        base_path=/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/deep_blending/StableSR
        /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
            --model_path  "$base_path/input_DS_$((DS_scale / 4))/$scene" --vis_video
        
        # Render smooth video: StableSR
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path  "$base_path/input_DS_$((DS_scale / 4))/$scene" --vis_video
        # echo "✅ Finished rendering | method: StableSR | scene: $scene with DS_scale: $DS_scale"
        
        
        # # # LR base path
        # # base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/my_data/new_resize/original_setting"
        # base_path="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/deep_blending/orig_new"

        # echo "🎨 Rendering original resolution: $scene"
        # # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        # #     --model_path "$base_path/load_DS_$DS_scale/$variation/$scene" --video_yt
        # # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        # #     --model_path "$base_path/input_DS_$DS_scale/$scene" --vis_video
        
        # # Render interpolated frames
        # /fs/nexus-scratch/ytchen/miniconda3/envs/msss/bin/python render.py \
        #     --model_path "$base_path/input_DS_$DS_scale/$scene" --interpolate

        # echo "✅ Finished rendering original $scene with DS_scale: $DS_scale"
        
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
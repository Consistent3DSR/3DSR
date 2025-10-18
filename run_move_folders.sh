# scenes=(
# #     # Aqua
# #     # Bedroom
# #     # Boats
# #     # Bridge
# #     # CreepyAttic
# #     # Hugo-1
# #     # Library
# #     # Museum-1
# #     # Museum-2
# #     # NightSnow
# #     # Playroom
# #     # Ponche
# #     # SaintAnne
# #     Shed
# #     # Street-10
# #     # Tree-18
#     #   DrJohnson
#     # room
# )

# scenes=(
#     # bicycle
#     # bonsai
#     # counter
#     # flowers
#     # garden
#     # kitchen
#     # room
#     stump
#     treehill
# )
scenes=(
    # fern
    # flower
    # fortress    
    # leaves
    # orchids
    # horns
    room
    trex
)
# scenes=(
#     "Family"
#     "Horse"
#     "Lighthouse"
#     "M60"
#     "Playground"
#     "Train"
#     "Truck"
# )

dataset_name="nerf_llff_data"
# dataset_name="deep_blending"
factor=2
base_dir="/fs/nexus-projects/dyn3Dscene/Codes/DiSR-NeRF/outputs/LLFF_rm_val_pose_0515"
for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"
    # For SRGS
    base_path=/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/llff/proposed/load_DS_8/train_proposed_DS_2_fidelity_wt_1_iter_5000_stop_densify_2500_0305/${scene}/ours_37000/DS_2/gt_2

    mkdir -p images
    mv "${base_path}"/DJI*.png images

    # For DiSR-NeRF
    # cur_dir=${base_dir}_${scene}/save/sr_val
    # echo "${cur_dir}/results"

    # rm -r ${cur_dir}/results
    # mkdir -p ${cur_dir}/results
    # find "${cur_dir}" -maxdepth 1 -type f -name "it10000_*.png" ! -name "*_gt.png" -exec cp {} "${cur_dir}/results/" \;
    
    # # # Get all matching files sorted by name
    # folder=${cur_dir}/results
    # files=($(ls "${folder}"/it10000_*.png | sort))
    # count=0

    # for f in "${files[@]}"; do
    #     new_name=$(printf "%05d.png" "$count")
    #     mv "$f" "${folder}/${new_name}"
    #     ((count++))
    # done

    # # Cpopy colmap sparse to SR_results
    # cp -r /fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data/${scene}/sparse /fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/SwinIR/${dataset_name}/${scene}
    # echo "✅ Copied colmap sparse to SR_results for scene: $scene"
    
    # # Move SR images to folder: images
    # mkdir -p /fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/SwinIR/${dataset_name}/${scene}/images_${factor}
    # mv /fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/SwinIR/${dataset_name}/${scene}/*.png /fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/SwinIR/${dataset_name}/${scene}/images_${factor}

    # Resize images to 4 divisible size
    # mv /fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data/${scene}/images /fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data/${scene}/images_orig
    # python /fs/nexus-projects/dyn3Dscene/Codes/utils/image_resize.py \
    #     --input_path /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images_orig \
    #     --output_folder /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images \
    #     --ratio 1 
    #     # --new_w 1332 \
    #     # --new_h 876

    # echo "✅ Completed resizing for scene: $scene to 4 divisible size"
    echo "--------------------------------------"

    # echo "Processing scene: $scene"

    # python /fs/nexus-projects/dyn3Dscene/Codes/utils/image_resize.py \
    #     --input_path /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images \
    #     --output_folder /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images_4 \
    #     --ratio 0.25 

    # echo "✅ Completed resizing for scene: $scene to x4 smaller"
    # echo "--------------------------------------"
done

# ===========================================================================
# dataset_name="deep_blending"
# for scene in ${scenes[@]}; do
#     echo "Processing scene: $scene"
#     # mv /fs/nexus-projects/dyn3Dscene/Codes/datasets/deep_blending/${scene}/images  /fs/nexus-projects/dyn3Dscene/Codes/datasets/deep_blending/${scene}/images_orig
#     # # rm /fs/nexus-projects/dyn3Dscene/Codes/datasets/deep_blending/${scene}/images_4/*.jpg
#     # # mkdir -p /fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/deep_blending/${scene}/images

#     # python /fs/nexus-projects/dyn3Dscene/Codes/utils/image_resize.py \
#     #     --input_path /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images_orig \
#     #     --output_folder /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images \
#     #     --ratio 1 \
#     #     --new_w 3672 \
#     #     --new_h 1968
    
#     # echo "✅ Completed resizing for scene: $scene to 4 divisible size"
#     # echo "--------------------------------------"

#     # echo "Processing scene: $scene"

#     # python /fs/nexus-projects/dyn3Dscene/Codes/utils/image_resize.py \
#     #     --input_path /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images \
#     #     --output_folder /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/images_4 \
#     #     --ratio 0.25 
#     # mv /fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/deep_blending/${scene}/*.png /fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/deep_blending/${scene}/images
#     cp -r /fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}/${scene}/sparse /fs/nexus-projects/dyn3Dscene/Codes/datasets/SR_results/StableSR/${dataset_name}/${scene}
        
# done

scenes=(
    # "fern" 
    # "flower" 
    # "fortress" 
    # "leaves"
    # "horns" 
    # "orchids" 
    "room" 
    # "trex"
    # "bicycle"
    # "bonsai"
    # "counter"
    # "flowers"
    # "garden"
    # "kitchen"
    # "room"
    # "stump"
    # "treehill"
)
gpu=0
factor=8

dataset_name="nerf_llff_data"
dataset="/fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}"
dataset_gt="/fs/nexus-projects/dyn3Dscene/Codes/datasets/${dataset_name}"
dataset_name="llff"
exp_dir="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/llff/orig/input_DS_8"
# exp_dir="/fs/nexus-projects/SR3D/Results/outputs/mip-splatting-multiresolution/load_DS_8/train_proposed_DS_2_full_dataset_fidelity_wt_1_iter_5000_0303"
# exp_dir="/fs/nexus-projects/SR3D/Results/outputs/mip-splatting-multiresolution/load_DS_8/train_proposed_DS_2_fidelity_wt_1_iter_5000_stop_densify_2500_0304"

for scene in "${scenes[@]}"; do
    # for iteration in 5000 10000 15000 20000 35000 37000; do
    for iteration in 30000; do
        echo "🎨 Rendering scene: $scene at iteration ${iteration}"
        OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py \
            -m ${exp_dir}/${scene} -r $((factor)) --data_device cpu --skip_train --iteration ${iteration} --vis_video
        # OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python render.py \
        #     -m ${exp_dir}/${scene} -r $((factor)) --data_device cpu --skip_train --iteration ${iteration}
    done
    echo "✅ Completed rendering for scene: $scene"

    # echo "📊 Evaluating scene: $scene"
    # OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${gpu} python metrics.py \
    #     -m ${exp_dir}/${scene} -g ${dataset_gt}/${scene}

    # echo "✅ Completed evaluation for scene: $scene"

done
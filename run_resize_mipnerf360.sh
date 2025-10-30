# Dataset path settings
input_dir=/fs/nexus-projects/dyn3Dscene/Codes/data/orig
output_dir=/fs/nexus-projects/dyn3Dscene/Codes/datasets/mipnerf360_test_new

# mipnerf360
scenes=(
#    "bicycle"
   "bonsai"
#    "counter"
#    "flowers"
#    "garden"
#    "kitchen"
#    "room"
#    "stump"
#    "treehill"
)

for scene in "${scenes[@]}"; do
    python utils/resize_images.py \
        --input_dir ${input_dir}/${scene} \
        --output_dir ${output_dir}/${scene}

    cp -r ${input_dir}/${scene}/sparse ${output_dir}/${scene}/sparse
    cp ${input_dir}/${scene}/poses_bounds.npy ${output_dir}/${scene}/poses_bounds.npy
done
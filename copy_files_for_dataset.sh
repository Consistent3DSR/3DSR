#!/bin/bash

# List of scenes
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# Base directories
src_base="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data"
dst_base="/fs/nexus-projects/dyn3Dscene/Codes/datasets/llff/StableSR"

# Files/Folders to copy
files=("sparse" "database.db" "poses_bounds.npy" "simplices.npy")

# Loop through each scene
for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"

    # Create destination folder if it doesn't exist
    mkdir -p "${dst_base}/${scene}"

    # Copy the "images" folder
    src_images="${src_base}/${scene}/images"
    dst_images="${dst_base}/${scene}/images"

    if [ -d "$src_images" ]; then
        echo "Copying images folder to ${dst_images}/"
        cp -r "$src_images" "$dst_images"
    else
        echo "⚠️ Warning: images folder does not exist in $src_images"
    fi

    # Loop through each file/folder to copy
    for file in "${files[@]}"; do
        src_path="${src_base}/${scene}/${file}"
        dst_path="${dst_base}/${scene}/${file}"

        if [ -e "$src_path" ]; then
            echo "Copying $file to ${dst_base}/${scene}/"
            cp -r "$src_path" "$dst_path"
        else
            echo "⚠️ Warning: $file does not exist in $src_path"
        fi
    done

    echo "✅ Completed copying for scene: $scene"
    echo "--------------------------------------"
done
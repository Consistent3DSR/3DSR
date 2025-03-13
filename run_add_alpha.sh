#!/bin/bash

# Define scenes
scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")  # List of scenes

# Define base directories
folder1_base="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_synthetic/original"  # Folder with alpha images
folder2_base="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_synthetic/StableSR/resized_4"  # Folder with images to modify
output_base="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_synthetic/StableSR/resized_4"  # Output folder

# Loop through each scene
for scene in "${scenes[@]}"; do
    folder1="${folder1_base}/${scene}/train"
    folder2="${folder2_base}/${scene}/train"
    output_folder="${output_base}/${scene}/train"

    # Check if both folders exist before processing
    if [ -d "$folder1" ] && [ -d "$folder2" ]; then
        echo "🚀 Processing scene: $scene"
        python change_rgba.py --folder1 "$folder1" --folder2 "$folder2" --output_folder "$output_folder"
        echo "✅ Completed: $scene"
    else
        echo "⚠️ Skipping scene $scene (one or both folders do not exist)"
    fi
done

echo "🎉 All scenes processed successfully!"
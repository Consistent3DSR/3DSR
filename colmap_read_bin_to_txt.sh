#!/bin/bash

# Define the list of scenes to process
scenes=("fern" "flower" "trex")  # Replace with actual scene names

# Define paths
CONVERT_SCRIPT="colmap_converter.py"  # Python script that does the conversion


# Loop through each scene
for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"
    INPUT_BASE_DIR="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data/new_data/nerf_llff_data/${scene}/sparse/0"   # Base directory containing COLMAP .bin models
    OUTPUT_BASE_DIR="/fs/nexus-projects/dyn3Dscene/Codes/datasets/nerf_llff_data/new_data/nerf_llff_data/${scene}/sparse/0" # Directory to save .txt models

    # Ensure output directory exists
    mkdir -p "$OUTPUT_BASE_DIR"

    # Define input and output paths
    INPUT_MODEL="${INPUT_BASE_DIR}"
    OUTPUT_MODEL="${OUTPUT_BASE_DIR}"

    # Ensure scene output directory exists
    mkdir -p "$OUTPUT_MODEL"

    # Run conversion
    python "$CONVERT_SCRIPT" \
        --input_model "$INPUT_MODEL" \
        --input_format ".bin" \
        --output_model "$OUTPUT_MODEL" \
        --output_format ".txt"

    echo "✅ Conversion completed for scene: $scene"
done

echo "🎉 All scenes processed successfully!"